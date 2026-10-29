from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from project_paths import OUTPUTS_ROOT
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_literature21_ferreira_mmas_fixed_split import (
    DATA_ROOT,
    REFERENCE_POINTS,
    SEEDS,
    SENSORS,
    SKAB_ROOT,
    WORKSPACE,
    data_key,
    data_paths,
    evaluate_predictions,
    json_default,
    load_sensor_frames,
    load_test_labels,
    safe_name,
    sha256,
    write_json,
)

OUTPUT_ROOT = OUTPUTS_ROOT / "literature23_niemi_lstm_pot_fixed_split"
NIEMI_PDF = WORKSPACE / "references" / "niemi_2025.pdf"

for key, value in {
    "TEMP": WORKSPACE / ".tmp",
    "TMP": WORKSPACE / ".tmp",
    "KERAS_HOME": WORKSPACE / ".keras",
    "MPLCONFIGDIR": WORKSPACE / ".matplotlib",
    "PYTHONPYCACHEPREFIX": WORKSPACE / ".pycache",
}.items():
    os.environ[key] = str(value)
    Path(value).mkdir(parents=True, exist_ok=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import genpareto
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NiemiConfig:
    window_size: int = 100
    lstm_units: tuple[int, int] = (80, 80)
    dropout: float = 0.30
    optimizer: str = "adam"
    loss: str = "mse"
    epochs: int = 10
    batch_size: int = 64
    validation_split: float = 0.20
    shuffle: bool = True
    early_stopping_patience: int = 1
    early_stopping_min_delta: float = 0.0
    restore_best_weights: bool = True
    smoothing_fraction: float = 0.05
    pot_initial_level: float = 0.98
    pot_risk: float = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer the best Niemi et al. (2025) configuration, LSTM + "
            "smoothed absolute prediction error + POT, to the fixed SKAB protocol."
        )
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / stem
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{stem}_{suffix:02d}"
        suffix += 1
    for name in (
        "audit",
        "figures",
        "logs",
        "metrics",
        "models",
        "predictions",
        "scores",
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir


def log(path: Path, message: str) -> None:
    line = f"[{pd.Timestamp.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def set_seed(seed: int) -> None:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def make_windows(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= width:
        raise ValueError(f"Need more than {width} values, got {len(values)}")
    x = np.stack([values[i : i + width] for i in range(len(values) - width)])
    y = values[width:]
    return x, y


def make_test_windows(
    reference: np.ndarray, test: np.ndarray, width: int
) -> np.ndarray:
    combined = np.concatenate([reference[-width:], test], axis=0)
    return np.stack([combined[i : i + width] for i in range(len(test))])


def build_lstm(config: NiemiConfig) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(config.window_size, len(SENSORS))),
            tf.keras.layers.LSTM(config.lstm_units[0], return_sequences=True),
            tf.keras.layers.Dropout(config.dropout),
            tf.keras.layers.LSTM(config.lstm_units[1], return_sequences=False),
            tf.keras.layers.Dropout(config.dropout),
            tf.keras.layers.Dense(len(SENSORS), activation="linear"),
        ],
        name="niemi_lstm_predictor",
    )
    model.compile(optimizer=config.optimizer, loss=config.loss)
    return model


def absolute_prediction_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.asarray(actual) - np.asarray(predicted)), axis=1)


def smooth_errors(
    reference_error: np.ndarray,
    test_error: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    span = max(2, int(round(len(reference_error) * fraction)))
    combined = np.concatenate([reference_error, test_error])
    smoothed = (
        pd.Series(combined, dtype=float)
        .ewm(span=span, adjust=False, min_periods=1)
        .mean()
        .to_numpy()
    )
    return smoothed[: len(reference_error)], smoothed[len(reference_error) :], span


def fit_static_pot_threshold(
    reference_scores: np.ndarray,
    level: float,
    risk: float,
) -> dict[str, Any]:
    values = np.asarray(reference_scores, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite scores passed to POT")
    if not 0.0 < level < 1.0 - risk < 1.0:
        raise ValueError("POT requires 0 < level < 1-risk < 1")
    initial_threshold = float(np.quantile(values, level))
    excesses = values[values > initial_threshold] - initial_threshold
    if len(excesses) < 5:
        initial_threshold = float(np.quantile(values, 0.95))
        excesses = values[values > initial_threshold] - initial_threshold
    if len(excesses) < 3:
        raise RuntimeError("Too few reference peaks for POT fitting")

    shape, _, scale = genpareto.fit(excesses, floc=0.0)
    shape = float(shape)
    scale = float(scale)
    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid GPD fit")
    ratio = risk * len(values) / len(excesses)
    if shape == 0.0 or abs(shape) < 1e-8:
        threshold = initial_threshold - scale * np.log(ratio)
    else:
        threshold = initial_threshold + scale / shape * (ratio ** (-shape) - 1.0)
    if not np.isfinite(threshold) or threshold <= initial_threshold:
        threshold = float(np.max(values) + max(np.std(values), 1e-9) * 0.01)
        fit_status = "gpd_fallback_above_training_max"
    else:
        fit_status = "gpd"
    return {
        "threshold": float(threshold),
        "initial_threshold": initial_threshold,
        "tail_count": int(len(excesses)),
        "sample_count": int(len(values)),
        "shape": shape,
        "scale": scale,
        "level": float(level),
        "risk": float(risk),
        "fit_status": fit_status,
    }


def train_and_score_file(
    frame: pd.DataFrame,
    seed: int,
    file_index: int,
    key: str,
    config: NiemiConfig,
    model_dir: Path,
) -> dict[str, Any]:
    reference = frame.iloc[:REFERENCE_POINTS][SENSORS].to_numpy(dtype=np.float32)
    test = frame.iloc[REFERENCE_POINTS:][SENSORS].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(reference)
    reference_scaled = scaler.transform(reference).astype(np.float32)
    test_scaled = scaler.transform(test).astype(np.float32)
    x_reference, y_reference = make_windows(reference_scaled, config.window_size)
    x_test = make_test_windows(reference_scaled, test_scaled, config.window_size)

    model_seed = seed * 1000 + file_index
    set_seed(model_seed)
    model = build_lstm(config)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        restore_best_weights=config.restore_best_weights,
        verbose=0,
    )
    history = model.fit(
        x_reference,
        y_reference,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=config.validation_split,
        shuffle=config.shuffle,
        callbacks=[callback],
        verbose=0,
    )
    reference_prediction = model.predict(x_reference, batch_size=config.batch_size, verbose=0)
    test_prediction = model.predict(x_test, batch_size=config.batch_size, verbose=0)
    reference_error = absolute_prediction_error(y_reference, reference_prediction)
    test_error = absolute_prediction_error(test_scaled, test_prediction)
    reference_smoothed, test_smoothed, smoothing_span = smooth_errors(
        reference_error, test_error, config.smoothing_fraction
    )
    pot = fit_static_pot_threshold(
        reference_smoothed, config.pot_initial_level, config.pot_risk
    )
    alarms = test_smoothed > pot["threshold"]
    changepoints = alarms & ~np.concatenate([[False], alarms[:-1]])

    model_path = model_dir / f"{safe_name(key)}.keras"
    model.save(model_path)
    tf.keras.backend.clear_session()
    return {
        "reference_error": reference_error,
        "reference_smoothed": reference_smoothed,
        "test_error": test_error,
        "test_smoothed": test_smoothed,
        "alarm": alarms.astype(np.int8),
        "changepoint": changepoints.astype(np.int8),
        "pot": pot,
        "smoothing_span": smoothing_span,
        "train_loss": float(history.history["loss"][-1]),
        "validation_loss": float(history.history["val_loss"][-1]),
        "epochs_completed": len(history.history["loss"]),
        "model_path": str(model_path),
    }


def reference_comparison(niemi_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [column for column in niemi_metrics.select_dtypes(include=[np.number]).columns if column != "seed"]
    means = pd.DataFrame([{"method": "Niemi LSTM+EWMA+POT", "aggregation": "five-seed mean", **niemi_metrics[numeric].mean().to_dict()}])
    stds = pd.DataFrame([{"method": "Niemi LSTM+EWMA+POT", **niemi_metrics[numeric].std(ddof=1).to_dict()}])
    return means, stds


def plot_comparison(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        ("point_f1", "Point F1"),
        ("point_far_percent", "Point FAR (%)"),
        ("point_mar_percent", "Point MAR (%)"),
        ("event_tps", "Detected changepoints"),
        ("event_missed_cps", "Missed changepoints"),
        ("event_fps", "Event FP"),
        ("average_detection_delay_seconds", "Mean delay (s)"),
        ("nab_standard", "NAB Standard"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    colors = ["#777777", "#9B59B6", "#2F80ED"]
    for ax, (column, title) in zip(axes.flat, columns):
        values = frame[column].astype(float).to_numpy()
        bars = ax.bar(frame["method"], values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("SKAB fixed 400/401+ split and official evaluation protocol")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, means: pd.DataFrame, stds: pd.DataFrame, run_dir: Path) -> None:
    selected = [
        "point_f1",
        "point_far_percent",
        "point_mar_percent",
        "event_tps",
        "event_missed_cps",
        "event_fps",
        "average_detection_delay_seconds",
        "nab_standard",
    ]
    lines = [
        "# 文献[23] Niemi et al. (2025) 对比实验结论",
        "",
        "## 实验身份",
        "",
        "- 数据：SKAB公开泵循环试验台数据，不是真实火电厂现场数据。",
        "- 固定划分：每个文件第1-400行为训练/参考集，第401行至末行为最终测试集。",
        "- 对比方法：Niemi等论文中总体最优的LSTM + 平滑绝对预测误差 + POT配置。",
        "- 论文明确参数：100点窗口、最多10轮、验证损失上升时早停。",
        "- 原始LSTM默认结构：两层80单元LSTM、dropout 0.3、Adam、MSE、batch size 64、20%验证集。",
        "- 未披露POT参数：采用预先锁定的level=0.98、risk=0.001，不读取标签调参。",
        "- 评价：全部预测先落盘，再调用SKAB官方core/metrics.py统一评价。",
        "",
        "## 本方法结果",
        "",
        "| 方法 | 点级F1 | FAR(%) | MAR(%) | 事件TP | 漏检事件 | 事件FP | 平均延迟(s) | NAB Standard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    std_lookup = stds.set_index("method")
    for _, row in means.iterrows():
        method = str(row["method"])
        std = std_lookup.loc[method]
        cells = []
        for column in selected:
            value = float(row[column])
            deviation = float(std[column])
            if method == "Official Conv-AE":
                cells.append(f"{value:.4f}" if column == "point_f1" else f"{value:.2f}")
            else:
                cells.append(
                    f"{value:.4f}±{deviation:.4f}"
                    if column == "point_f1"
                    else f"{value:.2f}±{deviation:.2f}"
                )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 输出位置", "", f"`{run_dir}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_root.resolve())
    log_path = run_dir / "logs" / "run.log"
    config = NiemiConfig()
    try:
        required = [
            SKAB_ROOT / "core" / "metrics.py",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required files: " + "; ".join(missing))

        log(log_path, "Loading sensor-only SKAB frames; test labels remain unread.")
        paths = data_paths()
        frames = load_sensor_frames(paths)
        keys = list(frames)
        split_rows = []
        for key in keys:
            frame = frames[key]
            split_rows.append(
                {
                    "file": key,
                    "total_rows": len(frame),
                    "reference_rows": REFERENCE_POINTS,
                    "reference_start": frame.index[0],
                    "reference_end": frame.index[REFERENCE_POINTS - 1],
                    "test_rows": len(frame) - REFERENCE_POINTS,
                    "test_start_position": REFERENCE_POINTS + 1,
                    "test_start": frame.index[REFERENCE_POINTS],
                    "test_end": frame.index[-1],
                    "final_test_shuffle": False,
                }
            )
        pd.DataFrame(split_rows).to_csv(
            run_dir / "audit" / "data_split_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )

        lock = {
            "locked_before_training_and_test_label_access": True,
            "paper": {
                "citation": "Niemi et al. (2025), DOI 10.3850/978-981-94-3281-3_ESREL-SRA-E2025-P4996-cd",
                "source_pdf": str(NIEMI_PDF),
                "source_pdf_sha256": sha256(NIEMI_PDF) if NIEMI_PDF.is_file() else None,
            },
            "selected_paper_configuration": "best overall: LSTM + smoothed absolute error + POT",
            "dataset": "SKAB public pump-loop testbed",
            "dataset_is_not": "real thermal-power-plant field data",
            "fixed_split": {
                "training_and_reference": "rows 1-400 of every file",
                "final_test": "rows 401-end of every file",
                "cross_file_split": False,
                "final_test_shuffle": False,
            },
            "seeds": SEEDS,
            "configuration": asdict(config),
            "pot_mode": "static threshold fitted only on smoothed reference errors",
            "pot_parameter_note": (
                "The paper does not disclose POT level/risk; level=0.98 and risk=0.001 "
                "were locked as standard transfer defaults before labels."
            ),
            "predicted_changepoint": "causal 0-to-1 transition of point alarm",
            "evaluation": {
                "module": str(SKAB_ROOT / "core" / "metrics.py"),
                "window_width": "60s",
                "anomaly_window_destination": "righter",
            },
        }
        lock_path = run_dir / "configuration_lock.json"
        write_json(lock_path, lock)
        lock_hash = sha256(lock_path)
        (run_dir / "configuration_lock.sha256").write_text(
            lock_hash + "  configuration_lock.json\n", encoding="ascii"
        )

        all_predictions: dict[int, dict[str, dict[str, pd.Series]]] = {}
        score_manifest: list[dict[str, Any]] = []
        training_rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            log(log_path, f"Starting seed {seed}.")
            seed_models = run_dir / "models" / f"seed{seed}"
            seed_scores = run_dir / "scores" / f"seed{seed}"
            seed_models.mkdir()
            seed_scores.mkdir()
            all_predictions[seed] = {}
            for file_index, key in enumerate(keys):
                result = train_and_score_file(
                    frames[key], seed, file_index, key, config, seed_models
                )
                test_index = frames[key].index[REFERENCE_POINTS:]
                point = pd.Series(result["alarm"], index=test_index, dtype=int)
                cp = pd.Series(result["changepoint"], index=test_index, dtype=int)
                all_predictions[seed][key] = {"point": point, "changepoint": cp}
                score_path = seed_scores / f"{safe_name(key)}.npz"
                np.savez_compressed(
                    score_path,
                    reference_error=result["reference_error"],
                    reference_smoothed=result["reference_smoothed"],
                    test_error=result["test_error"],
                    test_smoothed=result["test_smoothed"],
                    threshold=np.asarray([result["pot"]["threshold"]]),
                    alarm=result["alarm"],
                    changepoint=result["changepoint"],
                )
                score_manifest.append(
                    {
                        "seed": seed,
                        "file": key,
                        "score_path": str(score_path),
                        "score_sha256": sha256(score_path),
                        "model_path": result["model_path"],
                        "model_sha256": sha256(Path(result["model_path"])),
                    }
                )
                training_rows.append(
                    {
                        "seed": seed,
                        "file": key,
                        "epochs_completed": result["epochs_completed"],
                        "train_loss": result["train_loss"],
                        "validation_loss": result["validation_loss"],
                        "smoothing_span": result["smoothing_span"],
                        **{f"pot_{name}": value for name, value in result["pot"].items()},
                        "test_alarm_points": int(result["alarm"].sum()),
                        "test_changepoints": int(result["changepoint"].sum()),
                    }
                )
                log(
                    log_path,
                    f"seed={seed} file={file_index + 1}/34 {key} "
                    f"epochs={result['epochs_completed']} "
                    f"alarms={int(result['alarm'].sum())} "
                    f"CPs={int(result['changepoint'].sum())}",
                )
            prediction_path = run_dir / "predictions" / f"seed{seed}_predictions.pkl"
            with prediction_path.open("wb") as handle:
                pickle.dump(all_predictions[seed], handle, protocol=pickle.HIGHEST_PROTOCOL)
            log(log_path, f"Persisted seed {seed} predictions.")

        pd.DataFrame(score_manifest).to_csv(
            run_dir / "audit" / "score_and_model_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(training_rows).to_csv(
            run_dir / "metrics" / "training_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        prediction_manifest = [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted((run_dir / "predictions").glob("*.pkl"))
        ]
        write_json(
            run_dir / "audit" / "pre_evaluation_integrity.json",
            {
                "configuration_hash_matches": sha256(lock_path) == lock_hash,
                "prediction_files": prediction_manifest,
                "all_five_seed_predictions_persisted": len(prediction_manifest) == 5,
                "labels_loaded": False,
            },
        )

        log(log_path, "All predictions saved. Loading rows 401+ labels for official evaluation.")
        labels = load_test_labels(paths)
        metric_rows: list[dict[str, Any]] = []
        per_file_rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            points = {key: all_predictions[seed][key]["point"] for key in keys}
            cps = {key: all_predictions[seed][key]["changepoint"] for key in keys}
            metrics = evaluate_predictions(keys, labels, points, cps)
            metric_rows.append({"seed": seed, **metrics})
            for key in keys:
                per_file_rows.append(
                    {
                        "seed": seed,
                        "file": key,
                        **evaluate_predictions(
                            [key], labels, {key: points[key]}, {key: cps[key]}
                        ),
                    }
                )
            log(
                log_path,
                f"Evaluated seed={seed}: F1={metrics['point_f1']:.4f}, "
                f"FAR={metrics['point_far_percent']:.3f}%, "
                f"MAR={metrics['point_mar_percent']:.3f}%, eventFP={metrics['event_fps']}",
            )

        metric_frame = pd.DataFrame(metric_rows)
        metric_frame.to_csv(
            run_dir / "metrics" / "five_seed_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(per_file_rows).to_csv(
            run_dir / "metrics" / "per_file_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        numeric = [
            column
            for column in metric_frame.select_dtypes(include=[np.number]).columns
            if column != "seed"
        ]
        summary = (
            metric_frame[numeric]
            .agg(["mean", "std"])
            .T.reset_index()
            .rename(columns={"index": "metric"})
        )
        summary.to_csv(
            run_dir / "metrics" / "five_seed_mean_std.csv",
            index=False,
            encoding="utf-8-sig",
        )
        means, stds = reference_comparison(metric_frame)
        means.to_csv(
            run_dir / "metrics" / "method_summary_means.csv",
            index=False,
            encoding="utf-8-sig",
        )
        stds.to_csv(
            run_dir / "metrics" / "method_summary_stds.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_comparison(means, run_dir / "figures" / "method_summary.png")
        write_report(run_dir / "实验结论.md", means, stds, run_dir)

        training_frame = pd.DataFrame(training_rows)
        final_integrity = {
            "success": True,
            "configuration_hash_matches": sha256(lock_path) == lock_hash,
            "file_count": len(keys),
            "seed_count": len(SEEDS),
            "model_count": len(list((run_dir / "models").rglob("*.keras"))),
            "score_count": len(list((run_dir / "scores").rglob("*.npz"))),
            "prediction_count": len(list((run_dir / "predictions").glob("*.pkl"))),
            "training_row_count": len(training_frame),
            "epochs_within_paper_maximum": bool((training_frame["epochs_completed"] <= config.epochs).all()),
            "pot_fit_status_counts": training_frame["pot_fit_status"].value_counts().to_dict(),
            "fixed_split_all_files": bool(
                (pd.DataFrame(split_rows)["reference_rows"] == REFERENCE_POINTS).all()
                and (pd.DataFrame(split_rows)["test_start_position"] == REFERENCE_POINTS + 1).all()
            ),
            "evaluation_module_sha256": sha256(SKAB_ROOT / "core" / "metrics.py"),
            "niemi_five_seed_mean": metric_frame[numeric].mean().to_dict(),
            "niemi_five_seed_std": metric_frame[numeric].std(ddof=1).to_dict(),
        }
        write_json(run_dir / "final_integrity.json", final_integrity)
        (run_dir / "SUCCESS.txt").write_text(
            "SUCCESS\n"
            f"run_dir={run_dir}\n"
            f"configuration_sha256={lock_hash}\n",
            encoding="utf-8",
        )
        log(log_path, f"SUCCESS: {run_dir}")
        print(means.to_string(index=False), flush=True)
    except Exception:
        error = traceback.format_exc()
        log(log_path, error)
        (run_dir / "FAILED.txt").write_text(error, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
