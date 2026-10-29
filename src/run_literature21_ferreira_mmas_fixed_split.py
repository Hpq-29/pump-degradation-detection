from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


from project_paths import PACKAGE_ROOT as WORKSPACE, SKAB_ROOT, OUTPUTS_ROOT

from project_paths import DATA_ROOT
OUTPUT_ROOT = OUTPUTS_ROOT / "literature21_ferreira_mmas_fixed_split"
FERREIRA_PDF = WORKSPACE / "references" / "ferreira_2026.pdf"

SENSORS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]
REFERENCE_POINTS = 400
SEEDS = [0, 1, 2, 3, 4]

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
sys.path.insert(0, str(SKAB_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from core.metrics import chp_score


@dataclass(frozen=True)
class FerreiraConfig:
    hidden_neurons: tuple[int, ...] = (7, 6, 5, 6, 7)
    hidden_activation: str = "relu"
    output_activation: str = "linear"
    optimizer: str = "adam"
    loss: str = "mse"
    epochs: int = 200
    batch_size: int = 512
    validation_split: float = 0.20
    shuffle: bool = True
    mmas_window: int = 10
    threshold_multiplier: float = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Ferreira et al. (2026) AE+MMAS on SKAB using the fixed "
            "400/401+ split and the official SKAB evaluation implementation."
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def safe_name(key: str) -> str:
    return key.replace("\\", "_").replace("/", "_").replace(".csv", "")


def data_paths() -> list[Path]:
    paths = sorted(
        path for path in DATA_ROOT.rglob("*.csv") if "anomaly-free" not in str(path)
    )
    if len(paths) != 34:
        raise RuntimeError(f"Expected 34 SKAB anomaly files, found {len(paths)}")
    return paths


def data_key(path: Path) -> str:
    return str(path.relative_to(DATA_ROOT)).replace("/", "\\")


def load_sensor_frames(paths: list[Path]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        frame = pd.read_csv(
            path,
            sep=";",
            usecols=["datetime", *SENSORS],
            parse_dates=["datetime"],
        ).set_index("datetime").sort_index()
        if len(frame) <= REFERENCE_POINTS:
            raise RuntimeError(f"Fewer than 401 rows in {path}")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise RuntimeError(f"Invalid timestamp index in {path}")
        if frame[SENSORS].isna().any().any():
            raise RuntimeError(f"Missing sensor data in {path}")
        frames[data_key(path)] = frame
    return dict(sorted(frames.items()))


def load_test_labels(paths: list[Path]) -> dict[str, pd.DataFrame]:
    labels: dict[str, pd.DataFrame] = {}
    for path in paths:
        frame = pd.read_csv(
            path,
            sep=";",
            usecols=["datetime", "anomaly", "changepoint"],
            parse_dates=["datetime"],
        ).set_index("datetime").sort_index()
        labels[data_key(path)] = frame.iloc[REFERENCE_POINTS:].astype(int)
    return dict(sorted(labels.items()))


def set_seed(seed: int) -> None:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_autoencoder(config: FerreiraConfig) -> tf.keras.Model:
    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(len(SENSORS),))]
    layers.extend(
        tf.keras.layers.Dense(units, activation=config.hidden_activation)
        for units in config.hidden_neurons
    )
    layers.append(
        tf.keras.layers.Dense(len(SENSORS), activation=config.output_activation)
    )
    model = tf.keras.Sequential(layers, name="ferreira_deep_autoencoder")
    model.compile(optimizer=config.optimizer, loss=config.loss)
    return model


def squared_l2_score(values: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    residual = np.asarray(values, dtype=float) - np.asarray(reconstructed, dtype=float)
    return np.sum(np.square(residual), axis=1)


def moving_minimum(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rolling(window=window, min_periods=window).min().to_numpy()


def train_and_score_file(
    frame: pd.DataFrame,
    seed: int,
    file_index: int,
    key: str,
    config: FerreiraConfig,
    model_dir: Path,
) -> dict[str, Any]:
    reference = frame.iloc[:REFERENCE_POINTS][SENSORS].to_numpy(dtype=np.float32)
    test = frame.iloc[REFERENCE_POINTS:][SENSORS].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(reference)
    reference_scaled = scaler.transform(reference).astype(np.float32)
    test_scaled = scaler.transform(test).astype(np.float32)

    model_seed = seed * 1000 + file_index
    set_seed(model_seed)
    model = build_autoencoder(config)
    history = model.fit(
        reference_scaled,
        reference_scaled,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        shuffle=config.shuffle,
        verbose=0,
    )

    reference_reconstruction = model.predict(
        reference_scaled, batch_size=config.batch_size, verbose=0
    )
    test_reconstruction = model.predict(
        test_scaled, batch_size=config.batch_size, verbose=0
    )
    reference_as = squared_l2_score(reference_scaled, reference_reconstruction)
    test_as = squared_l2_score(test_scaled, test_reconstruction)

    reference_mmas = moving_minimum(reference_as, config.mmas_window)
    valid_reference_mmas = reference_mmas[~np.isnan(reference_mmas)]
    if not len(valid_reference_mmas):
        raise RuntimeError(f"No valid training MMAS values for {key}")
    threshold = config.threshold_multiplier * float(np.max(valid_reference_mmas))

    history_as = np.concatenate([reference_as[-(config.mmas_window - 1) :], test_as])
    test_mmas = moving_minimum(history_as, config.mmas_window)[
        config.mmas_window - 1 :
    ]
    if len(test_mmas) != len(test):
        raise AssertionError("MMAS alignment failed")
    alarms = test_mmas > threshold
    changepoints = alarms & ~np.concatenate([[False], alarms[:-1]])

    model_path = model_dir / f"{safe_name(key)}.keras"
    model.save(model_path)
    tf.keras.backend.clear_session()

    return {
        "reference_as": reference_as,
        "reference_mmas": reference_mmas,
        "test_as": test_as,
        "test_mmas": test_mmas,
        "threshold": float(threshold),
        "alarm": alarms.astype(np.int8),
        "changepoint": changepoints.astype(np.int8),
        "train_loss": float(history.history["loss"][-1]),
        "validation_loss": float(history.history["val_loss"][-1]),
        "epochs_completed": len(history.history["loss"]),
        "model_path": str(model_path),
    }


def evaluate_predictions(
    keys: list[str],
    labels: dict[str, pd.DataFrame],
    points: dict[str, pd.Series],
    cps: dict[str, pd.Series],
) -> dict[str, float]:
    true_point = [labels[key]["anomaly"] for key in keys]
    true_cp = [labels[key]["changepoint"] for key in keys]
    predicted_point = [points[key] for key in keys]
    predicted_cp = [cps[key] for key in keys]
    tp, tn, fp, fn = chp_score(
        true_point, predicted_point, metric="confusion_matrix", verbose=False
    )
    delay, missed, event_fp, total_cp = chp_score(
        true_cp,
        predicted_cp,
        metric="average_time",
        window_width="60s",
        anomaly_window_destination="righter",
        verbose=False,
    )
    nab = chp_score(
        true_cp,
        predicted_cp,
        metric="nab",
        window_width="60s",
        anomaly_window_destination="righter",
        verbose=False,
    )
    delay_seconds = np.nan if pd.isna(delay) else pd.Timedelta(delay).total_seconds()
    denominator = tp + (fn + fp) / 2.0
    return {
        "point_tp": int(tp),
        "point_tn": int(tn),
        "point_fp": int(fp),
        "point_fn": int(fn),
        "point_f1": float(tp / denominator) if denominator else 0.0,
        "point_far_percent": float(fp / (fp + tn) * 100.0) if fp + tn else 0.0,
        "point_mar_percent": float(fn / (fn + tp) * 100.0) if fn + tp else 0.0,
        "event_true_cps": int(total_cp),
        "event_tps": int(total_cp - missed),
        "event_missed_cps": int(missed),
        "event_missed_rate": float(missed / total_cp) if total_cp else np.nan,
        "event_fps": int(event_fp),
        "average_detection_delay_seconds": float(delay_seconds),
        "nab_standard": float(nab["Standard"]),
        "nab_low_fp": float(nab["LowFP"]),
        "nab_low_fn": float(nab["LowFN"]),
    }


def evaluate_per_file(
    key: str,
    labels: dict[str, pd.DataFrame],
    point: pd.Series,
    cp: pd.Series,
) -> dict[str, float]:
    return evaluate_predictions([key], labels, {key: point}, {key: cp})


def reference_comparison(ferreira_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [column for column in ferreira_metrics.select_dtypes(include=[np.number]).columns if column != "seed"]
    means = pd.DataFrame([{"method": "Ferreira AE+MMAS", "aggregation": "five-seed mean", **ferreira_metrics[numeric].mean().to_dict()}])
    stds = pd.DataFrame([{"method": "Ferreira AE+MMAS", **ferreira_metrics[numeric].std(ddof=1).to_dict()}])
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
    colors = ["#777777", "#E67E22", "#2F80ED"]
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


def write_report(
    path: Path,
    means: pd.DataFrame,
    stds: pd.DataFrame,
    run_dir: Path,
) -> None:
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
        "# 文献[21] Ferreira et al. (2026) 对比实验结论",
        "",
        "## 实验身份",
        "",
        "- 数据：SKAB公开泵循环试验台数据，不是真实火电厂现场数据。",
        "- 固定划分：每个文件第1-400行为训练/参考集，第401行至末行为最终测试集，不跨文件、不打乱最终测试时序。",
        "- 对比方法：逐文件z-score标准化；8-7-6-5-6-7-8深度全连接AE；平方L2重构误差；10点MMAS；阈值为训练MMAS最大值的1.5倍。",
        "- 训练：Adam、MSE、ReLU隐藏层、线性输出、200轮、batch size 512、validation split 20%、shuffle=True。",
        "- 评价：全部预测先落盘，再读取测试标签，调用SKAB官方core/metrics.py。",
        "- 说明：SKAB没有与原文等价的泵启停状态字段，因此没有臆造原文的idle-state过滤；其余可迁移核心步骤保持一致。",
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
        values: list[str] = []
        for column in selected:
            value = float(row[column])
            deviation = float(std[column])
            if method == "Official Conv-AE":
                values.append(f"{value:.4f}" if column == "point_f1" else f"{value:.2f}")
            else:
                values.append(
                    f"{value:.4f}±{deviation:.4f}"
                    if column == "point_f1"
                    else f"{value:.2f}±{deviation:.2f}"
                )
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 输出位置",
            "",
            f"`{run_dir}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_root.resolve())
    log_path = run_dir / "logs" / "run.log"
    config = FerreiraConfig()
    try:
        required = [
            SKAB_ROOT / "core" / "metrics.py",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required files: " + "; ".join(missing))

        log(log_path, "Loading sensor-only SKAB frames; labels remain unread.")
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
                "citation": "Ferreira et al. (2026), DOI 10.1109/OJIM.2026.3670416",
                "source_pdf": str(FERREIRA_PDF),
                "source_pdf_sha256": sha256(FERREIRA_PDF) if FERREIRA_PDF.is_file() else None,
            },
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
            "score": "squared L2 reconstruction error",
            "point_alarm": "MMAS > 1.5 * max(training MMAS)",
            "predicted_changepoint": "causal 0-to-1 transition of point alarm",
            "evaluation": {
                "module": str(SKAB_ROOT / "core" / "metrics.py"),
                "window_width": "60s",
                "anomaly_window_destination": "righter",
            },
            "adaptation_note": (
                "SKAB has no operation-state field equivalent to the source paper, "
                "so idle-state filtering is not fabricated."
            ),
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
                    reference_as=result["reference_as"],
                    reference_mmas=result["reference_mmas"],
                    test_as=result["test_as"],
                    test_mmas=result["test_mmas"],
                    threshold=np.asarray([result["threshold"]]),
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
                        "threshold": result["threshold"],
                        "train_loss": result["train_loss"],
                        "validation_loss": result["validation_loss"],
                        "epochs_completed": result["epochs_completed"],
                        "test_alarm_points": int(result["alarm"].sum()),
                        "test_changepoints": int(result["changepoint"].sum()),
                    }
                )
                log(
                    log_path,
                    f"seed={seed} file={file_index + 1}/34 {key} "
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
        prediction_manifest = []
        for path in sorted((run_dir / "predictions").glob("*.pkl")):
            prediction_manifest.append(
                {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            )
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
                        **evaluate_per_file(key, labels, points[key], cps[key]),
                    }
                )
            log(
                log_path,
                f"Evaluated seed={seed}: F1={metrics['point_f1']:.4f}, "
                f"FAR={metrics['point_far_percent']:.3f}%, "
                f"MAR={metrics['point_mar_percent']:.3f}%, "
                f"eventFP={metrics['event_fps']}",
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

        final_integrity = {
            "success": True,
            "configuration_hash_matches": sha256(lock_path) == lock_hash,
            "file_count": len(keys),
            "seed_count": len(SEEDS),
            "model_count": len(list((run_dir / "models").rglob("*.keras"))),
            "score_count": len(list((run_dir / "scores").rglob("*.npz"))),
            "prediction_count": len(list((run_dir / "predictions").glob("*.pkl"))),
            "fixed_split_all_files": bool(
                (pd.DataFrame(split_rows)["reference_rows"] == REFERENCE_POINTS).all()
                and (pd.DataFrame(split_rows)["test_start_position"] == REFERENCE_POINTS + 1).all()
            ),
            "evaluation_module_sha256": sha256(SKAB_ROOT / "core" / "metrics.py"),
            "ferreira_five_seed_mean": metric_frame[numeric].mean().to_dict(),
            "ferreira_five_seed_std": metric_frame[numeric].std(ddof=1).to_dict(),
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
