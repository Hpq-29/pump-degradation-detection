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
    REFERENCE_POINTS,
    SEEDS,
    SKAB_ROOT,
    WORKSPACE,
    data_paths,
    evaluate_predictions,
    load_sensor_frames,
    load_test_labels,
    safe_name,
    sha256,
    write_json,
)

OUTPUT_ROOT = OUTPUTS_ROOT / "literature24_wang_ksom_bayes_fixed_split"
WANG_PDF = WORKSPACE / "references" / "wang_2025.pdf"

for key, value in {
    "TEMP": WORKSPACE / ".tmp",
    "TMP": WORKSPACE / ".tmp",
    "MPLCONFIGDIR": WORKSPACE / ".matplotlib",
    "PYTHONPYCACHEPREFIX": WORKSPACE / ".pycache",
}.items():
    os.environ[key] = str(value)
    Path(value).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class WangConfig:
    signal_column: str = "Pressure"
    feature_window_points: int = 5
    feature_step_points: int = 1
    grid_rows: int = 3
    grid_columns: int = 3
    kernel: str = "RBF"
    rbf_alpha: float = 0.3
    som_epochs: int = 100
    learning_rate_start: float = 0.5
    learning_rate_end: float = 0.05
    neighborhood_radius_start: float = 1.5
    neighborhood_radius_end: float = 0.5
    posterior_threshold_sigma: float = 3.0
    variance_floor: float = 1e-6


FEATURE_NAMES = (
    "maximum",
    "minimum",
    "range",
    "mean",
    "slope",
    "rms",
    "skewness",
    "kurtosis_pearson",
    "frequency_center",
    "rms_frequency",
    "frequency_variance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer Wang et al. (2025) RBF-KSOM plus Bayesian posterior "
            "inference to the fixed SKAB protocol."
        )
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Development smoke test only; formal results must omit this option.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Development smoke test only; formal results use the locked five seeds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Development smoke test only; formal results use 100 epochs.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only allowed for label-free development smoke tests.",
    )
    return parser.parse_args()


def make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / stem
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{stem}_{suffix:02d}"
        suffix += 1
    for name in ("audit", "figures", "logs", "metrics", "models", "predictions", "scores"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir


def log(path: Path, message: str) -> None:
    line = f"[{pd.Timestamp.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def window_features(window: np.ndarray) -> np.ndarray:
    values = np.asarray(window, dtype=np.float64)
    maximum = float(np.max(values))
    minimum = float(np.min(values))
    average = float(np.mean(values))
    centered = values - average
    second_moment = float(np.mean(centered**2))
    if second_moment <= np.finfo(float).eps:
        skewness = 0.0
        kurtosis = 3.0
    else:
        skewness = float(np.mean(centered**3) / second_moment**1.5)
        kurtosis = float(np.mean(centered**4) / second_moment**2)

    frequencies = np.fft.rfftfreq(len(values), d=1.0)
    power = np.abs(np.fft.rfft(values)) ** 2
    power_sum = float(np.sum(power))
    if power_sum <= np.finfo(float).eps:
        frequency_center = 0.0
        rms_frequency = 0.0
        frequency_variance = 0.0
    else:
        frequency_center = float(np.sum(frequencies * power) / power_sum)
        rms_frequency = float(np.sqrt(np.sum(frequencies**2 * power) / power_sum))
        frequency_variance = float(
            np.sqrt(np.sum((frequencies - frequency_center) ** 2 * power) / power_sum)
        )

    return np.asarray(
        [
            maximum,
            minimum,
            maximum - minimum,
            average,
            (maximum - minimum) / len(values),
            float(np.sqrt(np.mean(values**2))),
            skewness,
            kurtosis,
            frequency_center,
            rms_frequency,
            frequency_variance,
        ],
        dtype=np.float64,
    )


def make_feature_matrix(values: np.ndarray, width: int, step: int = 1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < width:
        raise ValueError(f"Need at least {width} values, got {len(values)}")
    starts = range(0, len(values) - width + 1, step)
    return np.stack([window_features(values[start : start + width]) for start in starts])


def make_test_feature_matrix(reference: np.ndarray, test: np.ndarray, width: int) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    test = np.asarray(test, dtype=np.float64).reshape(-1)
    combined = np.concatenate([reference[-(width - 1) :], test])
    return np.stack([window_features(combined[i : i + width]) for i in range(len(test))])


def rbf_kernel_distance_squared(
    samples: np.ndarray, prototypes: np.ndarray, alpha: float
) -> np.ndarray:
    euclidean_squared = np.sum(
        (np.asarray(samples)[:, None, :] - np.asarray(prototypes)[None, :, :]) ** 2,
        axis=2,
    )
    return 2.0 * (1.0 - np.exp(-alpha * euclidean_squared))


def train_kernel_som(
    features: np.ndarray, seed: int, config: WangConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    unit_count = config.grid_rows * config.grid_columns
    if len(features) < unit_count:
        raise ValueError(f"Need at least {unit_count} feature vectors, got {len(features)}")
    rng = np.random.default_rng(seed)
    prototypes = features[rng.choice(len(features), unit_count, replace=False)].copy()
    coordinates = np.asarray(
        [(row, column) for row in range(config.grid_rows) for column in range(config.grid_columns)],
        dtype=np.float64,
    )
    total_steps = config.som_epochs * len(features)
    step_index = 0
    for _ in range(config.som_epochs):
        for sample_index in rng.permutation(len(features)):
            progress = step_index / max(total_steps - 1, 1)
            learning_rate = config.learning_rate_start * (
                config.learning_rate_end / config.learning_rate_start
            ) ** progress
            radius = config.neighborhood_radius_start * (
                config.neighborhood_radius_end / config.neighborhood_radius_start
            ) ** progress
            sample = features[sample_index]
            kernel_distances = rbf_kernel_distance_squared(
                sample[None, :], prototypes, config.rbf_alpha
            )[0]
            best_unit = int(np.argmin(kernel_distances))
            topology_squared = np.sum((coordinates - coordinates[best_unit]) ** 2, axis=1)
            neighborhood = np.exp(-topology_squared / (2.0 * radius**2))
            prototypes += learning_rate * neighborhood[:, None] * (sample - prototypes)
            step_index += 1
    final_distances = rbf_kernel_distance_squared(features, prototypes, config.rbf_alpha)
    assignments = np.argmin(final_distances, axis=1)
    return prototypes, assignments, coordinates


def fit_bayesian_distance_model(
    reference_features: np.ndarray,
    prototypes: np.ndarray,
    assignments: np.ndarray,
    config: WangConfig,
) -> dict[str, np.ndarray]:
    distances = rbf_kernel_distance_squared(reference_features, prototypes, config.rbf_alpha)
    unit_count = len(prototypes)
    assigned_distances = distances[np.arange(len(reference_features)), assignments]
    global_mean = float(np.mean(assigned_distances))
    global_sigma = max(float(np.std(assigned_distances, ddof=0)), config.variance_floor)
    means = np.empty(unit_count, dtype=np.float64)
    sigmas = np.empty(unit_count, dtype=np.float64)
    counts = np.bincount(assignments, minlength=unit_count).astype(np.int64)
    for unit in range(unit_count):
        unit_values = distances[assignments == unit, unit]
        if len(unit_values) == 0:
            means[unit] = global_mean
            sigmas[unit] = global_sigma
        else:
            means[unit] = float(np.mean(unit_values))
            sigmas[unit] = max(float(np.std(unit_values, ddof=0)), config.variance_floor)
    priors = np.maximum(counts.astype(np.float64), np.finfo(float).tiny)
    priors /= np.sum(priors)
    return {"means": means, "sigmas": sigmas, "priors": priors, "counts": counts}


def normality_scores(
    features: np.ndarray,
    prototypes: np.ndarray,
    model: dict[str, np.ndarray],
    alpha: float,
) -> np.ndarray:
    distances = rbf_kernel_distance_squared(features, prototypes, alpha)
    means = model["means"][None, :]
    sigmas = model["sigmas"][None, :]
    log_likelihood = (
        -0.5 * ((distances - means) / sigmas) ** 2
        - np.log(sigmas)
        - 0.5 * np.log(2.0 * np.pi)
    )
    log_weighted = log_likelihood + np.log(model["priors"])[None, :]
    posterior = np.exp(log_weighted - logsumexp(log_weighted, axis=1, keepdims=True))
    return np.max(posterior, axis=1)


def train_and_score_file(
    frame: pd.DataFrame,
    seed: int,
    file_index: int,
    key: str,
    config: WangConfig,
    model_dir: Path,
) -> dict[str, Any]:
    reference_signal = frame.iloc[:REFERENCE_POINTS][config.signal_column].to_numpy(dtype=float)
    test_signal = frame.iloc[REFERENCE_POINTS:][config.signal_column].to_numpy(dtype=float)
    reference_features = make_feature_matrix(
        reference_signal, config.feature_window_points, config.feature_step_points
    )
    test_features = make_test_feature_matrix(
        reference_signal, test_signal, config.feature_window_points
    )
    scaler = StandardScaler().fit(reference_features)
    reference_scaled = scaler.transform(reference_features)
    test_scaled = scaler.transform(test_features)

    model_seed = seed * 1000 + file_index
    prototypes, assignments, coordinates = train_kernel_som(reference_scaled, model_seed, config)
    distance_model = fit_bayesian_distance_model(
        reference_scaled, prototypes, assignments, config
    )
    reference_scores = normality_scores(
        reference_scaled, prototypes, distance_model, config.rbf_alpha
    )
    threshold_mean = float(np.mean(reference_scores))
    threshold_std = float(np.std(reference_scores, ddof=0))
    threshold = threshold_mean - config.posterior_threshold_sigma * threshold_std
    test_scores = normality_scores(test_scaled, prototypes, distance_model, config.rbf_alpha)
    alarms = test_scores <= threshold
    changepoints = alarms & ~np.concatenate([[False], alarms[:-1]])

    model_path = model_dir / f"{safe_name(key)}.npz"
    np.savez_compressed(
        model_path,
        prototypes=prototypes,
        topology_coordinates=coordinates,
        cluster_distance_means=distance_model["means"],
        cluster_distance_sigmas=distance_model["sigmas"],
        cluster_priors=distance_model["priors"],
        cluster_counts=distance_model["counts"],
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        normality_threshold=np.asarray([threshold]),
        model_seed=np.asarray([model_seed]),
    )
    return {
        "reference_scores": reference_scores,
        "test_scores": test_scores,
        "alarm": alarms.astype(np.int8),
        "changepoint": changepoints.astype(np.int8),
        "threshold": threshold,
        "threshold_mean": threshold_mean,
        "threshold_std": threshold_std,
        "cluster_counts": distance_model["counts"],
        "empty_clusters": int(np.sum(distance_model["counts"] == 0)),
        "model_path": str(model_path),
    }


def reference_comparison(wang_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [column for column in wang_metrics.select_dtypes(include=[np.number]).columns if column != "seed"]
    means = pd.DataFrame([{"method": "Wang RBF-KSOM+Bayes", "aggregation": "five-seed mean", **wang_metrics[numeric].mean().to_dict()}])
    stds = pd.DataFrame([{"method": "Wang RBF-KSOM+Bayes", **wang_metrics[numeric].std(ddof=1).to_dict()}])
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
    colors = ["#777777", "#D97706", "#2F80ED"]
    for ax, (column, title) in zip(axes.flat, columns):
        values = frame[column].astype(float).to_numpy()
        display_values = np.nan_to_num(values, nan=0.0)
        bars = ax.bar(frame["method"], display_values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            label = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("SKAB fixed 400/401+ split and official evaluation protocol")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_cell(value: float, deviation: float, is_official: bool, is_f1: bool = False) -> str:
    if not np.isfinite(value):
        return "NA"
    digits = 4 if is_f1 else 2
    if is_official:
        return f"{value:.{digits}f}"
    if not np.isfinite(deviation):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f}±{deviation:.{digits}f}"


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
        "# 文献[24] Wang et al. (2025) 对比实验结论",
        "",
        "## 实验身份",
        "",
        "- 数据：SKAB公开泵循环试验台数据，不是真实火电厂现场数据。",
        "- 固定划分：每个异常文件第1-400行为训练/参考集，第401行至末行为最终测试集。",
        "- 对比方法：Pressure单通道5点滑窗11维时频特征、3×3 RBF-KSOM、距离高斯分布与贝叶斯后验正常度。",
        "- 原文明示参数：3×3网络、RBF核alpha=0.3、后验阈值为训练正常度均值减3倍标准差。",
        "- 未披露参数：SOM训练轮数、学习率、邻域半径与初始化方式采用配置锁中的固定默认值，不读取标签调参。",
        "- 评价：全部预测先落盘，再加载第401行以后的标签并调用SKAB官方core/metrics.py。",
        "",
        "## 三种方法结果",
        "",
        "| 方法 | 点级F1 | FAR(%) | MAR(%) | 事件TP | 漏检事件 | 事件FP | 平均延迟(s) | NAB Standard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    std_lookup = stds.set_index("method")
    for _, row in means.iterrows():
        method = str(row["method"])
        std = std_lookup.loc[method]
        cells = [
            format_cell(
                float(row[column]),
                float(std[column]),
                method == "Official Conv-AE",
                column == "point_f1",
            )
            for column in selected
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 输出位置", "", f"`{run_dir}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_root.resolve())
    log_path = run_dir / "logs" / "run.log"
    config_values = asdict(WangConfig())
    if args.epochs is not None:
        config_values["som_epochs"] = args.epochs
    config = WangConfig(**config_values)
    selected_seeds = SEEDS if args.seeds is None else args.seeds
    is_formal = (
        args.limit_files is None
        and args.seeds is None
        and args.epochs is None
        and not args.skip_evaluation
    )
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
        if args.limit_files is not None:
            keys = keys[: args.limit_files]
            frames = {key: frames[key] for key in keys}
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
            run_dir / "audit" / "data_split_manifest.csv", index=False, encoding="utf-8-sig"
        )

        lock = {
            "locked_before_training_and_test_label_access": True,
            "formal_run": is_formal,
            "paper": {
                "citation": (
                    "Wang et al. (2025), An Unsupervised Anomaly Detection Method for "
                    "Nuclear Reactor Coolant Pumps Based on Kernel Self-Organizing Map "
                    "and Bayesian Posterior Inference, Energies 18, 2887, "
                    "DOI 10.3390/en18112887"
                ),
                "source_pdf": str(WANG_PDF),
                "source_pdf_sha256": sha256(WANG_PDF) if WANG_PDF.is_file() else None,
            },
            "selected_paper_configuration": "3x3 RBF-KSOM alpha=0.3 plus Bayesian posterior",
            "dataset": "SKAB public pump-loop testbed",
            "dataset_is_not": "real thermal-power-plant field data",
            "fixed_split": {
                "training_and_reference": "rows 1-400 of every file",
                "final_test": "rows 401-end of every file",
                "cross_file_split": False,
                "final_test_shuffle": False,
            },
            "seeds": selected_seeds,
            "configuration": asdict(config),
            "feature_mapping": {
                "paper": "one pressure signal, 5-hour window and 1-hour step, 11 time/frequency features",
                "SKAB_transfer": (
                    "SKAB Pressure column only; 5-sample causal window and 1-sample step; "
                    "the same 11 feature definitions"
                ),
            },
            "undisclosed_parameter_note": (
                "The paper does not disclose SOM epochs, learning-rate schedule, "
                "neighborhood-radius schedule, or initialization. The fixed values in "
                "configuration were locked before test-label access."
            ),
            "bayesian_implementation_note": (
                "For each cluster, Gaussian likelihood is evaluated at that cluster's "
                "kernel distance, multiplied by the empirical cluster prior, normalized "
                "across clusters, and the maximum posterior is the normality score."
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
        for seed in selected_seeds:
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
                    reference_normality=result["reference_scores"],
                    test_normality=result["test_scores"],
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
                        "normality_threshold": result["threshold"],
                        "reference_normality_mean": result["threshold_mean"],
                        "reference_normality_std": result["threshold_std"],
                        "empty_clusters": result["empty_clusters"],
                        "cluster_counts": json.dumps(result["cluster_counts"].tolist()),
                        "test_alarm_points": int(result["alarm"].sum()),
                        "test_changepoints": int(result["changepoint"].sum()),
                    }
                )
                log(
                    log_path,
                    f"seed={seed} file={file_index + 1}/{len(keys)} {key} "
                    f"threshold={result['threshold']:.6f} "
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
        training_frame = pd.DataFrame(training_rows)
        training_frame.to_csv(
            run_dir / "metrics" / "training_summary.csv", index=False, encoding="utf-8-sig"
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
                "expected_prediction_count": len(selected_seeds),
                "all_predictions_persisted": len(prediction_manifest) == len(selected_seeds),
                "labels_loaded": False,
            },
        )

        if args.skip_evaluation:
            (run_dir / "SMOKE_SUCCESS.txt").write_text(
                f"LABEL_FREE_SMOKE_SUCCESS\nrun_dir={run_dir}\n", encoding="utf-8"
            )
            log(log_path, f"LABEL-FREE SMOKE SUCCESS: {run_dir}")
            return

        log(log_path, "All predictions saved. Loading rows 401+ labels for official evaluation.")
        labels = load_test_labels(paths)
        metric_rows: list[dict[str, Any]] = []
        per_file_rows: list[dict[str, Any]] = []
        for seed in selected_seeds:
            points = {key: all_predictions[seed][key]["point"] for key in keys}
            cps = {key: all_predictions[seed][key]["changepoint"] for key in keys}
            metrics = evaluate_predictions(keys, labels, points, cps)
            metric_rows.append({"seed": seed, **metrics})
            if len(keys) == 34:
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
            run_dir / "metrics" / "five_seed_metrics.csv", index=False, encoding="utf-8-sig"
        )
        if per_file_rows:
            pd.DataFrame(per_file_rows).to_csv(
                run_dir / "metrics" / "per_file_metrics.csv", index=False, encoding="utf-8-sig"
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
            run_dir / "metrics" / "five_seed_mean_std.csv", index=False, encoding="utf-8-sig"
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
            "formal_run": is_formal,
            "configuration_hash_matches": sha256(lock_path) == lock_hash,
            "file_count": len(keys),
            "seed_count": len(selected_seeds),
            "model_count": len(list((run_dir / "models").rglob("*.npz"))),
            "score_count": len(list((run_dir / "scores").rglob("*.npz"))),
            "prediction_count": len(list((run_dir / "predictions").glob("*.pkl"))),
            "training_row_count": len(training_frame),
            "empty_cluster_total": int(training_frame["empty_clusters"].sum()),
            "fixed_split_all_files": bool(
                (pd.DataFrame(split_rows)["reference_rows"] == REFERENCE_POINTS).all()
                and (pd.DataFrame(split_rows)["test_start_position"] == REFERENCE_POINTS + 1).all()
            ),
            "evaluation_module_sha256": sha256(SKAB_ROOT / "core" / "metrics.py"),
            "wang_five_seed_mean": metric_frame[numeric].mean().to_dict(),
            "wang_five_seed_std": metric_frame[numeric].std(ddof=1).to_dict(),
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
