"""PyTorch reproduction of the paper method and its three component ablations.

The experiment retrains one Conv-AE per SKAB file for five random seeds.  It
keeps the historical 400/401+ split, model topology, optimizer settings,
early-stopping rule, post-processing configuration and official SKAB metrics.
No literature comparison method is run by this script.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from project_paths import PACKAGE_ROOT as WORKSPACE, SKAB_ROOT, OUTPUTS_ROOT

from project_paths import DATA_ROOT
OUTPUT_ROOT = OUTPUTS_ROOT / "pytorch_method_ablation"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / ".matplotlib"))
sys.path.insert(0, str(SKAB_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.metrics import chp_score

SENSORS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS",
]
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["full", "total_only", "no_vote", "no_event_confirmation"]
VARIANT_NAMES = {
    "full": "完整方法",
    "total_only": "仅总体证据",
    "no_vote": "去掉持续性投票",
    "no_event_confirmation": "去掉第二阶段事件确认",
}
REFERENCE_POINTS = 400
N_STEPS = 60
MAX_EPOCHS = 100
BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 128
VALIDATION_FRACTION = 0.10
PATIENCE = 5
LEARNING_RATE = 0.001
ADAM_EPSILON = 1e-7
EPSILON = 1e-12


@dataclass(frozen=True)
class FixedConfig:
    point_gate: str = "simes_and"
    point_alpha: float = 0.02
    point_vote_width: int = 5
    point_vote_required: int = 3
    front_back_width: int = 10
    event_beta: float = 0.05
    event_consecutive: int = 1
    merge_gap_seconds: int = 60
    direction_weight: float = 0.0
    direction_margin: float = 0.25
    minimum_hold_seconds: int = 120
    maximum_hold_seconds: int = 600
    rescue_horizon_seconds: int = 60


class ConvAE(nn.Module):
    """Channels-first equivalent of the historical Keras 60x8 -> 60x1 model."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = nn.Conv1d(8, 32, kernel_size=7, stride=2, padding=3)
        self.dropout1 = nn.Dropout(0.2)
        self.encoder2 = nn.Conv1d(32, 16, kernel_size=7, stride=2, padding=3)
        self.decoder1 = nn.ConvTranspose1d(
            16, 16, kernel_size=7, stride=2, padding=3, output_padding=1
        )
        self.dropout2 = nn.Dropout(0.2)
        self.decoder2 = nn.ConvTranspose1d(
            16, 32, kernel_size=7, stride=2, padding=3, output_padding=1
        )
        self.output = nn.ConvTranspose1d(32, 1, kernel_size=7, stride=1, padding=3)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keras Conv1D/Conv1DTranspose use Glorot uniform and zero biases.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.encoder1(x))
        x = self.dropout1(x)
        x = torch.relu(self.encoder2(x))
        x = torch.relu(self.decoder1(x))
        x = self.dropout2(x)
        x = torch.relu(self.decoder2(x))
        return self.output(x)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def safe_name(key: str) -> str:
    return key.replace("\\", "_").replace("/", "_").replace(".csv", "")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def data_paths() -> list[Path]:
    paths = sorted(
        path for path in DATA_ROOT.rglob("*.csv") if "anomaly-free" not in str(path)
    )
    if len(paths) != 34:
        raise RuntimeError(f"Expected 34 anomaly files, found {len(paths)}")
    return paths


def data_key(path: Path) -> str:
    return str(path.relative_to(DATA_ROOT)).replace("/", "\\")


def read_sensor_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path, sep=";", usecols=["datetime", *SENSORS], parse_dates=["datetime"]
    ).set_index("datetime").sort_index()
    if frame.index.has_duplicates or frame[SENSORS].isna().any().any():
        raise RuntimeError(f"Invalid sensor data: {path}")
    return frame


def standardize(reference: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference_values = reference[SENSORS].to_numpy(dtype=np.float64)
    test_values = test[SENSORS].to_numpy(dtype=np.float64)
    mean = reference_values.mean(axis=0)
    scale = reference_values.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (
        ((reference_values - mean) / scale).astype(np.float32),
        ((test_values - mean) / scale).astype(np.float32),
        mean,
        scale,
    )


def create_sequences(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.stack(
        [values[i : i + N_STEPS] for i in range(len(values) - N_STEPS + 1)]
    )


def split_train_validation(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    split_at = int(len(sequences) * (1.0 - VALIDATION_FRACTION))
    return sequences[:split_at], sequences[split_at:]


def train_model(sequences: np.ndarray, seed: int) -> tuple[ConvAE, list[dict], int, float]:
    set_seed(seed)
    train, validation = split_train_validation(sequences)
    x_train = torch.from_numpy(train.transpose(0, 2, 1).copy())
    x_validation = torch.from_numpy(validation.transpose(0, 2, 1).copy())
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train), batch_size=BATCH_SIZE, shuffle=True, generator=generator,
        num_workers=0, drop_last=False,
    )
    model = ConvAE()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=ADAM_EPSILON
    )
    history: list[dict] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = torch.mean((output.expand(-1, batch.shape[1], -1) - batch) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            total_items += len(batch)
        train_loss = total_loss / total_items
        model.eval()
        with torch.inference_mode():
            validation_output = model(x_validation)
            validation_loss = float(torch.mean(
                (validation_output.expand(-1, x_validation.shape[1], -1) - x_validation) ** 2
            ))
        history.append({"epoch": epoch, "loss": train_loss, "val_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    best_epoch = min(history, key=lambda row: row["val_loss"])["epoch"]
    return model, history, int(best_epoch), elapsed


def score_model(model: ConvAE, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = torch.from_numpy(sequences.transpose(0, 2, 1).copy())
    loader = DataLoader(TensorDataset(data), batch_size=PREDICT_BATCH_SIZE, shuffle=False)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            outputs.append(model(batch).cpu().numpy())
    prediction = np.concatenate(outputs, axis=0).transpose(0, 2, 1)
    channels = np.mean(np.abs(sequences - prediction), axis=1)
    return channels.sum(axis=1), channels


def conformal_p(calibration: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(calibration, dtype=float))
    values = np.asarray(values, dtype=float)
    greater_equal = len(ordered) - np.searchsorted(ordered, values, side="left")
    return (greater_equal + 1.0) / (len(ordered) + 1.0)


def simes_p(p_channels: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(p_channels, dtype=float), axis=1)
    count = ordered.shape[1]
    return np.minimum(
        1.0, np.min(ordered * (count / np.arange(1, count + 1, dtype=float)), axis=1)
    )


def causal_vote(flags: np.ndarray, width: int, required: int) -> np.ndarray:
    flags = np.asarray(flags, dtype=int)
    cumulative = np.concatenate([[0], np.cumsum(flags)])
    result = np.zeros(len(flags), dtype=bool)
    for position in range(len(flags)):
        start = max(0, position - width + 1)
        result[position] = cumulative[position + 1] - cumulative[start] >= required
    return result


def health_arrays(reference_total: np.ndarray, reference_channels: np.ndarray,
                  test_total: np.ndarray, test_channels: np.ndarray) -> dict[str, np.ndarray]:
    reference_p_total = conformal_p(reference_total, reference_total)
    test_p_total = conformal_p(reference_total, test_total)
    reference_p_channels = np.column_stack([
        conformal_p(reference_channels[:, j], reference_channels[:, j])
        for j in range(reference_channels.shape[1])
    ])
    test_p_channels = np.column_stack([
        conformal_p(reference_channels[:, j], test_channels[:, j])
        for j in range(reference_channels.shape[1])
    ])
    q95 = float(np.quantile(reference_total, 0.95))
    median = float(np.median(reference_total))
    mad = max(float(np.median(np.abs(reference_total - median))), EPSILON)

    def health(total: np.ndarray, p_total: np.ndarray, p_channels: np.ndarray) -> np.ndarray:
        return (
            -np.log(p_total + EPSILON)
            + np.mean(-np.log(p_channels + EPSILON), axis=1)
            + np.maximum(0.0, (total - q95) / mad)
        )
    return {
        "reference_p_total": reference_p_total,
        "test_p_total": test_p_total,
        "reference_p_channels": reference_p_channels,
        "test_p_channels": test_p_channels,
        "reference_health": health(reference_total, reference_p_total, reference_p_channels),
        "test_health": health(test_total, test_p_total, test_p_channels),
        "q95": np.asarray(q95), "median": np.asarray(median), "mad": np.asarray(mad),
    }


def low_flags(p_total: np.ndarray, p_channels: np.ndarray, config: FixedConfig) -> np.ndarray:
    base = (p_total <= config.point_alpha) & (simes_p(p_channels) <= config.point_alpha)
    return causal_vote(base, config.point_vote_width, config.point_vote_required)


def event_statistics(index: pd.DatetimeIndex, health: np.ndarray, flags: np.ndarray,
                     width: int) -> list[dict[str, Any]]:
    rows = []
    for boundary in range(width, len(health) - width + 1):
        alarm_position = boundary + width - 1
        front_health = float(np.mean(health[boundary - width : boundary]))
        back_health = float(np.mean(health[boundary : boundary + width]))
        front_ratio = float(np.mean(flags[boundary - width : boundary]))
        back_ratio = float(np.mean(flags[boundary : boundary + width]))
        signed_health = back_health - front_health
        signed_ratio = back_ratio - front_ratio
        rows.append({
            "boundary": boundary, "alarm_time": index[alarm_position],
            "signed_health": signed_health, "signed_ratio": signed_ratio,
            "front_ratio": front_ratio, "back_ratio": back_ratio,
            "d_value": abs(signed_health) + 2.0 * abs(signed_ratio),
        })
    return rows


def confirmed_events(reference_rows: list[dict], test_rows: list[dict],
                     config: FixedConfig) -> list[dict]:
    calibration = np.asarray([row["d_value"] for row in reference_rows], dtype=float)
    event_p = conformal_p(calibration, np.asarray([row["d_value"] for row in test_rows]))
    selected = [
        {**row, "event_p": float(p)} for row, p in zip(test_rows, event_p)
        if p <= config.event_beta
    ]
    triggered = []
    run_length = 0
    previous_boundary = None
    for row in selected:
        boundary = int(row["boundary"])
        run_length = run_length + 1 if previous_boundary is not None and boundary == previous_boundary + 1 else 1
        previous_boundary = boundary
        if run_length >= config.event_consecutive:
            triggered.append({**row, "run_length": run_length})
    merged = []
    for row in triggered:
        if not merged or (row["alarm_time"] - merged[-1]["alarm_time"]).total_seconds() > config.merge_gap_seconds:
            merged.append(row)
    return merged


def transition_events(index: pd.DatetimeIndex, flags: np.ndarray, gap: int) -> list[dict]:
    changes = np.diff(np.asarray(flags, dtype=int), prepend=0)
    events = []
    for i in np.flatnonzero(changes):
        if not events or (index[i] - events[-1]["alarm_time"]).total_seconds() > gap:
            events.append({
                "boundary": int(i), "alarm_time": index[i],
                "transition_direction": int(changes[i]),
                "direction_source": "persistent_flag_transition",
            })
    return events


def state_and_rescue(index: pd.DatetimeIndex, events: list[dict],
                     config: FixedConfig) -> pd.Series:
    values = np.zeros(len(index), dtype=int)
    event_position, state = 0, 0
    entered_at = None
    for array_position, timestamp in enumerate(index):
        while event_position < len(events) and events[event_position]["alarm_time"] <= timestamp:
            event = events[event_position]
            direction = float(event["signed_health"]) + config.direction_weight * float(event["signed_ratio"])
            if direction > config.direction_margin:
                if state == 0:
                    entered_at = pd.Timestamp(event["alarm_time"])
                state = 1
            elif direction < -config.direction_margin:
                held = 0.0 if entered_at is None else (pd.Timestamp(event["alarm_time"]) - entered_at).total_seconds()
                if state == 1 and held >= config.minimum_hold_seconds:
                    state, entered_at = 0, None
            event_position += 1
        if state == 1 and config.maximum_hold_seconds > 0 and entered_at is not None \
                and (timestamp - entered_at).total_seconds() >= config.maximum_hold_seconds:
            state, entered_at = 0, None
        values[array_position] = state
    for event in events:
        start_time = pd.Timestamp(event["alarm_time"])
        end_time = start_time + pd.Timedelta(seconds=config.rescue_horizon_seconds)
        values[int(index.searchsorted(start_time, side="left")) :
               int(index.searchsorted(end_time, side="right"))] = 1
    return pd.Series(values, index=index, dtype=int)


def cp_series(index: pd.DatetimeIndex, events: list[dict]) -> pd.Series:
    result = pd.Series(0, index=index, dtype=int)
    valid = pd.DatetimeIndex([row["alarm_time"] for row in events]).intersection(index)
    result.loc[valid] = 1
    return result


def predict_variant(arrays: dict[str, np.ndarray], reference_index: pd.DatetimeIndex,
                    score_index: pd.DatetimeIndex, output_index: pd.DatetimeIndex,
                    config: FixedConfig, variant: str) -> tuple[pd.Series, pd.Series, list[dict]]:
    used = replace(config, point_vote_width=1, point_vote_required=1) if variant == "no_vote" else config
    if variant == "total_only":
        q95, mad = float(arrays["q95"]), float(arrays["mad"])
        reference_health = -np.log(arrays["reference_p_total"] + EPSILON) + np.maximum(
            0.0, (arrays["reference_total"] - q95) / mad
        )
        test_health = -np.log(arrays["test_p_total"] + EPSILON) + np.maximum(
            0.0, (arrays["test_total"] - q95) / mad
        )
        reference_flags = causal_vote(
            arrays["reference_p_total"] <= config.point_alpha,
            config.point_vote_width, config.point_vote_required,
        )
        test_flags = causal_vote(
            arrays["test_p_total"] <= config.point_alpha,
            config.point_vote_width, config.point_vote_required,
        )
    else:
        reference_health, test_health = arrays["reference_health"], arrays["test_health"]
        reference_flags = low_flags(arrays["reference_p_total"], arrays["reference_p_channels"], used)
        test_flags = low_flags(arrays["test_p_total"], arrays["test_p_channels"], used)
    if variant == "no_event_confirmation":
        events = transition_events(score_index, test_flags, config.merge_gap_seconds)
        adapter = [
            {**event, "signed_health": float(event["transition_direction"]), "signed_ratio": 0.0}
            for event in events
        ]
        point = state_and_rescue(output_index, adapter, config)
    else:
        events = confirmed_events(
            event_statistics(reference_index, reference_health, reference_flags, config.front_back_width),
            event_statistics(score_index, test_health, test_flags, config.front_back_width), config,
        )
        point = state_and_rescue(output_index, events, config)
    return point, cp_series(output_index, events), events


def evaluate_predictions(keys: list[str], labels: dict[str, pd.DataFrame],
                         points: dict[str, pd.Series], cps: dict[str, pd.Series]) -> dict[str, float]:
    true_point = [labels[key]["anomaly"] for key in keys]
    true_cp = [labels[key]["changepoint"] for key in keys]
    predicted_point = [points[key] for key in keys]
    predicted_cp = [cps[key] for key in keys]
    tp, tn, fp, fn = chp_score(true_point, predicted_point, metric="confusion_matrix", verbose=False)
    delay, missed, event_fp, total_cp = chp_score(
        true_cp, predicted_cp, metric="average_time", window_width="60s",
        anomaly_window_destination="righter", verbose=False,
    )
    nab = chp_score(
        true_cp, predicted_cp, metric="nab", window_width="60s",
        anomaly_window_destination="righter", verbose=False,
    )
    delay_seconds = np.nan if pd.isna(delay) else pd.Timedelta(delay).total_seconds()
    denominator = tp + (fn + fp) / 2.0
    return {
        "point_tp": int(tp), "point_tn": int(tn), "point_fp": int(fp), "point_fn": int(fn),
        "point_f1": float(tp / denominator) if denominator else 0.0,
        "point_far_percent": float(fp / (fp + tn) * 100.0) if fp + tn else 0.0,
        "point_mar_percent": float(fn / (fn + tp) * 100.0) if fn + tp else 0.0,
        "event_true_cps": int(total_cp), "event_tps": int(total_cp - missed),
        "event_missed_cps": int(missed),
        "event_missed_rate": float(missed / total_cp) if total_cp else np.nan,
        "event_fps": int(event_fp), "average_detection_delay_seconds": float(delay_seconds),
        "nab_standard": float(nab["Standard"]), "nab_low_fp": float(nab["LowFP"]),
        "nab_low_fn": float(nab["LowFN"]),
    }


def check_implementation() -> None:
    set_seed(0)
    model = ConvAE()
    x = torch.zeros(2, 8, 60)
    assert model(x).shape == (2, 1, 60)
    assert sum(parameter.numel() for parameter in model.parameters()) == 11073
    dummy = np.zeros((341, 60, 8), dtype=np.float32)
    train, validation = split_train_validation(dummy)
    assert len(train) == 306 and len(validation) == 35
    assert np.allclose(conformal_p(np.array([1., 2., 2.]), np.array([2., 3.])), [.75, .25])
    assert np.allclose(simes_p(np.array([[.01, .5], [.2, .2]])), [.02, .2])
    assert np.array_equal(causal_vote(np.array([1, 0, 1, 1]), 5, 3), [0, 0, 0, 1])
    index = pd.date_range("2020-01-01", periods=150, freq="s")
    flags = np.zeros(150, dtype=int); flags[10:70] = 1; flags[71:140] = 1
    assert [row["boundary"] for row in transition_events(index, flags, 60)] == [10, 71, 140]


def make_run_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = OUTPUT_ROOT / stem
    suffix = 1
    while run.exists():
        run = OUTPUT_ROOT / f"{stem}_{suffix:02d}"; suffix += 1
    for folder in ["audit", "models", "scores", "predictions", "metrics", "figures", "source"]:
        (run / folder).mkdir(parents=True, exist_ok=True)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Train the first file for three epochs only.")
    args = parser.parse_args()
    check_implementation()
    if args.check_only:
        print("PASS: PyTorch topology, validation split and post-processing checks.")
        return
    global MAX_EPOCHS
    if args.smoke:
        MAX_EPOCHS = 3
    run = make_run_dir()
    log_path = run / "run.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    try:
        shutil.copy2(__file__, run / "source" / Path(__file__).name)
        shutil.copy2(SKAB_ROOT / "core" / "metrics.py", run / "source" / "skab_metrics.py")
        config = FixedConfig()
        protocol = {
            "formal_run": not args.smoke,
            "framework": f"PyTorch {torch.__version__}", "device": "CPU",
            "architecture": [
                "Conv1d(8,32,k=7,s=2,p=same)+ReLU", "Dropout(0.2)",
                "Conv1d(32,16,k=7,s=2,p=same)+ReLU",
                "ConvTranspose1d(16,16,k=7,s=2,p=same)+ReLU", "Dropout(0.2)",
                "ConvTranspose1d(16,32,k=7,s=2,p=same)+ReLU",
                "ConvTranspose1d(32,1,k=7,s=1,p=same)",
            ],
            "parameter_count": 11073, "input_shape": [60, 8], "output_shape": [60, 1],
            "training": {
                "optimizer": "Adam", "learning_rate": LEARNING_RATE,
                "betas": [0.9, 0.999], "epsilon": ADAM_EPSILON,
                "loss": "MSE, one-channel output broadcast to eight input channels",
                "batch_size": BATCH_SIZE, "maximum_epochs": MAX_EPOCHS,
                "validation_fraction": VALIDATION_FRACTION,
                "validation_selection": "last 10% before per-epoch training shuffle",
                "early_stopping_patience": PATIENCE, "restore_best_weights": True,
                "initialization": "Xavier/Glorot uniform; zero bias",
            },
            "data": {"files": 34, "seeds": SEEDS, "reference_rows_per_file": 400,
                     "test_start_row_one_based": 401, "window": 60},
            "detection": asdict(config), "variants": {
                "full": "Complete paper method.",
                "total_only": "Remove channel Simes gate and channel term in health.",
                "no_vote": "Change causal vote from 3-of-5 to 1-of-1.",
                "no_event_confirmation": "Use persistent-flag transitions without front/back event calibration.",
            },
            "excluded": "No literature comparison methods are trained or evaluated.",
        }
        write_json(run / "protocol_lock.json", protocol)
        protocol_hash = sha256(run / "protocol_lock.json")
        paths = data_paths()
        data_hashes = {str(path): sha256(path) for path in paths}
        keys = [data_key(path) for path in paths]
        predictions = {variant: {seed: {} for seed in SEEDS} for variant in VARIANTS}
        training_rows, scaler_rows, score_rows, event_rows = [], [], [], []
        seeds_to_run = [0] if args.smoke else SEEDS
        paths_to_run = paths[:1] if args.smoke else paths
        keys_to_run = keys[:1] if args.smoke else keys
        log(f"RUN_DIR={run}")
        log(f"PyTorch={torch.__version__}; device=CPU; files={len(paths_to_run)}; seeds={seeds_to_run}")
        for seed in seeds_to_run:
            (run / "models" / f"seed{seed}").mkdir()
            (run / "scores" / f"seed{seed}").mkdir()
            for file_index, path in enumerate(paths_to_run):
                key = data_key(path)
                frame = read_sensor_frame(path)
                reference, test = frame.iloc[:REFERENCE_POINTS], frame.iloc[REFERENCE_POINTS:]
                reference_values, test_values, mean, scale = standardize(reference, test)
                for sensor, sensor_mean, sensor_scale in zip(SENSORS, mean, scale):
                    scaler_rows.append({"seed": seed, "file": key, "sensor": sensor,
                                        "mean": sensor_mean, "scale": sensor_scale})
                reference_sequences = create_sequences(reference_values)
                test_sequences = create_sequences(test_values)
                assert reference_sequences.shape == (341, 60, 8)
                model_seed = seed * 1000 + file_index
                model, history, best_epoch, training_seconds = train_model(reference_sequences, model_seed)
                model_path = run / "models" / f"seed{seed}" / f"{safe_name(key)}_modelseed{model_seed}.pt"
                torch.save({
                    "state_dict": model.state_dict(), "model_seed": model_seed,
                    "architecture": protocol["architecture"], "framework": protocol["framework"],
                }, model_path)
                for row in history:
                    training_rows.append({"seed": seed, "file": key, "model_seed": model_seed,
                                          "best_epoch": best_epoch, "training_seconds": training_seconds, **row})
                reference_total, reference_channels = score_model(model, reference_sequences)
                test_total, test_channels = score_model(model, test_sequences)
                del model
                arrays = health_arrays(reference_total, reference_channels, test_total, test_channels)
                arrays.update({
                    "reference_total": reference_total, "reference_channels": reference_channels,
                    "test_total": test_total, "test_channels": test_channels,
                })
                reference_index = reference.index[N_STEPS - 1 :]
                score_index = test.index[N_STEPS - 1 :]
                assert len(score_index) == len(test_total)
                for variant in VARIANTS:
                    point, cp, events = predict_variant(
                        arrays, reference_index, score_index, test.index, config, variant
                    )
                    predictions[variant][seed][key] = {
                        "point": point, "changepoint": cp, "events": events,
                    }
                    for event in events:
                        event_rows.append({"variant": variant, "seed": seed, "file": key, **event})
                score_path = run / "scores" / f"seed{seed}" / f"{safe_name(key)}.npz"
                np.savez_compressed(
                    score_path, reference_timestamps=reference_index.astype("int64").to_numpy(),
                    test_timestamps=score_index.astype("int64").to_numpy(), **arrays,
                )
                score_rows.append({
                    "seed": seed, "file": key, "model_seed": model_seed,
                    "epochs_completed": len(history), "best_epoch": best_epoch,
                    "training_seconds": training_seconds, "model_path": str(model_path),
                    "model_sha256": sha256(model_path), "score_path": str(score_path),
                    "score_sha256": sha256(score_path),
                })
                log(f"seed={seed} file={file_index + 1}/{len(paths_to_run)} {key}: "
                    f"epochs={len(history)}, best={best_epoch}, time={training_seconds:.2f}s")
        pd.DataFrame(training_rows).to_csv(run / "models" / "training_history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(scaler_rows).to_csv(run / "audit" / "scalers.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(score_rows).to_csv(run / "audit" / "model_score_manifest.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(event_rows).to_csv(run / "predictions" / "events.csv", index=False, encoding="utf-8-sig")
        prediction_path = run / "predictions" / "all_variants.pkl"
        with prediction_path.open("wb") as handle:
            pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for variant in VARIANTS:
            for seed in seeds_to_run:
                frames = [pd.DataFrame({
                    "file": key, "datetime": predictions[variant][seed][key]["point"].index,
                    "point": predictions[variant][seed][key]["point"].to_numpy(),
                    "changepoint": predictions[variant][seed][key]["changepoint"].to_numpy(),
                }) for key in keys_to_run]
                pd.concat(frames, ignore_index=True).to_csv(
                    run / "predictions" / f"{variant}_seed{seed}.csv.gz",
                    index=False, compression="gzip",
                )
        write_json(run / "audit" / "pre_evaluation.json", {
            "predictions_saved_before_labels_loaded": True,
            "prediction_sha256": sha256(prediction_path), "protocol_sha256": protocol_hash,
            "models": len(score_rows), "prediction_sets": len(VARIANTS) * len(score_rows),
        })
        if args.smoke:
            write_json(run / "SMOKE_SUCCESS.json", {
                "status": "PASS", "model_output_shape": [60, 1],
                "training_loss_first": training_rows[0]["loss"],
                "training_loss_last": training_rows[-1]["loss"],
                "run_dir": run,
            })
            log("SMOKE SUCCESS; labels were not loaded.")
            return
        log("All 170 models and 680 prediction sets saved; loading labels for evaluation.")
        labels = {}
        for path in paths:
            frame = pd.read_csv(
                path, sep=";", usecols=["datetime", "anomaly", "changepoint"],
                parse_dates=["datetime"],
            ).set_index("datetime").sort_index().iloc[REFERENCE_POINTS:].astype(int)
            labels[data_key(path)] = frame
        metrics, per_file = [], []
        for variant in VARIANTS:
            for seed in SEEDS:
                points = {key: predictions[variant][seed][key]["point"] for key in keys}
                cps = {key: predictions[variant][seed][key]["changepoint"] for key in keys}
                result = evaluate_predictions(keys, labels, points, cps)
                assert result["event_tps"] + result["event_missed_cps"] == 127
                metrics.append({"variant": variant, "method": VARIANT_NAMES[variant], "seed": seed, **result})
                for key in keys:
                    per_file.append({"variant": variant, "method": VARIANT_NAMES[variant],
                                     "seed": seed, "file": key,
                                     **evaluate_predictions([key], labels, {key: points[key]}, {key: cps[key]})})
                log(f"evaluated {variant} seed={seed}: TP={result['event_tps']}, "
                    f"missed={result['event_missed_cps']}, FP={result['event_fps']}, "
                    f"NAB={result['nab_standard']:.2f}")
        metrics_frame = pd.DataFrame(metrics)
        metrics_frame.to_csv(run / "metrics" / "per_seed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(per_file).to_csv(run / "metrics" / "per_file.csv", index=False, encoding="utf-8-sig")
        metric_names = [column for column in metrics_frame.select_dtypes(include=[np.number]).columns if column != "seed"]
        summary = pd.DataFrame([
            {"variant": variant, "method": VARIANT_NAMES[variant], "metric": metric,
             "mean": metrics_frame.loc[metrics_frame.variant == variant, metric].mean(),
             "std": metrics_frame.loc[metrics_frame.variant == variant, metric].std(ddof=1)}
            for variant in VARIANTS for metric in metric_names
        ])
        summary.to_csv(run / "metrics" / "summary_mean_std.csv", index=False, encoding="utf-8-sig")
        full = metrics_frame[metrics_frame.variant == "full"].set_index("seed")
        paired = []
        for variant in VARIANTS[1:]:
            current = metrics_frame[metrics_frame.variant == variant].set_index("seed")
            for metric in metric_names:
                delta = current[metric] - full[metric]
                paired.append({
                    "variant": variant, "metric": metric,
                    "mean_delta_variant_minus_full": delta.mean(), "std_delta": delta.std(ddof=1),
                    "paired_seed_deltas": json.dumps(delta.tolist()),
                })
        pd.DataFrame(paired).to_csv(run / "metrics" / "paired_differences.csv", index=False, encoding="utf-8-sig")
        create_report(run, summary)
        create_figure(run, summary)
        manifest = pd.DataFrame(score_rows)
        unchanged = all(sha256(path) == digest for path, digest in [(Path(p), h) for p, h in data_hashes.items()])
        hashes_match = all(
            sha256(Path(row.model_path)) == row.model_sha256 and sha256(Path(row.score_path)) == row.score_sha256
            for row in manifest.itertuples(index=False)
        )
        integrity = {
            "status": "PASS", "framework": f"PyTorch {torch.__version__}", "device": "CPU",
            "models_trained": len(score_rows), "expected_models": 170,
            "prediction_sets": len(VARIANTS) * len(score_rows), "expected_prediction_sets": 680,
            "labels_loaded_after_predictions_saved": True, "data_files_unchanged": unchanged,
            "model_and_score_hashes_match": hashes_match,
            "protocol_hash_matches": sha256(run / "protocol_lock.json") == protocol_hash,
            "literature_comparisons_run": False,
        }
        assert all([len(score_rows) == 170, unchanged, hashes_match, integrity["protocol_hash_matches"]])
        write_json(run / "audit" / "integrity.json", integrity)
        (run / "SUCCESS.txt").write_text(
            "PyTorch formal training, evaluation and three ablations completed successfully.\n",
            encoding="utf-8",
        )
        log("SUCCESS: PyTorch method and three ablations completed.")
        print(summary[summary.metric.isin([
            "event_tps", "event_missed_cps", "event_fps",
            "average_detection_delay_seconds", "nab_standard", "point_f1",
            "point_far_percent", "point_mar_percent",
        ])].to_string(index=False))
    except Exception:
        error = traceback.format_exc()
        (run / "FAILURE.txt").write_text(error, encoding="utf-8")
        log(error)
        raise


def create_report(run: Path, summary: pd.DataFrame) -> None:
    wanted = [
        ("event_tps", "事件TP", 2), ("event_missed_cps", "漏检", 2),
        ("event_fps", "事件FP", 2), ("average_detection_delay_seconds", "平均延迟/s", 2),
        ("nab_standard", "NAB Standard", 2), ("point_f1", "点级F1", 4),
        ("point_far_percent", "FAR/%", 2), ("point_mar_percent", "MAR/%", 2),
    ]
    lookup = summary.set_index(["variant", "metric"])
    table = ["| 方法 | " + " | ".join(label for _, label, _ in wanted) + " |",
             "|---|" + "---:|" * len(wanted)]
    for variant in VARIANTS:
        cells = []
        for metric, _, digits in wanted:
            row = lookup.loc[(variant, metric)]
            cells.append(f"{row['mean']:.{digits}f}±{row['std']:.{digits}f}")
        table.append("| " + VARIANT_NAMES[variant] + " | " + " | ".join(cells) + " |")
    report = f"""# PyTorch本文方法与消融实验结果

本文方法已迁移至PyTorch {torch.__version__}，在CPU上重新训练。数据划分、Conv-AE结构、训练参数、检测参数和官方评价口径均与锁定方案保持一致。实验包含34个SKAB文件、5个随机种子，共训练170个模型；消融复用各自种子和文件对应的新重构误差，不重复训练网络。本次未运行任何文献对比方法。

下表为每个种子先汇总34个文件，再计算5个种子的均值±样本标准差：

""" + "\n".join(table) + """

## 消融定义

- 完整方法：总体与通道经验共形证据、3/5持续性投票及前后窗口事件确认全部保留。
- 仅总体证据：去掉通道Simes门控及健康指标中的通道项。
- 去掉持续性投票：把3/5投票改为单点判定（1/1）。
- 去掉第二阶段事件确认：不计算前后窗口事件统计量及事件级p值，直接以持续异常标志的上升沿和下降沿触发事件。

平均延迟仅基于成功检出的事件计算；不同版本的检出事件集合不同，不能脱离漏检和误报单独比较。五个随机种子反映初始化和训练次序造成的波动，不是五套独立数据，也不用于宣称统计显著性。

## 可复现材料

- `protocol_lock.json`：运行前锁定的网络、训练、检测及消融设置。
- `models/`与`models/training_history.csv`：170个PyTorch模型及逐轮损失。
- `scores/`：每个模型生成的参考段和测试段重构误差、p值及健康指标。
- `predictions/`：完整方法和三组消融的逐点预测与事件记录。
- `metrics/`：逐种子、逐文件、均值标准差及与完整方法的配对差值。
- `audit/`：数据、模型、分数、协议和预测保存顺序的完整性记录。
"""
    (run / "实验结果说明.md").write_text(report, encoding="utf-8")


def create_figure(run: Path, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    names = ["Full", "Total only", "No vote", "No stage 2"]
    specs = [
        ("event_missed_cps", "Missed events (lower is better)"),
        ("event_fps", "False event alarms (lower is better)"),
        ("nab_standard", "NAB Standard (higher is better)"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        data = summary[summary.metric == metric].set_index("variant").loc[VARIANTS]
        bars = ax.bar(names, data["mean"], yerr=data["std"], capsize=4,
                      color=["#28679C", "#899EB0", "#DCAD5D", "#AC7474"])
        for bar, value in zip(bars, data["mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10); ax.tick_params(axis="x", labelrotation=22)
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True); ax.margins(y=.18)
    fig.suptitle("PyTorch component ablation: 5 seeds, 34 SKAB files", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(run / "figures" / "ablation_summary.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
