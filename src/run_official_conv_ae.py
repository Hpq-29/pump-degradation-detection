import os
import sys
import pickle
import time
from pathlib import Path

from project_paths import PACKAGE_ROOT as WORKSPACE, SKAB_ROOT, DATA_ROOT, OUTPUTS_ROOT
from datetime import datetime
import argparse
OUTPUT_DIR = OUTPUTS_ROOT / "official_conv_ae" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")

os.environ["TEMP"] = str(WORKSPACE / ".tmp")
os.environ["TMP"] = str(WORKSPACE / ".tmp")
os.environ["KERAS_HOME"] = str(WORKSPACE / ".keras")
os.environ["MPLCONFIGDIR"] = str(WORKSPACE / ".matplotlib")
os.environ["PYTHONPYCACHEPREFIX"] = str(WORKSPACE / ".pycache")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"





sys.path.insert(0, str(SKAB_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from core.Conv_AE import Conv_AE
from core.metrics import chp_score
from core.utils import create_sequences


def load_preprocess_skab():
    datasets = []
    for path in sorted(DATA_ROOT.rglob("*.csv")):
        if "anomaly-free" in str(path):
            continue
        frame = pd.read_csv(path, sep=";", index_col="datetime", parse_dates=True)
        sensors = frame.drop(columns=["anomaly", "changepoint"])
        labels = frame[["anomaly", "changepoint"]]
        datasets.append([sensors.iloc[:400], sensors.iloc[400:], labels.iloc[:400], labels.iloc[400:]])
    if len(datasets) != 34:
        raise ValueError(f"Expected 34 anomaly files, got {len(datasets)}")
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        frames = load_preprocess_skab()
        assert all(len(item[0]) == 400 and len(item[1]) > 60 for item in frames)
        print("PASS: 34 SKAB files; fixed 400/401+ split; no training performed.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    n_steps = 60
    quantile = 0.999

    datasets = load_preprocess_skab()

    predicted_outlier = []
    predicted_cp = []
    true_outlier = []
    true_cp = []

    model = Conv_AE()

    total_start = time.time()

    for file_index, data in enumerate(datasets, start=1):
        X_train, X_test, y_train, y_test = data

        print(
            f"\n[{file_index}/{len(datasets)}] "
            f"开始训练，训练数据={X_train.shape}，"
            f"测试数据={X_test.shape}"
        )

        file_start = time.time()

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_sequence = create_sequences(
            scaler.transform(X_train),
            n_steps,
        )

        model.fit(X_sequence)

        if file_index == 1:
            print("第一个模型输入：", model.model.input_shape)
            print("第一个模型输出：", model.model.output_shape)

            with (
                OUTPUT_DIR / "first_model_summary.txt"
            ).open("w", encoding="utf-8") as file:
                model.model.summary(
                    print_fn=lambda line: file.write(line + "\n")
                )

        train_prediction = model.model.predict(
            X_sequence,
            verbose=0,
        )

        residuals = pd.Series(
            np.sum(
                np.mean(
                    np.abs(X_sequence - train_prediction),
                    axis=1,
                ),
                axis=1,
            )
        )

        ucl = residuals.quantile(quantile) * 4 / 3

        X_test_sequence = create_sequences(
            scaler.transform(X_test),
            n_steps,
        )

        test_prediction = model.model.predict(
            X_test_sequence,
            verbose=0,
        )

        cnn_residuals = pd.Series(
            np.sum(
                np.mean(
                    np.abs(X_test_sequence - test_prediction),
                    axis=1,
                ),
                axis=1,
            )
        )

        anomalous_data = cnn_residuals > ucl
        anomalous_data_indices = []

        for data_idx in range(
            n_steps - 1,
            len(X_test_sequence) - n_steps + 1,
        ):
            if np.all(
                anomalous_data[
                    data_idx - n_steps + 1 : data_idx
                ]
            ):
                anomalous_data_indices.append(data_idx)

        prediction = pd.Series(
            data=0,
            index=X_test.index,
            dtype=int,
        )
        prediction.iloc[anomalous_data_indices] = 1

        prediction_cp = prediction.diff().abs()
        prediction_cp.iloc[0] = prediction.iloc[0]

        predicted_outlier.append(prediction)
        predicted_cp.append(prediction_cp)
        true_outlier.append(y_test["anomaly"])
        true_cp.append(y_test["changepoint"])

        file_elapsed = time.time() - file_start

        print(
            f"[{file_index}/{len(datasets)}] 完成，"
            f"用时={file_elapsed:.2f}秒，"
            f"UCL={ucl:.6f}，"
            f"预测突变点={int(prediction_cp.sum())}"
        )

    total_elapsed = time.time() - total_start

    with (
        OUTPUT_DIR / "predicted_outlier.pkl"
    ).open("wb") as file:
        pickle.dump(predicted_outlier, file)

    with (
        OUTPUT_DIR / "predicted_changepoints.pkl"
    ).open("wb") as file:
        pickle.dump(predicted_cp, file)

    print("\n" + "=" * 70)
    print(f"34个文件全部完成，总用时：{total_elapsed:.2f}秒")
    print("=" * 70)

    print("\n点级指标：FAR、MAR、F1")
    binary = chp_score(
        true_outlier,
        predicted_outlier,
        metric="binary",
    )
    print(binary)

    print("\n平均检测延迟")
    average_delay = chp_score(
        true_cp,
        predicted_cp,
        metric="average_time",
        window_width="60s",
        anomaly_window_destination="righter",
    )
    print(average_delay)

    print("\nNAB指标")
    nab = chp_score(
        true_cp,
        predicted_cp,
        metric="nab",
        window_width="60s",
        anomaly_window_destination="righter",
    )
    print(nab)

    with (
        OUTPUT_DIR / "runtime.txt"
    ).open("w", encoding="utf-8") as file:
        file.write(f"总运行时间：{total_elapsed:.2f}秒\n")
        file.write("模型输入：(None, 60, 8)\n")
        file.write("模型输出：(None, 60, 1)\n")


if __name__ == "__main__":
    main()