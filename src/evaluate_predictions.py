from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from project_paths import DATA_ROOT, SKAB_ROOT, PACKAGE_ROOT
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(PACKAGE_ROOT / '.matplotlib'))
sys.path.insert(0, str(SKAB_ROOT))
import numpy as np
import pandas as pd
from core.metrics import chp_score

def evaluate(keys, labels, predictions):
    actual_point = [labels[k].anomaly for k in keys]
    actual_cp = [labels[k].changepoint for k in keys]
    predicted_point = [predictions[k].point for k in keys]
    predicted_cp = [predictions[k].changepoint for k in keys]
    tp, tn, fp, fn = chp_score(actual_point, predicted_point, metric='confusion_matrix', verbose=False)
    delay, missed, event_fp, total = chp_score(actual_cp, predicted_cp, metric='average_time',
        window_width='60s', anomaly_window_destination='righter', verbose=False)
    nab = chp_score(actual_cp, predicted_cp, metric='nab', window_width='60s',
        anomaly_window_destination='righter', verbose=False)
    return {'point_f1': float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0,
            'event_tps': int(total-missed), 'event_missed_cps': int(missed), 'event_fps': int(event_fp),
            'event_true_cps': int(total), 'average_detection_delay_seconds':
            float('nan') if pd.isna(delay) else pd.Timedelta(delay).total_seconds(),
            'nab_standard': float(nab['Standard']), 'nab_low_fp': float(nab['LowFP']),
            'nab_low_fn': float(nab['LowFN']), 'event_detection_rate_percent': (total-missed)/total*100,
            'point_tp': int(tp), 'point_tn': int(tn), 'point_fp': int(fp), 'point_fn': int(fn)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions', type=Path, help='CSV/CSV.GZ with file,datetime,point,changepoint columns')
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, parse_dates=['datetime'])
    predictions = {str(key).replace('/', chr(92)): part.set_index('datetime').sort_index()
                   for key, part in frame.groupby('file')}
    labels = {}
    for key, prediction in predictions.items():
        raw = pd.read_csv(DATA_ROOT / Path(key.replace(chr(92), '/')), sep=';', parse_dates=['datetime'])
        labels[key] = raw.set_index('datetime').sort_index().iloc[400:][['anomaly','changepoint']].astype(int)
        pd.testing.assert_index_equal(prediction.index, labels[key].index)
        if not prediction[['point','changepoint']].isin([0,1]).all().all():
            raise ValueError('Predictions must be binary')
    print(json.dumps(evaluate(sorted(labels), labels, predictions), indent=2))


if __name__ == '__main__':
    main()
