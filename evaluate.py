import argparse
from pathlib import Path

from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import PointMatcher
from traccuracy.metrics import DivisionMetrics, BasicMetrics

def evaluate(in_path, dataset, pred_path):

    gt_dir = Path(in_path).expanduser() / dataset / "01_GT" / "TRA"
    gt_trk_dir = gt_dir / "man_track.txt"

    print(gt_dir)
    print(gt_trk_dir)

    gt_data = load_ctc_data(str(gt_dir), str(gt_trk_dir), name="GT")

    print("Loaded dataset")

    pred_dir = str(Path(pred_path).expanduser() / dataset)
    pred_trk_dir = str(pred_dir / "man_track.txt")

    pred_data = load_ctc_data(pred_dir, pred_trk_dir, name="prediction")

    results, matched = run_metrics(
        gt_data=gt_data,
        pred_data=pred_data,
        matcher=PointMatcher(),
        metrics=[DivisionMetrics(), BasicMetrics()]
        )

    return results, matched

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tracking dataset.")
    parser.add_argument("--pred-path", type=str, required=True)
    parser.add_argument("--gt-path", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()