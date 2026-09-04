import argparse
from pathlib import Path

from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import PointMatcher
from traccuracy.metrics import DivisionMetrics, BasicMetrics

def evaluate(gt_path, pred_path):

    gt_dir = Path(gt_path).expanduser()
    gt_trk_dir = gt_dir / "man_track.txt"

    print(gt_dir)
    print(gt_trk_dir)

    gt_data = load_ctc_data(str(gt_dir), str(gt_trk_dir), name="GT")

    print("Loaded dataset")

    pred_dir = Path(pred_path).expanduser()
    pred_trk_dir = pred_dir / "man_track.txt"

    pred_data = load_ctc_data(str(pred_dir), str(pred_trk_dir), name="prediction")

    results, matched = run_metrics(
        gt_data=gt_data,
        pred_data=pred_data,
        matcher=PointMatcher(threshold=0.5),
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
    results, matched = evaluate(args.gt_path, args.pred_path)
    print(results)

if __name__ == "__main__":
    main()