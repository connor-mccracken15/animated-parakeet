"""
Run linking evaluation of a dataset's prediction graph
"""

from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import PointMatcher
from traccuracy.metrics import BasicMetrics, DivisionMetrics

def evaluate_dataset(dataset, threshold=5):
    gt = load_ctc_data(str(dataset.gt_dir), str(dataset.gt_dir / "man_track.txt"), name="gt")
    pred = load_ctc_data(str(dataset.pred_dir), str(dataset.pred_dir / "man_track.txt"), name="pred")

    results, matched = run_metrics(
        gt_data=gt,
        pred_data=pred,
        matcher=PointMatcher(threshold=threshold),
        metrics=[BasicMetrics(), DivisionMetrics(max_frame_buffer=1)],
    )

    return results