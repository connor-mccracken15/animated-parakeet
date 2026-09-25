from pathlib import Path

from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics

from data.lfct_dataset import LFCT_Dataset

def evaluate_dataset(dataset):

    results, matched = run_metrics(
        gt_data=dataset.to_traccuracy("gt"),
        pred_data=dataset.to_traccuracy("pred"),
        matcher=CTCMatcher(),
        metrics=[CTCMetrics()],
    )

    return results, matched