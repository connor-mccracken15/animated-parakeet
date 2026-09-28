"""
Run basic evaluation of a datasets prediction mask/graph
"""

from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics

def evaluate_dataset(dataset):

    results, matched = run_metrics(
        gt_data=dataset.to_traccuracy("gt"),
        pred_data=dataset.to_traccuracy("pred"),
        matcher=CTCMatcher(),
        metrics=[CTCMetrics()],
    )