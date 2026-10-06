"""
Runs distance-Hungarian on all cell types and frame rates, plots edge F1
"""

from pathlib import Path

import matplotlib.pyplot as plt

from data.dataset import Dataset
from evaluation.metrics import evaluate_dataset
from tracking.distance_hungarian import run_distance_hungarian

TYPES = ["HEK293", "MDA-MB-231", "MFC10A", "U87"]
N_FRAMES = [1, 2, 4, 8, 16, 32]

in_path = Path("~/projects/dissertation/data/lfct").expanduser()

def edge_f1(results):
    return next(r["results"]["Edge F1"] for r in results if r["metric"]["name"] == "BasicMetrics")

f1 = {t: [] for t in TYPES}

for t in TYPES:
    for n in N_FRAMES:
        d = in_path / t
        dataset = Dataset(img_path=d / f"{n:02d}_PC", gt_path=d / f"{n:02d}_GT", pred_path=d / f"{n:02d}_PRED")

        dataset.pred_masks, dataset.pred_graph = run_distance_hungarian(dataset)
        dataset.save_pred()

        f1[t].append(edge_f1(evaluate_dataset(dataset)))
        print(t, n, f1[t][-1])

for t in TYPES:
    plt.plot(N_FRAMES, f1[t], marker="o", label=t)

plt.xscale("log", base=2)
plt.xticks(N_FRAMES, N_FRAMES)
plt.xlabel("Frame step (every nth frame)")
plt.ylabel("Edge F1")
plt.legend()
plt.show()