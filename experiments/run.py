"""
Runs a tracking method on all cell types and frame rates
"""

import matplotlib.pyplot as plt

from tracking.lap import run_distance_hungarian
from experiments.experiment import Experiment

TYPES = ["HEK293", "MDA-MB-231", "MFC10A", "U87"]
N_FRAMES = [0, 1, 2, 3, 4]


def edge_f1(results):
    return next(r["results"]["Edge F1"] for r in results if r["metric"]["name"] == "BasicMetrics")


exp = Experiment("distance_hungarian", run_distance_hungarian, TYPES, N_FRAMES,
                 "~/projects/dissertation/lfct", "~/projects/dissertation/runs")

results = exp.run()

for t, res in results.items():
    plt.plot(list(res), [edge_f1(r) for r in res.values()], marker="o", label=t)
    
plt.xticks(N_FRAMES, N_FRAMES)
plt.xlabel("Frame step (every nth frame)")
plt.ylabel("Edge F1")
plt.legend()
plt.show()