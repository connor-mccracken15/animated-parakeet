from pathlib import Path
import datetime
import json

from data.dataset import Dataset
from evaluation.metrics import evaluate_dataset


class Experiment:

    def __init__(self, name, tracker, datasets, n_frames, in_dir, out_dir):
        self.name = name
        self.tracker = tracker
        self.datasets = datasets
        self.n_frames = n_frames
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.datetime = datetime.datetime.now()
        self.run_name = f"{name}_{self.datetime:%Y%m%d_%H%M%S}"
        self.results = {t: {} for t in datasets}

    def run(self):
        for t in self.datasets:
            for n in self.n_frames:
                d = self.in_path / t
                pred_path = self.run_path / t / f"{n:02d}_PRED"
                pred_path.mkdir(parents=True, exist_ok=True)

                ds = Dataset(pc_path=d / f"{n:02d}_PC", gt_path=d / f"{n:02d}_GT" / "TRA", pred_path=pred_path)
                ds.pred_masks, ds.pred_graph = self.tracker(ds)
                ds.save_pred()

                self.results[t][n] = evaluate_dataset(ds)
                self.save()
        return self.results

    def save(self):
        self.run_path.mkdir(parents=True, exist_ok=True)
        with open(self.run_path / "results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)