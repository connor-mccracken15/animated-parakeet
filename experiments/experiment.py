
from pathlib import Path
import datetime

class Experiment:

    def __init__(self, name, datasets, n_frames, in_dir, out_dir, results):
        self.name = name
        self.datasets = datasets
        self.n_frames = n_frames
        self.in_path = Path(in_dir).expanduser()
        self.out_path = Path(out_dir).expanduser()
        self.results = results
        self.datetime = datetime.datetime.now()