"""
View a dataset
"""

from pathlib import Path

from data.dataset import Dataset
from data.segmentation import run_cellpose
from visualisation.napari import view_dataset

in_path = Path("~/projects/dissertation/data/bacteria").expanduser()

dataset = Dataset(pc_path=in_path / "00",
                  gt_path=in_path / "00_GT" / "TRA",
                  seg_path=in_path / "00_GT" / "SEG"
                  )

view_dataset(dataset)