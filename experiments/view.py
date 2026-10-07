"""
View a dataset
"""

from pathlib import Path

from data.dataset import Dataset
from data.segmentation import run_cellpose
from visualisation.napari import view_dataset

in_path = Path("~/projects/dissertation/data/lfct/HEK293").expanduser()

dataset = Dataset(pc_path=in_path / "01_PC",
                  rfp_path=in_path / "01_RFP",
                  gt_path=in_path / "01_GT" / "TRA",
                  pred_path=in_path / "01_PRED" / "TRA",
                  seg_path=in_path / "01_GT" / "SEG"
                  )

dataset.masks_seg = run_cellpose(dataset)

dataset.save_seg()