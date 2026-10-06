"""
Runs an experiment. Currently setup for testing only
"""

from pathlib import Path

from data.dataset import Dataset
from data.drop_framerate import create_lfr
from data.segmentation import run_cellpose

from evaluation.metrics import evaluate_dataset
from tracking.iou_hungarian import run_iou_hungarian

from visualisation.napari_view import view_dataset

TYPES = ["HEK293", "MDA-MB-231", "MFC10A", "U87"]
N_FRAMES = [2, 4, 8, 16, 32]

in_path = Path("~/projects/dissertation/data/lfct").expanduser()

type = "U87"
n_frames = "01"

dataset_lfct = Dataset(img_path = in_path / type / f"{n_frames}_PC",  
                       gt_path = in_path / type / f"{n_frames}_GT", 
                       pred_path = in_path / type / f"{n_frames}_PRED",
                       seg_path = in_path / type / f"{n_frames}_SEG")


in_path = Path("~/projects/dissertation/data/deepcell/train").expanduser()

type = "020"

dataset_dc = Dataset(img_path = in_path / type / "01",  
                       gt_path = in_path / type / "01_GT", 
                       pred_path = in_path / type / "01_PRED")

view_dataset(dataset_dc)
