from dataset import Dataset
from pathlib import Path
import simulate
import view_dataset

dataset_name = "PhC-C2DH-U373"
root_path = "~/projects/dissertation/data/ctc/training"
version = "01"

test = Dataset(img_path=Path(root_path, dataset_name, version), 
               seg_path=Path(root_path, dataset_name, (version + "_ST"), "SEG"), 
               trk_gt_path = Path(root_path, dataset_name, (version + "_GT"), "TRA"))

test.trk_pred = simulate.run(test)

view_dataset.view(test)