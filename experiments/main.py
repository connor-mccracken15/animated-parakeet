from utils.ctc_dataset import CTC_Dataset
from pathlib import Path
import utils.view_dataset as view_dataset
import methods.iou_hungarian as iou

dataset_name = "PhC-C2DL-PSC"
root_path = "~/projects/dissertation/data/ctc"
version = "02"

print(Path(root_path, dataset_name, (version + "_GT"), "TRA", "man_track.txt"))

ctc_test = CTC_Dataset(img_path=Path(root_path, dataset_name, version),
                   seg_path=Path(root_path, dataset_name, (version + "_ST"), "SEG"), 
                   trk_mask_gt_path = Path(root_path, dataset_name, (version + "_GT"), "TRA"),
                   trk_graph_gt_path = Path(root_path, dataset_name, (version + "_GT"), "TRA", "man_track.txt"))



view_dataset.view(ctc_test)