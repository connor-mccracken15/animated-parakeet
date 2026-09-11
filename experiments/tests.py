from utils.dataset import Dataset
from pathlib import Path
import utils.simulate as simulate
import utils.view_dataset as view_dataset
import utils.ctc_convert as ctc_convert

dataset_name = "Fluo-N2DH-GOWT1"
root_path = "~/projects/dissertation/data/ctc/training"
version = "01"

ctc_test = Dataset(type="ctc",
                   img_path=Path(root_path, dataset_name, version),
                   seg_path=Path(root_path, dataset_name, (version + "_ST"), "SEG"), 
                   trk_mask_gt_path = Path(root_path, dataset_name, (version + "_GT"), "TRA"),
                   trk_graph_gt_path = Path(root_path, dataset_name, (version + "_GT"), "TRA", "man_track.txt"))

dataset_name = "MFC10A"
root_path = "~/projects/dissertation/data/lfct"
field = "3"
type = "PC"

print(Path(root_path, dataset_name, ("FLD_" + field + "_" + type)))

lfct_test = Dataset(img_path=Path(root_path, dataset_name, ("FLD_" + field + "_" + type)))

view_dataset.view(lfct_test)