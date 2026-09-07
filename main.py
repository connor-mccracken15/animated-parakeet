from dataset import Dataset
import simulate
import view_dataset

test = Dataset(img_path="~/projects/dissertation/data/ctc/training/BF-C2DL-HSC/01", seg_path="~/projects/dissertation/data/ctc/training/BF-C2DL-HSC/01_ST/SEG", trk_gt_path = "~/projects/dissertation/data/ctc/training/BF-C2DL-HSC/01_GT/TRA")
test.trk_pred = simulate.run(test)

view_dataset.view(test)