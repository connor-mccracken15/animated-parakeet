import napari
from utils.lfct_dataset import LFCT_Dataset

def view_dataset(dataset):
    viewer = napari.Viewer()
    viewer.add_image(dataset.imgs, name="video")

    if dataset.gt_masks is not None:
        viewer.add_labels(dataset.gt_masks, name="gt masks")

    if dataset.pred_masks is not None:
        viewer.add_labels(dataset.pred_masks, name="pred masks")

    napari.run()