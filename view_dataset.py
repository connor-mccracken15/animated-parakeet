from pathlib import Path
import napari

from dataset import Dataset

def view(dataset):
    viewer = napari.Viewer()
    viewer.add_image(dataset.img, name="video")

    if dataset.seg is not None:
        viewer.add_labels(dataset.seg, name="seg masks")

    if dataset.trk_gt is not None:
        viewer.add_labels(dataset.trk_gt, name="gt masks")

    if dataset.trk_pred is not None:
        viewer.add_labels(dataset.trk_pred, name="pred masks")

    napari.run()