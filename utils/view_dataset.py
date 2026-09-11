import napari

from dataset import Dataset

def view(dataset):
    viewer = napari.Viewer()
    viewer.add_image(dataset.img, name="video")

    if dataset.seg is not None:
        viewer.add_labels(dataset.seg, name="seg masks")

    if dataset.trk_mask_gt is not None:
        viewer.add_labels(dataset.trk_gt, name="gt masks")

    if dataset.trk_mask_pred is not None:
        viewer.add_labels(dataset.trk_pred, name="pred masks")

    napari.run()