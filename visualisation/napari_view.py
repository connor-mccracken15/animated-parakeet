"""
View dataset and assocaited masks in napari
"""

import napari
import numpy as np
from skimage.measure import regionprops

def view_dataset(dataset):
    viewer = napari.Viewer()
    viewer.add_image(dataset.imgs, name="video")

    for name, masks in [("gt", dataset.gt_masks), ("pred", dataset.pred_masks)]:

        if masks is None:
            continue

        viewer.add_labels(masks, name=f"{name} masks")

    napari.run()