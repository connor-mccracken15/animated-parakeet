"""
View dataset and assocaited masks in napari
"""

import napari
import numpy as np
from skimage.measure import regionprops


def _tracks(masks, graph):
    data = [[r.label, t, *r.centroid]
            for t, frame in enumerate(masks) for r in regionprops(frame)]
    data = np.array(data)
    data = data[np.lexsort((data[:, 1], data[:, 0]))]  # sort by track, then time
    parents = {int(l): [int(p)] for l, _, _, p in graph if p != 0}
    return data, parents


def view_dataset(dataset):
    viewer = napari.Viewer()
    viewer.add_image(dataset.imgs, name="video")

    for name, masks, graph in [("gt", dataset.gt_masks, dataset.gt_graph),
                               ("pred", dataset.pred_masks, dataset.pred_graph)]:
        if masks is None:
            continue
        viewer.add_labels(masks, name=f"{name} masks", opacity=0.5)
        data, parents = _tracks(masks, graph)
        viewer.add_tracks(data, graph=parents, name=f"{name} tracks", tail_length=10)

    napari.run()