"""
View dataset, masks and tracks in napari
"""

import napari
import numpy as np
from skimage.measure import regionprops

# Convert CTC masks + graph to napari tracks format
def _to_tracks(masks, graph):
    data = sorted((r.label, t, *r.centroid) for t, frame in enumerate(masks) for r in regionprops(frame))
    parents = {int(l): [int(p)] for l, _, _, p in np.atleast_2d(graph) if p > 0}
    return np.array(data), parents

def view_dataset(dataset):
    viewer = napari.Viewer()

    for name, imgs in [("pc", dataset.pc),
                       ("pc", dataset.rfp)]:

        if imgs is None:
            continue

        viewer.add_image(imgs, name={name})

    for name, masks, graph in [("gt", dataset.gt_masks, dataset.gt_graph),
                               ("pred", dataset.pred_masks, dataset.pred_graph),
                               ("seg", dataset.seg_masks, None)]:
        if masks is None:
            continue

        viewer.add_labels(masks, name=f"{name} masks")

        if graph is not None:
            data, parents = _to_tracks(masks, graph)
            viewer.add_tracks(data, graph=parents, name=f"{name} tracks", tail_length=10)

    napari.run()