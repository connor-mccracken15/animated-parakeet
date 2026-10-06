"""
Bayesian tracking (btrack) linking on label masks.
"""

import numpy as np
import btrack
from btrack.utils import segmentation_to_objects, update_segmentation
from btrack import datasets

# Convert btrack tracks to CTC masks + graph, splitting tracks at gaps
def _to_ctc(masks, tracks):
    pred = update_segmentation(masks, tracks)  # labels = btrack track IDs
    next_id = max(trk.ID for trk in tracks) + 1
    graph, last = [], {}

    for trk in sorted(tracks, key=lambda k: k.t[0]):  # parents before children
        frames = [t for t, d in zip(trk.t, trk.dummy) if not d]
        runs = np.split(np.array(frames), np.where(np.diff(frames) > 1)[0] + 1)
        parent = last.get(trk.parent, 0) if trk.parent not in (0, None, trk.ID) else 0

        for i, run in enumerate(runs):
            label = trk.ID
            if i > 0:  # new label after a gap, linked to previous segment
                label, next_id = next_id, next_id + 1
                for t in run:
                    pred[t][pred[t] == trk.ID] = label
            graph.append((label, run[0], run[-1], parent))
            parent = label

        last[trk.ID] = label

    return pred, np.array(graph)

def run_btrack(dataset, max_search_radius=20):
    masks = dataset.gt_masks
    objects = segmentation_to_objects(masks)

    with btrack.BayesianTracker() as tracker:
        tracker.configure(datasets.cell_config())
        tracker.max_search_radius = max_search_radius
        tracker.tracking_updates = ["MOTION"]
        tracker.append(objects)
        tracker.volume = ((0, masks.shape[2]), (0, masks.shape[1]))
        tracker.track(step_size=100)
        tracker.optimize()
        tracks = tracker.tracks

    return _to_ctc(masks, tracks)