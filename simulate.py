"""
Online tracking simulation. Walks the sequence one frame at a time, links each, and writes masks into dataset
"""

import numpy as np
import iou_hungarian


def _track_frame(seg, tracked_prev, next_id, iou_threshold):

    if tracked_prev is None:
        matches = {}
    else:
        matches = iou_hungarian.link(tracked_prev, seg, iou_threshold)

    label_to_id = {label: track_id for track_id, label in matches.items()}

    tracked = np.zeros_like(seg)
    for label in np.unique(seg):
        if label == 0:
            continue
        if label in label_to_id:
            track_id = label_to_id[label]
        else:
            track_id = next_id
            next_id += 1
        tracked[seg == label] = track_id

    return tracked, next_id


def run(dataset, iou_threshold=0.1):
    masks = dataset.seg
    trk_pred = np.zeros_like(masks)
    next_id = 1
    tracked_prev = None

    for k, seg in enumerate(masks):
        tracked, next_id = _track_frame(seg, tracked_prev, next_id, iou_threshold)
        trk_pred[k] = tracked
        tracked_prev = tracked

    return trk_pred