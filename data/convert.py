"""
Converts LFCT dataset graph from Trackmate format (xml) to CTC friendly format (txt) 
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from skimage.draw import disk
import copy
from skimage.draw import disk, polygon

# Load spots and the links between them
def _read_trackmate(xml):
    root = xml.getroot()
    pixel_size = float(root.find('.//ImageData').get('pixelwidth', 1))

    spots = {}
    for s in root.iter('Spot'):
        spots[int(s.get('ID'))] = {
            'frame': int(s.get('FRAME')),
            'x': float(s.get('POSITION_X')) / pixel_size,
            'y': float(s.get('POSITION_Y')) / pixel_size
        }

    children = {spot_id: [] for spot_id in spots}
    for e in root.iter('Edge'):
        a = int(e.get('SPOT_SOURCE_ID'))
        b = int(e.get('SPOT_TARGET_ID'))
        if spots[a]['frame'] > spots[b]['frame']:
            a, b = b, a
        children[a].append(b)

    return spots, children

# Label each spot, new label starts after every division, gap or merge
def _split_into_tracklets(spots, children):
    has_parent = {c for kids in children.values() for c in kids}
    to_visit = [(s, 0) for s in spots if s not in has_parent]
    labels, tracklets = {}, []

    while to_visit:
        spot, parent = to_visit.pop()
        if spot in labels:  # already reached via another parent (merge)
            continue

        label = len(tracklets) + 1
        start = spots[spot]['frame']

        # follow the cell until it divides, skips a frame, merges or ends
        labels[spot] = label
        while (len(children[spot]) == 1
               and children[spot][0] not in labels
               and spots[children[spot][0]]['frame'] == spots[spot]['frame'] + 1):
            spot = children[spot][0]
            labels[spot] = label

        tracklets.append((label, start, spots[spot]['frame'], parent))
        to_visit += [(child, label) for child in children[spot]]

    return labels, tracklets

# Paint each spot as a disc with its label
def _draw_masks(spots, labels, n_frames, height, width):

    masks = np.zeros((n_frames, height, width), np.uint16)
    for spot_id, s in spots.items():
        y = np.clip(s['y'], 0, height - 1)  # keep edge spots inside the image
        x = np.clip(s['x'], 0, width - 1)
        rows, cols = disk((y, x), 3, shape=(height, width))
        masks[s['frame'], rows, cols] = labels[spot_id]

    return masks

# Run conversion
def create_ctc(pc, xml_path):
    height, width = pc[0].shape[:2]

    spots, children = _read_trackmate(ET.parse(xml_path))
    labels, tracklets = _split_into_tracklets(spots, children)
    masks = _draw_masks(spots, labels, len(pc), height, width)

    return masks, np.array(tracklets)