"""
Converts LFCT dataset graph from Trackmate format (xml) to CTC friendly format (txt) 
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import tifffile
from skimage.draw import disk

# Load spots and the links between them
def read_trackmate(xml_path):
    root = ET.parse(xml_path).getroot()
    pixel_size = float(root.find('.//ImageData').get('pixelwidth', 1))

    spots = {}
    for s in root.iter('Spot'):
        spots[int(s.get('ID'))] = {
            'frame': int(s.get('FRAME')),
            'x': float(s.get('POSITION_X')) / pixel_size,
            'y': float(s.get('POSITION_Y')) / pixel_size,
            'radius': float(s.get('RADIUS')) / pixel_size,
        }

    children = {spot_id: [] for spot_id in spots}
    for e in root.iter('Edge'):
        a, b = int(e.get('SPOT_SOURCE_ID')), int(e.get('SPOT_TARGET_ID'))
        if spots[a]['frame'] > spots[b]['frame']:
            a, b = b, a
        children[a].append(b)

    return spots, children

# Label each spot, new label starts after every division
def split_into_tracklets(spots, children):
    has_parent = {c for kids in children.values() for c in kids}
    to_visit = [(s, 0) for s in spots if s not in has_parent]
    labels, tracklets = {}, []

    while to_visit:
        spot, parent = to_visit.pop()
        label = len(tracklets) + 1
        start = spots[spot]['frame']

        # follow the cell until it divides or ends
        labels[spot] = label
        while len(children[spot]) == 1:
            spot = children[spot][0]
            labels[spot] = label

        tracklets.append((label, start, spots[spot]['frame'], parent))
        to_visit += [(child, label) for child in children[spot]]

    return labels, tracklets

# Paint each spot as a disc with its label
def draw_masks(spots, labels, n_frames, height, width):

    masks = np.zeros((n_frames, height, width), np.uint16)
    for spot_id, s in spots.items():
        rr, cc = disk((s['y'], s['x']), 10, shape=(height, width))
        masks[s['frame'], rr, cc] = labels[spot_id]
        
    return masks

# Run conversion
def trackmate_to_ctc(in_path):

    gt_dir = Path(in_path, '01_GT', 'TRA').expanduser()
    img_dir = Path(in_path, '01_PC').expanduser()
    
    images = sorted(img_dir.glob('*.tif'))
    height, width = tifffile.imread(images[0]).shape[:2]

    spots, children = read_trackmate(gt_dir / 'tracking_trackmate.xml')
    labels, tracklets = split_into_tracklets(spots, children)
    masks = draw_masks(spots, labels, len(images), height, width)

    for t, mask in enumerate(masks):
        tifffile.imwrite(gt_dir / f'track{t:03d}.tif')

    with open(gt_dir / 'track_ctc.txt', 'w') as f:
        for tracklet in tracklets:
            f.write(' '.join(map(str, tracklet)) + '\n')