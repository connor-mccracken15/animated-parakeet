import tifffile
import numpy as np
from pathlib import Path
import napari

trk_dir = Path("~/projects/dissertation/data/baseline1/Fluo-N2DL-HeLa").expanduser()

trk_files = sorted(trk_dir.glob("*.tif"))
mask_stack = np.stack([tifffile.imread(f) for f in trk_files])

img_dir = Path("~/projects/dissertation/data/training/Fluo-N2DL-HeLa/01").expanduser()

img_files = sorted(img_dir.glob("*.tif"))
img_stack = np.stack([tifffile.imread(f) for f in img_files])

seg_dir = Path("~/projects/dissertation/data/training/Fluo-N2DL-HeLa/01_GT/TRA").expanduser()

seg_files = sorted(seg_dir.glob("*.tif"))
seg_stack = np.stack([tifffile.imread(f) for f in seg_files])

viewer = napari.Viewer()
viewer.add_image(img_stack, name="video")
viewer.add_labels(seg_stack, name="segmentation masks")
viewer.add_labels(mask_stack, name="trackastra masks")
napari.run()