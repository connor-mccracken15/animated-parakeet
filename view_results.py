import tifffile
import numpy as np
from pathlib import Path
import napari

ctc_dir = Path("C:/Users/Connor/Documents/University/Bath/Dissertation/Data/output")

tif_files = sorted(ctc_dir.glob("*.tif"))
mask_stack = np.stack([tifffile.imread(f) for f in tif_files])

viewer = napari.Viewer()
viewer.add_labels(mask_stack, name="ctc masks")
napari.run()