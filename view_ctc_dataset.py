import napari
import tifffile
import numpy as np
from pathlib import Path

# Get sorted list of tif files
folder = Path("C:/Users/Connor/Documents/University/Bath/Dissertation/Data/Cell Tracking Challenge/testing/Fluo-N2DH-SIM+/01")

files = sorted(folder.glob("*.tif"))

stack = np.stack([tifffile.imread(f) for f in files])

viewer = napari.Viewer()
viewer.add_image(stack, name="video")
napari.run()