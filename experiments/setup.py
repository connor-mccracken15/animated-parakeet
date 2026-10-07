"""
Generate initial datasets - converts from TrackMate to CTC format and creates low frame rate (LFR) sets
"""

from pathlib import Path

from data.convert import convert_to_ctc
from data.framerate import create_lfr_dataset
from data.segmentation import run_cellpose
from data.dataset import Dataset

TYPES = ["HEK293", "MDA-MB-231", "MFC10A", "U87"]
N_FRAMES = [1, 2, 4, 8, 16, 32]

def main():
    in_path = Path("~/projects/dissertation/data/lfct").expanduser()

    for t in TYPES:
        dataset = Dataset(pc_path = in_path / t / "01_PC",  
                          rfp_path = in_path / t / "01_RFP",
                          gt_path = in_path / t / "01_GT" / "TRA", 
                          pred_path = in_path / t / "01_PRED",
                          seg_path= in_path / t / "01_GT" / "SEG")

        print(f"Running {t} CTC conversion...")

        dataset.gt_masks, dataset.gt_graph = convert_to_ctc(pc=dataset.pc, xml_path=in_path / t / "01_GT" / "man_track.xml")
        dataset.save_gt()

        print(f"Running {t} Cellpose...")

        dataset.seg_masks = run_cellpose(dataset.pc, dataset.gt_masks)
        dataset.save_seg()

        for n in N_FRAMES:
            print(f"Creating {t} n frames {n} LFR creation...")
            dropped = create_lfr_dataset(dataset, n)

            dropped.pc_dir = in_path / t / f"{n:02d}_PC"
            dropped.rfp_dir = in_path / t / f"{n:02d}_RFP"
            dropped.gt_dir = in_path / t / f"{n:02d}_GT" / "TRA"
            dropped.pred_dir = in_path / t / f"{n:02d}_PRED"
            dropped.seg_dir = in_path / t / f"{n:02d}_GT" / "SEG"

            dropped.save_gt()
            dropped.save_pc()
            dropped.save_rfp()
            dropped.save_seg()

if __name__ == "__main__":
    main()