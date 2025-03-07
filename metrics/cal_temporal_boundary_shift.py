import os
import numpy as np
from skimage import measure
from scipy.spatial.distance import directed_hausdorff
from common import load_raster_mask, saveAsCSV, get_csv_name

def extract_boundary(mask):
    """Extract boundary points from a binary mask using edge detection."""
    contours = measure.find_contours(mask, 0.5)
    if len(contours) == 0:
        return np.array([])  # Return empty if no boundaries exist
    return np.vstack(contours)  # Stack all contour points into a single array

def temporal_boundary_shift(pred_2010, pred_2022):
    """Compute boundary shift distance between 2010 and 2022 predictions."""

    # Extract boundary points from both years
    boundary_2010 = extract_boundary(pred_2010)
    boundary_2022 = extract_boundary(pred_2022)

    # If any boundary is empty, return 0 (no meaningful shift)
    if boundary_2010.shape[0] == 0 or boundary_2022.shape[0] == 0:
        return 0  

    # Compute directed Hausdorff distance in both directions
    dist_2010_to_2022 = directed_hausdorff(boundary_2010, boundary_2022)[0]
    dist_2022_to_2010 = directed_hausdorff(boundary_2022, boundary_2010)[0]

    return max(dist_2010_to_2022, dist_2022_to_2010)  # Return symmetric Hausdorff Distance

def compute_temporal_boundary_shift(pred_2010_file, pred_2022_file):
    tbs_result = []
    pred_2010_dirs = {dir: root for root, dirs, _ in os.walk(pred_2010_file) for dir in dirs} # Pre-index 2010 directories for fast lookup
    for root, dirs, files in os.walk(pred_2022_file):
        for dir in dirs:
            path_2022 = os.path.join(root, dir, f"{dir}_boundary.tif")
            if os.path.exists(path_2022):
                if dir in pred_2010_dirs:
                    path_2010 = os.path.join(pred_2010_dirs[dir], dir, f"{dir}_boundary.tif")
                    if os.path.exists(path_2022) and os.path.exists(path_2010):
                        boundary_2022 = load_raster_mask(path_2022, channel=0)
                        boundary_2010 = load_raster_mask(path_2010, channel=0)
                        tbs = temporal_boundary_shift(boundary_2010, boundary_2022)   # Corrected function call (Replace with actual processing function)
                        print(f"TEMPORAL_BOUNDARY_SHIFT for {dir}: {tbs}")
                        tbs_result.append({"ID": dir, "TEMPORAL_BOUNDARY_SHIFT": tbs})
    return tbs_result

      
def compute_temporal_boundary_shifts():
    pred_2022_file = r"E:\Master_Chemnitz\Output\Result_2022\VNIR\648\result"
    pred_2010_file = r"E:\Master_Chemnitz\Output\Result_2010\VNIR"

    pred_2022_ndv_file = r"E:\Master_Chemnitz\Output\Result_2022\NDV\648\result"
    pred_2010_ndv_file = r"E:\Master_Chemnitz\Output\Result_2010\NDV"

    pred_2022_vnir_band3_file = r"E:\Master_Chemnitz\Output\Result_2022_band3\VNIR\648\result"
    pred_2010_vnir_band3_file = r"E:\Master_Chemnitz\Output\Result_2010_band3\VNIR"

    pred_2022_ndv_band3_file = r"E:\Master_Chemnitz\Output\Result_2022_band3\NDV\648\result"
    pred_2010_ndv_band3_file = r"E:\Master_Chemnitz\Output\Result_2010_band3\NDV"

    pred_2022_vnir_aug_file = r"E:\Master_Chemnitz\Output\Result_2022_band_aug\VNIR\648\result"
    pred_2010_vnir_aug_file = r"E:\Master_Chemnitz\Output\Result_2010_band_aug\VNIR"

    pred_2022_vnir_hp_file = r"E:\Master_Chemnitz\Output\Result_hp_2022\VNIR\648\result"
    pred_2010_vnir_hp_file = r"E:\Master_Chemnitz\Output\Result_hp_2010\VNIR"

    pred_2022_vnir_mix_cut_file = r"E:\Master_Chemnitz\Output\Result_mix_cut_2022\VNIR\648\result"
    pred_2010_vnir_mix_cut_file = r"E:\Master_Chemnitz\Output\Result_mix_cut_2010\VNIR"

    pred_2010_2022 =[(pred_2010_file, pred_2022_file),
                 (pred_2010_ndv_file, pred_2022_ndv_file), (pred_2010_vnir_band3_file, pred_2022_vnir_band3_file),
                 (pred_2010_ndv_band3_file, pred_2022_ndv_band3_file),(pred_2010_vnir_aug_file, pred_2022_vnir_aug_file),
                 (pred_2010_vnir_hp_file, pred_2022_vnir_hp_file), (pred_2010_vnir_mix_cut_file, pred_2022_vnir_mix_cut_file)
                 ]
    for pred2010, pred2022 in pred_2010_2022: #[(pred_2010_file, pred_2022_file)]:
        tbs_result = compute_temporal_boundary_shift(pred2010, pred2022)
        csv_name = get_csv_name(pred2010, 'tbs')
        outputPath = os.path.join(r"D:\Source\Test\MasterThesis\metrics\res_2010", csv_name)
        saveAsCSV(["ID","TEMPORAL_BOUNDARY_SHIFT"], outputPath, tbs_result)

    