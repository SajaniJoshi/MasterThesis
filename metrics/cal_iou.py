import numpy as np
from skimage.morphology import  disk, dilation
import geopandas as gpd
from common import binarize, load_raster_mask, clean_prediction, smooth_mask, get_gt_pred

def compute_raster_iou(pred_mask, gt_mask, dilation_size=2):
    """Compute IoU for raster-based masks, applying slight dilation."""
    
    # Ensure binary masks
    pred_mask = binarize(pred_mask)
    gt_mask = binarize(gt_mask)

    # Apply dilation to allow slight misalignments
    pred_mask_dilated = dilation(pred_mask, disk(dilation_size))
    gt_mask_dilated = dilation(gt_mask, disk(dilation_size))

    # Compute IoU
    intersection = np.logical_and(pred_mask_dilated, gt_mask_dilated).sum()
    union = np.logical_or(pred_mask_dilated, gt_mask_dilated).sum()

    return intersection / union if union > 0 else 0.0

def compute_shapefile_iou(original_shp, predicted_shp):
    """Compute IoU for vector-based shapefiles."""
    try:
        # Compute intersection and union
        intersection = gpd.overlay(original_shp, predicted_shp, how='intersection')
        union = gpd.overlay(original_shp, predicted_shp, how='union', keep_geom_type=False)

        # Compute IoU
        intersection_area = intersection.geometry.area.sum()
        union_area = union.geometry.area.sum()
        iou =intersection_area / union_area if union_area > 0 else 0.0
        print(f"Intersection over Union (IoU): {iou:.4f}")
        return iou, intersection_area
    except Exception as e:
        print("Error in calculating intersection/union:", e)

def cal_iou_raster_shape(ground_truth_raster, predicted_raster, ground_truth_shp, predicted_shp):
    ground_truth_mask = load_raster_mask(ground_truth_raster, channel=0) # Extract the segmentation mask (not boundary) from Channel 1
    predicted_mask = load_raster_mask(predicted_raster, channel=0)
    predicted_mask = clean_prediction(predicted_mask) # Apply Post-processing (Cleaning & Smoothing)
    predicted_mask = smooth_mask(predicted_mask)
    raster_iou = compute_raster_iou(predicted_mask, ground_truth_mask) # Compute IoU (Raster-based)
    ground_truth_shp, predicted_shp = get_gt_pred(ground_truth_shp, predicted_shp)
    shapefile_iou, intersection_area = compute_shapefile_iou(ground_truth_shp, predicted_shp) # Compute IoU (Shapefile-based)
    print(f"Raster-based IoU (Pixel-wise): {raster_iou:.4f}")
    print(f"Vector-based IoU (Pixel-wise): {shapefile_iou:.4f}")
    return raster_iou, shapefile_iou
