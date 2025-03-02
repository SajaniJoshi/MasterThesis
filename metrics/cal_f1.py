import numpy as np
import cv2
from skimage.morphology import dilation, disk
from sklearn.metrics import f1_score
from common import binarize, load_raster_mask

def preprocess_boundary(binary_image, dilate_size=1):
    """Extract boundaries and apply slight dilation."""
    # Convert to uint8
    binary_image = binary_image.astype(np.uint8)

    # Edge detection with optimized Canny thresholds
    edges = cv2.Canny(binary_image * 255, 75, 175)  # Adjusted thresholds

    # Dilation to allow slight boundary shifts (smaller size)
    edges_dilated = dilation(edges, disk(dilate_size))

    return edges_dilated > 0  # Convert to boolean mask

def compute_f1_score(ground_truth_path, prediction_path):
    """Compute the F1-score for boundary matching."""

    # Ensure binary format
    gt_binary = binarize(ground_truth_path)
    pred_binary = binarize(prediction_path)

    # Preprocess boundaries with minimal dilation
    gt_edges = preprocess_boundary(gt_binary, dilate_size=1)
    pred_edges = preprocess_boundary(pred_binary, dilate_size=1)

    # Convert to 1D boolean arrays for F1-score calculation
    gt_flat = gt_edges.flatten()
    pred_flat = pred_edges.flatten()

    # Compute F1-score
    f1 = f1_score(gt_flat, pred_flat, average='binary')
    print("Final Optimized F1-Score for Boundary Matching:", f1)
    return f1

def cal_f1_boundary(ground_truth_raster, predicted_raster_boundary):
    ground_truth_boundary= load_raster_mask(ground_truth_raster, channel=3)
    predicted_raster_boundary= load_raster_mask(predicted_raster_boundary, channel=0)
    return compute_f1_score(ground_truth_boundary, predicted_raster_boundary)

def calculate_f1_shp(ground_truth, predicted, intersection_area):
    ground_truth['class'] = 1
    predicted['class'] = 1
    total_predicted = predicted.geometry.area.sum()
    total_ground_truth = ground_truth.geometry.area.sum()
    precision = intersection_area / total_predicted if total_predicted else 0
    recall = intersection_area / total_ground_truth if total_ground_truth else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return f1




