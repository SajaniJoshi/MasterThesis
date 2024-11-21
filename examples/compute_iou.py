from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.validation import explain_validity
import matplotlib.pyplot as plt

class ComputeIou():
    def __init__(self, pred_polygons, gt_polygons):
        self.pred_polygons = pred_polygons
        self.gt_polygons = gt_polygons
        
    def validate_polygon(self, polygon):
        """
            Validates a Shapely polygon and provides feedback if invalid.
        """
        if not polygon.is_valid:
            print(f"Invalid Polygon: {explain_validity(polygon)}")
            return False
        return True

    # Helper function to visualize polygons
    def plot_polygon(self, polygon, color='blue'):
        if not polygon.is_empty and polygon.is_valid:
            x, y = polygon.exterior.xy
            plt.plot(x, y, color=color)
            plt.fill(x, y, alpha=0.5, color=color)
        else:
            print("Polygon is invalid or empty. Skipping visualization.")

    # Function to compute IoU safely
    def compute_iou_safe(self, polygon1, polygon2):
        if not self.validate_polygon(polygon1) or not self.validate_polygon(polygon2):
            print("Skipping invalid polygons.")
            return 0

        # Check for empty polygons
        if polygon1.is_empty or polygon2.is_empty:
            print("One or both polygons are empty. Skipping IoU computation.")
            return 0

        # Check for small or degenerate polygons
        if polygon1.area < 1e-6 or polygon2.area < 1e-6:
            print("One or both polygons are too small. Skipping IoU computation.")
            return 0

        # Attempt simplification
        try:
            polygon1 = polygon1.simplify(0.01, preserve_topology=True)
            polygon2 = polygon2.simplify(0.01, preserve_topology=True)
        except Exception as e:
            print(f"Error during simplification: {e}")
            return 0

        # Compute IoU
        try:
            intersection = polygon1.intersection(polygon2).area
            union = polygon1.union(polygon2).area
            return intersection / union if union != 0 else 0
        except Exception as e:
            print(f"Error during IoU computation: {e}")
            return 0
    
    def calculateIOU(self):
        iou_results = []
        for pred_polygon in tqdm(self.pred_polygons, desc="Processing Predictions"):
            for gt_polygon in self.gt_polygons:
                iou = self.compute_iou_safe(pred_polygon, gt_polygon)
                iou_results.append((pred_polygon, gt_polygon, iou))
        return iou_results
    
    def evaluate_segmentation_from_iou(self, iou_results, iou_threshold=0.5):
        """
            Returns:
            precision: Precision score.
            recall: Recall score.
            f1_score: F1 score.
        """
        matched_gt = set()  # Keeps track of matched ground-truth polygons
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives

        # Extract all unique ground-truth polygons for FN calculation
        all_gt_polygons = {id(gt_polygon) for _, gt_polygon, _ in iou_results}
        fn = len(all_gt_polygons)  # Initialize FN as the total number of ground-truth polygons

        # Iterate through IoU results to determine matches
        for pred_polygon, gt_polygon, iou in iou_results:
            if iou >= iou_threshold:
                gt_id = id(gt_polygon)
                if gt_id not in matched_gt:
                    tp += 1  # True Positive
                    matched_gt.add(gt_id)  # Mark this GT as matched
                    fn -= 1  # Decrease FN since this GT is matched
                else:
                    fp += 1  # Predicted polygon matches an already matched GT

        # Calculate FP from unmatched predictions
        unmatched_predictions = len({id(pred_polygon) for pred_polygon, _, _ in iou_results}) - tp
        fp += unmatched_predictions

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1_score

        
             

