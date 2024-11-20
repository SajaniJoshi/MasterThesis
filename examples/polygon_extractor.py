import cv2
import numpy as np
from shapely.geometry import Polygon
from mxnet import nd

class PolygonExtractor:
    def __init__(self):
        """
        Initialize the PolygonExtractor class.
        """
        pass

    def check_unique_labels(self, segmented_image):
        """
        Checks if the instance-segmented image contains unique labels for each object.
        """
        unique_labels = np.unique(segmented_image)
        print(f"Unique Labels Found: {unique_labels}")
        return unique_labels

    def relabel_connected_components(self, segmented_image):
        """
        Relabels a segmented image to ensure each object has a unique label.
        """
        unique_labels = np.unique(segmented_image)
        new_label_map = np.zeros_like(segmented_image, dtype=np.int32)
        current_label = 1  # Start labeling from 1 (0 is background)

        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            mask = (segmented_image == label).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
            for i in range(1, num_labels):  # Skip 0 (background in connectedComponents)
                new_label_map[labels == i] = current_label
                current_label += 1

        return new_label_map

    def extract_polygons(self, segmented_image):
        """
        Extracts polygons from an instance-segmented image.
        """
        polygons = []
        unique_labels = np.unique(segmented_image)  # Assuming each object has a unique label
        for label in unique_labels:
            if label == 0:  # Skip the background
                continue
            mask = (segmented_image == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                polygons.append(contour)
        return polygons

    def extract_mask_polygons(self, mask):
        """
        Extracts polygons from all slices of a 3D mask corresponding to a given ID.
        """
        polygons = []
        
        # Ensure mask is uint8 and values are 0-255
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        # Process each 2D mask slice
        for i in range(mask.shape[0]):  # Iterate over slices in the mask
            single_channel_mask = mask[i]
            # Ensure binary values for contours
            if single_channel_mask.max() <= 1:  # If in range 0-1, scale to 0-255
                single_channel_mask = (single_channel_mask * 255).astype(np.uint8)
            polygons.extend(self.extract_polygons(single_channel_mask))
        return polygons

    @staticmethod
    def approximate_polygons(contours, epsilon=0.01):
        """
        Approximates polygons using the Ramer-Douglas-Peucker algorithm.
        """
        approx_polygons = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
            approx_polygons.append(approx)
        return approx_polygons

    @staticmethod
    def contours_to_shapely_polygons(contours):
        """
        Converts contours into Shapely polygons.
        """
        polygons = []
        for contour in contours:
            if contour.shape[0] > 2:  # Valid polygons need at least 3 points
                polygons.append(Polygon(contour.squeeze()))
        return polygons

    @staticmethod
    def compute_iou(polygon1, polygon2):
        """
        Computes IoU between two polygons using Shapely.
        """
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        if union == 0:
            return 0
        return intersection / union

    def getPolygons(self, segmented_image, mask):
        """
        Processes the predicted and ground truth images, extracting Shapely polygons.
        """
        predicted_image = nd.array(segmented_image)  
        ground_truth_image = nd.array(mask)

        pred_polygons = self.extract_polygons(predicted_image.asnumpy())
        gt_polygons = self.extract_mask_polygons(ground_truth_image.asnumpy())

        pred_polygons = self.contours_to_shapely_polygons(self.approximate_polygons(pred_polygons))
        gt_polygons = self.contours_to_shapely_polygons(self.approximate_polygons(gt_polygons))
        return pred_polygons, gt_polygons
