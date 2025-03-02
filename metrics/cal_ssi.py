import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.ops import unary_union

def load_raster_image(path):
    """ Load a raster image with rasterio and convert it to an 8-bit grayscale image. """
    with rasterio.open(path) as src:
        img = src.read(1)  # Read the first band
        img_normalized = (img - img.min()) / (img.max() - img.min()) * 255
        img_uint8 = img_normalized.astype(np.uint8)
        return img_uint8

def find_contours(binary_image):
    """ Find contours in a binary image using OpenCV. """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def compactness(contour):
    """ Compute the compactness of a contour. """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0  # Avoid division by zero
    return (4 * np.pi * area) / (perimeter ** 2)

def shape_similarity_by_compactness(contour1, contour2):
    """ Calculate shape similarity index based on compactness of two contours. """
    comp1 = compactness(contour1)
    comp2 = compactness(contour2)
    ssi = 1 - abs(comp1 - comp2)  # Compute SSI
    return max(0, ssi)  # Ensure SSI is non-negative

def visualize_contours(image, contours):
    """ Visualize contours on the image. """
    vis_image = np.zeros_like(image)
    cv2.drawContours(vis_image, contours, -1, (255, 0, 0), 3)  # Draw in red
    plt.figure(figsize=(6, 6))
    plt.imshow(vis_image, cmap='gray')
    plt.title('Contours Visualized')
    plt.show()
    
def shape_similarity_index(ground_truth_path, predicted_mask_path):
    # Load the images
    ground_truth = load_raster_image(ground_truth_path)
    predicted_mask = load_raster_image(predicted_mask_path)
    
    # Threshold the images to get binary images
    _, binary_gt = cv2.threshold(ground_truth, 128, 255, cv2.THRESH_BINARY)
    _, binary_pred = cv2.threshold(predicted_mask, 128, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours_gt = find_contours(binary_gt)
    contours_pred = find_contours(binary_pred)
    
    # Assuming each image contains one main object
    if contours_gt and contours_pred:
        # Compute SSI based on compactness
        largest_contour_gt = max(contours_gt, key=cv2.contourArea)
        largest_contour_pred = max(contours_pred, key=cv2.contourArea)
        ssi_compactness = shape_similarity_by_compactness(largest_contour_gt, largest_contour_pred)
        print(f"Shape Similarity Index (SSI) based on Compactness: {ssi_compactness}")
        return ssi_compactness
        
        # Optionally visualize contours
        #visualize_contours(ground_truth, contours_gt)
        #visualize_contours(predicted_mask, contours_pred)
    else:
        print("No suitable contours found in one or both images.")
        
def calculate_shape_similarity_index(ground_truth, predicted):
    # Combine all geometries in each file into a single geometry to simplify the calculation
    ground_truth_geom = unary_union(ground_truth.geometry)
    predicted_geom = unary_union(predicted.geometry)
     # Calculate the area of intersection and union
    intersection = ground_truth_geom.intersection(predicted_geom).area
    union = ground_truth_geom.union(predicted_geom).area
     # Calculate Shape Similarity Index (SSI), equivalent to Jaccard Index here
    ssi = intersection / union
    print("Shape Similarity Index:", ssi)
    return ssi

