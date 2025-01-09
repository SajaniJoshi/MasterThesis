import os
import numpy as np
from rasterio.features import geometry_mask
import geopandas as gpd
from pyproj import CRS

def calculate_iou(mask, prediction):
    """
    Calculate the Intersection over Union (IoU) between a ground truth mask and a predicted mask.
    
    Parameters:
    - mask: Ground truth binary mask (numpy array)
    - prediction: Predicted binary mask (numpy array)
    
    Returns:
    - IoU score (float)
    """
    # Ensure the inputs are binary (0 or 1)
    mask = mask.astype(bool)
    prediction = prediction.astype(bool)

    # Calculate Intersection and Union
    intersection = np.logical_and(mask, prediction).sum()
    union = np.logical_or(mask, prediction).sum()

    if union == 0: # Handle edge case to avoid division by zero
        return 1.0 if intersection == 0 else 0.0

    iou_score = intersection / union  # Calculate IoU
    print(f'IOU:{iou_score}')
    return iou_score



# Function to calculate IoU between shapefile and GeoTIFF
def calculate_iou_2010(id, prediction, currentMetadata):
    mask_path = os.path.join(r'D:\Source\Output\IACS_2010', f'tile_{str(id)}.shp')
    gdf = gpd.read_file(mask_path)
    target_crs = CRS.from_user_input(currentMetadata.crs) # Ensure CRS is passed as a string
    gdf = gdf.to_crs(target_crs)   # Transform the CRS
    mask = geometry_mask(gdf.geometry, transform=currentMetadata.transform, invert=True, out_shape=currentMetadata.shape) # Create a mask from the shapefile geometries
    return calculate_iou(mask, prediction)