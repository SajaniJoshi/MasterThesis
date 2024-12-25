from osgeo import gdal, ogr
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from rasterio.features import rasterize
from affine import Affine
from my_Save import saveLosses

def read_tiff_as_array(tiff_path):
    """Read a TIFF file and return it as a NumPy array."""
    dataset = gdal.Open(tiff_path)
    band = dataset.GetRasterBand(1)  # Assuming single-band
    array = band.ReadAsArray()
    return array, dataset.GetGeoTransform(), dataset.GetProjection()

def reproject_shapefile(shapefile_path, target_crs):
    """Reproject a shapefile to match the CRS of the target TIFF."""
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

def rasterize_shapefile(gdf, reference_array_shape, reference_transform):
    """Rasterize a GeoDataFrame to match the reference array shape and transform."""
    shapes = [(geom, 1) for geom in gdf.geometry]
    rasterized = rasterize(
        shapes,
        out_shape=reference_array_shape,
        transform=reference_transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )
    return rasterized

def calculate_iou(predicted_mask, ground_truth_mask):
    """Calculate the Intersection over Union (IoU) for binary masks."""
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def calculate_ious(val_ids):
    ious = []
    for id in val_ids:
        print(f'ID: {id}')
        # Paths to the files
        shapefile_path =  rf"D:\Source\Output\Result\{id}\{id}.shp"
        tiff_path = rf'D:\Source\Input\Data\2022\BB\XX_Reference_Masks_ResUNetA\{id}.tif'

        # Read the TIFF file to get the ground truth mask and its CRS
        reference_array, gdal_transform, tiff_crs = read_tiff_as_array(tiff_path)
        target_crs = gpd.GeoDataFrame({'geometry': []}, crs=tiff_crs).crs  # Ensure correct CRS format

        # Reproject the shapefile to match the TIFF CRS
        gdf = reproject_shapefile(shapefile_path, target_crs)

        # Convert GDAL transform to Affine
        transform = Affine.from_gdal(*gdal_transform)

        # Rasterize the shapefile to create a predicted mask
        predicted_mask = rasterize_shapefile(gdf, reference_array.shape, transform)

        # Calculate IoU
        iou_score = calculate_iou(predicted_mask, reference_array)
        print(f"IoU Score: {iou_score}")

        ious.append({
                "ID": id,
                "IOU": iou_score
            })
    saveLosses(["ID", "IOU"], r"D:\Source\Output\Loss\IOU.csv", ious)

