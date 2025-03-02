import os
import csv 
import numpy as np
import rasterio
from skimage.morphology import remove_small_objects, closing, disk
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union, polygonize

def getType(input, title):
    method = "using all bands"
    color = 'purple'
    type = 'NDV'
    year = "2022"
    output = input.replace(".csv", "")
    output= f"{output}_{title}.png"
    if '2010' in input:
        year = '2010'
    if 'VNIR' in input:
        type = 'VNIR'
        color = 'lightblue'
    if '_hp_' in input:
        color = "#0065A2"
    if 'NDV' in input:
        type = 'NDV'
        color = 'lightGreen'
    if '_band3_' in input:
        method = "using 3 bands"
    elif '_aug_' in input:
        method = "using augmentation"
    elif '_hp_' in input:
        method = "using different hyperparameters"
    elif '_mix_cut_' in input:
        method = "using Mixup and Cutmix"
        color = '#FC6238'
    return method, color, type, year, output

def get_csv_name(pred_path, name):
    method = "all"
    year = 2022
    type= 'VNIR'
    if '2010' in pred_path:
        year = 2010
    if 'NDV' in pred_path:
        type = 'NDV'
    if "hp" in pred_path:
        method = "hp"
    elif "band3" in pred_path:
        method = "band3"
    elif "aug" in pred_path:
        method = "aug"
    elif "mix_cut" in pred_path:
        method = "mix_cut"
    return f"{name}_{type}_{method}_{year}.csv"

def saveAsCSV(headers, path,  data): 
        with open(path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
            print(f"Data saved to {path}")
            
def clean_prediction(pred_mask, min_size=500):
    """Remove small disconnected predictions (false positives)."""
    return remove_small_objects(pred_mask.astype(bool), min_size=min_size).astype(np.uint8)

def smooth_mask(mask):
    """Applies morphological closing to fill small gaps in prediction."""
    return closing(mask, disk(2))  # Use a disk of radius 2 to smooth boundaries

def binarize(mask, threshold=0.5):
    """Convert a grayscale image to binary (0 or 1)."""
    return (mask > threshold).astype(np.uint8)

def load_raster_mask(filepath, channel=0):
    """Load a specific channel from a raster mask (GeoTIFF)."""
    with rasterio.open(filepath) as src:
        mask = src.read(channel + 1)  # Rasterio uses 1-based indexing
    return mask

def inspect_and_repair_geometries(gdf):
    """
    Inspect and attempt to repair geometries in a GeoDataFrame.
    Returns a new GeoDataFrame with repaired geometries.
    """
    fixed_geometries = []
    for geometry in gdf.geometry:
        if geometry is None:
             fixed_geometries.append(None)
        elif not geometry.is_valid:
            # Clean the geometry using buffer and polygonize to handle complex cases
            cleaned = geometry.buffer(0)
            if not cleaned.is_valid or cleaned.is_empty:
                # Try a more aggressive cleaning technique
                cleaned = geometry.buffer(0).simplify(0.01, preserve_topology=True)
                # Decompose to simple components and try to rebuild the polygon
                if not cleaned.is_valid:
                    simple_polys = polygonize(cleaned.boundary)
                    cleaned = unary_union(list(simple_polys))
            fixed_geometries.append(cleaned)
        else:
            fixed_geometries.append(geometry)
    
    # Create a new GeoDataFrame
    new_gdf = gpd.GeoDataFrame(gdf.drop(columns=['geometry']), geometry=fixed_geometries, crs=gdf.crs)
    return new_gdf

def get_gt_pred(ground_truth_shp, predicted_shp):
    if os.path.exists(ground_truth_shp) and os.path.exists(predicted_shp):
        ground_truth = gpd.read_file(ground_truth_shp)
        predicted = gpd.read_file(predicted_shp)
        # Check for invalid geometries and fix them
        ground_truth = inspect_and_repair_geometries(ground_truth)
        predicted= inspect_and_repair_geometries(predicted)

        if predicted.crs != ground_truth.crs:
            predicted = predicted.to_crs(ground_truth.crs)
            return ground_truth, predicted
    return None, None

def get_res_path(main_path, year):
    path_NDV_all = os.path.join(main_path, f"Result_NDV_all_{year}.csv") 
    path_NDV_band3= os.path.join(main_path, f"Result_NDV_band3_{year}.csv")
    path_VNIR_all= os.path.join(main_path, f"Result_VNIR_all_{year}.csv")
    path_VNIR_aug= os.path.join(main_path, f"Result_VNIR_aug_{year}.csv")
    path_VNIR_band3= os.path.join(main_path, f"Result_VNIR_band3_{year}.csv")
    path_VNIR_hp= os.path.join(main_path, f"Result_VNIR_hp_{year}.csv")
    path_VNIR_mix_cut= os.path.join(main_path,f"Result_VNIR_mix_cut_{year}.csv")
    return path_NDV_all, path_NDV_band3, path_VNIR_all, path_VNIR_aug, path_VNIR_band3, path_VNIR_hp, path_VNIR_mix_cut

def get_csv_paths(year):
    if year == "2022":
        path_NDV_all, path_NDV_band3, path_VNIR_all, path_VNIR_aug, path_VNIR_band3, path_VNIR_hp, path_VNIR_mix_cut = get_res_path(r"D:\Source\Test\MasterThesis\metrics\res_2022","2022")
    elif year == "2010":
        path_NDV_all, path_NDV_band3, path_VNIR_all, path_VNIR_aug, path_VNIR_band3, path_VNIR_hp, path_VNIR_mix_cut = get_res_path(r"D:\Source\Test\MasterThesis\metrics\res_2010","2010")
    return path_NDV_all, path_NDV_band3, path_VNIR_all, path_VNIR_aug, path_VNIR_band3, path_VNIR_hp, path_VNIR_mix_cut
    


    
    