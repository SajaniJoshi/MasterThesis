import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Polygon, MultiPolygon
import pandas as pd

def lossPlot(train_losses, loss_path_plot):
    df = pd.DataFrame(train_losses)
    print(df.columns)
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df["Current Epoch"], df["Training Loss"], label="Training Loss", marker='o')
    plt.plot(df["Current Epoch"], df["Validation Loss"], label="Validation Loss", marker='s')
    plt.title("Training and Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    output_path = rf"{loss_path_plot}\loss.png"
    plt.savefig(output_path, format='png', dpi=300)
    plt.show()
    # Close the plot to free memory
    plt.close()
    
def plotAll(id, originalimage, pred_segm, pred_bound, pred_dists, instanceSegment, result_path):
        fig, ax =plt.subplots(1,5, figsize=(30,30))
        ax[0].imshow(originalimage)
        ax[1].imshow(pred_segm, cmap='binary')
        ax[2].imshow(pred_bound, cmap='binary')
        ax[3].imshow(pred_dists, cmap='binary')
        ax[4].imshow(instanceSegment, cmap=plt.get_cmap('prism'), interpolation=None)
        ax[4].set_title(f'Instance Segmentation {id}')
        ax[0].set_title(f'Original {id}')
        ax[1].set_title(f'Extent Mask {id}')
        ax[2].set_title(f'Boundary Mask {id}')
        ax[3].set_title(f'Distance Mask {id}')
        output_path = rf"{result_path}\{id}_all.png"
        plt.savefig(output_path, format='png', dpi=150)
        # Show the plot
        plt.show()
        plt.close(fig)  # Close the figure to free up memory
        
def visualize_segmentation(id, segmented_image, result_path):
    """
    Visualizes an instance-segmented image with unique labels using a color map.
    """
    plt.figure(figsize=(15, 15))
    plt.imshow(segmented_image, cmap='tab20')  # 'tab20' is a colormap with 20 distinct colors
    plt.colorbar()
    plt.title("Instance-Segmented Image with Labels")
    plt.show()
    
    fig, ax =plt.subplots(figsize=(15,15))
    ax.imshow(segmented_image, cmap=plt.get_cmap('prism'), interpolation=None)
    ax.set_title('Instance Segmentation')
    path=  os.path.join(result_path, f"{id}_prism.tiff")
    #fig.savefig(path)
    #plt.close(fig)

def writePredictionImage(id, name, pre_img, orignalMeta, result_path):
    try:
        path=  os.path.join(result_path, f"{id}_{name}.tif")
        meta = orignalMeta.getMetadata(pre_img)
        with rasterio.open(path, "w", **meta) as dst:
                dst.write(pre_img, 1)  # Write to band 1
    except Exception as e:
        print(f"Error creating prediction image: {e}")


def CreatePly(id,currentMetadata,inst, result_path):
    inst = inst.astype(np.uint8)  # Use np.uint8 if values are integers in the range 0–255
    mask = inst > 0  # Define the mask for valid data
    polygons = []
    for geom, value in shapes(inst, mask=mask, transform=currentMetadata.transform):
        polygons.append({"geometry": shape(geom), "value": value})
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(polygons, crs=currentMetadata.crs)
    output_shapefile_path=  os.path.join(result_path, f"{id}")
    os.makedirs(output_shapefile_path, exist_ok=True)

    # Save the polygons as a Shapefile
    gdf.to_file(output_shapefile_path)
    print(f"Saved polygons to {output_shapefile_path}")
    return output_shapefile_path
     
    
def visualize_all(id, img, currentMetadata, outputs, pred_segm, pred_bound, inst, result_path): 
    pred_dists = outputs[2][0,1,:,:].asnumpy() 
    output_shapefile_path = CreatePly(id,currentMetadata,inst, result_path)
    writePredictionImage(id, "extend", pred_segm, currentMetadata, output_shapefile_path)
    writePredictionImage(id, "boundary", pred_bound, currentMetadata, output_shapefile_path)
    writePredictionImage(id, "distance", pred_dists, currentMetadata, output_shapefile_path)
    writePredictionImage(id, "inst_seg", inst, currentMetadata, output_shapefile_path)
    if img.ndim == 4 and img.shape[0] == 1:  
        abc = img[0]  # Remove batch dimension
    if abc.ndim == 3 and abc.shape[0] >= 3:  # RGB or multi-channel image
        rgb_image = np.transpose(abc[:3], (1, 2, 0))
             
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    rgb_image = (rgb_image * 255).astype(np.uint8)  # Convert to uint8
    if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:
        rgb_image = np.transpose(rgb_image, (1, 2, 0))  # Convert from (3, H, W) to (H, W, 3)  
    plotAll(id, rgb_image.asnumpy(), pred_segm, pred_bound, pred_dists, inst, output_shapefile_path)
    print(f'Complete plotting {id}')