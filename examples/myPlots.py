import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


result_path = r"D:\Source\Output\Result"

def lossPlot(train_losses, loss_path_plot):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_path_plot)
    plt.show()
    
def plotAll(id, originalimage, pred_segm, pred_bound, pred_dists, instanceSegment):
        fig, ax =plt.subplots(1,5, figsize=(30,30))
        ax[0].imshow(originalimage)
        ax[1].imshow(pred_segm)
        ax[2].imshow(pred_bound)
        ax[3].imshow(pred_dists)
        ax[4].imshow(instanceSegment, cmap=plt.get_cmap('prism'), interpolation=None)
        ax[4].set_title(f'Instance Segmentation {id}')
        ax[0].set_title(f'Original {id}')
        ax[1].set_title(f'Extent Mask {id}')
        ax[2].set_title(f'Boundary Mask {id}')
        ax[3].set_title(f'Distance Mask {id}')
        #fig.savefig(os.path.join("D:\Source\Test\data\Output\Fractal\Result", f"{id}_all.tiff"))
        #plt.close(fig)

    
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

def writePredictionImage(id, name, pre_img, orignalMeta):
   path=  os.path.join(result_path, f"{id}_{name}.tif")
   meta = orignalMeta.getMetadata(pre_img)
   with rasterio.open(path, "w", **meta) as dst:
                dst.write(pre_img, 1)  # Write to band 1


def CreatePly(id,currentMetadata,inst):
    inst = inst.astype(np.uint8)  # Use np.uint8 if values are integers in the range 0–255
    mask = inst > 0  # Define the mask for valid data
    polygons = []
    for geom, value in shapes(inst, mask=mask, transform=currentMetadata.transform):
        polygons.append({"geometry": shape(geom), "value": value})
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(polygons, crs=currentMetadata.crs)
    output_shapefile_path=  os.path.join(result_path, f"{id}")

    # Save the polygons as a Shapefile
    gdf.to_file(output_shapefile_path)
    print(f"Saved polygons to {output_shapefile_path}")
     
    
def visualize_all(id, img, currentMetadata, outputs, pred_segm, pred_bound, inst): 
    pred_dists = outputs[2][0,1,:,:].asnumpy() 
    writePredictionImage(id, "extend", pred_segm, currentMetadata)
    writePredictionImage(id, "boundary", pred_bound, currentMetadata)
    writePredictionImage(id, "distance", pred_dists, currentMetadata)
    writePredictionImage(id, "inst_seg", inst, currentMetadata)
    CreatePly(id,currentMetadata,inst)
    if img.ndim == 4 and img.shape[0] == 1:  
        abc = img[0]  # Remove batch dimension
    if abc.ndim == 3 and abc.shape[0] >= 3:  # RGB or multi-channel image
        rgb_image = np.transpose(abc[:3], (1, 2, 0))
             
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    rgb_image = (rgb_image * 255).astype(np.uint8)  # Convert to uint8
    if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:
        rgb_image = np.transpose(rgb_image, (1, 2, 0))  # Convert from (3, H, W) to (H, W, 3)  
    plotAll(id, rgb_image.asnumpy(), pred_segm, pred_bound, pred_dists, inst)