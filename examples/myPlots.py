import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Polygon, MultiPolygon
import pandas as pd
from PIL import Image

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
        
def writePredictionImage(id, name, pre_img, orignalMeta, result_path):
    try:
        path=  os.path.join(result_path, f"{id}_{name}.tif")
        #path_png=  os.path.join(result_path, f"{id}_{name}.png")
        #Image.fromarray(pre_img).save(path_png)
        meta = orignalMeta.getMetadata(pre_img)
        with rasterio.open(path, "w", **meta) as dst:
                dst.write(pre_img, 1)  # Write to band 1
                
    except Exception as e:
        print(f"Error creating prediction image: {e}")

def CreatePly(id,currentMetadata,inst, result_path):
    inst = inst.astype(np.int32) 
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
    return output_shapefile_path

class plotPredictedImape:
    def __init__(self, id, ori, mask, extent, distance, boundary, instSeg, path):
        self.id = id
        self.ori = ori
        self.mask = mask
        self.extent = extent
        self.distance = distance
        self.boundary = boundary
        self.instSeg = instSeg
        self.path = path

def plot_images_with_masks(images, results_path):
    num_images = len(images)
    #fig, axes = plt.subplots(num_images, 7, figsize=(28, num_images * 4), gridspec_kw={'wspace': 0, 'hspace': 0.1})
    fig, axes = plt.subplots(num_images, 7, figsize=(28, num_images * 4))
    
    for i, image_data in enumerate(images):
        # Extract data
        id = image_data.id
        original = image_data.ori.asnumpy()
        original = np.squeeze(original)  # Shape becomes (4, 256, 256)
        original = np.transpose(original[:3, :, :], (1, 2, 0))  # Shape becomes (256, 256, 3)
        original = (original - original.min()) / (original.max() - original.min()) # Normalize to [0.0, 1.0]
        extent = image_data.extent
        boundary = image_data.boundary
        distance = image_data.distance
        ground_truth = image_data.mask.asnumpy()
        ground_truth = np.squeeze(ground_truth)  # Shape becomes (6, 256, 256)
        instance_segmentation = image_data.instSeg

        # Compute Agreement Mask
        # Field pixels: `ground_truth[0] == 1`
        binary_field = ground_truth[0]
        # Boundary pixels: `ground_truth[3] == 1`
        binary_boundary = ground_truth[3]
        # Background pixels: All bands are 0
        binary_background = np.all(ground_truth == 0, axis=0)
        # Step 2: Agreement Mask Initialization
        agreement_mask = np.zeros_like(instance_segmentation, dtype=np.uint8)
        # Step 3: Compute Agreement Mask
        # Green: Correct Detection
        correct_detection = (instance_segmentation > 0) & (binary_field == 1)
        agreement_mask[correct_detection] = 1
        # Yellow: Omission
        omission = (instance_segmentation == 0) & (binary_field == 1)
        agreement_mask[omission] = 2

        # Magenta: Incorrect Detection
        incorrect_detection = (instance_segmentation > 0) & binary_background
        agreement_mask[incorrect_detection] = 3

        # Plot each panel
        axes[i, 0].imshow(original, cmap='gray')
        axes[i, 0].set_title(f"{id} - Original")
        axes[i, 0].axis('off')

        extent_plot = axes[i, 1].imshow(extent, cmap='viridis')
        axes[i, 1].set_title(f"{id} - Extent")
        axes[i, 1].axis('off')
        #plt.colorbar(extent_plot, ax=axes[i, 1], orientation='horizontal')

        boundary_plot = axes[i, 2].imshow(boundary, cmap='magma')
        axes[i, 2].set_title(f"{id} - Boundary")
        axes[i, 2].axis('off')
        #plt.colorbar(boundary_plot, ax=axes[i, 2], orientation='horizontal')

        distance_plot = axes[i, 3].imshow(distance, cmap='plasma')
        axes[i, 3].set_title(f"{id} - Distance")
        axes[i, 3].axis('off')
        #plt.colorbar(distance_plot, ax=axes[i, 3], orientation='horizontal')

        ground_truth = np.argmax(ground_truth, axis=0)  # Shape becomes (256, 256)
        axes[i, 4].imshow(ground_truth, cmap='binary')
        axes[i, 4].set_title(f"{id} - Ground Truth")
        axes[i, 4].axis('off')

        instance_plot = axes[i, 5].imshow(instance_segmentation, cmap='prism', interpolation=None)
        axes[i, 5].set_title(f"{id} - Instance Segmentation")
        axes[i, 5].axis('off')

        # Colormap for agreement
        agreement_cmap = plt.matplotlib.colors.ListedColormap(['black', 'green', 'yellow', 'magenta'])
        agreement_plot = axes[i, 6].imshow(agreement_mask, cmap=agreement_cmap)
        axes[i, 6].set_title(f"{id} - Agreement")
        axes[i, 6].axis('off')

        # Add colorbar for each subplot
        #fig.colorbar(agreement_plot, ax=axes[i, 6], ticks=[0, 1, 2, 3], orientation='horizontal')
        # Add legend below the Agreement Mask
        if i == num_images - 1:  # Add legend only once at the bottom
          colors = ['black', 'green', 'yellow', 'magenta']
          labels = ['Background', 'Correct Detection', 'Omission', 'Incorrect Detection']
          patches = [plt.matplotlib.patches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
          fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize='large')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.1)  # Reduce spacing
    print(results_path)
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    plt.show() 

def plotIOUS(iou_path, outputPath):
    df = pd.read_csv(iou_path)
    iou_values= df['IOU']
    print(iou_values)
    bins = 10

    # Plot Histogram
    plt.figure(figsize=(10, 6))
    counts, edges, _ = plt.hist(iou_values, bins=bins, color='purple', alpha=0.7, edgecolor='black')
    
    # Convert counts to percentages
    print('Count:', counts)
    total_counts = sum(counts)
    print('total_counts:', total_counts)
    percentages = (counts / total_counts) * 100

    # Clear the plot and re-plot with percentages
    plt.clf()
    plt.bar((edges[:-1] + edges[1:]) / 2, percentages, width=np.diff(edges), color='purple', alpha=0.7, edgecolor='black')
     
    # Label axes and title
    plt.xlabel('IoU Value')
    plt.ylabel('Percentage')
    plt.title('Distribution of IoU Values')
    
    # Save and show the plot
    plt.savefig(outputPath, format='png', dpi=150)
    plt.show()

