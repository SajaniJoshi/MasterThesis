import os
import numpy as np
import matplotlib.pyplot as plt
import const
from common import  get_ori_perd_2022_2010

def plot_images_with_masks_com(idsList, name, ncols):
    images = get_ori_perd_2022_2010(idsList)
    num_images = len(images)
    if num_images == 0:
        print("No images to display.")
        return
    
    fig, axes = plt.subplots(num_images, ncols, figsize=(28, num_images * 4))
    if num_images == 1:
        axes = np.array([axes])  # Ensure axes is 2-dimensional
    axes = axes.flatten()  # Simplify indexing

    for i, image_data in enumerate(images):
        # Extract data as per provided structure
        id = image_data.id
        original = image_data.ori
        original = np.squeeze(original)
        original = np.transpose(original[:3, :, :], (1, 2, 0))
        original = (original - original.min()) / (original.max() - original.min())
        extent = image_data.extent
        boundary = image_data.boundary
        distance = image_data.distance
        instance_segmentation = image_data.instSeg
        year = 2022
        if not image_data.is2022:
            year = 2010
        
        ax_idx = i * ncols  # Compute starting index for current row
        axes[ax_idx].imshow(original, cmap='gray')
        axes[ax_idx].set_title(f"{id} - Original {(year)}", fontsize=20)
        axes[ax_idx].axis('off')

        axes[ax_idx + 1].imshow(extent, cmap='viridis')
        axes[ax_idx + 1].set_title(f"{id} - Extent {(year)}", fontsize=20)
        axes[ax_idx + 1].axis('off')

        axes[ax_idx + 2].imshow(boundary, cmap='magma')
        axes[ax_idx + 2].set_title(f"{id} - Boundary {(year)}", fontsize=20)
        axes[ax_idx + 2].axis('off')

        axes[ax_idx + 3].imshow(distance, cmap='plasma')
        axes[ax_idx + 3].set_title(f"{id} - Distance {(year)}", fontsize=20)
        axes[ax_idx + 3].axis('off')
        
        axes[ax_idx + 4].imshow(instance_segmentation, cmap='prism')
        axes[ax_idx + 4].set_title(f"{id} - Instance Segmentation {(year)}", fontsize=20)
        axes[ax_idx + 4].axis('off')

    plt.tight_layout()  # Adjust padding as needed
    plt.subplots_adjust(top=0.95, bottom=-0.4,  hspace=0.1)  # Adjust bottom to fit the legend if needed
    results_path =  os.path.join(const.result_path_2022, name)
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    plt.show()