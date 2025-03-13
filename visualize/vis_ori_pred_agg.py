import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from common import  get_ori_pred


def plot_images_with_masks(idsList, name, is2022):
    results_path = rf"D:\Source\Test\MasterThesis\visualize\{name}"
    images = get_ori_pred(idsList, is2022)
    num_images = len(images)
    if num_images == 0:
        print("No images to display.")
        return

    fig, axes = plt.subplots(num_images, 7, figsize=(28, num_images * 4))
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
        ground_truth = image_data.mask
        ground_truth = np.squeeze(ground_truth)
        instance_segmentation = image_data.instSeg

        # Additional masks computation here
        binary_field = ground_truth[0] == 1
        binary_boundary = ground_truth[3] == 1
        binary_background = np.all(ground_truth == 0, axis=0)
        
        agreement_mask = np.zeros_like(instance_segmentation, dtype=np.uint8)
        agreement_mask[(instance_segmentation > 0) & binary_field] = 1  # Detected instances in field
        agreement_mask[(instance_segmentation == 0) & binary_field] = 2  # Missed instances in field
        agreement_mask[(instance_segmentation > 0) & binary_background] = 3  # False positive detection in background
        

        ax_idx = i * 7  # Compute starting index for current row
        axes[ax_idx].imshow(original, cmap='gray')
        axes[ax_idx].set_title(f"{id} - Original", fontsize=20)
        axes[ax_idx].axis('off')

        axes[ax_idx + 1].imshow(extent, cmap='viridis')
        axes[ax_idx + 1].set_title(f"{id} - Extent", fontsize=20)
        axes[ax_idx + 1].axis('off')

        axes[ax_idx + 2].imshow(boundary, cmap='magma')
        axes[ax_idx + 2].set_title(f"{id} - Boundary", fontsize=20)
        axes[ax_idx + 2].axis('off')

        axes[ax_idx + 3].imshow(distance, cmap='plasma')
        axes[ax_idx + 3].set_title(f"{id} - Distance", fontsize=20)
        axes[ax_idx + 3].axis('off')

        ground_truth = np.argmax(ground_truth, axis=0)
        axes[ax_idx + 4].imshow(ground_truth, cmap='binary')
        axes[ax_idx + 4].set_title(f"{id} - Ground Truth", fontsize=20)
        axes[ax_idx + 4].axis('off')

        axes[ax_idx + 5].imshow(instance_segmentation, cmap='prism')
        axes[ax_idx + 5].set_title(f"{id} - Instance Segmentation", fontsize=20)
        axes[ax_idx + 5].axis('off')

        agreement_cmap = ListedColormap(['black', 'green', 'yellow', 'magenta'])
        axes[ax_idx + 6].imshow(agreement_mask, cmap=agreement_cmap)
        axes[ax_idx + 6].set_title(f"{id} - Agreement", fontsize=20)
        axes[ax_idx + 6].axis('off')
        
    plt.tight_layout()  # Adjust padding as needed
    plt.subplots_adjust(top=0.95, bottom=-0.05, hspace=0.1)  # Adjust bottom to fit the legend if needed

    if num_images > 0:  # Add legend only if there are images
        colors = ['black', 'green', 'yellow', 'magenta']
        labels = ['Background', 'Correct Detection', 'Omission', 'Incorrect Detection']
        patches = [plt.matplotlib.patches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
        fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=20)

    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    plt.show()