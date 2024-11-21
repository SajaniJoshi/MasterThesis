import os
import matplotlib.pyplot as plt

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
    
def visualize_segmentation(id, segmented_image, result_path):
    """
    Visualizes an instance-segmented image with unique labels using a color map.
    """
    plt.figure(figsize=(30, 30))
    plt.imshow(segmented_image, cmap='tab20')  # 'tab20' is a colormap with 20 distinct colors
    plt.colorbar()
    plt.title("Instance-Segmented Image with Labels")
    plt.show()
    
    fig, ax =plt.subplots(figsize=(15,15))
    ax.imshow(segmented_image, cmap=plt.get_cmap('prism'), interpolation=None)
    ax.set_title('Instance Segmentation')
    path=  os.path.join(result_path, f"{id}_prism.tiff")
    plt.savefig(path)