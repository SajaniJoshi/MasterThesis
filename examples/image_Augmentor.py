import numpy as np
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet.gluon.data import Dataset

# Augmentor class
class SatelliteImageAugmentor:
    def __init__(self, flip_prob=0.5, rotate_prob=0.5, noise_prob=0.3):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob

    def plot_image(self, img, label):
        plt.figure(figsize=(6, 6))
        plt.imshow(np.clip(img.transpose(1, 2, 0), 0, 1))  # Transpose to HxWxC for display
        plt.title(label)
        plt.axis('off')
        plt.show()

    def random_flip(self, img, mask):
        if np.random.rand() < self.flip_prob:
            print('Applying Horizontal Flip')
            img = np.flip(img, axis=2)  # Horizontal flip
            mask = np.flip(mask, axis=2)
            #self.plot_image(img, "After Horizontal Flip")
        if np.random.rand() < self.flip_prob:
            print('Applying Vertical Flip')
            img = np.flip(img, axis=1)  # Vertical flip
            mask = np.flip(mask, axis=1)
            #self.plot_image(img, "After Vertical Flip")
        return img, mask

    def random_rotate(self, img, mask):
        if np.random.rand() < self.rotate_prob:
            angle = np.random.choice([90, 180, 270])
            print(f'Applying Rotation: {angle} degrees')
            img = np.rot90(img, k=angle // 90, axes=(1, 2))
            mask = np.rot90(mask, k=angle // 90, axes=(1, 2))
            #self.plot_image(img, f"After Rotation ({angle} degrees)")
        return img, mask

    def add_noise(self, img):
        if np.random.rand() < self.noise_prob:
            print('Adding Gaussian Noise')
            noise = np.random.normal(0, 0.01, img.shape)  # Gaussian noise
            img = img + noise
            img = np.clip(img, 0, 1)  # Clip values to valid range [0, 1]
            #self.plot_image(img, "After Adding Noise")
        return img

    def augment_with_mask(self, img, mask):
        #self.plot_image(img, "Original Image")
        img, mask = self.random_flip(img, mask)
        img, mask = self.random_rotate(img, mask)
        img = self.add_noise(img)  # Add noise only to the image
        #self.plot_image(img, "Final Augmented Image")
        return img, mask

# Custom Dataset
class GeoTiffDataset(Dataset):
    def __init__(self, image_dict, mask_dict, augmentor=None):
        """
        Args:
            image_dict (dict): Dictionary of GeoTiffMetadata objects for images.
            mask_dict (dict): Dictionary of GeoTiffMetadata objects for masks.
            augmentor (SatelliteImageAugmentor): Data augmentation pipeline.
        """
        assert set(image_dict.keys()) == set(mask_dict.keys()), "Image and mask IDs must match."
        self.image_dict = image_dict
        self.mask_dict = mask_dict
        self.augmentor = augmentor
        self.ids = list(image_dict.keys())  # List of IDs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img = self.image_dict[id_].image  # Access image from GeoTiffMetadata
        mask = self.mask_dict[id_].image  # Access mask from GeoTiffMetadata

        # Ensure images and masks are NumPy arrays
        if not isinstance(img, np.ndarray):
            raise ValueError(f"Image for ID {id_} is not a NumPy array.")
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"Mask for ID {id_} is not a NumPy array.")

        # Apply augmentations if provided
        if self.augmentor:
            img, mask = self.augmentor.augment_with_mask(img, mask)

        # Convert to MXNet NDArray
        img_ndarray = mx.nd.array(img, dtype='float32')
        mask_ndarray = mx.nd.array(mask, dtype='float32')

        return img_ndarray, mask_ndarray
