import numpy as np
import mxnet as mx
from sklearn.model_selection import train_test_split
from examples.image_Augmentor import GeoTiffDataset, SatelliteImageAugmentor

class TrainValDataLoader:
    
    def __init__(self, image_dict, mask_dict, augmentation=False):
        """
        Initializes the data loader with images and masks dictionaries.
        :param image_dict: Dictionary of images.
        :param mask_dict: Dictionary of corresponding masks.
        :param augmentation: Boolean flag to enable data augmentation.
        """
        self.augmentation = augmentation
        self.image_dict = image_dict
        self.mask_dict = mask_dict
        self.train_ids, self.val_ids = train_test_split(list(image_dict.keys()), test_size=0.2, random_state=42)

    def create_sub_dict(self, ids, original_dict):
        """ Filter dictionary by keys. """
        return {id: original_dict[id] for id in ids}

    def prepare_data_loaders(self, batch_size=4):
        """
        Prepares train and validation data loaders.
        :param batch_size: Batch size for data loaders.
        :return: Tuple of (train_loader, val_loader)
        """
        train_loader = self.get_data_loader(self.train_ids, batch_size, self.augmentation)
        val_loader = self.get_data_loader(self.val_ids, batch_size, augment=False)
        return train_loader, val_loader

    def get_data_loader(self, ids, batch_size, augment):
        """
        Creates a data loader for the given IDs.
        :param ids: List of image/ mask ids to include.
        :param batch_size: Batch size for the DataLoader.
        :param augment: Boolean flag to enable augmentation.
        :return: DataLoader for the given image and mask ids.
        """
        images = mx.nd.array(np.array([self.image_dict[id].image for id in ids]))
        masks = mx.nd.array(np.array([self.mask_dict[id].image for id in ids]))

        if augment:
            image_dict = self.create_sub_dict(ids, self.image_dict)
            mask_dict = self.create_sub_dict(ids, self.mask_dict)
            augmentor = SatelliteImageAugmentor()
            dataset = GeoTiffDataset(image_dict, mask_dict, augmentor=augmentor)
        else:
            dataset = mx.gluon.data.ArrayDataset(images, masks)

        return mx.gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=augment, num_workers=0)

