import os
import re
import numpy as np
import rasterio
from mxnet import nd
from metadata import GeoTiffMetadata


class ImageDict:
    def __init__(self, directory, isMask):
        """
        Initialize the ImageDict class.

        :param directory: Path to the directory containing TIFF files
        :param isMask: Boolean indicating if the files are masks
        """
        self.directory = directory
        self.isMask = isMask

    @staticmethod
    def extract_first_number(filename):
        """
        Extract the first sequence of digits from a filename.

        :param filename: Name of the file as a string
        :return: Extracted number as an integer, or None if no match is found
        """
        filename = str(filename)  # Ensure filename is a string
        match = re.match(r"(\d+)", filename)  # Match the first sequence of digits
        if match:
            return int(match.group(1))  # Convert matched string to integer
        else:
            print(f"Error: No match found in '{filename}'")
            return None

    def load_tif_files(self):
        """
        Load TIFF files from the directory into a dictionary keyed by an extracted ID.

        :return: Dictionary with extracted IDs as keys and GeoTIFF metadata as values
        """
        image_dict = {}
        try:
            count = 0
            for filename in os.listdir(self.directory):  # Use the instance variable 'directory'
                if filename.endswith(".tif"):
                    file_path = os.path.join(self.directory, filename)
                    with rasterio.open(file_path) as src:
                        # Read the image as a NumPy array
                        image = src.read()

                        # Extract ID from the filename
                        id = self.extract_first_number(filename)
                        if id is not None:
                            if not self.isMask:  # Use the instance variable 'isMask'
                                # Scale and convert to float32 for non-mask images
                                image = np.clip(image / 10000.0, 0, 1)
                                image = image.astype('float32')

                            # Assuming GeoTiffMetadata is a defined class to handle metadata
                            metadata = GeoTiffMetadata(src, image)
                            
                            # Add to dictionary based on mask status or filename condition
                            if self.isMask or 'VNIR' in filename:
                                image_dict[id] = metadata
                                count += 1
                            if count == 5:
                                break
        

        except Exception as e:
            print(f"Error loading TIFF files: {e}")
        return image_dict

    def getImage(self, id, image_dict, ctx):
        """
        Retrieve an image by ID from the image dictionary.

        :param id: ID of the image to retrieve
        :param image_dict: Dictionary containing images
        :return: The image as an NDArray with shape (1, channels, height, width), or None if ID is not found
        """
        if id in image_dict:
            img = nd.array(image_dict[id].image)  # Convert to MXNet NDArray
            if img.ndim == 3:
                img = img.expand_dims(axis=0)  # Resulting shape: (1, channels, height, width)
                return img.as_in_context(ctx)
            else:
                print(f"Image with ID {id} is not 3-dimensional.")
                return None
        else:
            print(f"ID {id} is not in the dictionary.")
            return None
