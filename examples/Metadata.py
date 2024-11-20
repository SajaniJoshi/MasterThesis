class GeoTiffMetadata():
    def __init__(self, src, img):
        self.profile = src.profile
        self.transform = src.transform
        self.crs = src.crs
        self.shape = (src.height, src.width)  # Shape (height, width)
        self.bounds = src.bounds  # Bounding box
        self.image = img
    
    def getMetadata(currentMetadata, prediction_to_save): # Extract metadata from currentMetadata
        metadata = {
                "driver": "GTiff",
                "height": currentMetadata.shape[0],
                "width": currentMetadata.shape[1],
                "count": 1,
                "dtype": prediction_to_save.dtype.name,
                "crs": currentMetadata.crs,
                "transform": currentMetadata.transform,
                }
        return metadata