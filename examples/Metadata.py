class GeoTiffMetadata():
    def __init__(self, src, img):
        self.profile = src.profile
        self.transform = src.transform
        self.crs = src.crs
        self.shape = (src.height, src.width)  # Shape (height, width)
        self.bounds = src.bounds  # Bounding box
        self.image = img
    
    def getMetadata(self, prediction_to_save): # Extract metadata from currentMetadata
        metadata = {
                "driver": "GTiff",
                "height": self.shape[0],
                "width": self.shape[1],
                "count": 1,
                "dtype": prediction_to_save.dtype.name,
                "crs": self.crs,
                "transform": self.transform,
                }
        return metadata