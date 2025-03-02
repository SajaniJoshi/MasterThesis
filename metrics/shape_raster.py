import geopandas as gpd
import rasterio
from rasterio.features import rasterize

def to_raster(shapefile_path):
    # Load the shapefile
    fields = gpd.read_file(shapefile_path)

    # Get the bounding box of the shapefile
    bounds = fields.total_bounds  # [minx, miny, maxx, maxy]
    width_height = 256  # Desired raster dimensions
    # Define the affine transform
    transform = rasterio.transform.from_bounds(*bounds, width_height, width_height)
    raster_path = shapefile_path.replace(r"E:\Master_Chemnitz\Output\IACS_2010", r"E:\Master_Chemnitz\Output\mask_2010")
    raster_path =raster_path.replace(".shp", ".tif")

    # Rasterize the shapefile
    with rasterio.open(
        raster_path,  # Output raster file
        'w',
        driver='GTiff',
        height=width_height,
        width=width_height,
        count=1,
        dtype=rasterio.uint8,
        crs=fields.crs,
        transform=transform
    ) as raster:
        out_image = rasterize(
            shapes=((geom, 1) for geom in fields.geometry),
            out_shape=(width_height, width_height),
            transform=transform,
            fill=0,  # Set background pixels
            all_touched=True,
            dtype=rasterio.uint8
    )
        raster.write(out_image, 1)
to_raster(r"E:\Master_Chemnitz\Output\IACS_2010\tile_494.shp")
# Visualize the raster
#with rasterio.open('agricultural_fields_256x256.tif') as src:
    #image = src.read(1)  # Read the first band

#plt.imshow(image, cmap='viridis')
#plt.colorbar(label='Pixel Value')
#plt.title('Rasterized Agricultural Fields (256x256)')
#plt.show()
