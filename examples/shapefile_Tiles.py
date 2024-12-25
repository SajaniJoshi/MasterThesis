import os 
import geopandas as gpd


def split_shapefile_by_tiles():

    # Load the shapefiles
    big_gdf = gpd.read_file(r"D:\Source\Input\Data\IACS\BB_Antragsdaten_2010_FBS_c1.shp")
    tile_gdf = gpd.read_file(r"D:\Source\Input\Data\Tiles\global_tiles_dnn_ls_bb.shp")
    output_directory = r"D:\Source\Output\IACS_2010"
    os.makedirs(output_directory, exist_ok=True)

    # Ensure both shapefiles have the same CRS
    if big_gdf.crs != tile_gdf.crs:
        tile_gdf = tile_gdf.to_crs(big_gdf.crs)

          # Ensure both shapefiles have the same CRS
    if big_gdf.crs != tile_gdf.crs:
        tile_gdf = tile_gdf.to_crs(big_gdf.crs)

    # Fix invalid geometries with buffer(0)
    big_gdf["geometry"] = big_gdf["geometry"].apply(
        lambda geom: geom.buffer(0) if not geom.is_valid else geom
    )
    tile_gdf["geometry"] = tile_gdf["geometry"].apply(
        lambda geom: geom.buffer(0) if not geom.is_valid else geom
    )

    # Loop through each tile and intersect with the big shapefile
    for idx, tile in tile_gdf.iterrows():
        tile_geometry = tile.geometry

        # Intersect the big shapefile with the tile geometry
        intersected = big_gdf[big_gdf.geometry.intersects(tile_geometry)]

        # Clip the intersected geometries to the tile boundary
        clipped = gpd.clip(intersected, tile_geometry)

        # Check if the clipped GeoDataFrame is empty
        if clipped.empty:
            print(f"No data for tile {idx}, skipping.")
            continue

        # Save the resulting shapefile
        tile_id = int(tile.get('id', idx) if 'id' in tile else idx)  # Use 'id' if present, otherwise index
        print(f'tile_id: {tile_id}')
        output_path = os.path.join(output_directory, f"tile_{tile_id}.shp")
        clipped.to_file(output_path)
        print(f"Saved: {output_path}")

