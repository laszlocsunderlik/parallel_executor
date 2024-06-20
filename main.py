import os
import time
from typing import Optional, Union
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import mapping
from colorama import Fore

# Crop types
CROP_TYPES = {
    'SOY': 5,
    'CORN': 1,
    'SPRING_WHEAT': 23,
    'WINTER_WHEAT': 24
}

def get_states(
        boundaries: Union[gpd.GeoDataFrame, str],
        desired_epsg_crs: Optional[str] = None,
        region_filter: Optional[list[str]] = None,
        region_sub_filter: Optional[list[str]] = None,
        state_limit: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Extract states from GIS shape file.  This function has knowledge of Natural Earth admin level data model for structuring and
    encoding adm1 code, region, sub-region, and is able to filter to retrieve a particular subset of states.

    :param boundaries: a GeoDataFrame containing state boundaries and metadata
    :param desired_epsg_crs: if supplied, transforms the results to the specified crs. For example, to receive the
        state boundary geometries in WGS84, which is `EPSG 4326 https://epsg.io/4326`, you pass the value '4326'
    :param region_filter: if supplied, restricts results to states within the given region(s)
    :param region_sub_filter: if supplied, restricts results to states within the given sub-region(s)
    :param state_limit: if supplied, restricts results to the given number of states
    :return: a GeoDataFrame with the matching states.
    """

    if not isinstance(boundaries, pd.DataFrame):
        boundaries = gpd.read_file(boundaries)

    if desired_epsg_crs:
        current_epsg_crs = boundaries.crs.to_string().split(":")[1]
        if current_epsg_crs != desired_epsg_crs:
            boundaries = boundaries.to_crs("EPSG:" + desired_epsg_crs)
    print(len(boundaries))

    excluded_states = ['Alaska', 'Hawaii']
    boundaries = boundaries.loc[(boundaries['iso_a2'] == 'US') & (~boundaries['name'].isin(excluded_states))]

    print(Fore.YELLOW + f'num locations: {boundaries["adm1_code"].nunique()}', flush=True)
    print(Fore.YELLOW + f'num regions: {boundaries["region"].nunique()}', flush=True)
    print(Fore.YELLOW + f'num sub-regions: {boundaries["region_sub"].nunique()}', flush=True)
    print(Fore.YELLOW + f'num geometries: {len(boundaries)}', flush=True)

    if region_filter:
        boundaries = boundaries.loc[boundaries['region'].isin(region_filter)]
        print(Fore.BLUE + f'filtering for regions: {region_filter} leaves {len(boundaries)} region', flush=True)
    if region_sub_filter:
        boundaries = boundaries.loc[boundaries['region_sub'].isin(region_sub_filter)]
        print(Fore.BLUE + f'filtering for sub-regions: {region_sub_filter} leaves {len(boundaries)} sub-regions', flush=True)

    return boundaries

def get_crs_from_raster(raster_path: str) -> str:
    """
    Extract the raster file crs value. For example, the raster file is in WGS84, 
    which is `EPSG 4326 https://epsg.io/4326`, return the value '4326'.

    :param raster_path (str): the path where the raster file is
    :return: a string with the EPSG value like '4326'
    """
    with rasterio.open(raster_path) as src:
        crs = str(src.crs).split(":")[1]
    return crs

def process_chunk(masked_raster: np.ndarray, crop_value: int, window: Window) -> float:
    """
    Processes a chunk of the masked raster to calculate the crop coverage for a specific crop type.
    
    param: masked_raster (np.ndarray): The masked raster array containing crop data
    param: crop_value (int): The integer value representing the crop type in the raster
    param: window (Window): A rasterio Window object representing the chunk of the raster to process
    return: float, the crop coverage in hectares for the specified chunk
    """
    data = masked_raster[window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]
    crop_mask = data == crop_value
    pixel_area_ha = 30 * 30 / 10000
    crop_coverage_ha = np.sum(crop_mask) * pixel_area_ha
    return crop_coverage_ha

def calculate_crop_coverage(masked_raster: np.ndarray, crop_value: int, chunk_size: int) -> float:
    """
    Calculates the total crop coverage for a specific crop type by processing the masked raster in chunks.

    param: masked_raster (np.ndarray): The masked raster array containing crop data
    param: crop_value (int): The integer value representing the crop type in the raster
    param: chunk_size (int): The size of the chunks to process the raster in (in pixels)
    return: float, the total crop coverage in hectares for the specified crop type
    """
    height, width = masked_raster.shape
    total_coverage = 0.0
    for i in range(0, width, chunk_size):
        for j in range(0, height, chunk_size):
            window = Window(i, j, min(chunk_size, width - i), min(chunk_size, height - j))
            total_coverage += process_chunk(masked_raster, crop_value, window)
    return total_coverage

def process_raster(raster_path: str, state_df: gpd.GeoDataFrame, chunk_size: Optional[int] = 1000) -> pd.DataFrame:
    """
    Processes a raster file to calculate crop coverage for each state in the provided GeoDataFrame.
    
    param: raster_path (str): Path to the raster file to be processed.
    param: state_df (gpd.GeoDataFrame): A GeoDataFrame containing state geometries and associated data.
    param: chunk_size (int, optional): The size of the chunks (in pixels) to divide the raster into for processing.
           Defaults to 1000.
    return: pd.DataFrame: A DataFrame containing the crop coverage data for each state.
            Columns: ['state', 'year', 'soy_coverage_ha', 'corn_coverage_ha', 'spring_wheat_coverage_ha', 'winter_wheat_coverage_ha'].
    """
    year = os.path.basename(raster_path).split('_')[0]

    with rasterio.open(raster_path) as src:
        print(f"Processing raster: {raster_path}")
        results = []
        for _, state in state_df.iterrows():
            state_name = state['name']
            geometry = [mapping(state['geometry'])]
            print(Fore.RED + f'Processing state: {state_name} for {year}', flush=True)
            out_image, _ = mask(src, geometry, crop=True)
            masked_raster = out_image[0]

            soy_coverage = calculate_crop_coverage(masked_raster, CROP_TYPES.get('SOY'), chunk_size)
            print(Fore.GREEN + f'SOY coverage in ha: {soy_coverage}', flush=True)
            corn_coverage = calculate_crop_coverage(masked_raster, CROP_TYPES.get('CORN'), chunk_size)
            print(Fore.GREEN + f'CORN coverage in ha: {corn_coverage}', flush=True)
            spring_wheat_coverage = calculate_crop_coverage(masked_raster, CROP_TYPES.get('SPRING_WHEAT'), chunk_size)
            print(Fore.GREEN + f'SPRING_WHEAT coverage in ha: {spring_wheat_coverage}', flush=True)
            winter_wheat_coverage = calculate_crop_coverage(masked_raster, CROP_TYPES.get('WINTER_WHEAT'), chunk_size)
            print(Fore.GREEN + f'WINTER_WHEAT coverage in ha: {winter_wheat_coverage}', flush=True)

            results.append({
                'state': state_name,
                'year': year,
                'soy_coverage_ha': soy_coverage,
                'corn_coverage_ha': corn_coverage,
                'spring_wheat_coverage_ha': spring_wheat_coverage,
                'winter_wheat_coverage_ha': winter_wheat_coverage
            })

    results_df = pd.DataFrame(results)
    return results_df

def main():
    wd = os.getcwd()
    state_boundaries = gpd.read_file(f"{wd}/data/ne_50m_admin_1_states_provinces/ne_50m_admin_1_states_provinces.shp")

    raster_files = [
        f"{wd}/data/2020_30m_cdls/2020_30m_cdls.tif",
        f"{wd}/data/2021_30m_cdls/2021_30m_cdls.tif",
        f"{wd}/data/2022_30m_cdls/2022_30m_cdls.tif",
        f"{wd}/data/2023_30m_cdls/2023_30m_cdls.tif"
    ]

    all_results = []

    for raster_path in raster_files:
        crs = get_crs_from_raster(raster_path)
        transformed_states = get_states(boundaries=state_boundaries, desired_epsg_crs=crs)
        results_df = process_raster(raster_path, transformed_states)
        all_results.append(results_df)

    final_results_df = pd.concat(all_results, ignore_index=True)
    final_results_df.to_csv("/app/output/simple_out.csv")

if __name__ == '__main__':
    start_time = time.time()

    print("-------------------------------------------------------------------------------------------------")
    print("---------------------------------------  SCRIPT STARTED  ----------------------------------------")
    print("-------------------------------------------------------------------------------------------------")

    main()

    run_time = time.time() - start_time
    print("--------------------------------------  SCRIPT FINISHED  ----------------------------------------")
    print(f"--------------------------- Script finished in {run_time:.1f} seconds --------------------------")
    print("-------------------------------------------------------------------------------------------------")
    print("\n")
