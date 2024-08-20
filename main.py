import asyncio
import concurrent.futures
import functools
import os
import time
from typing import Optional, Union, Dict

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

def get_states(boundaries: Union[gpd.GeoDataFrame, str], desired_epsg_crs: Optional[str] = None) -> gpd.GeoDataFrame:
    if not isinstance(boundaries, pd.DataFrame):
        boundaries = gpd.read_file(boundaries)

    if desired_epsg_crs:
        current_epsg_crs = boundaries.crs.to_string().split(":")[1]
        if current_epsg_crs != desired_epsg_crs:
            boundaries = boundaries.to_crs("EPSG:" + desired_epsg_crs)

    excluded_states = ['Alaska', 'Hawaii']
    boundaries = boundaries.loc[(boundaries['iso_a2'] == 'US') & (~boundaries['name'].isin(excluded_states))]

    return boundaries

def get_crs_from_raster(src: rasterio.DatasetReader) -> str:
    return str(src.crs).split(":")[1]
    

def process_chunk(masked_raster: np.ndarray, crop_value: int, window: Window) -> float:
    data = masked_raster[window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]
    crop_mask = data == crop_value
    pixel_area_ha = 30 * 30 / 10000
    crop_coverage_ha = np.sum(crop_mask) * pixel_area_ha
    return crop_coverage_ha

def calculate_crop_coverage(masked_raster: np.ndarray, crop_values: Dict[str, int], chunk_size: int) -> Dict[str, float]:
    height, width = masked_raster.shape
    total_coverage = {crop: 0.0 for crop in crop_values.keys()}
    
    for i in range(0, width, chunk_size):
        for j in range(0, height, chunk_size):
            window = Window(i, j, min(chunk_size, width - i), min(chunk_size, height - j))
            
            for crop_name, crop_value in crop_values.items():
                total_coverage[crop_name] += process_chunk(masked_raster, crop_value, window)
    
    return total_coverage

async def async_process_crop_type(masked_raster: np.ndarray, crop_values: Dict[str, int], chunk_size: int) -> Dict[str, float]:
    loop = asyncio.get_running_loop()
    partial_func = functools.partial(calculate_crop_coverage, masked_raster, crop_values, chunk_size)
    
    with concurrent.futures.ThreadPoolExecutor() as pool:
        total_coverage = await loop.run_in_executor(pool, partial_func)
    
    return total_coverage

async def process_raster(
        raster: rasterio.DatasetReader, 
        state_df: gpd.GeoDataFrame, 
        year: str,
        chunk_size: Optional[int] = 1000) -> pd.DataFrame:
    results = []
    
    for _, state in state_df.iterrows():
        state_name = state['name']
        geometry = [mapping(state['geometry'])]
        print(Fore.RED + f'Processing state: {state_name} for {year}', flush=True)
        out_image, _ = mask(raster, geometry, crop=True)
        masked_raster = out_image[0]

        tasks = [
            async_process_crop_type(masked_raster, {crop_name: crop_value}, chunk_size)
            for crop_name, crop_value in CROP_TYPES.items()
        ]

        coverages = await asyncio.gather(*tasks)

        result = {
            'state': state_name,
            'year': year,
        }
        for coverage, (crop_name, _) in zip(coverages, CROP_TYPES.items()):
            result[f'{crop_name.lower()}_coverage_ha'] = coverage[crop_name]

        results.append(result)

    results_df = pd.DataFrame(results)
    return results_df

async def main(chunk_size):
    wd = os.getcwd()
    state_boundaries = gpd.read_file(f"{wd}/data/ne_50m_admin_1_states_provinces/ne_50m_admin_1_states_provinces.shp")
    raster_file = f"{wd}/data/2020_30m_cdls/2020_30m_cdls.tif"
    year = os.path.basename(raster_file).split('_')[0]

    all_results = []
    with rasterio.open(raster_file) as src:
        print(f"Processing raster: {src}")

        crs = get_crs_from_raster(src)
        transformed_states = get_states(boundaries=state_boundaries, desired_epsg_crs=crs)
        results_df = await process_raster(src, transformed_states, year, chunk_size)
        all_results.append(results_df)

    final_results_df = pd.concat(all_results, ignore_index=True)
    final_results_df.to_csv("output/simple_out.csv")

if __name__ == '__main__':
    start_time = time.time()

    print("-------------------------------------------------------------------------------------------------")
    print("---------------------------------------  SCRIPT STARTED  ----------------------------------------")
    print("-------------------------------------------------------------------------------------------------")

    asyncio.run(main(chunk_size=5000))

    run_time = time.time() - start_time
    print("--------------------------------------  SCRIPT FINISHED  ----------------------------------------")
    print(f"--------------------------- Script finished in {run_time:.1f} seconds --------------------------")
    print("-------------------------------------------------------------------------------------------------")
    print("\n")
