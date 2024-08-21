import asyncio
import concurrent.futures
import os
import time
from typing import Optional, Union, Dict, Iterable
import functools

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

def partition_raster(masked_raster: np.ndarray, chunk_size: int) -> Iterable:
    height, width = masked_raster.shape
    for i in range(0, width, chunk_size):
        for j in range(0, height, chunk_size):
            window = Window(i, j, min(chunk_size, width - i), min(chunk_size, height - j))
            yield masked_raster[window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]

def process_partition(window_raster: np.ndarray, crop_values: list[int]) -> Dict[int, float]:
    total_coverage = {crop_value: 0.0 for crop_value in crop_values}
    pixel_area_ha = 30 * 30 / 10000

    for crop_value in crop_values:
        crop_mask = window_raster == crop_value
        total_coverage[crop_value] += np.sum(crop_mask) * pixel_area_ha

    return total_coverage

def merge_coverages(coverage1: Dict[int, float], coverage2: Dict[int, float]) -> dict[int, float]:
    merged_coverage = coverage1.copy()  
    for crop_value, coverage in coverage2.items():
        if crop_value in merged_coverage:
            merged_coverage[crop_value] += coverage
        else:
            merged_coverage[crop_value] = coverage
    return merged_coverage

def process_single_state(
        state: gpd.GeoSeries,
        raster_file: str,
        year: str,
        chunk_size: int) -> dict[str, float]:

    state_name = state['name']
    geometry = [mapping(state['geometry'])]
    print(Fore.RED + f'Processing state: {state_name} for {year}', flush=True)

    with rasterio.open(raster_file) as src:
        out_image, _ = mask(src, geometry, crop=True)
        masked_raster = out_image[0]

        crop_values = list(CROP_TYPES.values())
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            partitions = list(partition_raster(masked_raster, chunk_size))
            results = executor.map(functools.partial(process_partition, crop_values=crop_values), partitions)

            total_coverage = functools.reduce(merge_coverages, results)

        coverages = {f'{crop_name.lower()}_coverage_ha': total_coverage[crop_value]
                     for crop_name, crop_value in CROP_TYPES.items()}

    result = {
        'state': state_name,
        'year': year,
    }
    result.update(coverages)

    return result

async def async_process_states(
        raster_file: str, 
        state_df: gpd.GeoDataFrame, 
        year: str,
        chunk_size: int) -> pd.DataFrame:
    
    loop = asyncio.get_running_loop()
    tasks = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for _, state in state_df.iterrows():
            tasks.append(loop.run_in_executor(
                pool, process_single_state, state, raster_file, year, chunk_size
            ))

        results = await asyncio.gather(*tasks)
    
    results_df = pd.DataFrame(results)
    return results_df

async def main(chunk_size: int):
    wd = os.getcwd()
    state_boundaries = gpd.read_file(f"{wd}/data/ne_50m_admin_1_states_provinces/ne_50m_admin_1_states_provinces.shp")
    raster_file = f"{wd}/data/2020_30m_cdls/2020_30m_cdls.tif"
    year = os.path.basename(raster_file).split('_')[0]

    with rasterio.open(raster_file) as src:
        print(f"Processing raster: {src}")

        crs = get_crs_from_raster(src)
        transformed_states = get_states(boundaries=state_boundaries, desired_epsg_crs=crs)
        results_df = await async_process_states(raster_file, transformed_states, year, chunk_size)

    results_df.to_csv("output/simple_out3.csv")

if __name__ == '__main__':
    start_time = time.time()

    print("-------------------------------------------------------------------------------------------------")
    print("---------------------------------------  SCRIPT STARTED  ----------------------------------------")
    print("-------------------------------------------------------------------------------------------------")

    asyncio.run(main(chunk_size=500))

    run_time = time.time() - start_time
    print("--------------------------------------  SCRIPT FINISHED  ----------------------------------------")
    print(f"--------------------------- Script finished in {run_time:.1f} seconds --------------------------")
    print("-------------------------------------------------------------------------------------------------")
    print("\n")
    '''
    Key Points:
        Nested Parallelism:

        The outer ProcessPoolExecutor handles state-level parallelism.
        The inner ProcessPoolExecutor handles partition-level parallelism within each state.
        merge_coverages:

        Merges the dictionaries from each partition into a single dictionary per state.
        process_partition:

        Processes a single partition to calculate the crop coverage for all crop types.
    '''