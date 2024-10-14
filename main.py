import asyncio
import concurrent.futures
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

class CropTypes:
    SOY = 5
    CORN = 1
    SPRING_WHEAT = 23
    WINTER_WHEAT = 24

class StateProcessor:
    def __init__(self, boundaries: Union[gpd.GeoDataFrame, str], desired_epsg_crs: Optional[str] = None):
        self.boundaries = self._load_boundaries(boundaries)
        self.desired_epsg_crs = desired_epsg_crs

    def _load_boundaries(self, boundaries: Union[gpd.GeoDataFrame, str]) -> gpd.GeoDataFrame:
        if not isinstance(boundaries, pd.DataFrame):
            boundaries = gpd.read_file(boundaries)
        return boundaries

    def get_states(self) -> gpd.GeoDataFrame:
        if self.desired_epsg_crs:
            current_epsg_crs = self.boundaries.crs.to_string().split(":")[1]
            if current_epsg_crs != self.desired_epsg_crs:
                self.boundaries = self.boundaries.to_crs("EPSG:" + self.desired_epsg_crs)

        excluded_states = ['Alaska', 'Hawaii']
        return self.boundaries.loc[(self.boundaries['iso_a2'] == 'US') & (~self.boundaries['name'].isin(excluded_states))]

class RasterProcessor:
    @staticmethod
    def get_crs_from_raster(src: rasterio.DatasetReader) -> str:
        return str(src.crs).split(":")[1]

    @staticmethod
    def process_chunk(masked_raster: np.ndarray, crop_value: int, window: Window) -> float:
        data = masked_raster[window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]
        crop_mask = data == crop_value
        pixel_area_ha = 30 * 30 / 10000
        crop_coverage_ha = np.sum(crop_mask) * pixel_area_ha
        return crop_coverage_ha

    @classmethod
    def calculate_crop_coverage(cls, masked_raster: np.ndarray, crop_value: int, chunk_size: int) -> float:
        height, width = masked_raster.shape
        total_coverage = 0.0
        for i in range(0, width, chunk_size):
            for j in range(0, height, chunk_size):
                window = Window(i, j, min(chunk_size, width - i), min(chunk_size, height - j))
                total_coverage += cls.process_chunk(masked_raster, crop_value, window)
        return total_coverage

class StateCropProcessor:
    def __init__(self, raster_file: str, chunk_size: int):
        self.raster_file = raster_file
        self.chunk_size = chunk_size

    def process_single_state(self, state: gpd.GeoSeries, year: str) -> Dict[str, float]:
        state_name = state['name']
        geometry = [mapping(state['geometry'])]
        print(Fore.RED + f'Processing state: {state_name} for {year}', flush=True)

        with rasterio.open(self.raster_file) as src:
            out_image, _ = mask(src, geometry, crop=True)
            masked_raster = out_image[0]

            coverages = {}
            for crop_name, crop_value in vars(CropTypes).items():
                if not crop_name.startswith('__'):
                    coverages[f'{crop_name.lower()}_coverage_ha'] = RasterProcessor.calculate_crop_coverage(
                        masked_raster, crop_value, self.chunk_size
                    )

        result = {
            'state': state_name,
            'year': year,
        }
        result.update(coverages)

        print(Fore.GREEN + f"Result for {state_name}: {result}")

        return result

class CropCoverageAnalyzer:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    async def async_process_states(self, raster_file: str, state_df: gpd.GeoDataFrame, year: str) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        tasks = []

        state_processor = StateCropProcessor(raster_file, self.chunk_size)

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            for _, state in state_df.iterrows():
                tasks.append(loop.run_in_executor(
                    pool, state_processor.process_single_state, state, year
                ))

            results = await asyncio.gather(*tasks)
        results_df = pd.DataFrame(results)
        return results_df

    async def run(self):
        wd = os.getcwd()
        state_boundaries = gpd.read_file(f"{wd}/data/ne_50m_admin_1_states_provinces/ne_50m_admin_1_states_provinces.shp")
        raster_file = f"{wd}/data/2020_30m_cdls/2020_30m_cdls.tif"
        year = os.path.basename(raster_file).split('_')[0]

        with rasterio.open(raster_file) as src:
            print(f"Processing raster: {src}")

            crs = RasterProcessor.get_crs_from_raster(src)
            state_processor = StateProcessor(state_boundaries, desired_epsg_crs=crs)
            transformed_states = state_processor.get_states()
            results_df = await self.async_process_states(raster_file, transformed_states, year)

        results_df.to_csv("output/simple_out3.csv")

if __name__ == '__main__':
    start_time = time.time()

    print("-------------------------------------------------------------------------------------------------")
    print("---------------------------------------  SCRIPT STARTED  ----------------------------------------")
    print("-------------------------------------------------------------------------------------------------")

    analyzer = CropCoverageAnalyzer(chunk_size=2000)
    asyncio.run(analyzer.run())

    run_time = time.time() - start_time
    print("--------------------------------------  SCRIPT FINISHED  ----------------------------------------")
    print(f"--------------------------- Script finished in {run_time:.1f} seconds --------------------------")
    print("-------------------------------------------------------------------------------------------------")
    print("\n")
