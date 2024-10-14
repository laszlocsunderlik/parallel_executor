# CropCoverage Analyzer

## Overview

CropCoverage Analyzer is a high-performance, scalable solution for processing large-scale geospatial raster data to calculate crop coverage across the United States. This project leverages cutting-edge technologies and optimized algorithms to efficiently analyze Cropland Data Layer (CDL) rasters, providing valuable insights into agricultural land use patterns.

## Key Features

- **Asynchronous Processing**: Utilizes Python's `asyncio` for concurrent execution, maximizing CPU utilization.
- **Multiprocessing**: Employs a process pool to parallelize state-level computations, dramatically reducing processing time.
- **Efficient Raster Analysis**: Implements a chunked processing approach to handle large raster files without excessive memory usage.
- **Geospatial Data Handling**: Seamlessly integrates `geopandas` and `rasterio` for robust geospatial operations.
- **Flexible CRS Handling**: Automatically aligns coordinate reference systems between vector and raster data.
- **Configurable Crop Types**: Easily extendable to analyze different crop types as needed.

## Technical Stack

- Python 3.8+
- NumPy & Pandas for data manipulation
- Rasterio for raster data processing
- GeoPandas for vector data handling
- Colorama for enhanced console output
- Asyncio for async process
- Process pool for multiprocessing
- Sliding window for raster calculation

## Quick Start

### Build your Docker image

```
docker build -t your_image_name .
```
### Run the Docker container to mount local folders

```
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output your_image_name
```