#!/usr/bin/env python3


### The usual python imports for the notebook
###The usual python imports for the notebook
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import numpy as np
from scipy.ndimage import median_filter
from scipy.constants import speed_of_light
from scipy.signal import convolve2d, medfilt
from scipy.ndimage import sobel
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter
import dask.array as da
from dask_image import ndfilters
from scipy.ndimage import generic_filter
import scipy.ndimage.filters as ndfilters
from scipy.interpolate import interp1d
import os
#######

def medianfilter_array(arr, ws=32):
    """Apply a median filter to an array using Dask.
    
    Works with both xarray and numpy arrays.
    
    Parameters:
    - arr: xarray.DataArray or numpy array to be filtered.
    - ws: window size for the median filter (default is 32).
    
    Returns:
    - Filtered array of the same type as the input.
    """
    chunksize = (ws * 8, ws * 8)
    if isinstance(arr, xr.DataArray):
        inn = arr.values
    else:
        inn = arr
    
    arrb = da.from_array(inn, chunks=chunksize)
    arrfilt = da.map_overlap(median_filter, arrb, depth=ws // 2, size=(ws, ws), mode='reflect', boundary='reflect').compute()
    
    if isinstance(arr, xr.DataArray):
        out = arr.copy()
        out.values = arrfilt
    else:
        out = arrfilt
    return out


def open_geotiff(path, fill_value=0):
    """Open a GeoTIFF file with GDAL and replace NaN values with a specified fill value."""
    try:
        bovl = gdal.Open(path, gdal.GA_ReadOnly)
        if bovl is None:
            raise Exception("Failed to open the GeoTIFF file.")

        band = bovl.GetRasterBand(1)
        bovl_data = band.ReadAsArray()

        # Replace NaN values with the specified fill_value
        bovl_data[np.isnan(bovl_data)] = fill_value

        return bovl_data
    except Exception as e:
        print("Error:", e)
        return None

def export_to_tiff(output_filename, data_array, reference_tif_path):
    """Export a NumPy array to a GeoTIFF file using a reference TIFF for geospatial properties."""
    try:
        # Open the reference TIFF to read its spatial properties
        ref_ds = gdal.Open(reference_tif_path)
        if ref_ds is None:
            raise Exception("Failed to open the reference GeoTIFF file.")
        
        geotransform = ref_ds.GetGeoTransform()
        projection = ref_ds.GetProjection()

        driver = gdal.GetDriverByName("GTiff")
        
        # Get the dimensions of the data array
        row, col = data_array.shape
        
        # Create the output GeoTIFF
        outdata = driver.Create(output_filename, col, row, 1, gdal.GDT_Float32)
        
        # Set the geotransform and projection from the reference TIFF
        outdata.SetGeoTransform(geotransform)
        outdata.SetProjection(projection)
        
        # Write data to the raster band
        outdata.GetRasterBand(1).WriteArray(data_array)
        
        # Flush the cache to disk to write changes
        outdata.FlushCache()
        
        # Cleanup
        ref_ds = None
        outdata = None

        print(f"Exported data to {output_filename} successfully.")

    except Exception as e:
        print("Error:", e)

########

# Main script
homedir = os.getcwd()
spec_frame = ['021D', '014A', '116A', '123D']
spec_frame = ['021D']
for folder in spec_frame:
    folder_path = os.path.join(homedir, folder)
    print(folder_path)
    if os.path.isdir(folder_path):
        for tifs in os.listdir(folder_path):
            if tifs.endswith('geo.azi.tif'):
                output_name = tifs.replace('azi.tif', 'msk.azi.tif')
                output_path = os.path.join(folder_path, output_name)
                if not os.path.exists(output_path):
                    tif_path = os.path.join(folder_path, tifs)
                    tif_arr = open_geotiff(tif_path, fill_value=np.NaN)
                    tif_med = medianfilter_array(tif_arr, ws=64)
                    diff = np.abs(tif_arr - tif_med)
                    mask = np.ones(diff.shape)
                    mask[diff > 1] = 0
                    tif_masked = tif_arr * mask
                    tif_masked = np.where(mask == 0, np.NaN, tif_masked)
                    export_to_tiff(output_path, tif_masked, tif_path)
            elif tifs.endswith('geo.rng.tif'):
                output_name = tifs.replace('rng.tif', 'msk.rng.tif')
                output_path = os.path.join(folder_path, output_name)
                if not os.path.exists(output_path):
                    tif_path = os.path.join(folder_path, tifs)
                    tif_arr = open_geotiff(tif_path, fill_value=np.NaN)
                    tif_med = medianfilter_array(tif_arr, ws=32)
                    diff = np.abs(tif_arr - tif_med)
                    mask = np.ones(diff.shape)
                    mask[diff > 1] = 0
                    tif_masked = tif_arr * mask
                    tif_masked = np.where(mask == 0, np.NaN, tif_masked)
                    export_to_tiff(output_path, tif_masked, tif_path)
