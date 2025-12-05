from PIL import Image
from typing import Callable
from utils.imiltyping import *
import numpy as np
import functools
import time
import tracemalloc
import torch

def image_from_imagearr(img_arr: ImageLikeArray) -> Image.Image:
    """Given an ImageLikeArray returns as a PIL Image

    Args:
        input_array (ImageLikeArray): Array Image

    Returns:
        Image: PIL Image 
    """
    if isinstance(img_arr, GrayImageLikeArray):
        img_arr = img_arr.reshape(img_arr.shape[:2])
    return Image.fromarray(img_arr)

def imagearr_from_image(input_image: Image) -> ImageLikeArray:
    """Given an input PIL Image returns as a image array

    Args:
        input_image (Image): PIL Image

    Returns:
        ImageLikeArray: Image Array
    """
    img_arr = np.asarray(input_image)
    if isinstance(img_arr, NumberField2D):
        img_arr = img_arr.reshape(*img_arr.shape, 1)
    return img_arr

def image_open(path: str) -> ImageLikeArray:
    """Given a path returns an ImageLikeArray

    Args:
        path (str): Path to image

    Returns:
        ImageLikeArray: Image as an array
    """
    img = Image.open(path)
    ret = imagearr_from_image(img)
    return ret

def gpu_benchmark(func: Callable):
    """
    Decorator to benchmark a function's execution time and memory usage.
    Prints the time taken and peak memory used by the function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gpu_peak_mem = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_peak_mem = torch.cuda.max_memory_allocated() / (1024**2) # in MiB
        print(f"[BENCHMARK] {func.__name__} executed in {end_time - start_time:.6f}s, "
              f"peak memory usage: {peak / 1024:.2f} KiB"
              + (f", GPU peak memory: {gpu_peak_mem:.2f} MiB" if gpu_peak_mem is not None else ""))
        return result

    return wrapper

def benchmark(func: Callable):
    """
    Decorator to benchmark the execution time and RAM usage of a function.
    Prints elapsed wall time and peak memory usage in KiB.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(
            f"[BENCHMARK] {func.__name__} executed in {end_time - start_time:.6f}s, "
            f"peak memory usage: {peak / 1024:.2f} KiB"
        )
        return result

    return wrapper
