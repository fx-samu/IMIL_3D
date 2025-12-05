from PIL import Image
import cv2
import numpy as np

from utils.imiltyping import *


def naive_texture(img_arr: ImageLikeArray) -> ImageLikeArray:
    height, width = img_arr.shape[:2]
    grid_size = max(width, height)
    texture = np.zeros((8 * grid_size, 8 * grid_size, 3), dtype=np.uint8)
    avg_color = avg_color = np.mean(img_arr, axis=(0, 1))
    texture[:] = avg_color
    
    src_transform = np.float32([
        [0, 0], 
        [width, 0],
        [width, height],
        [0,height]
    ])
    dest_transform = np.float32([
        [2*grid_size, 0], 
        [2*grid_size, 2*grid_size], 
        [0, 2*grid_size],
        [0, 0]
    ])
    
    warp_matrix = cv2.getPerspectiveTransform(src_transform, dest_transform)
    warped      = cv2.warpPerspective(img_arr, warp_matrix, (2*grid_size, 2*grid_size))
    
    h, w = warped.shape[:2]
    x_off = 3 * grid_size
    y_off = 2 * grid_size
    texture[y_off:y_off + h, x_off:x_off + w, :warped.shape[2]] = warped
    
    return texture

# def naive_texture(img_arr: ImageLikeArray) -> ImageLikeArray:
#     """Will create a basic texture to fill in a cube in blender

#     Args:
#         img (ImageLikeArray):Input image as a texture

#     Returns:
#         ImageLikeArray: Texture that can be mapped to blender to a cube
#     """
#     width, height = img_arr.shape[:2]
#     grid_size = max(width, height)
#     texture = np.zeros((8 * grid_size, 8 * grid_size, 3), dtype=np.uint8)
#     npy_img = np.array(img_arr)
    
#     avg_color = avg_color = np.mean(npy_img) if npy_img.ndim == 2 else np.mean(npy_img, axis=(0, 1))
#     texture[:] = avg_color
    
#     src_transform = np.array([
#         [0, height], 
#         [width, height],
#         [width, 0],
#         [0,0]
#     ], dtype=np.float32)
#     dest_transform = np.array([
#         [0, 0], 
#         [0, 2*grid_size], 
#         [2*grid_size, 2*grid_size], 
#         [2*grid_size, 0]
#     ], dtype=np.float32)
    
#     transform = cv2.getPerspectiveTransform(src_transform, dest_transform)
#     wrp_img = cv2.warpPerspective(npy_img, transform, (2*grid_size, 2*grid_size))
    
#     h, w = wrp_img.shape[:2]
#     x_off = 3 * grid_size
#     y_off = 2 * grid_size
#     texture[y_off:y_off + h, x_off:x_off + w, :wrp_img.shape[2]] = wrp_img
    
#     return texture