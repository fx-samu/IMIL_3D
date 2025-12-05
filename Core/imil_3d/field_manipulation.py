import numpy as np
from nptyping import NDArray, Shape
import cv2
from imil_3d.field_recognition import best_wh_quad, map_corners
from imil_3d.imiltyping import *
from imil_3d.functions import image_from_imagearr

def image_resize(img_arr: ImageLikeArray, mp: float = 2.5) -> ImageLikeArray:
    """
    Resize a PIL Image so that its total number of pixels is approximately equal to the specified number of megapixels.

    Args:
        img (Image): Input PIL Image to be resized.
        mp (float, optional): Target megapixels (in millions of pixels). Defaults to 2.5.

    Returns:
        Image: Resized PIL Image. If the original image is already below the target megapixels, it is returned unchanged.
    """
    img = image_from_imagearr(img_arr)
    width, height   = img.size
    scale_factor    = (mp * 1e6 / ( width * height ))**(1/2)
    if scale_factor < 1:
        width   = int(width * scale_factor)
        height  = int(height * scale_factor)
        img     = img.resize((width, height))
    
    return np.asarray(img)

def field_mask(img_arr: ImageLikeArray, mask: BoolField2D) -> ImageLikeArray:
    """Given an image array and a mask, applies the mask

    Args:
        img_arr (ImageLikeArray): An image
        mask (BoolField2D): Mask of the image

    Returns:
        ImageLikeArray: Image with mask applied 
    """
    if img_arr.ndim == 3:
        masked = img_arr * mask[..., None]
    else:
        raise ValueError("img_arr is not ImageLikeArray")
    return masked

def unwarp(img_arr: ImageLikeArray, inp_corners: ListPoint2D, 
           width: int = None, height: int = None, orientation: int = 0) -> ImageLikeArray:
    """Given an ImageLikeArray and corners ordered in CCW will give you an unwarp perspective

    Args:
        img_arr (ImageLikeArray): ImageLikeArray to unwarp
        inp_corners (ListPoint2D): Corners in [top-left, top-right, bottom-right, bottom-left]
        width (int, optional): Width of the final warp.Defaults maps to best_wh_quad.
        height (int, optional): Height of the final warp. Defaults maps to best_wh_quad.
        orientation (int, optional): 90 Degree rotation. Defaults to 0.
        
    Returns:
        NDArray: Unwarped input 
    """
    if img_arr.ndim < 2:
        raise ValueError("inp_array must be at least 2D (image like array)")
    if np.asarray(inp_corners).shape != (4, 2):
        raise ValueError("inp_points must be array-like of shape (4, 2) for four corner coordinates")
    
    if not width or not height:
        width, height = best_wh_quad(inp_corners)
    
    dest_points     = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    warp_matrix     = cv2.getPerspectiveTransform(inp_corners, dest_points)
    warped          = cv2.warpPerspective(img_arr, warp_matrix, (int(width), int(height)))
    
    rotated_warped  = np.rot90(warped, k=orientation % 4)
    return rotated_warped

def map_unwarp(img_arr: ImageLikeArray, inp_corners: ListPoint2D, 
           width: int = None, height: int = None, orientation: int = 0) -> ImageLikeArray:
    """Given an ImageLikeArray and corners, will estimate best CCW orientation with corners_map 
    and give you an unwarp perspective

    Args:
        img_arr (ImageLikeArray): ImageLikeArray to unwarp
        inp_corners (ListPoint2D): Corners in [top-left, top-right, bottom-right, bottom-left]
        width (int, optional): Width of the final warp.Defaults maps to best_wh_quad.
        height (int, optional): Height of the final warp. Defaults maps to best_wh_quad.
        orientation (int, optional): 90 Degree rotation. Defaults to 0.
        
    Returns:
        NDArray: Unwarped input 
    """
    out_corners = map_corners(inp_corners)
    return unwarp(img_arr, out_corners, width, height, orientation)

def field_normalization(input_field: NumberField2D) -> NumberField2D:
    """
    Normalizes a numerical field to the [0, 1] range.

    Args:
        input_field (NumberField2D): 2D array-like structure (field) of numbers.

    Returns:
        NumberField2D: Normalized field, same shape as input.
    """
    arr = np.asarray(input_field, dtype=np.float32)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def field_normalization_mask(input_field: NumberField2D, input_mask: BoolField2D) -> NumberField2D:
    arr =  np.asarray(input_field, dtype=np.float32)
    zeroed_arr = np.where(input_mask, arr, 0)
    second_min = np.unique(zeroed_arr)[1]
    max_val = np.max(zeroed_arr)
    if max_val == second_min:
        return np.zeros_like(arr)
    ret_arr = (zeroed_arr - second_min) / (max_val - second_min)
    ret_arr = np.where(input_mask, ret_arr, 0)
    return ret_arr