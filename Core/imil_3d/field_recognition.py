# from a raw image to detection of the actual object
from nptyping import Floating
from imil_3d.imiltyping import * 
from imil_3d.functions import gpu_benchmark

from segment_anything import SamPredictor, sam_model_registry
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import cv2
import torch
import numpy as np

import skimage as ski
from itertools import combinations

import matplotlib.pyplot as plt

def yolo_bounding_box(img_arr: ImageLikeArray) -> ListPoint2D:
    # TODO: Actually implement this with some kind of object detection
    raise NotImplementedError

def magic_bounding_box(img_n: int, img_mp: float) -> ListPoint2D:
    # Magical function that somehow seems to know what is the bounding box in O(1)
    bbox = dict[tuple, ListPoint2D]()
    bbox[(1, 2.5)]  = np.array([[225, 75], [1075, 1800]], dtype = np.int16) # 2.5 MP 1.jpg
    bbox[(1, 1)]    = np.array([[140, 45], [670, 1125]], dtype = np.int16)  # 1 MP 1.jpg
    bbox[(1, 0.1)]  = np.array([[45, 15], [210, 355]], dtype = np.int16)    # 0.1MP 1.jpg
    bbox[(2, 0.5)]  = np.array([[75, 0], [745, 799]], dtype = np.int16)     # 0.5 MP 2.jpg
    bbox[(3, 0.5)]  = np.array([[140, 75], [495, 775]], dtype = np.int16)   # 0.5 MP 3.jpg
    bbox[(4, 1)]    = np.array([[120, 10], [675, 1050]], dtype = np.int16)
    bbox[(5, 1)]    = np.array([[150, 250], [475, 960]], dtype = np.int16)
    bbox[(6, 1)]    = np.array([[250, 100], [625, 1050]], dtype = np.int16)
    bbox[(7, 1)]    = np.array([[250, 60], [625, 1050]], dtype = np.int16)
    bbox[(11, 2.5)] = np.array([[375, 300], [950, 1500]], dtype = np.int16)
    bbox[(13, 1)]   = np.array([[200, 10], [600, 900]], dtype = np.int16)
    bbox[(12, 0.5)]   = np.array([[100, 50], [500, 750]], dtype = np.int16)
       
         
    ret = bbox.get((img_n, img_mp), None)
    
    if type(ret) == type(None):
        raise NotImplementedError("The magical function hasn't been implemented yet for your image soonTM.")
    return ret

def sam_mask(img_arr: ImageLikeArray, bounding_box: ListPoint2D, 
             sam_path: str, model_type: str = "vit_b", device: str = "cuda") -> BoolField2D:
    """Generates a mask for the object in the bounding box with SAM

    Args:
        img_arr (ImageLikeArray): Image Like Array
        bounding_box (ListPoint2D): List of 2 points 2D of the corners (sup left, inf right) bounding box of the object
        sam_path (str): Sam checkpoint path
        model_type (str, optional): Sam configuration to use. Defaults to "vit_b".
        device (str, optional): Device to use for inference. Defaults to "cuda".

    Returns:
        BoolField2D: Mask of the object
    """
    sam_model = sam_model_registry[model_type](checkpoint=sam_path)
    sam_model.to(device=device)
    predictor = SamPredictor(sam_model)
    
    predictor.set_image(img_arr)
    mask, _, _ = predictor.predict(box = bounding_box, multimask_output = False)
    
    del predictor, sam_model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    mask = mask.reshape(mask.shape[1], mask.shape[2])
    return mask

def sam2_mask(img_arr: ImageLikeArray, bounding_box: ListPoint2D,
              sam_path: str, config_path:str, device: str = "cuda") -> BoolField2D:
    """Generates a mask for the object in the bounding box with SAMv2

    Args:
        img_arr (ImageLikeArray): Image Like Array
        bounding_box (ListPoint2D): List of 2 points 2D of the corners (sup left, inf right) bounding box of the object
        sam_path (str): SAMv2 checkpoint path
        config_path (str): Sam configuration to use
        device (str, optional): Device to use for inference. Defaults to "cuda".

    Returns:
        BoolField2D: Mask of the object
    """
    sam2_model = build_sam2(config_path, sam_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    bounding_box_list = bounding_box.reshape(2*bounding_box.shape[0])
    
    predictor.set_image(img_arr)
    mask, _, _ = predictor.predict(box = bounding_box_list, multimask_output = False)
    
    del predictor, sam2_model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    mask = mask.reshape(mask.shape[1], mask.shape[2]).astype(bool)
    return mask

def corners_rect(img_arr: ImageLikeArray) -> ListPoint2D:
    """Finds the corners of the largest object in an image by fiting the minimal area rectangle to it

    Args:
        img_array (ImageLikeArray): Image input

    Raises:
        ValueError: If there is no connected object found in the image

    Returns:
        ListPoint2D: The corners of the rectangle
    """
    # Convert to grayscale if multi-channel
    mask                = (img_arr != 0).any(axis=-1).astype(np.uint8) if img_arr.ndim == 3 else (img_arr != 0).astype(np.uint8)
    num_labels, labels  = cv2.connectedComponents(mask, connectivity=8)
    
    if num_labels > 1:
        largest = max(range(1, num_labels), key=lambda lab: (labels==lab).sum())
        mask    = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        raise ValueError("No contours found in the mask.")
    
    cnt     = max(contours, key=cv2.contourArea)
    rect    = cv2.minAreaRect(cnt)
    corners = cv2.boxPoints(rect)
        
    return corners

def corners_hough(img_arr: ImageLikeArray, treshold: float = 0.25) -> ListPoint2D:
    """Finds the corners of the largest object in an image by finding hough lines

    Args:
        img_arr (ImageLikeArray): Image input
        treshold (float): percentage peak minimum acceptable for a peak in hough transform
        
    Raises:
        Exception: if perfect 4 Hough lines could not be fit to the object

    Returns:
        ListPoint2D: The corners of the object fitted
    """
    def compute_intersection(line1, line2):
        # Each line is in (angle, dist) form: rho = x*cos(theta) + y*sin(theta)
        theta1, rho1 = line1
        theta2, rho2 = line2
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([rho1, rho2])
        # If lines are nearly parallel, skip
        if np.abs(np.linalg.det(A)) < 1e-1:
            return None
        x, y = np.linalg.solve(A, b)
        return [x, y]
    
    mask = (img_arr != 0).any(axis=-1).astype(np.uint8) if img_arr.ndim == 3 else (img_arr != 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    
    if num_labels > 1:
        largest = max(range(1, num_labels), key=lambda lab: (labels==lab).sum())
        mask = (labels == largest)
    borders = ski.feature.canny(mask)
    
    height, width   = img_arr.shape[:2]
    hypotenuse      = np.sqrt(width**2 + height**2) 
        
    tested_angles       = np.linspace(-np.pi / 2, np.pi / 2, 720, endpoint=False)    
    hspace, theta, d    = ski.transform.hough_line(borders, theta=tested_angles)
    _, angles, dists    = ski.transform.hough_line_peaks(hspace, theta, d, 
                                                         threshold=treshold*np.max(hspace), 
                                                         min_distance=int(0.05*hypotenuse), 
                                                         num_peaks=4)
    
    intersections = []
    for (angle1, dist1), (angle2, dist2) in combinations(zip(angles, dists), 2):
        pt = compute_intersection((angle1, dist1), (angle2, dist2))
        if pt is not None:
            intersections.append(pt)
    intersections = np.array(intersections)
    
    """
    DEBUG IMAGE
    """
    fig, ax = plt.subplots()
    # ax.imshow(img_arr if img_arr.ndim == 2 else img_arr[..., :3], cmap='gray')
    
    # Save a debug image with the detected lines and intersections as stars
    img_uint8 = (img_arr * 255).astype(np.uint8) if img_arr.max() <= 1 else img_arr.astype(np.uint8)
    debug_img = img_uint8 if img_arr.ndim == 3 else cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    # Draw Hough lines
    for angle, dist in zip(angles, dists):
        # Hough line: x*cos(theta) + y*sin(theta) = dist
        a = np.cos(angle)
        b = np.sin(angle)
        x0 = a * dist
        y0 = b * dist
        # Draw the line segment for visualization
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(debug_img, pt1, pt2, (0, 255, 0), 1)
    # Draw intersections
    for idx, (x, y) in enumerate(intersections):
        cv2.drawMarker(debug_img, (int(round(x)), int(round(y))), (0,0,255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
        cv2.putText(
            debug_img,
            str(idx + 1),
            (int(round(x)) + 10, int(round(y)) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )
    debug_img
    
    # plt.imshow(debug_img[..., ::-1])
    # plt.show()
    
    """
    END DEBUG
    """
    
    if intersections.shape[0] != 4:
        raise Exception("No corners found with hough")
    
    return intersections

def corners_rdp(img_arr: ImageLikeArray) -> ListPoint2D:
    """Finds the corners of the largest countour object in an image by it 
    with Ramer–Douglas–Peucker algorithm

    Args:
        img_arr (ImageLikeArray): Image input
    Raises:
        Exception: If it can't find RDP simplification that fits a quad
    Returns:
        ListPoint2D: The corners of the object fitted
    """
    def binary_search(fun: Callable, args: Callable, args_range: int, target: Any):
        l, x, r = 0, args_range//2, args_range
        while (ret := fun(*args(x))) != target:
            if ret > target:
                l = x
            else:
                r = x
            x = (l + r) // 2
        return ret, x
    
    mask                = (img_arr != 0).any(axis=-1).astype(np.uint8) if img_arr.ndim == 3 else (img_arr != 0).astype(np.uint8)
    num_labels, labels  = cv2.connectedComponents(mask, connectivity=8)
    
    if num_labels > 1:
        largest = max(range(1, num_labels), key=lambda lab: (labels==lab).sum())
        mask    = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        raise ValueError("No contours found in the mask.")
    
    cnt     = max(contours, key=cv2.contourArea)
    
    peri    = cv2.arcLength(cnt, True)
    args    = lambda i: (cnt, peri*range(1, 1001, 1)[i]/1000, True)
    _, indx = binary_search(lambda *x: len(cv2.approxPolyDP(*x)), args, 999, 4)
    corners = cv2.approxPolyDP(*args(indx))
    # approx  = cv2.approxPolyDP(cnt, 1*eps, True)
    
    return corners.reshape((4,2))

def best_line_hough(img_arr: ImageLikeArray) -> NDArray[Shape["Thetha, Rho"], np.float32]:
    """Given an image like array find the best fitting line
    
    Args:
        img_arr (ImageLikeArray): Image like array

    Returns:
        NDArray[Shape["Thetha, Rho"], np.float32]: Line in Hesse normal form Rho = xCos(Thetha) + ySen(Theta)
    """
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    hspace, theta, d = ski.transform.hough_line(img_arr, theta=tested_angles)
    _, angles, dists    = ski.transform.hough_line_peaks(hspace, theta, d, num_peaks=1)
    
    return np.float([angles, dists], dtype=np.float32)

def best_wh_quad(corners: ListPoint2D) -> (float, float):
    """Given 4 points in the plane returns minimal width height as if it were to be a rectangle

    Args:
        corners (ListPoint2D): list of corners 
    
    Returns:
        float, float: The minimun width, height of a quad corner
    """
    corner_combinations = combinations(corners, 2)
    distances           = map(lambda x: np.linalg.norm(x[0]-x[1]), corner_combinations)
    distances_sorted    = sorted(distances)
    width, height       = distances_sorted[0], distances_sorted[2] 
    return width, height 

def map_corners(corners: ListPoint2D) -> ListPoint2D:
    """Given 4 corners, will try to figure out the vector closest to direction (0, -1)
    and will map the same corners in order from top left counterclockwise

    Args:
        corners (ListPoint2D): 4 Corners

    Returns:
        ListPoint2D: Mapped corners [top-left, top-right, bottom-right, bottom-left]
    """
    
    quad_origin         = np.mean(corners, axis=0)
    corner_combinations = combinations(corners, 2)
    up_reference_vect   = np.float32([0,-1])
    up_vect             = up_reference_vect
    min_cos_dist        = float('inf')
    for pair_point in corner_combinations:
        direction   = np.mean(pair_point, axis=0) - quad_origin
        length      = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        norm_vect   = direction/length
        cosine_dist = 1 - np.dot(norm_vect, up_reference_vect)
        if cosine_dist < min_cos_dist:
            min_cos_dist    = cosine_dist
            up_vect         = norm_vect
    
    # Map each corner to a quadrant (0-3) relative to the norm_vector and assign to ret
    # norm_vector defines the 'up' direction (quadrant 0)
    # The four quadrants are defined CCW starting from this direction
    right_vect  = np.array([-up_vect[1], up_vect[0]]) 
    ret         = np.zeros((4,2), dtype=np.float32)
    assigned    = [False] * 4
    for corner in corners:
        vec = corner - quad_origin
        x = np.dot(vec, right_vect)
        y = np.dot(vec, up_vect)
        if y > 0 and x < 0:          # top-left
            idx = 0
        elif x >= 0 and y > 0:       # top-right
            idx = 1
        elif y < 0 and x > 0:        # bottom-right
            idx = 2
        else:                        # bottom-left (y <= 0 and x <= 0)
            idx = 3
        
        if assigned[idx]:
            raise ValueError(f"Corner collision at quadrant {idx}. Input may not form a valid quad.")
        assigned[idx] = True
        ret[idx] = corner
    
    return ret