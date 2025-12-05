#!/usr/bin/python
# Objective: image -> 3D model with height map texture
from nptyping.shape import Shape
import imil_3d as im3 
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configuration
ASSETS_DIR      = "../Assets"
CHECKPOINT_DIR  = "../checkpoints"

KHACHKAR_NUM    = 1
KHACHKAR_RES    = 0.1  # Megapixels
DEFAULT_IMG     = f"{ASSETS_DIR}/Khachckars/{KHACHKAR_NUM}.jpg"

SAM2_PT         = f"{CHECKPOINT_DIR}/sam2.1_hiera_base_plus.pt"
SAM2_CFG        = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Corner detection methods
CORNER_METHODS = {
    1: ("Minimum Area Rectangle", im3.field_recognition.corners_rect),
    2: ("Hough Line Transform", im3.field_recognition.corners_hough),
    3: ("Ramer-Douglas-Peucker", im3.field_recognition.corners_rdp),
}

def select_corner_method() -> int:
    """Prompts user to select a corner detection method"""
    print("\n=== Corner Detection Method ===")
    for key, (name, _) in CORNER_METHODS.items():
        print(f"  {key}. {name}")
    
    while True:
        try:
            choice = int(input("\nSelect method (1-3): "))
            if choice in CORNER_METHODS:
                return choice
        except ValueError:
            pass
        print("Invalid choice. Please enter 1, 2, or 3.")

@im3.functions.benchmark
def main():
    # Select corner detection method
    method_choice = select_corner_method()
    method_name, corner_fn = CORNER_METHODS[method_choice]
    print(f"\nUsing: {method_name}\n")
    
    # Load and preprocess image
    img_arr     = im3.functions.image_open(DEFAULT_IMG)
    img_arr     = im3.field_manipulation.image_resize(img_arr, mp=KHACHKAR_RES)
    
    # Segment object with SAM2
    bbox        = im3.field_recognition.magic_bounding_box(KHACHKAR_NUM, KHACHKAR_RES)
    mask        = im3.field_recognition.sam2_mask(img_arr, bbox, SAM2_PT, SAM2_CFG)
    img_masked  = im3.field_manipulation.field_mask(img_arr, mask)
    
    # Detect corners using selected method
    corners     = corner_fn(img_masked)
    
    # Unwarp perspective and generate height map
    unwarp_img      = im3.field_manipulation.map_unwarp(img_masked, corners)
    unwarp_height   = im3.height_map.grayscale(unwarp_img)
    final_mask      = unwarp_height != 0
    
    print(f"\n=== Output Shapes ===")
    print(f"  Height map: {unwarp_height.shape}")
    print(f"  Mask:       {final_mask.shape}")
    print(f"  Image:      {unwarp_img.shape}")
    
    # Generate mesh from height map
    norm_hmap       = im3.field_manipulation.field_normalization_mask(unwarp_height, final_mask)
    raw_mesh        = im3.mesh_works.mesh_hmap(norm_hmap * 0.05)
    f_mask_map      = im3.mesh_works.face_mask_from_vertex_mask(raw_mesh, final_mask)
    f_front_mesh    = im3.mesh_works.mask_face_submesh(raw_mesh, f_mask_map)
    
    # Create solid and apply texture
    solidified_mesh = im3.mesh_works.solidify_mesh_box(f_front_mesh, 0.1)
    mean_color      = unwarp_img[final_mask].mean(axis=0).astype(np.uint8)
    textured_mesh   = im3.mesh_works.apply_texture_to_solid(solidified_mesh, unwarp_img, mean_color)
    
    # Export meshes
    im3.mesh_works.mesh_export(f_front_mesh, f"{ASSETS_DIR}/Models/Front_{KHACHKAR_NUM}.stl")
    im3.mesh_works.mesh_export(textured_mesh, f"{ASSETS_DIR}/Models/Object_{KHACHKAR_NUM}.glb")
    print("\n=== Exported ===")
    print("  f_mesh.stl (geometry only)")
    print("  solidified_mesh.glb (with texture)")

if __name__ == "__main__":
    main()
