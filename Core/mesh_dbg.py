import numpy as np
import utils
import os
import trimesh as tm
from trimesh import Trimesh

os.chdir(os.path.dirname(os.path.abspath(__file__)))

@imil_3d.functions.benchmark
def main():
    # tm.util.attach_to_log()
    
    height_map      = np.load("height_map.npy")
    v_mask_map      = np.load("mask_map.npy")
    img_arr         = imil_3d.functions.image_open("unwarp_img.png")
    norm_hmap       = imil_3d.field_manipulation.field_normalization_mask(height_map, v_mask_map)
    hmap            = imil_3d.field_manipulation.field_normalization(height_map)
    
    raw_mesh        = imil_3d.mesh_works.mesh_hmap(norm_hmap*0.05)
    f_mask_map      = imil_3d.mesh_works.face_mask_from_vertex_mask(raw_mesh, v_mask_map)
    
    f_front_mesh    = imil_3d.mesh_works.mask_face_submesh(raw_mesh, f_mask_map)

    solidified_mesh    = imil_3d.mesh_works.solidify_mesh_box(f_front_mesh, 0.1)
    
    # Calculate mean color from masked region
    mean_color = img_arr[v_mask_map].mean(axis=0).astype(np.uint8)
    
    # Apply texture to solid mesh
    textured_mesh = imil_3d.mesh_works.apply_texture_to_solid(solidified_mesh, img_arr, mean_color)
    
    # Export (use .glb for texture support, .stl for geometry-only)
    imil_3d.mesh_works.mesh_export(f_front_mesh, "f_mesh.stl")
    imil_3d.mesh_works.mesh_export(textured_mesh, "solidified_mesh.glb")


if __name__ == "__main__": 
    main()
    
