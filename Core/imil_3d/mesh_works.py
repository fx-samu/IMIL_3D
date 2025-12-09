import trimesh as tm
from trimesh import Trimesh
from trimesh.visual import TextureVisuals
from .imiltyping import *
from scipy.ndimage import binary_dilation
import numpy as np
import cv2
import skimage as ski
from PIL import Image

import matplotlib.pyplot as plt

def mesh_hmap(height_map: NumberField2D, flip_x: bool = True) -> Trimesh:
    """Creates a triangulated mesh from a 2D height map
    
    Args:
        height_map (NumberField2D): 2D array where values represent Z heights
        flip_x (bool, optional): Flip horizontally to correct mirror. Defaults to True.
    
    Returns:
        Trimesh: Triangulated surface mesh with vertices at grid positions
    """
    # Flip horizontally if needed (corrects X-axis mirror without breaking normals)
    if flip_x:
        height_map = height_map[:, ::-1]
    
    height, width = height_map.shape
    width_ratio   = width/height

    # Create grid of (x, y) coordinates
    xs = np.linspace(0, width_ratio, width)
    ys = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(xs, ys)

    # Flatten the grid and build vertices (x, y, z)
    verts = np.column_stack((xv.flatten(), yv.flatten(), height_map.flatten()))

    # Faces: connect each adjacent grid square as two triangles
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            # Calculate vertex indices for the four corners of the quad
            idx_bl = i * width + j       # bottom left
            idx_br = i * width + (j+1)   # bottom right
            idx_tl = (i+1) * width + j   # top left
            idx_tr = (i+1) * width + (j+1) # top right

            # Lower triangle (bl, tr, tl)
            faces.append([idx_bl, idx_tr, idx_tl])
            # Upper triangle (bl, br, tr)
            faces.append([idx_bl, idx_br, idx_tr])

    faces = np.array(faces)

    mesh = tm.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh

def mask_vertex_mesh(mesh: Trimesh, vertex_mask_map: BoolField2D) -> Trimesh:
    """Removes vertices from mesh based on a boolean mask
    
    Args:
        mesh (Trimesh): Input mesh to filter
        vertex_mask_map (BoolField2D): Boolean mask matching vertex grid shape
    
    Returns:
        Trimesh: Mesh with masked vertices removed
    """
    flat_mask_map = vertex_mask_map.flatten()
    assert len(flat_mask_map) == len(mesh.vertices)
    assert flat_mask_map.dtype == bool
    ret_mesh = mesh.copy()
    ret_mesh.update_vertices(flat_mask_map)
    return ret_mesh.process(validate=True)

def mask_face_submesh(mesh: Trimesh, face_mask_map: BoolField2D) -> Trimesh:
    """Extracts a submesh containing only faces where mask is True
    
    Args:
        mesh (Trimesh): Input mesh to filter
        face_mask_map (BoolField2D): Boolean mask for faces to keep
    
    Returns:
        Trimesh: Submesh with only masked faces and referenced vertices
    """
    ret_mesh = mesh.submesh([face_mask_map], append=True, repair=True)
    ret_mesh.remove_unreferenced_vertices()
    ret_mesh.process(validate=True)
    return ret_mesh

def face_mask_from_vertex_mask(mesh: Trimesh, vertex_mask_map: BoolField2D, 
                               flip_x: bool = True) -> BoolField2D:
    """Converts a vertex mask to a face mask (face is True if all its vertices are True)
    
    Args:
        mesh (Trimesh): Input mesh
        vertex_mask_map (BoolField2D): Boolean mask for vertices
        flip_x (bool, optional): Flip mask horizontally to match mesh_hmap. Defaults to True.
    
    Returns:
        BoolField2D: Face mask where True means all face vertices were masked
    """
    if flip_x:
        vertex_mask_map = vertex_mask_map[:, ::-1]
    flat_mask_map = vertex_mask_map.flatten()
    f_mask = flat_mask_map[mesh.faces].all(axis=1)
    return f_mask

def face_mask_by_sampling(mesh: Trimesh, vertex_mask_map: BoolField2D, 
                          width_ratio: float) -> np.ndarray:
    """Creates face mask by sampling vertex mask at face center positions
    
    Args:
        mesh (Trimesh): Input mesh from mesh_hmap
        vertex_mask_map (BoolField2D): Boolean mask in image space
        width_ratio (float): Width/height ratio used in mesh_hmap
    
    Returns:
        np.ndarray: Boolean array where True means face center is inside mask
    """
    height, width = vertex_mask_map.shape
    
    # Compute face centers
    face_centers = mesh.vertices[mesh.faces].mean(axis=1)  # (n_faces, 3)
    
    # Map back to pixel coordinates
    x_normalized = face_centers[:, 0]  # [0, width_ratio]
    y_normalized = face_centers[:, 1]  # [0, 1]
    
    x_pixels = (x_normalized / width_ratio * (width - 1)).astype(int)
    y_pixels = (y_normalized * (height - 1)).astype(int)
    
    # Clamp to valid range
    x_pixels = np.clip(x_pixels, 0, width - 1)
    y_pixels = np.clip(y_pixels, 0, height - 1)
    
    # Sample mask at face centers
    face_mask = vertex_mask_map[y_pixels, x_pixels]
    
    return face_mask

def pixel_contour(vertex_mask_map: BoolField2D) -> NumberField2D:
    """Detects corner points along the contour of a binary mask
    
    Args:
        vertex_mask_map (BoolField2D): Binary mask of the region
    
    Returns:
        NumberField2D: Array of (y, x) coordinates of detected corner points
    """
    height, width = vertex_mask_map.shape
    
    # Pad with zeros
    padded = np.pad(vertex_mask_map, pad_width=1, mode='constant', constant_values=False)
    
    # Vertical edges - mark BOTH sides
    digital_sobel_v = np.zeros((height, width), dtype=bool)
    transitions_v = padded[:, 1:] != padded[:, :-1]
    digital_sobel_v |= transitions_v[1:-1, :width]
    digital_sobel_v |= transitions_v[1:-1, 1:width+1]
    
    # Horizontal edges - mark BOTH sides
    digital_sobel_h = np.zeros((height, width), dtype=bool)
    transitions_h = padded[1:, :] != padded[:-1, :]
    digital_sobel_h |= transitions_h[:height, 1:-1]
    digital_sobel_h |= transitions_h[1:height+1, 1:-1]
    
    corners_test = digital_sobel_v & digital_sobel_h
    
    # Dilate edges
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)
    
    digital_sobel_v_dilated = binary_dilation(digital_sobel_v, structure=structure)
    digital_sobel_h_dilated = binary_dilation(digital_sobel_h, structure=structure)
    
    # Display the mask with the corners marked as stars
    # plt.figure(figsize=(6, 6))
    # plt.imshow(vertex_mask_map, cmap='gray')
    # y, x = np.nonzero(corners)
    # plt.scatter(x, y, marker='*', color='red', s=100, label='Corners')
    # plt.title("Vertex Mask Map with Corners")
    # plt.legend()
    # plt.show()
    
    # Corners = intersection
    corners = digital_sobel_v_dilated & digital_sobel_h_dilated
    
    # Filter 1: Keep only corners in mask region
    corners = corners & vertex_mask_map
    
    # Filter 2: Remove corners with exactly 1 False neighbor (these are edges, not corners)
    # Count False neighbors in 4-connected neighborhood
    padded_mask = np.pad(vertex_mask_map, pad_width=1, mode='constant', constant_values=False)
    
    false_neighbors = np.zeros((height, width), dtype=int)
    false_neighbors += ~padded_mask[:-2, 1:-1]   # top
    false_neighbors += ~padded_mask[2:, 1:-1]    # bottom
    false_neighbors += ~padded_mask[1:-1, :-2]   # left
    false_neighbors += ~padded_mask[1:-1, 2:]    # right
    
    # Keep corners with 0, 2, 3, or 4 False neighbors (exclude 1)
    valid_corners = (false_neighbors != 1)
    corners = corners & valid_corners
    # Corners = intersection
    corners = digital_sobel_v_dilated & digital_sobel_h_dilated
    corners = corners & vertex_mask_map
    
    # Get all 8 neighbors
    padded_mask = np.pad(vertex_mask_map, pad_width=1, mode='constant', constant_values=False)
    
    # Orthogonal neighbors
    top    = padded_mask[:-2, 1:-1]
    bottom = padded_mask[2:, 1:-1]
    left   = padded_mask[1:-1, :-2]
    right  = padded_mask[1:-1, 2:]
    
    # Diagonal neighbors
    top_left     = padded_mask[:-2, :-2]
    top_right    = padded_mask[:-2, 2:]
    bottom_left  = padded_mask[2:, :-2]
    bottom_right = padded_mask[2:, 2:]
    
    # Count False orthogonal neighbors
    false_neighbors = (~top).astype(int) + (~bottom).astype(int) + \
                      (~left).astype(int) + (~right).astype(int)
    
    # Exclude edges (exactly 1 False orthogonal neighbor)
    valid_corners = (false_neighbors != 1)
    
    # If all 4 orthogonal are True, check diagonals
    all_orthogonal_true = top & bottom & left & right
    all_diagonal_true = top_left & top_right & bottom_left & bottom_right
    
    # Interior point = all orthogonal True AND all diagonal True
    is_interior = all_orthogonal_true & all_diagonal_true
    valid_corners = valid_corners & ~is_interior
    
    # Filter adjacent False for 2-False case
    has_two_false = (false_neighbors == 2)
    adjacent_false = ((~top & ~left) | (~top & ~right) | 
                      (~bottom & ~left) | (~bottom & ~right))
    valid_corners = valid_corners & (~has_two_false | adjacent_false)
    
    corners = corners & valid_corners
        

    # Display the mask with the corners marked as stars
    # plt.figure(figsize=(6, 6))
    # plt.imshow(vertex_mask_map, cmap='gray')
    # y, x = np.nonzero(corners)
    # plt.scatter(x, y, marker='*', color='red', s=100, label='Corners')
    # plt.title("Vertex Mask Map with Corners")
    # plt.legend()
    # plt.show()
    
    return np.transpose(np.nonzero(corners))

def mesh_contour_mask(contour_points: NumberField2D, vertex_mask_map: BoolField2D) -> Trimesh:
    """Creates a flat mesh from contour corner points, masked by vertex mask
    
    Args:
        contour_points (NumberField2D): Array of (y, x) corner coordinates
        vertex_mask_map (BoolField2D): Boolean mask to filter resulting faces
    
    Returns:
        Trimesh: Flat mesh (z=0) with faces only inside the masked region
    """
    height, width = vertex_mask_map.shape
    width_ratio   = width/height
    
    # Get unique x and y coordinates from corners (sorted)
    y_coords = np.unique(contour_points[:, 0])
    x_coords = np.unique(contour_points[:, 1])
    
    # Map to normalized space
    xs = np.linspace(0, width_ratio, width)
    ys = np.linspace(0, 1, height)
    
    corner_xs = xs[x_coords.astype(int)]
    corner_ys = ys[y_coords.astype(int)]
    
    # Create grid from corner positions
    xv, yv = np.meshgrid(corner_xs, corner_ys)
    verts = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(xv.size)])
    
    # Create faces (quads as triangles)
    grid_height, grid_width = len(corner_ys), len(corner_xs)
    faces = []
    for i in range(grid_height - 1):
        for j in range(grid_width - 1):
            idx_bl = i * grid_width + j
            idx_br = i * grid_width + (j+1)
            idx_tl = (i+1) * grid_width + j
            idx_tr = (i+1) * grid_width + (j+1)
            
            faces.append([idx_bl, idx_tr, idx_tl])
            faces.append([idx_bl, idx_br, idx_tr])
    
    mesh = tm.Trimesh(vertices=verts, faces=faces, process=False)
    
    face_mask = face_mask_by_sampling(mesh, vertex_mask_map, width_ratio)

    mesh = mask_face_submesh(mesh, face_mask)
    return mesh

def imprint_front_on_solid(solid_back: Trimesh, front_mesh: Trimesh, 
                           front_offset: float = 0.0) -> Trimesh:
    """Combines a front relief mesh onto a solid back using boolean union
    
    Args:
        solid_back (Trimesh): Base solid mesh
        front_mesh (Trimesh): Relief surface to imprint on top
        front_offset (float, optional): Z offset for front mesh. Defaults to 0.0.
    
    Returns:
        Trimesh: Combined solid mesh with imprinted front
    """
    # 1. Position front mesh at the top of solid
    back_top_z = solid_back.vertices[:, 2].max()
    
    # Move front mesh to sit on top
    front_mesh_positioned = front_mesh.copy()
    front_mesh_positioned.vertices[:, 2] += (back_top_z + front_offset)
    
    # 2. Make front mesh into a solid (extrude downward to overlap with back)
    extrusion_depth = solid_back.vertices[:, 2].max() - solid_back.vertices[:, 2].min() + 0.1
    front_solid = extrude_mesh(front_mesh_positioned, [0, 0, -extrusion_depth])
    
    # 3. Boolean union to combine
    combined = tm.boolean.union([solid_back, front_solid], engine='blender')
    try:
        pass
    except:
        # Fallback if blender engine not available
        combined = tm.boolean.union([solid_back, front_solid])
    
    return combined

def extrude_mesh(mesh: Trimesh, direction: list = [0, 0, -1], 
                 height: float = 0.1) -> Trimesh:
    """Extrudes a planar mesh along a direction to create a solid
    
    Args:
        mesh (Trimesh): Input planar mesh to extrude
        direction (list, optional): Extrusion direction vector. Defaults to [0, 0, -1].
        height (float, optional): Extrusion distance. Defaults to 0.1.
    
    Returns:
        Trimesh: Extruded solid mesh with top, bottom, and side faces
    """
    import numpy as np
    import trimesh
    
    # 1. Duplicate vertices, offset by direction
    offset = np.array(direction) * height
    top_verts = mesh.vertices.copy()
    bottom_verts = mesh.vertices + offset
    
    # 2. Combine vertices
    all_verts = np.vstack([top_verts, bottom_verts])
    
    # 3. Create faces
    n_verts = len(mesh.vertices)
    
    # Top faces (original)
    top_faces = mesh.faces.copy()
    
    # Bottom faces (flipped normals)
    bottom_faces = mesh.faces[:, ::-1] + n_verts
    
    # 4. Find boundary edges using face adjacency
    # Boundary edges are those that appear in only one face
    edges = mesh.edges
    edge_counts = {}
    for edge in edges:
        edge_key = tuple(sorted(edge))
        edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
    
    # Boundary edges appear only once
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    
    # 5. Create side faces
    side_faces = []
    for edge in boundary_edges:
        v0, v1 = edge
        v0_bottom = v0 + n_verts
        v1_bottom = v1 + n_verts
        
        # Create two triangles for this edge
        side_faces.append([v0, v1, v1_bottom])
        side_faces.append([v0, v1_bottom, v0_bottom])
    
    # 6. Combine all faces
    all_faces = np.vstack([top_faces, bottom_faces, np.array(side_faces)])
    
    # 7. Create solid mesh
    solid = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
    
    return solid

def solidify_mesh_box(mesh: Trimesh, height: float, flatten_boundary: bool = True) -> Trimesh:
    """Creates a solid box by extruding mesh downward with flat bottom
    
    Args:
        mesh (Trimesh): Input surface mesh (typically from mesh_hmap)
        height (float): Extrusion depth below the mesh's minimum Z
        flatten_boundary (bool): If True, flatten boundary vertices to min_z for clean edges
    
    Returns:
        Trimesh: Watertight solid with top surface, flat bottom, and closed sides.
                 Vertex layout: [top, bottom, side_boundary] for proper UV separation.
    """
    import trimesh

    # 1. Get top and bottom vertices
    top_verts = mesh.vertices.copy()
    min_z = np.min(top_verts[:, 2])
    bottom_z = min_z - height
    n_verts = len(top_verts)

    # 2. Top faces: use mesh.faces (same indices)
    top_faces = mesh.faces.copy()
    # 3. Bottom faces: flip winding and shift indices to bottom_verts
    bottom_faces = mesh.faces[:, ::-1] + n_verts

    # 4. Detect boundary edges, preserving original winding from faces
    edges = mesh.edges  # shape (n_edges, 2) - preserves face winding order
    edges_sorted = np.sort(edges, axis=1)  # canonical form for counting
    
    # Find which original edges are boundaries (appear once when canonicalized)
    _, inverse, counts = np.unique(edges_sorted, axis=0, return_inverse=True, return_counts=True)
    is_boundary = counts[inverse] == 1
    boundary_edges = edges[is_boundary]
    n_boundary_edges = len(boundary_edges)

    # 4.5. Flatten boundary vertices to min_z for clean frame edges
    boundary_vert_indices = np.unique(boundary_edges.flatten())
    if flatten_boundary:
        top_verts[boundary_vert_indices, 2] = min_z
    
    # Create bottom vertices after boundary flattening
    bottom_verts = top_verts.copy()
    bottom_verts[:, 2] = bottom_z

    # 5. Create duplicated boundary vertices for side faces (UV separation)
    # Get unique boundary vertex indices and create mapping to new indices
    dup_base_idx = 2 * n_verts  # Duplicates start after top+bottom
    
    # Map original vertex index -> duplicated vertex index
    orig_to_dup = np.zeros(n_verts, dtype=np.intp)
    orig_to_dup[boundary_vert_indices] = dup_base_idx + np.arange(len(boundary_vert_indices))
    
    # Create the duplicated vertices (same position as top boundary verts)
    dup_boundary_verts = top_verts[boundary_vert_indices]
    
    # 6. Stack all vertices: [top, bottom, duplicated_boundary]
    all_verts = np.vstack([top_verts, bottom_verts, dup_boundary_verts])

    # 7. Side face generation using DUPLICATED vertices (not shared with top)
    v0_orig = boundary_edges[:, 0]
    v1_orig = boundary_edges[:, 1]
    
    # Map to duplicated indices for the top edge of side faces
    v0_dup = orig_to_dup[v0_orig]
    v1_dup = orig_to_dup[v1_orig]
    
    # Bottom indices (from original bottom vertex range)
    v0_b = v0_orig + n_verts
    v1_b = v1_orig + n_verts
    
    # Side faces use: duplicated top + bottom (all will get solid UVs)
    side_faces = np.empty((n_boundary_edges * 2, 3), dtype=np.intp)
    side_faces[0::2] = np.column_stack([v0_dup, v0_b, v1_b])
    side_faces[1::2] = np.column_stack([v0_dup, v1_b, v1_dup])

    all_faces = np.vstack([top_faces, bottom_faces, side_faces])
    
    # # DEBUG: Verify vertex separation
    # print(f"\n[DEBUG solidify_mesh_box]")
    # print(f"  n_top_verts: {n_verts}")
    # print(f"  n_boundary_verts: {len(boundary_vert_indices)}")
    # print(f"  n_boundary_edges: {n_boundary_edges}")
    # print(f"  Total vertices: {len(all_verts)} (expected: {2*n_verts + len(boundary_vert_indices)})")
    # print(f"  Vertex ranges: top[0:{n_verts}], bottom[{n_verts}:{2*n_verts}], dup[{2*n_verts}:{len(all_verts)}]")
    # print(f"  Side face vertex indices range: [{side_faces.min()}, {side_faces.max()}]")
    # print(f"  Side faces using top range [0,{n_verts})? {(side_faces < n_verts).any()}")
    # if (side_faces < n_verts).any():
    #     bad_indices = side_faces[side_faces < n_verts]
    #     print(f"  !!! PROBLEM: Side faces reference original top vertices: {np.unique(bad_indices)[:10]}...")

    solid = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    return solid


def apply_texture_to_solid(mesh: Trimesh, inp_image: ImageLikeArray, 
                           mean_color: NumerPoint3D, flip_x: bool = True,
                           frame_boundary: bool = True) -> Trimesh:
    """Applies a texture to a solidified mesh with UV mapping
    
    Args:
        mesh (Trimesh): Solid mesh from solidify_mesh_box
        inp_image (ImageLikeArray): Texture image (grayscale, RGB, or RGBA)
        mean_color (np.ndarray): RGB color [r,g,b] for side and bottom faces
        flip_x (bool, optional): Flip texture horizontally to match mesh_hmap. Defaults to True.
        frame_boundary (bool, optional): If True, boundary vertices get solid color (frame effect). Defaults to True.
    
    Returns:
        Trimesh: New mesh with TextureVisuals applied
    """
    # Flip texture to match mesh orientation
    if flip_x:
        inp_image = inp_image[:, ::-1]
    
    height, width = inp_image.shape[:2]
    
    # Derive n_top_verts from mesh geometry
    # solidify_mesh_box layout: [top, bottom, duplicated_boundary]
    # Bottom vertices are at the absolute min_z
    absolute_min_z = mesh.vertices[:, 2].min()
    is_bottom = np.isclose(mesh.vertices[:, 2], absolute_min_z)
    n_top_verts = is_bottom.sum()  # bottom count == top count
    
    # Get top vertices and find boundary (flattened to top's min_z)
    top_verts = mesh.vertices[:n_top_verts]
    top_min_z = top_verts[:, 2].min()
    is_boundary_top = np.isclose(top_verts[:, 2], top_min_z)
    
    # Mesh x ranges [0, width_ratio], y ranges [0, 1]
    width_ratio = mesh.vertices[:, 0].max()
    
    # Ensure at least RGB (convert grayscale if needed)
    if inp_image.ndim == 2:
        inp_image = np.stack([inp_image] * 3, axis=-1)
    
    n_channels = inp_image.shape[2]
    mean_color = np.asarray(mean_color, dtype=np.uint8)[:n_channels]
    
    # Create texture atlas: add solid color strip on the right for side/bottom faces
    solid_strip_width = 4
    atlas_width = width + solid_strip_width
    atlas = np.zeros((height, atlas_width, n_channels), dtype=np.uint8)
    atlas[:, :width, :] = inp_image
    atlas[:, width:, :] = mean_color  # Solid color strip
    
    # Fill all black (zero) pixels with mean color to prevent UV bleeding artifacts
    black_mask = (atlas == 0).all(axis=-1)
    atlas[black_mask] = mean_color
    
    # UV coordinate for the solid color region (center of the strip)
    solid_u = (width + solid_strip_width / 2) / atlas_width
    solid_v = 0.5
    
    # Create UV coordinates for all vertices
    # Layout: [top_verts (texture), bottom_verts (solid), boundary_dups (solid)]
    n_total_verts = len(mesh.vertices)
    uv = np.zeros((n_total_verts, 2), dtype=np.float64)
    
    # Top vertices (0 to n_top_verts-1): map x,y to u,v in main texture region
    u_scale = width / atlas_width
    uv[:n_top_verts, 0] = (top_verts[:, 0] / width_ratio) * u_scale
    uv[:n_top_verts, 1] = top_verts[:, 1]  # y is already [0, 1]
    
    # If frame_boundary, assign boundary top vertices solid color UVs
    if frame_boundary:
        uv[:n_top_verts][is_boundary_top, 0] = solid_u
        uv[:n_top_verts][is_boundary_top, 1] = solid_v
    
    # ALL other vertices (bottom + duplicated boundary): solid color UV
    # This includes [n_top_verts:2*n_top_verts] (bottom) and [2*n_top_verts:] (side dups)
    uv[n_top_verts:, 0] = solid_u
    uv[n_top_verts:, 1] = solid_v
    
    # Flip V coordinate (image y=0 is top, UV v=0 is bottom)
    uv[:, 1] = 1.0 - uv[:, 1]
    
    # # DEBUG: Verify UV assignments
    # print(f"\n[DEBUG apply_texture_to_solid]")
    # print(f"  n_top_verts (detected): {n_top_verts}")
    # print(f"  n_total_verts: {n_total_verts}")
    # print(f"  Atlas size: {atlas_width}x{height}, texture region: [0,{width}], solid region: [{width},{atlas_width}]")
    # print(f"  solid_u: {solid_u:.4f} (should be > {width/atlas_width:.4f})")
    # print(f"  Top verts UV u range: [{uv[:n_top_verts, 0].min():.4f}, {uv[:n_top_verts, 0].max():.4f}]")
    # print(f"  Top verts UV v range: [{uv[:n_top_verts, 1].min():.4f}, {uv[:n_top_verts, 1].max():.4f}]")
    # print(f"  Non-top verts UV u (all should be {solid_u:.4f}): unique={np.unique(uv[n_top_verts:, 0])}")
    
    # Check if any top vertex UVs are outside texture region
    top_u_max = width / atlas_width
    # if (uv[:n_top_verts, 0] > top_u_max).any():
    #     print(f"  !!! PROBLEM: Some top vertex UVs exceed texture region ({top_u_max:.4f})")
    
    # Create PIL Image for trimesh
    pil_mode = 'RGBA' if n_channels == 4 else 'RGB'
    pil_image = Image.fromarray(atlas, mode=pil_mode)
    
    # Create material and texture visuals
    material = tm.visual.material.SimpleMaterial(image=pil_image)
    texture_visuals = TextureVisuals(uv=uv, material=material)
    
    # Create new mesh with texture
    textured_mesh = tm.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        visual=texture_visuals,
        process=False
    )
    
    return textured_mesh


def mesh_export(mesh: Trimesh, path: str, orientation: int = 2):
    """Exports a mesh to file with optional rotation (format auto-detected from extension)
    
    Args:
        mesh (Trimesh): Mesh to export (with or without texture)
        path (str): Output file path (.stl, .obj, .glb, etc.)
        orientation (int, optional): Rotation in 90° CCW steps around Z. Defaults to 2.
    """
    mesh = mesh.copy()
    
    # Apply rotation if needed
    if orientation % 4 != 0:
        angle_rad = np.deg2rad(90 * (orientation % 4))
        rotmat = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                 0,                1]
        ])
        mesh.vertices = mesh.vertices @ rotmat.T
    
    mesh.export(path)