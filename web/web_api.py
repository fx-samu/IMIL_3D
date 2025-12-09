"""
IMIL_3D Web API
Simple FastAPI backend for the web interface
"""
import sys
import os

# Add Core to path so we can import imil_3d
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Core'))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
from PIL import Image
import io
import base64
import uuid
import tempfile
import json

import imil_3d as im3

app = FastAPI(title="IMIL_3D API")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config - adjust paths as needed
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
SAM2_PT = os.path.join(CHECKPOINT_DIR, 'sam2.1_hiera_base_plus.pt')
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Temp storage for session data (in production, use proper session/cache)
sessions = {}


def image_to_base64(img_arr: np.ndarray) -> str:
    """Convert numpy array to base64 PNG"""
    img = Image.fromarray(img_arr.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def mask_to_base64(mask: np.ndarray) -> str:
    """Convert boolean mask to base64 PNG (white on black)"""
    img = Image.fromarray((mask * 255).astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def draw_corners_on_image(img_arr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw corners as red dots on image copy"""
    import cv2
    img_copy = img_arr.copy()
    for i, (x, y) in enumerate(corners):
        cv2.circle(img_copy, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.putText(img_copy, str(i+1), (int(x)+8, int(y)-8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img_copy


@app.post("/api/process")
async def process_image(
    image: UploadFile = File(...),
    bbox_x1: int = Form(...),
    bbox_y1: int = Form(...),
    bbox_x2: int = Form(...),
    bbox_y2: int = Form(...),
    megapixels: float = Form(0.5)
):
    """
    Process uploaded image:
    1. Resize to target megapixels
    2. Run SAM2 segmentation with bounding box
    3. Run 3 corner detection methods
    
    Returns base64 encoded images for display
    """
    try:
        # Read and decode image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_arr = np.array(pil_image)
        
        original_h, original_w = img_arr.shape[:2]
        
        # Resize image
        img_resized = im3.field_manipulation.image_resize(img_arr, mp=megapixels)
        resized_h, resized_w = img_resized.shape[:2]
        
        # Scale bounding box to resized dimensions
        scale_x = resized_w / original_w
        scale_y = resized_h / original_h
        
        bbox_scaled = np.array([
            [int(bbox_x1 * scale_x), int(bbox_y1 * scale_y)],
            [int(bbox_x2 * scale_x), int(bbox_y2 * scale_y)]
        ], dtype=np.int16)
        
        # Run SAM2 segmentation
        mask = im3.field_recognition.sam2_mask(img_resized, bbox_scaled, SAM2_PT, SAM2_CFG)
        
        # Apply mask to image
        img_masked = im3.field_manipulation.field_mask(img_resized, mask)
        
        # Run 3 corner detection methods
        corner_results = {}
        corner_methods = {
            'rect': ('Minimum Area Rectangle', im3.field_recognition.corners_rect),
            'hough': ('Hough Lines', im3.field_recognition.corners_hough),
            'rdp': ('Ramer-Douglas-Peucker', im3.field_recognition.corners_rdp),
        }
        
        for key, (name, fn) in corner_methods.items():
            try:
                corners = fn(img_masked)
                corners_img = draw_corners_on_image(img_masked, corners)
                corner_results[key] = {
                    'name': name,
                    'corners': corners.tolist(),
                    'preview': image_to_base64(corners_img),
                    'success': True
                }
            except Exception as e:
                corner_results[key] = {
                    'name': name,
                    'corners': None,
                    'preview': None,
                    'success': False,
                    'error': str(e)
                }
        
        # Generate session ID and store data for later GLB generation
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            'img_masked': img_masked,
            'mask': mask,
            'corners': {k: v['corners'] for k, v in corner_results.items() if v['success']}
        }
        
        return JSONResponse({
            'session_id': session_id,
            'resized_image': image_to_base64(img_resized),
            'mask': mask_to_base64(mask),
            'masked_image': image_to_base64(img_masked),
            'corners': corner_results,
            'dimensions': {
                'original': [original_w, original_h],
                'resized': [resized_w, resized_h]
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GenerateRequest(BaseModel):
    session_id: str
    corner_method: str  # 'rect', 'hough', or 'rdp'
    depth_scale: float = 0.05
    box_depth: float = 0.1


@app.post("/api/generate")
async def generate_glb(request: GenerateRequest):
    """
    Generate GLB from selected corner method
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    
    if request.corner_method not in session['corners']:
        raise HTTPException(status_code=400, detail=f"Corner method '{request.corner_method}' not available")
    
    try:
        img_masked = session['img_masked']
        corners = np.array(session['corners'][request.corner_method])
        
        # Unwarp perspective
        unwarp_img = im3.field_manipulation.map_unwarp(img_masked, corners)
        
        # Generate height map
        height_map = im3.height_map.grayscale(unwarp_img)
        final_mask = height_map != 0
        
        # Normalize height map
        norm_hmap = im3.field_manipulation.field_normalization_mask(height_map, final_mask)
        
        # Create mesh
        raw_mesh = im3.mesh_works.mesh_hmap(norm_hmap * request.depth_scale)
        f_mask_map = im3.mesh_works.face_mask_from_vertex_mask(raw_mesh, final_mask)
        front_mesh = im3.mesh_works.mask_face_submesh(raw_mesh, f_mask_map)
        
        # Solidify
        solid_mesh = im3.mesh_works.solidify_mesh_box(front_mesh, request.box_depth)
        
        # Apply texture
        mean_color = unwarp_img[final_mask].mean(axis=0).astype(np.uint8)
        textured_mesh = im3.mesh_works.apply_texture_to_solid(solid_mesh, unwarp_img, mean_color)
        
        # Export to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
        im3.mesh_works.mesh_export(textured_mesh, temp_file.name)
        
        # Read GLB and return as base64
        with open(temp_file.name, 'rb') as f:
            glb_data = f.read()
        
        os.unlink(temp_file.name)
        
        return JSONResponse({
            'glb': base64.b64encode(glb_data).decode('utf-8'),
            'success': True
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up session data"""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "ok"}


# Serve static files at /static, with index.html at root
static_dir = os.path.join(os.path.dirname(__file__), 'static')

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(static_dir, 'index.html'))

@app.get("/style.css")
async def serve_css():
    return FileResponse(os.path.join(static_dir, 'style.css'), media_type='text/css')

@app.get("/app.js")
async def serve_js():
    return FileResponse(
        os.path.join(static_dir, 'app.js'),
        media_type='application/javascript',
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

