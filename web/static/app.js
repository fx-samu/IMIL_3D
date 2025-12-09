/**
 * IMIL-3D Web Interface
 */

// State
let currentImage = null;
let bbox = { x1: 0, y1: 0, x2: 0, y2: 0 };
let isDrawing = false;
let sessionId = null;
let selectedMethod = null;
let glbBlob = null;

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const imageInput = document.getElementById('image-input');
const canvasContainer = document.getElementById('canvas-container');
const canvas = document.getElementById('image-canvas');
const ctx = canvas.getContext('2d');
const megapixelsInput = document.getElementById('megapixels');
const processBtn = document.getElementById('process-btn');
const generateBtn = document.getElementById('generate-btn');
const depthScaleInput = document.getElementById('depth-scale');
const boxDepthInput = document.getElementById('box-depth');
const backBtn = document.getElementById('back-btn');
const restartBtn = document.getElementById('restart-btn');
const downloadBtn = document.getElementById('download-btn');
const modelViewer = document.getElementById('model-viewer');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');

// Steps
const stepUpload = document.getElementById('step-upload');
const stepCorners = document.getElementById('step-corners');
const stepViewer = document.getElementById('step-viewer');

// === Upload Handling ===

uploadArea.addEventListener('click', () => imageInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--accent)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '';
    if (e.dataTransfer.files.length) {
        handleImageFile(e.dataTransfer.files[0]);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleImageFile(e.target.files[0]);
    }
});

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            setupCanvas();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function setupCanvas() {
    // Size canvas to image (max 800px width)
    const maxWidth = 800;
    const scale = Math.min(1, maxWidth / currentImage.width);
    
    canvas.width = currentImage.width * scale;
    canvas.height = currentImage.height * scale;
    
    // Store scale for bbox calculation
    canvas.dataset.scale = scale;
    canvas.dataset.originalWidth = currentImage.width;
    canvas.dataset.originalHeight = currentImage.height;
    
    drawImage();
    
    uploadArea.classList.add('hidden');
    canvasContainer.classList.remove('hidden');
    processBtn.disabled = true;
    bbox = { x1: 0, y1: 0, x2: 0, y2: 0 };
}

function drawImage() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    
    // Draw bbox if set
    if (bbox.x2 !== bbox.x1 && bbox.y2 !== bbox.y1) {
        ctx.strokeStyle = '#4a9eff';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(
            bbox.x1,
            bbox.y1,
            bbox.x2 - bbox.x1,
            bbox.y2 - bbox.y1
        );
        ctx.setLineDash([]);
    }
}

// === Canvas Drawing ===

canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    // Account for CSS scaling: map displayed coords to canvas coords
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    bbox.x1 = (e.clientX - rect.left) * scaleX;
    bbox.y1 = (e.clientY - rect.top) * scaleY;
    bbox.x2 = bbox.x1;
    bbox.y2 = bbox.y1;
    isDrawing = true;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    // Account for CSS scaling: map displayed coords to canvas coords
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    bbox.x2 = (e.clientX - rect.left) * scaleX;
    bbox.y2 = (e.clientY - rect.top) * scaleY;
    
    drawImage();
});

canvas.addEventListener('mouseup', () => {
    if (!isDrawing) return;
    isDrawing = false;
    
    // Normalize bbox (ensure x1 < x2, y1 < y2)
    if (bbox.x1 > bbox.x2) [bbox.x1, bbox.x2] = [bbox.x2, bbox.x1];
    if (bbox.y1 > bbox.y2) [bbox.y1, bbox.y2] = [bbox.y2, bbox.y1];
    
    // Enable process button if bbox is valid
    const minSize = 20;
    processBtn.disabled = (bbox.x2 - bbox.x1 < minSize || bbox.y2 - bbox.y1 < minSize);
});

canvas.addEventListener('mouseleave', () => {
    if (isDrawing) {
        isDrawing = false;
    }
});

// === Process Image ===

processBtn.addEventListener('click', async () => {
    if (!currentImage) return;
    
    showLoading('Processing image with SAM2...');
    
    try {
        // Scale bbox back to original image coordinates
        const scale = parseFloat(canvas.dataset.scale);
        const scaledBbox = {
            x1: Math.round(bbox.x1 / scale),
            y1: Math.round(bbox.y1 / scale),
            x2: Math.round(bbox.x2 / scale),
            y2: Math.round(bbox.y2 / scale)
        };
        
        // Create form data
        const formData = new FormData();
        
        // Convert current image to blob
        const imageBlob = await new Promise(resolve => {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = currentImage.width;
            tempCanvas.height = currentImage.height;
            tempCanvas.getContext('2d').drawImage(currentImage, 0, 0);
            tempCanvas.toBlob(resolve, 'image/png');
        });
        
        formData.append('image', imageBlob, 'image.png');
        formData.append('bbox_x1', scaledBbox.x1);
        formData.append('bbox_y1', scaledBbox.y1);
        formData.append('bbox_x2', scaledBbox.x2);
        formData.append('bbox_y2', scaledBbox.y2);
        formData.append('megapixels', megapixelsInput.value);
        
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }
        
        const result = await response.json();
        sessionId = result.session_id;
        
        displayResults(result);
        
    } catch (error) {
        alert('Error: ' + error.message);
        console.error(error);
    } finally {
        hideLoading();
    }
});

function displayResults(result) {
    // Show mask
    document.getElementById('mask-preview').src = 'data:image/png;base64,' + result.mask;
    
    // Show corner results
    const methods = ['rect', 'hough', 'rdp'];
    methods.forEach(method => {
        const card = document.querySelector(`[data-method="${method}"]`);
        const img = document.getElementById(`corners-${method}`);
        const status = card.querySelector('.status');
        
        const data = result.corners[method];
        
        if (data.success) {
            img.src = 'data:image/png;base64,' + data.preview;
            status.textContent = '✓ Detected';
            status.style.color = 'var(--success)';
            card.classList.remove('error');
        } else {
            img.src = '';
            img.alt = 'Detection failed';
            status.textContent = '✗ ' + (data.error || 'Failed');
            status.style.color = 'var(--error)';
            card.classList.add('error');
        }
    });
    
    // Reset selection
    selectedMethod = null;
    generateBtn.disabled = true;
    document.querySelectorAll('.result-card.selected').forEach(el => el.classList.remove('selected'));
    
    // Switch to step 2
    stepUpload.classList.remove('active');
    stepCorners.classList.remove('hidden');
    stepCorners.classList.add('active');
}

// === Corner Selection ===

document.querySelectorAll('.result-card.selectable').forEach(card => {
    card.addEventListener('click', () => {
        if (card.classList.contains('error')) return;
        
        // Deselect others
        document.querySelectorAll('.result-card.selected').forEach(el => el.classList.remove('selected'));
        
        // Select this one
        card.classList.add('selected');
        selectedMethod = card.dataset.method;
        generateBtn.disabled = false;
    });
});

// === Generate GLB ===

generateBtn.addEventListener('click', async () => {
    if (!sessionId || !selectedMethod) return;
    
    showLoading('Generating 3D mesh...');
    
    try {
        const depthScale = parseFloat(depthScaleInput.value);
        const boxDepth = parseFloat(boxDepthInput.value);
        
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                corner_method: selectedMethod,
                depth_scale: depthScale,
                box_depth: boxDepth
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Generation failed');
        }
        
        const result = await response.json();
        
        // Convert base64 to blob URL
        const glbBytes = atob(result.glb);
        const glbArray = new Uint8Array(glbBytes.length);
        for (let i = 0; i < glbBytes.length; i++) {
            glbArray[i] = glbBytes.charCodeAt(i);
        }
        glbBlob = new Blob([glbArray], { type: 'model/gltf-binary' });
        const glbUrl = URL.createObjectURL(glbBlob);
        
        // Set model viewer source
        modelViewer.src = glbUrl;
        
        // Set download link
        downloadBtn.href = glbUrl;
        
        // Switch to step 3
        stepCorners.classList.remove('active');
        stepViewer.classList.remove('hidden');
        stepViewer.classList.add('active');
        
    } catch (error) {
        alert('Error: ' + error.message);
        console.error(error);
    } finally {
        hideLoading();
    }
});

// === Navigation ===

backBtn.addEventListener('click', () => {
    stepCorners.classList.remove('active');
    stepUpload.classList.add('active');
});

restartBtn.addEventListener('click', () => {
    // Clean up session
    if (sessionId) {
        fetch(`/api/session/${sessionId}`, { method: 'DELETE' }).catch(() => {});
    }
    
    // Reset state
    currentImage = null;
    bbox = { x1: 0, y1: 0, x2: 0, y2: 0 };
    sessionId = null;
    selectedMethod = null;
    glbBlob = null;
    
    // Reset UI
    uploadArea.classList.remove('hidden');
    canvasContainer.classList.add('hidden');
    processBtn.disabled = true;
    imageInput.value = '';
    
    // Switch to step 1
    stepViewer.classList.remove('active');
    stepCorners.classList.add('hidden');
    stepUpload.classList.add('active');
});

// === Loading ===

function showLoading(text) {
    loadingText.textContent = text;
    loading.classList.remove('hidden');
}

function hideLoading() {
    loading.classList.add('hidden');
}

