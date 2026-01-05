/**
 * img2img.js - Image-to-image and mask canvas functionality
 *
 * Handles img2img image upload, mask drawing, and parameters.
 */

// =============================================================================
// Img2Img Image Upload
// =============================================================================

function setupImg2ImgUpload() {
    const dropzone = document.getElementById('img2imgDropzone');
    const input = document.getElementById('img2imgImageInput');

    if (!dropzone || !input) return;

    dropzone.addEventListener('click', () => input.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            await handleImg2ImgImageUpload(files[0]);
        }
    });

    input.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            await handleImg2ImgImageUpload(e.target.files[0]);
        }
    });
}

async function handleImg2ImgImageUpload(file) {
    const validation = validateImageFile(file);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }

    try {
        const base64 = await fileToBase64(file);

        // Create image to get dimensions
        const img = new Image();
        img.onload = () => {
            AppState.img2imgImage = {
                base64: base64,
                width: img.width,
                height: img.height,
            };

            // Show preview
            const preview = document.getElementById('img2imgImagePreview');
            const previewImg = document.getElementById('img2imgPreviewImg');
            const dimensions = document.getElementById('img2imgImageInfo');
            const clearBtn = document.getElementById('img2imgClearImage');

            if (preview && previewImg) {
                previewImg.src = base64;
                preview.classList.remove('hidden');
            }

            if (dimensions) {
                dimensions.textContent = `${img.width} x ${img.height}`;
            }

            if (clearBtn) {
                clearBtn.classList.remove('hidden');
            }

            // Initialize mask canvas
            initMaskCanvas(img.width, img.height);
        };
        img.src = base64;

    } catch (err) {
        console.error('Failed to load img2img image:', err);
        showError('Failed to load image');
    }
}

function clearImg2ImgImage() {
    AppState.img2imgImage = null;

    const preview = document.getElementById('img2imgImagePreview');
    const previewImg = document.getElementById('img2imgPreviewImg');
    const dimensions = document.getElementById('img2imgImageInfo');
    const maskCanvas = document.getElementById('img2imgMaskCanvas');
    const clearBtn = document.getElementById('img2imgClearImage');

    if (preview) preview.classList.add('hidden');
    if (previewImg) previewImg.src = '';
    if (dimensions) dimensions.textContent = '';
    if (clearBtn) clearBtn.classList.add('hidden');
    if (maskCanvas) {
        const ctx = maskCanvas.getContext('2d');
        ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    }

    AppState.img2imgMaskCtx = null;
}

// =============================================================================
// Mask Canvas
// =============================================================================

function initMaskCanvas(width, height) {
    const maskCanvas = document.getElementById('img2imgMaskCanvas');
    const maskControls = document.getElementById('img2imgMaskControls');

    if (!maskCanvas) return;

    // Set canvas size to match image
    maskCanvas.width = width;
    maskCanvas.height = height;

    const ctx = maskCanvas.getContext('2d');
    AppState.img2imgMaskCtx = ctx;

    // Clear canvas (transparent = preserve)
    ctx.clearRect(0, 0, width, height);

    // Show mask controls
    if (maskControls) {
        maskControls.classList.remove('hidden');
    }

    // Setup drawing events
    setupMaskDrawing(maskCanvas);
}

function setupMaskDrawing(canvas) {
    const brushTool = document.getElementById('img2imgBrushTool');
    const eraserTool = document.getElementById('img2imgEraserTool');
    const brushSizeSlider = document.getElementById('img2imgBrushSize');

    let currentTool = 'brush';
    let brushSize = 30;

    // Tool selection
    if (brushTool) {
        brushTool.addEventListener('click', () => {
            currentTool = 'brush';
            brushTool.classList.add('bg-purple-600');
            brushTool.classList.remove('bg-gray-700');
            if (eraserTool) {
                eraserTool.classList.remove('bg-purple-600');
                eraserTool.classList.add('bg-gray-700');
            }
        });
    }

    if (eraserTool) {
        eraserTool.addEventListener('click', () => {
            currentTool = 'eraser';
            eraserTool.classList.add('bg-purple-600');
            eraserTool.classList.remove('bg-gray-700');
            if (brushTool) {
                brushTool.classList.remove('bg-purple-600');
                brushTool.classList.add('bg-gray-700');
            }
        });
    }

    if (brushSizeSlider) {
        brushSizeSlider.addEventListener('input', () => {
            brushSize = parseInt(brushSizeSlider.value);
            const sizeValue = document.getElementById('img2imgBrushSizeValue');
            if (sizeValue) sizeValue.textContent = brushSize;
        });
    }

    // Drawing
    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        if (e.touches) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY,
            };
        }

        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    function drawOnMask(x, y) {
        const ctx = AppState.img2imgMaskCtx;
        if (!ctx) return;

        ctx.beginPath();
        ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);

        if (currentTool === 'brush') {
            ctx.fillStyle = 'rgba(255, 255, 255, 1)'; // White = edit
            ctx.fill();
        } else {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.fill();
            ctx.globalCompositeOperation = 'source-over';
        }
    }

    function startDrawing(e) {
        e.preventDefault();
        AppState.isDrawing = true;
        const coords = getCanvasCoords(e);
        AppState.lastX = coords.x;
        AppState.lastY = coords.y;
        drawOnMask(coords.x, coords.y);
    }

    function draw(e) {
        if (!AppState.isDrawing) return;
        e.preventDefault();

        const coords = getCanvasCoords(e);

        // Draw line from last point to current (for smooth strokes)
        const ctx = AppState.img2imgMaskCtx;
        if (ctx) {
            const dx = coords.x - AppState.lastX;
            const dy = coords.y - AppState.lastY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const steps = Math.max(1, Math.floor(dist / (brushSize / 4)));

            for (let i = 0; i <= steps; i++) {
                const t = i / steps;
                const x = AppState.lastX + dx * t;
                const y = AppState.lastY + dy * t;
                drawOnMask(x, y);
            }
        }

        AppState.lastX = coords.x;
        AppState.lastY = coords.y;
    }

    function stopDrawing() {
        AppState.isDrawing = false;
    }

    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    // Touch events
    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);
}

function clearMask() {
    const maskCanvas = document.getElementById('img2imgMaskCanvas');
    if (!maskCanvas || !AppState.img2imgMaskCtx) return;

    AppState.img2imgMaskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
}

function invertMask() {
    const maskCanvas = document.getElementById('img2imgMaskCanvas');
    if (!maskCanvas || !AppState.img2imgMaskCtx) return;

    const ctx = AppState.img2imgMaskCtx;
    const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
        // Invert alpha channel (255 - current value)
        const alpha = data[i + 3];
        if (alpha > 0) {
            // Was painted (edit), make transparent (preserve)
            data[i + 3] = 0;
        } else {
            // Was transparent (preserve), make white (edit)
            data[i] = 255;
            data[i + 1] = 255;
            data[i + 2] = 255;
            data[i + 3] = 255;
        }
    }

    ctx.putImageData(imageData, 0, 0);
}

// =============================================================================
// Mask Export
// =============================================================================

function getMaskBase64() {
    const maskCanvas = document.getElementById('img2imgMaskCanvas');
    if (!maskCanvas) return null;

    // Create a grayscale version of the mask
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = maskCanvas.width;
    tempCanvas.height = maskCanvas.height;
    const tempCtx = tempCanvas.getContext('2d');

    // Fill with black (preserve)
    tempCtx.fillStyle = 'black';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Draw mask (white areas = edit)
    tempCtx.drawImage(maskCanvas, 0, 0);

    return tempCanvas.toDataURL('image/png');
}

function hasMask() {
    const maskCanvas = document.getElementById('img2imgMaskCanvas');
    if (!maskCanvas || !AppState.img2imgMaskCtx) return false;

    const imageData = AppState.img2imgMaskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const data = imageData.data;

    // Check if any pixel has alpha > 0 (meaning something was drawn)
    for (let i = 3; i < data.length; i += 4) {
        if (data[i] > 0) return true;
    }

    return false;
}

// =============================================================================
// Img2Img Parameters
// =============================================================================

function getImg2ImgParams() {
    const enableCheckbox = document.getElementById('img2imgEnabled');
    if (!enableCheckbox || !enableCheckbox.checked) {
        return null;
    }

    if (!AppState.img2imgImage) {
        return null;
    }

    const strengthSlider = document.getElementById('img2imgStrength');

    const params = {
        image: AppState.img2imgImage.base64,
        strength: strengthSlider ? parseFloat(strengthSlider.value) : 0.75,
    };

    // Add mask if differential editing is enabled and mask exists
    const maskEnabled = document.getElementById('img2imgMaskEnabled');
    if (maskEnabled && maskEnabled.checked && hasMask()) {
        params.mask_image = getMaskBase64();
    }

    return params;
}

// =============================================================================
// Event Binding
// =============================================================================

function initImg2ImgEvents() {
    setupImg2ImgUpload();

    const clearBtn = document.getElementById('img2imgClearImage');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearImg2ImgImage);
    }

    const clearMaskBtn = document.getElementById('img2imgClearMask');
    if (clearMaskBtn) {
        clearMaskBtn.addEventListener('click', clearMask);
    }

    const invertMaskBtn = document.getElementById('img2imgInvertMask');
    if (invertMaskBtn) {
        invertMaskBtn.addEventListener('click', invertMask);
    }

    // Strength slider value display
    const strengthSlider = document.getElementById('img2imgStrength');
    const strengthValue = document.getElementById('img2imgStrengthValue');
    if (strengthSlider && strengthValue) {
        strengthSlider.addEventListener('input', () => {
            strengthValue.textContent = formatNumber(strengthSlider.value, 2);
        });
    }

    // Toggle img2img section visibility
    const enableCheckbox = document.getElementById('img2imgEnabled');
    const controls = document.getElementById('img2imgControls');
    if (enableCheckbox && controls) {
        enableCheckbox.addEventListener('change', () => {
            controls.classList.toggle('hidden', !enableCheckbox.checked);
        });
    }

    // Toggle differential mask controls
    const maskEnabled = document.getElementById('img2imgMaskEnabled');
    const maskControls = document.getElementById('img2imgMaskControls');
    if (maskEnabled && maskControls) {
        maskEnabled.addEventListener('change', () => {
            maskControls.classList.toggle('hidden', !maskEnabled.checked);
        });
    }
}

// Export for use by other modules
window.handleImg2ImgImageUpload = handleImg2ImgImageUpload;
window.clearImg2ImgImage = clearImg2ImgImage;
window.initMaskCanvas = initMaskCanvas;
window.clearMask = clearMask;
window.invertMask = invertMask;
window.getMaskBase64 = getMaskBase64;
window.hasMask = hasMask;
window.getImg2ImgParams = getImg2ImgParams;
window.initImg2ImgEvents = initImg2ImgEvents;
