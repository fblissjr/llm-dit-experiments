/**
 * image-utils.js - Shared image loading utilities
 *
 * Provides functions for loading images into various input targets,
 * drag-and-drop handling, and workflow continuity features.
 */

// =============================================================================
// Image Target Definitions
// =============================================================================

const ImageTargets = {
    IMG2IMG: 'img2img',
    VL: 'vl',
    QWEN_EDIT: 'qwen-edit',
    QWEN_COMBINE: 'qwen-combine',
};

// =============================================================================
// Core Image Loading Functions
// =============================================================================

/**
 * Load a base64 image into the Img2Img input
 * @param {string} base64 - The base64 image data (with or without data: prefix)
 * @param {number} width - Optional image width
 * @param {number} height - Optional image height
 */
async function loadImageIntoImg2Img(base64, width, height) {
    // Ensure base64 has data prefix
    const imageData = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;

    // Create image to get dimensions if not provided
    const img = new Image();

    return new Promise((resolve, reject) => {
        img.onload = () => {
            const w = width || img.width;
            const h = height || img.height;

            AppState.img2imgImage = {
                base64: imageData,
                width: w,
                height: h,
            };

            // Update preview
            const preview = document.getElementById('img2imgImagePreview');
            const previewImg = document.getElementById('img2imgPreviewImg');
            const dimensions = document.getElementById('img2imgImageInfo');
            const clearBtn = document.getElementById('img2imgClearImage');

            if (preview && previewImg) {
                previewImg.src = imageData;
                preview.classList.remove('hidden');
            }

            if (dimensions) {
                dimensions.textContent = `${w} x ${h}`;
            }

            if (clearBtn) {
                clearBtn.classList.remove('hidden');
            }

            // Initialize mask canvas
            if (typeof initMaskCanvas === 'function') {
                initMaskCanvas(w, h);
            }

            resolve({ width: w, height: h });
        };

        img.onerror = () => reject(new Error('Failed to load image'));
        img.src = imageData;
    });
}

/**
 * Load a base64 image into the VL conditioning input
 * @param {string} base64 - The base64 image data
 */
async function loadImageIntoVL(base64) {
    const imageData = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;

    AppState.vlImagePreview = imageData;
    // Clear any previous embeddings since this is a new image
    AppState.vlEmbeddingsId = null;

    const preview = document.getElementById('vlImagePreview');
    const previewImg = document.getElementById('vlPreviewImg');
    const extractBtn = document.getElementById('vlExtractBtn');
    const embeddingsStatus = document.getElementById('vlEmbeddingsStatus');

    if (preview && previewImg) {
        previewImg.src = imageData;
        preview.classList.remove('hidden');
    }

    if (extractBtn) {
        extractBtn.disabled = false;
    }

    if (embeddingsStatus) {
        embeddingsStatus.classList.add('hidden');
    }
}

/**
 * Load a base64 image into Qwen Image Edit single edit input
 * @param {string} base64 - The base64 image data
 */
async function loadImageIntoQwenEdit(base64) {
    const imageData = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;

    AppState.singleEditImage = imageData;

    const preview = document.getElementById('singleEditPreview');
    const previewImg = document.getElementById('singleEditPreviewImg');
    const editBtn = document.getElementById('singleEditBtn');

    if (preview && previewImg) {
        previewImg.src = imageData;
        preview.classList.remove('hidden');
    }

    if (editBtn) {
        editBtn.disabled = false;
    }
}

/**
 * Add a base64 image to the Qwen Image Edit multi-combine list
 * @param {string} base64 - The base64 image data
 */
async function addImageToQwenCombine(base64) {
    const imageData = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;

    if (!AppState.multiCombineImages) {
        AppState.multiCombineImages = [];
    }

    AppState.multiCombineImages.push(imageData);

    // Re-render the preview
    if (typeof renderMultiPreview === 'function') {
        renderMultiPreview();
    }

    if (typeof updateMultiCombineState === 'function') {
        updateMultiCombineState();
    }
}

/**
 * Universal function to load an image into a specific target
 * @param {string} base64 - The base64 image data
 * @param {string} target - Target identifier (from ImageTargets)
 * @param {Object} options - Optional parameters (width, height)
 */
async function loadImageIntoTarget(base64, target, options = {}) {
    switch (target) {
        case ImageTargets.IMG2IMG:
            return loadImageIntoImg2Img(base64, options.width, options.height);
        case ImageTargets.VL:
            return loadImageIntoVL(base64);
        case ImageTargets.QWEN_EDIT:
            return loadImageIntoQwenEdit(base64);
        case ImageTargets.QWEN_COMBINE:
            return addImageToQwenCombine(base64);
        default:
            console.warn('Unknown image target:', target);
    }
}

// =============================================================================
// UI Navigation Helpers
// =============================================================================

/**
 * Expand a collapsible section by its ID
 * Handles both <details> elements and collapsible-content divs
 * @param {string} sectionId - The ID of the section to expand
 */
function expandSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (!element) return;

    // Handle <details> elements
    if (element.tagName === 'DETAILS') {
        element.open = true;
        return;
    }

    // Handle collapsible-content divs
    if (element.classList.contains('collapsible-content')) {
        element.classList.add('expanded');

        // Update the toggle button aria-expanded
        const toggle = document.querySelector(`[aria-controls="${sectionId}"]`);
        if (toggle) {
            toggle.setAttribute('aria-expanded', 'true');
        }
    }
}

/**
 * Scroll to an element smoothly
 * @param {string|HTMLElement} elementOrId - Element or ID to scroll to
 */
function scrollToElement(elementOrId) {
    const element = typeof elementOrId === 'string'
        ? document.getElementById(elementOrId)
        : elementOrId;

    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

/**
 * Enable a checkbox and expand its associated controls
 * @param {string} checkboxId - The checkbox ID
 * @param {string} controlsId - The controls section ID
 */
function enableFeatureAndExpand(checkboxId, controlsId) {
    const checkbox = document.getElementById(checkboxId);
    const controls = document.getElementById(controlsId);

    if (checkbox) {
        checkbox.checked = true;
        checkbox.dispatchEvent(new Event('change'));
    }

    if (controls) {
        controls.classList.remove('hidden');
    }
}

// =============================================================================
// Unified Action Handlers
// =============================================================================

/**
 * Use an image as Img2Img input
 * @param {string} base64 - The base64 image data
 * @param {number} width - Image width
 * @param {number} height - Image height
 */
async function useAsImg2Img(base64, width, height) {
    // Expand the img2img details section
    expandSection('img2imgSection');

    // Enable img2img mode
    enableFeatureAndExpand('img2imgEnabled', 'img2imgControls');

    // Load the image
    await loadImageIntoImg2Img(base64, width, height);

    // Scroll to the section
    scrollToElement('img2imgSection');
}

/**
 * Use an image as VL conditioning reference
 * @param {string} base64 - The base64 image data
 */
async function useAsVLReference(base64) {
    // Expand the VL details section
    expandSection('vlSection');

    // Load the image
    await loadImageIntoVL(base64);

    // Scroll to the section
    scrollToElement('vlSection');
}

/**
 * Use an image in Qwen Image Edit
 * @param {string} base64 - The base64 image data
 */
async function useInQwenEdit(base64) {
    // Switch to Qwen-Image model if not already
    if (typeof switchModelType === 'function') {
        switchModelType('qwenimage');
    }

    // Load the image
    await loadImageIntoQwenEdit(base64);

    // Scroll to the section
    scrollToElement('qwenImageSection');
}

/**
 * Add an image to Qwen Image Edit combine list
 * @param {string} base64 - The base64 image data
 */
async function addToCombine(base64) {
    // Switch to Qwen-Image model if not already
    if (typeof switchModelType === 'function') {
        switchModelType('qwenimage');
    }

    // Add to combine list
    await addImageToQwenCombine(base64);

    // Scroll to the section
    scrollToElement('qwenImageSection');
}

// =============================================================================
// Current Result State
// =============================================================================

/**
 * Store the current generated result for use in workflows
 * @param {string} base64 - The base64 image data
 * @param {number} width - Image width
 * @param {number} height - Image height
 */
function setCurrentResult(base64, width, height) {
    AppState.currentResultBase64 = base64;
    AppState.currentResultWidth = width;
    AppState.currentResultHeight = height;
}

/**
 * Get the current result data
 * @returns {Object|null} The current result or null
 */
function getCurrentResult() {
    if (!AppState.currentResultBase64) return null;
    return {
        base64: AppState.currentResultBase64,
        width: AppState.currentResultWidth,
        height: AppState.currentResultHeight,
    };
}

// =============================================================================
// Drag and Drop Utilities
// =============================================================================

/**
 * Setup a dropzone to accept internal image transfers
 * @param {HTMLElement} dropzone - The dropzone element
 * @param {string} targetType - The target type (from ImageTargets)
 */
function setupDropzoneForInternalImages(dropzone, targetType) {
    if (!dropzone) return;

    dropzone.addEventListener('dragover', (e) => {
        if (e.dataTransfer.types.includes('application/x-image-transfer')) {
            e.preventDefault();
            dropzone.classList.add('dragover', 'drop-target-active');
        }
    });

    dropzone.addEventListener('dragleave', (e) => {
        if (!dropzone.contains(e.relatedTarget)) {
            dropzone.classList.remove('dragover', 'drop-target-active');
        }
    });

    dropzone.addEventListener('drop', async (e) => {
        dropzone.classList.remove('dragover', 'drop-target-active');

        const transferData = e.dataTransfer.getData('application/x-image-transfer');
        if (transferData) {
            e.preventDefault();
            try {
                const data = JSON.parse(transferData);
                await loadImageIntoTarget(data.base64, targetType, {
                    width: data.width,
                    height: data.height,
                });
            } catch (err) {
                console.error('Failed to process dropped image:', err);
            }
        }
    });
}

/**
 * Make a history item draggable
 * @param {HTMLElement} element - The element to make draggable
 * @param {Object} imageData - The image data { base64, width, height }
 */
function makeHistoryItemDraggable(element, imageData) {
    element.setAttribute('draggable', 'true');

    element.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('application/x-image-transfer', JSON.stringify(imageData));
        e.dataTransfer.effectAllowed = 'copy';

        // Add visual feedback
        document.body.classList.add('dragging-image');
        element.classList.add('dragging');

        // Highlight valid drop targets
        highlightDropTargets(true);
    });

    element.addEventListener('dragend', () => {
        document.body.classList.remove('dragging-image');
        element.classList.remove('dragging');
        highlightDropTargets(false);
    });
}

/**
 * Highlight or unhighlight all valid drop targets
 * @param {boolean} highlight - Whether to highlight or not
 */
function highlightDropTargets(highlight) {
    const targets = [
        'img2imgDropzone',
        'vlImageDropzone',
        'singleEditDropzone',
        'multiCombineDropzone',
    ];

    targets.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.toggle('drop-target-highlight', highlight);
        }
    });
}

// =============================================================================
// Mobile Utilities
// =============================================================================

const isMobile = () => window.matchMedia('(max-width: 768px)').matches;
const isTouch = () => 'ontouchstart' in window;

/**
 * Setup long-press detection for touch devices
 * @param {HTMLElement} element - The element to setup
 * @param {Function} callback - Callback when long-press detected
 * @param {number} duration - Press duration in ms (default 500)
 */
function setupLongPress(element, callback, duration = 500) {
    let timer = null;
    let startX, startY;
    const moveThreshold = 10;

    element.addEventListener('touchstart', (e) => {
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;

        timer = setTimeout(() => {
            callback(e);
        }, duration);
    }, { passive: true });

    element.addEventListener('touchmove', (e) => {
        if (timer) {
            const dx = Math.abs(e.touches[0].clientX - startX);
            const dy = Math.abs(e.touches[0].clientY - startY);
            if (dx > moveThreshold || dy > moveThreshold) {
                clearTimeout(timer);
                timer = null;
            }
        }
    }, { passive: true });

    element.addEventListener('touchend', () => {
        if (timer) {
            clearTimeout(timer);
            timer = null;
        }
    });

    element.addEventListener('touchcancel', () => {
        if (timer) {
            clearTimeout(timer);
            timer = null;
        }
    });
}

// =============================================================================
// Initialization
// =============================================================================

function initImageUtils() {
    // Initialize AppState properties if they don't exist
    if (!AppState.currentResultBase64) AppState.currentResultBase64 = null;
    if (!AppState.currentResultWidth) AppState.currentResultWidth = null;
    if (!AppState.currentResultHeight) AppState.currentResultHeight = null;

    // Setup dropzones for internal image transfers
    setupDropzoneForInternalImages(document.getElementById('img2imgDropzone'), ImageTargets.IMG2IMG);
    setupDropzoneForInternalImages(document.getElementById('vlImageDropzone'), ImageTargets.VL);
    setupDropzoneForInternalImages(document.getElementById('singleEditDropzone'), ImageTargets.QWEN_EDIT);
    setupDropzoneForInternalImages(document.getElementById('multiCombineDropzone'), ImageTargets.QWEN_COMBINE);
}

// Export for use by other modules
window.ImageTargets = ImageTargets;
window.loadImageIntoImg2Img = loadImageIntoImg2Img;
window.loadImageIntoVL = loadImageIntoVL;
window.loadImageIntoQwenEdit = loadImageIntoQwenEdit;
window.addImageToQwenCombine = addImageToQwenCombine;
window.loadImageIntoTarget = loadImageIntoTarget;
window.useAsImg2Img = useAsImg2Img;
window.useAsVLReference = useAsVLReference;
window.useInQwenEdit = useInQwenEdit;
window.addToCombine = addToCombine;
window.setCurrentResult = setCurrentResult;
window.getCurrentResult = getCurrentResult;
window.makeHistoryItemDraggable = makeHistoryItemDraggable;
window.setupLongPress = setupLongPress;
window.isMobile = isMobile;
window.isTouch = isTouch;
window.initImageUtils = initImageUtils;
