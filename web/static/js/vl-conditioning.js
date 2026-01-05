/**
 * vl-conditioning.js - Vision-Language conditioning functionality
 *
 * Handles VL image upload, embedding extraction, and generation parameters.
 */

// =============================================================================
// VL Status Check
// =============================================================================

async function checkVLStatus() {
    const vlSection = document.getElementById('vlSection');
    const vlStatus = document.getElementById('vlStatus');

    try {
        const data = await ApiClient.getVLStatus();
        AppState.features.vlEnabled = data.available;

        if (vlSection) {
            vlSection.classList.toggle('hidden', !data.available);
        }

        if (vlStatus) {
            if (data.available) {
                vlStatus.textContent = 'VL Ready';
                vlStatus.classList.add('text-green-400');
                vlStatus.classList.remove('text-gray-500');
            } else {
                vlStatus.textContent = data.reason || 'VL Unavailable';
                vlStatus.classList.remove('text-green-400');
                vlStatus.classList.add('text-gray-500');
            }
        }

        // Load VL config if available
        if (data.available) {
            await loadVLConfig();
        }

    } catch (err) {
        console.error('Failed to check VL status:', err);
        AppState.features.vlEnabled = false;
    }
}

async function loadVLConfig() {
    try {
        const data = await ApiClient.getVLConfig();

        // Apply VL config defaults
        const vlAlphaSlider = document.getElementById('vlAlpha');
        const vlAlphaValue = document.getElementById('vlAlphaValue');

        if (data.default_alpha !== undefined && vlAlphaSlider && vlAlphaValue) {
            vlAlphaSlider.value = data.default_alpha;
            vlAlphaValue.textContent = formatNumber(data.default_alpha, 2);
        }

    } catch (err) {
        console.error('Failed to load VL config:', err);
    }
}

// =============================================================================
// VL Image Upload
// =============================================================================

function setupVLImageUpload() {
    const dropzone = document.getElementById('vlImageDropzone');
    const input = document.getElementById('vlImageInput');

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
            await handleVLImageUpload(files[0]);
        }
    });

    input.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            await handleVLImageUpload(e.target.files[0]);
        }
    });
}

async function handleVLImageUpload(file) {
    const validation = validateImageFile(file);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }

    try {
        const base64 = await fileToBase64(file);
        AppState.vlImagePreview = base64;

        const preview = document.getElementById('vlImagePreview');
        const previewImg = document.getElementById('vlPreviewImg');
        const extractBtn = document.getElementById('vlExtractBtn');

        if (preview && previewImg) {
            previewImg.src = base64;
            preview.classList.remove('hidden');
        }

        if (extractBtn) {
            extractBtn.disabled = false;
        }

    } catch (err) {
        console.error('Failed to load VL image:', err);
        showError('Failed to load image');
    }
}

function clearVLImage() {
    AppState.vlImagePreview = null;
    AppState.vlEmbeddingsId = null;

    const preview = document.getElementById('vlImagePreview');
    const previewImg = document.getElementById('vlPreviewImg');
    const extractBtn = document.getElementById('vlExtractBtn');
    const embeddingsStatus = document.getElementById('vlEmbeddingsStatus');

    if (preview) preview.classList.add('hidden');
    if (previewImg) previewImg.src = '';
    if (extractBtn) extractBtn.disabled = true;
    if (embeddingsStatus) embeddingsStatus.classList.add('hidden');
}

// =============================================================================
// VL Embedding Extraction
// =============================================================================

async function extractVLEmbeddings() {
    if (!AppState.vlImagePreview) {
        showError('Please upload an image first');
        return;
    }

    const extractBtn = document.getElementById('vlExtractBtn');
    const embeddingsStatus = document.getElementById('vlEmbeddingsStatus');
    const vlHiddenLayer = document.getElementById('vlHiddenLayer');

    setButtonLoading(extractBtn, 'Extracting...');

    try {
        // Clear previous embeddings
        if (AppState.vlEmbeddingsId) {
            await ApiClient.deleteVLCache(AppState.vlEmbeddingsId);
        }

        const data = {
            image: AppState.vlImagePreview,
            hidden_layer: vlHiddenLayer ? parseInt(vlHiddenLayer.value) : -6,
        };

        const result = await ApiClient.extractVLEmbeddings(data);

        if (result.embeddings_id) {
            AppState.vlEmbeddingsId = result.embeddings_id;

            if (embeddingsStatus) {
                embeddingsStatus.classList.remove('hidden');
                embeddingsStatus.textContent = `Embeddings ready (${result.token_count || '?'} tokens)`;
            }
        }

    } catch (err) {
        console.error('Failed to extract VL embeddings:', err);
        showError('Failed to extract embeddings');
    } finally {
        resetButton(extractBtn);
    }
}

// =============================================================================
// VL Alpha Presets
// =============================================================================

function updateVLPresetButtons(selectedAlpha) {
    const presets = document.querySelectorAll('.vl-alpha-preset');
    presets.forEach(btn => {
        const presetValue = parseFloat(btn.dataset.alpha);
        if (Math.abs(presetValue - selectedAlpha) < 0.01) {
            btn.classList.add('bg-purple-600');
            btn.classList.remove('bg-gray-700');
        } else {
            btn.classList.remove('bg-purple-600');
            btn.classList.add('bg-gray-700');
        }
    });
}

function setupVLAlphaPresets() {
    const presets = document.querySelectorAll('.vl-alpha-preset');
    const vlAlphaSlider = document.getElementById('vlAlpha');
    const vlAlphaValue = document.getElementById('vlAlphaValue');

    presets.forEach(btn => {
        btn.addEventListener('click', () => {
            const alpha = parseFloat(btn.dataset.alpha);
            if (vlAlphaSlider && vlAlphaValue) {
                vlAlphaSlider.value = alpha;
                vlAlphaValue.textContent = formatNumber(alpha, 2);
            }
            updateVLPresetButtons(alpha);
        });
    });

    if (vlAlphaSlider) {
        vlAlphaSlider.addEventListener('input', () => {
            const alpha = parseFloat(vlAlphaSlider.value);
            if (vlAlphaValue) {
                vlAlphaValue.textContent = formatNumber(alpha, 2);
            }
            updateVLPresetButtons(alpha);
        });
    }
}

// =============================================================================
// VL Parameters for Generation
// =============================================================================

function getVLParams() {
    if (!AppState.vlEmbeddingsId) {
        return null;
    }

    const vlAlphaSlider = document.getElementById('vlAlpha');
    const vlHiddenLayer = document.getElementById('vlHiddenLayer');
    const vlBlendMode = document.getElementById('vlBlendMode');

    return {
        embeddings_id: AppState.vlEmbeddingsId,
        alpha: vlAlphaSlider ? parseFloat(vlAlphaSlider.value) : 0.5,
        hidden_layer: vlHiddenLayer ? parseInt(vlHiddenLayer.value) : -6,
        blend_mode: vlBlendMode ? vlBlendMode.value : 'linear',
    };
}

// =============================================================================
// Event Binding
// =============================================================================

function initVLEvents() {
    setupVLImageUpload();
    setupVLAlphaPresets();

    const extractBtn = document.getElementById('vlExtractBtn');
    if (extractBtn) {
        extractBtn.addEventListener('click', extractVLEmbeddings);
    }

    const clearBtn = document.getElementById('vlClearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearVLImage);
    }
}

// Export for use by other modules
window.checkVLStatus = checkVLStatus;
window.loadVLConfig = loadVLConfig;
window.handleVLImageUpload = handleVLImageUpload;
window.clearVLImage = clearVLImage;
window.extractVLEmbeddings = extractVLEmbeddings;
window.updateVLPresetButtons = updateVLPresetButtons;
window.getVLParams = getVLParams;
window.initVLEvents = initVLEvents;
