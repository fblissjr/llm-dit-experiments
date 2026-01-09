/**
 * qwen-image.js - Qwen-Image model specific functionality
 *
 * Handles Qwen-Image decomposition, layer editing, and multi-image combining.
 */

// =============================================================================
// Model Status Checks
// =============================================================================

async function checkQwenImageStatus() {
    const qwenImageSection = document.getElementById('qwenImageSection');
    const qwenImageStatus = document.getElementById('qwenImageStatus');

    try {
        const data = await ApiClient.getQwenImageStatus();
        AppState.features.qwenImageEnabled = data.available;

        if (qwenImageSection) {
            qwenImageSection.classList.toggle('hidden', !data.available);
        }

        if (qwenImageStatus) {
            if (data.available) {
                qwenImageStatus.textContent = 'Ready';
                qwenImageStatus.classList.add('text-green-400');
                qwenImageStatus.classList.remove('text-gray-500');
            } else {
                qwenImageStatus.textContent = data.reason || 'Unavailable';
                qwenImageStatus.classList.remove('text-green-400');
                qwenImageStatus.classList.add('text-gray-500');
            }
        }

    } catch (err) {
        console.error('Failed to check Qwen-Image status:', err);
        AppState.features.qwenImageEnabled = false;
    }
}

async function checkQwenImage2512Status() {
    const qwenImage2512Section = document.getElementById('qwenImage2512Section');
    const qwenImage2512Status = document.getElementById('qwenImage2512Status');

    try {
        const data = await ApiClient.getQwenImage2512Status();
        AppState.features.qwenImage2512Enabled = data.available;

        if (qwenImage2512Section) {
            qwenImage2512Section.classList.toggle('hidden', !data.available);
        }

        if (qwenImage2512Status) {
            if (data.available) {
                qwenImage2512Status.textContent = 'Ready';
                qwenImage2512Status.classList.add('text-green-400');
                qwenImage2512Status.classList.remove('text-gray-500');
            } else {
                qwenImage2512Status.textContent = data.reason || 'Unavailable';
                qwenImage2512Status.classList.remove('text-green-400');
                qwenImage2512Status.classList.add('text-gray-500');
            }
        }

    } catch (err) {
        console.error('Failed to check Qwen-Image-2512 status:', err);
        AppState.features.qwenImage2512Enabled = false;
    }
}

// =============================================================================
// UI Section Visibility
// =============================================================================

function updateQiSections() {
    const modelType = getSelectedModelType();

    const zimageSection = document.getElementById('zImageControls');
    const qwenImageSection = document.getElementById('qwenImageSection');
    const ltx2Section = document.getElementById('ltx2Section');

    // Z-Image section shown for both zimage and qwenimage2512 (Qwen T2I uses same form)
    if (zimageSection) {
        zimageSection.classList.toggle('hidden', modelType !== 'zimage' && modelType !== 'qwenimage2512');
    }
    if (qwenImageSection) {
        qwenImageSection.classList.toggle('hidden', modelType !== 'qwenimage');
    }
    if (ltx2Section) {
        ltx2Section.classList.toggle('hidden', modelType !== 'ltx2');
    }
}

function getSelectedModelType() {
    const zimageBtn = document.getElementById('modelTypeZImage');
    const qwenImageBtn = document.getElementById('modelTypeQwenImage');
    const qwenImage2512Btn = document.getElementById('modelTypeQwenImage2512');
    const ltx2Btn = document.getElementById('modelTypeLTX2');

    if (zimageBtn && zimageBtn.classList.contains('bg-blue-600')) return 'zimage';
    if (qwenImageBtn && qwenImageBtn.classList.contains('bg-blue-600')) return 'qwenimage';
    if (qwenImage2512Btn && qwenImage2512Btn.classList.contains('bg-blue-600')) return 'qwenimage2512';
    if (ltx2Btn && ltx2Btn.classList.contains('bg-blue-600')) return 'ltx2';

    return 'zimage'; // default
}

function switchModelType(type) {
    const buttons = {
        'zimage': document.getElementById('modelTypeZImage'),
        'qwenimage': document.getElementById('modelTypeQwenImage'),
        'qwenimage2512': document.getElementById('modelTypeQwenImage2512'),
        'ltx2': document.getElementById('modelTypeLTX2'),
    };

    // Update button styles
    Object.entries(buttons).forEach(([key, btn]) => {
        if (!btn) return;
        if (key === type) {
            btn.classList.add('bg-blue-600');
            btn.classList.remove('bg-gray-700');
        } else {
            btn.classList.remove('bg-blue-600');
            btn.classList.add('bg-gray-700');
        }
    });

    // Update section visibility
    updateQiSections();

    // Update model defaults
    if (typeof setModelDefaults === 'function') {
        setModelDefaults(type);
    }

    // Reload resolution presets for new model
    if (typeof ResolutionSelector !== 'undefined' && ResolutionSelector.loadConstraints) {
        // Map UI model type to API model type
        const modelMap = {
            'zimage': 'zimage',
            'qwenimage': 'qwenimage-layered',
            'qwenimage2512': 'qwenimage-t2i',
        };
        ResolutionSelector.loadConstraints(modelMap[type] || 'zimage');
    }
}

// =============================================================================
// Qwen-Image Decomposition
// =============================================================================

function setupQiImageUpload() {
    const dropzone = document.getElementById('qiImageDropzone');
    const input = document.getElementById('qiImageInput');

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
            await handleQiImageUpload(files[0]);
        }
    });

    input.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            await handleQiImageUpload(e.target.files[0]);
        }
    });
}

async function handleQiImageUpload(file) {
    const validation = validateImageFile(file);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }

    try {
        const base64 = await fileToBase64(file);
        AppState.qiInputImage = base64;

        const preview = document.getElementById('qiImagePreview');
        const previewImg = document.getElementById('qiPreviewImg');
        const decomposeBtn = document.getElementById('qiDecomposeBtn');

        if (preview && previewImg) {
            previewImg.src = base64;
            preview.classList.remove('hidden');
        }

        if (decomposeBtn) {
            decomposeBtn.disabled = false;
        }

    } catch (err) {
        console.error('Failed to load Qwen-Image input:', err);
        showError('Failed to load image');
    }
}

async function decomposeImage() {
    if (!AppState.qiInputImage) {
        showError('Please upload an image first');
        return;
    }

    const decomposeBtn = document.getElementById('qiDecomposeBtn');
    const promptEl = document.getElementById('qiPrompt');
    const layerNumEl = document.getElementById('qiLayerNum');
    const resolutionEl = document.getElementById('qiResolution');

    setButtonLoading(decomposeBtn, 'Decomposing...');

    try {
        const data = {
            image: AppState.qiInputImage,
            prompt: promptEl ? promptEl.value : 'A colorful scene',
            layer_num: layerNumEl ? parseInt(layerNumEl.value) : 4,
            resolution: resolutionEl ? parseInt(resolutionEl.value) : 640,
            seed: null, // Random
        };

        const result = await ApiClient.qwenImageDecompose(data);

        if (result.layers) {
            AppState.qiDecomposedLayers = result.layers;
            renderDecomposedLayers(result.layers);
        }

    } catch (err) {
        console.error('Failed to decompose image:', err);
        showError('Decomposition failed');
    } finally {
        resetButton(decomposeBtn);
    }
}

function renderDecomposedLayers(layers) {
    const container = document.getElementById('qiLayersContainer');
    if (!container) return;

    container.innerHTML = layers.map((layer, index) => `
        <div class="layer-card bg-gray-800 rounded-lg p-2">
            <img src="data:image/png;base64,${escapeHtml(layer.image)}" class="w-full aspect-square object-cover rounded mb-2" alt="Layer ${index + 1}">
            <div class="flex gap-1">
                <button class="layer-edit flex-1 px-2 py-1 bg-purple-600 hover:bg-purple-700 text-white text-xs rounded" data-index="${index}">
                    Edit
                </button>
                <button class="layer-download px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded" data-index="${index}">
                    Save
                </button>
            </div>
        </div>
    `).join('');

    container.classList.remove('hidden');

    // Add event listeners
    container.querySelectorAll('.layer-edit').forEach(btn => {
        btn.addEventListener('click', () => {
            const index = parseInt(btn.dataset.index);
            openEditModal(index);
        });
    });

    container.querySelectorAll('.layer-download').forEach(btn => {
        btn.addEventListener('click', () => {
            const index = parseInt(btn.dataset.index);
            downloadLayer(index);
        });
    });
}

function downloadLayer(index) {
    const layer = AppState.qiDecomposedLayers[index];
    if (!layer) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${layer.image}`;
    link.download = `layer_${index + 1}.png`;
    link.click();
}

// =============================================================================
// Layer Editing Modal
// =============================================================================

function openEditModal(layerIndex) {
    const modal = document.getElementById('editLayerModal');
    const previewImg = document.getElementById('editLayerPreview');
    const indexInput = document.getElementById('editLayerIndex');

    if (!modal || !AppState.qiDecomposedLayers[layerIndex]) return;

    const layer = AppState.qiDecomposedLayers[layerIndex];

    if (previewImg) {
        previewImg.src = `data:image/png;base64,${layer.image}`;
    }
    if (indexInput) {
        indexInput.value = layerIndex;
    }

    modal.classList.remove('hidden');
}

function closeEditModal() {
    const modal = document.getElementById('editLayerModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

async function executeLayerEdit() {
    const indexInput = document.getElementById('editLayerIndex');
    const instructionEl = document.getElementById('editInstruction');
    const editBtn = document.getElementById('executeEditBtn');

    if (!indexInput || !instructionEl) return;

    const layerIndex = parseInt(indexInput.value);
    const layer = AppState.qiDecomposedLayers[layerIndex];

    if (!layer) {
        showError('Layer not found');
        return;
    }

    setButtonLoading(editBtn, 'Editing...');

    try {
        const data = {
            layer_image: `data:image/png;base64,${layer.image}`,
            instruction: instructionEl.value,
        };

        const result = await ApiClient.qwenImageEditLayer(data);

        if (result.edited_image) {
            // Update the layer
            AppState.qiDecomposedLayers[layerIndex].image = result.edited_image.split(',')[1] || result.edited_image;
            renderDecomposedLayers(AppState.qiDecomposedLayers);
            closeEditModal();
        }

    } catch (err) {
        console.error('Failed to edit layer:', err);
        showError('Edit failed');
    } finally {
        resetButton(editBtn);
    }
}

// =============================================================================
// Single Image Edit
// =============================================================================

function setupSingleEditUpload() {
    const dropzone = document.getElementById('singleEditDropzone');
    const input = document.getElementById('singleEditInput');

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
            await handleSingleEditImage(files[0]);
        }
    });

    input.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            await handleSingleEditImage(e.target.files[0]);
        }
    });
}

async function handleSingleEditImage(file) {
    const validation = validateImageFile(file);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }

    try {
        const base64 = await fileToBase64(file);
        AppState.singleEditImage = base64;

        const preview = document.getElementById('singleEditPreview');
        const previewImg = document.getElementById('singleEditPreviewImg');
        const editBtn = document.getElementById('singleEditBtn');

        if (preview && previewImg) {
            previewImg.src = base64;
            preview.classList.remove('hidden');
        }

        if (editBtn) {
            editBtn.disabled = false;
        }

    } catch (err) {
        console.error('Failed to load single edit image:', err);
        showError('Failed to load image');
    }
}

// =============================================================================
// Multi-Image Combine
// =============================================================================

function setupMultiCombineUpload() {
    const dropzone = document.getElementById('multiCombineDropzone');
    const input = document.getElementById('multiCombineInput');

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
        await handleMultiFiles(e.dataTransfer.files);
    });

    input.addEventListener('change', async (e) => {
        await handleMultiFiles(e.target.files);
    });
}

async function handleMultiFiles(files) {
    for (const file of files) {
        const validation = validateImageFile(file);
        if (!validation.valid) {
            showError(validation.error);
            continue;
        }

        try {
            const base64 = await fileToBase64(file);
            AppState.multiCombineImages.push(base64);
        } catch (err) {
            console.error('Failed to load file:', err);
        }
    }

    renderMultiPreview();
    updateMultiCombineState();
}

function renderMultiPreview() {
    const container = document.getElementById('multiPreviewContainer');
    if (!container) return;

    container.innerHTML = AppState.multiCombineImages.map((img, index) => `
        <div class="relative">
            <img src="${escapeHtml(img)}" class="w-16 h-16 object-cover rounded">
            <button class="multi-image-remove absolute -top-1 -right-1 w-4 h-4 bg-red-600 rounded-full text-white text-xs" data-index="${index}">x</button>
        </div>
    `).join('');

    container.classList.toggle('hidden', AppState.multiCombineImages.length === 0);

    // Event delegation for remove buttons
    container.querySelectorAll('.multi-image-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(e.target.dataset.index, 10);
            removeMultiImage(index);
        });
    });
}

function removeMultiImage(index) {
    AppState.multiCombineImages.splice(index, 1);
    renderMultiPreview();
    updateMultiCombineState();
}

function updateMultiCombineState() {
    const combineBtn = document.getElementById('multiCombineBtn');
    if (combineBtn) {
        combineBtn.disabled = AppState.multiCombineImages.length < 2;
    }
}

async function executMultiCombine() {
    if (AppState.multiCombineImages.length < 2) {
        showError('At least 2 images required');
        return;
    }

    const combineBtn = document.getElementById('multiCombineBtn');
    const instructionEl = document.getElementById('multiCombineInstruction');

    setButtonLoading(combineBtn, 'Combining...');

    try {
        const data = {
            images: AppState.multiCombineImages,
            instruction: instructionEl ? instructionEl.value : 'Combine these images',
        };

        const result = await ApiClient.qwenImageEditMulti(data);

        if (result.combined_image) {
            // Show result
            openImageModal(result.combined_image);
        }

    } catch (err) {
        console.error('Failed to combine images:', err);
        showError('Combine failed');
    } finally {
        resetButton(combineBtn);
    }
}

// =============================================================================
// Event Binding
// =============================================================================

function initQwenImageEvents() {
    // Model type buttons
    const modelButtons = [
        { id: 'modelTypeZImage', type: 'zimage' },
        { id: 'modelTypeQwenImage', type: 'qwenimage' },
        { id: 'modelTypeQwenImage2512', type: 'qwenimage2512' },
        { id: 'modelTypeLTX2', type: 'ltx2' }
    ];
    modelButtons.forEach(({ id, type }) => {
        const btn = document.getElementById(id);
        if (btn) {
            btn.addEventListener('click', () => {
                switchModelType(type);
            });
        }
    });

    // Decomposition
    setupQiImageUpload();
    const decomposeBtn = document.getElementById('qiDecomposeBtn');
    if (decomposeBtn) {
        decomposeBtn.addEventListener('click', decomposeImage);
    }

    // Edit modal
    const closeEditBtn = document.getElementById('closeEditModal');
    if (closeEditBtn) {
        closeEditBtn.addEventListener('click', closeEditModal);
    }

    const executeEditBtn = document.getElementById('executeEditBtn');
    if (executeEditBtn) {
        executeEditBtn.addEventListener('click', executeLayerEdit);
    }

    // Single edit
    setupSingleEditUpload();

    // Multi-combine
    setupMultiCombineUpload();
    const multiCombineBtn = document.getElementById('multiCombineBtn');
    if (multiCombineBtn) {
        multiCombineBtn.addEventListener('click', executMultiCombine);
    }
}

// Export for use by other modules
window.checkQwenImageStatus = checkQwenImageStatus;
window.checkQwenImage2512Status = checkQwenImage2512Status;
window.updateQiSections = updateQiSections;
window.getSelectedModelType = getSelectedModelType;
window.switchModelType = switchModelType;
window.handleQiImageUpload = handleQiImageUpload;
window.decomposeImage = decomposeImage;
window.openEditModal = openEditModal;
window.closeEditModal = closeEditModal;
window.executeLayerEdit = executeLayerEdit;
window.handleMultiFiles = handleMultiFiles;
window.removeMultiImage = removeMultiImage;
window.initQwenImageEvents = initQwenImageEvents;
