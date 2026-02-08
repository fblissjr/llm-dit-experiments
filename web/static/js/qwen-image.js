/**
 * qwen-image.js - Qwen-Image model specific functionality
 *
 * Handles Qwen-Image editing and multi-image combining.
 */

// =============================================================================
// Model Status Checks
// =============================================================================

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
    const flux2Section = document.getElementById('flux2Section');
    const sharedParams = document.getElementById('sharedParams');

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
    if (flux2Section) {
        flux2Section.classList.toggle('hidden', modelType !== 'flux2');
    }

    // Hide shared Z-Image params for models that have their own controls
    // (FLUX.2 and LTX-2 have dedicated steps/guidance in their sections)
    if (sharedParams) {
        const hideShared = modelType === 'flux2' || modelType === 'ltx2';
        sharedParams.classList.toggle('hidden', hideShared);
    }
}

function getSelectedModelType() {
    const zimageBtn = document.getElementById('modelTypeZImage');
    const qwenImageBtn = document.getElementById('modelTypeQwenImage');
    const qwenImage2512Btn = document.getElementById('modelTypeQwenImage2512');
    const ltx2Btn = document.getElementById('modelTypeLTX2');
    const flux2Btn = document.getElementById('modelTypeFLUX2');

    if (zimageBtn && zimageBtn.classList.contains('bg-blue-600')) return 'zimage';
    if (qwenImageBtn && qwenImageBtn.classList.contains('bg-blue-600')) return 'qwenimage';
    if (qwenImage2512Btn && qwenImage2512Btn.classList.contains('bg-blue-600')) return 'qwenimage2512';
    if (ltx2Btn && ltx2Btn.classList.contains('bg-blue-600')) return 'ltx2';
    if (flux2Btn && flux2Btn.classList.contains('bg-blue-600')) return 'flux2';

    return 'zimage'; // default
}

function switchModelType(type) {
    const buttons = {
        'zimage': document.getElementById('modelTypeZImage'),
        'qwenimage': document.getElementById('modelTypeQwenImage'),
        'qwenimage2512': document.getElementById('modelTypeQwenImage2512'),
        'ltx2': document.getElementById('modelTypeLTX2'),
        'flux2': document.getElementById('modelTypeFLUX2'),
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

    // FLUX.2 has its own controls - update defaults when switching to it
    if (type === 'flux2' && typeof updateFlux2Defaults === 'function') {
        const modelSelect = document.getElementById('flux2Model');
        if (modelSelect) {
            updateFlux2Defaults(modelSelect.value);
        }
    }

    // Reload resolution presets for new model
    if (typeof ResolutionSelector !== 'undefined' && ResolutionSelector.loadConstraints) {
        // Map UI model type to API model type
        const modelMap = {
            'zimage': 'zimage',
            'qwenimage': 'qwenimage-edit',
            'qwenimage2512': 'qwenimage-t2i',
        };
        ResolutionSelector.loadConstraints(modelMap[type] || 'zimage');
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
        { id: 'modelTypeLTX2', type: 'ltx2' },
        { id: 'modelTypeFLUX2', type: 'flux2' }
    ];
    modelButtons.forEach(({ id, type }) => {
        const btn = document.getElementById(id);
        if (btn) {
            btn.addEventListener('click', () => {
                switchModelType(type);
            });
        }
    });

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
window.checkQwenImage2512Status = checkQwenImage2512Status;
window.updateQiSections = updateQiSections;
window.getSelectedModelType = getSelectedModelType;
window.switchModelType = switchModelType;
window.handleMultiFiles = handleMultiFiles;
window.removeMultiImage = removeMultiImage;
window.initQwenImageEvents = initQwenImageEvents;
