/**
 * rewriter.js - Prompt rewriter functionality
 *
 * Handles prompt rewriting with transformers or API backends.
 */

// =============================================================================
// State
// =============================================================================

let rewriterModels = [];
let rewriterMode = 'transformers';

// =============================================================================
// Mode Toggle
// =============================================================================

function setupRewriterModeToggle() {
    const toggleBtns = document.querySelectorAll('[data-rewriter-mode]');
    toggleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.rewriterMode;
            setRewriterMode(mode);
        });
    });
}

function setRewriterMode(mode) {
    rewriterMode = mode;

    // Update toggle button styles
    const toggleBtns = document.querySelectorAll('[data-rewriter-mode]');
    toggleBtns.forEach(btn => {
        if (btn.dataset.rewriterMode === mode) {
            btn.classList.add('bg-blue-600');
            btn.classList.remove('bg-gray-700');
        } else {
            btn.classList.remove('bg-blue-600');
            btn.classList.add('bg-gray-700');
        }
    });

    // Show/hide relevant controls
    const transformersControls = document.getElementById('rewriterTransformersControls');
    const apiControls = document.getElementById('rewriterApiControls');

    if (transformersControls) {
        transformersControls.classList.toggle('hidden', mode !== 'transformers');
    }
    if (apiControls) {
        apiControls.classList.toggle('hidden', mode !== 'api');
    }

    // Load models for the selected mode
    if (mode === 'transformers') {
        loadRewriterModels();
    } else {
        loadRewriters();
    }
}

// =============================================================================
// Rewriter Loading
// =============================================================================

async function loadRewriters() {
    const rewriterSelect = document.getElementById('rewriterSelect');
    if (!rewriterSelect) return;

    try {
        const data = await ApiClient.getRewriters();
        const rewriters = data.rewriters || [];

        rewriterSelect.innerHTML = '';
        rewriters.forEach(rw => {
            const option = document.createElement('option');
            option.value = rw.id;
            option.textContent = rw.name;
            if (rw.description) option.title = rw.description;
            rewriterSelect.appendChild(option);
        });

    } catch (err) {
        console.error('Failed to load rewriters:', err);
    }
}

async function loadRewriterModels() {
    const modelSelect = document.getElementById('rewriterModelSelect');
    if (!modelSelect) return;

    try {
        const data = await ApiClient.getRewriterModels();
        rewriterModels = data.models || [];

        modelSelect.innerHTML = '';
        rewriterModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            if (model.supports_vision) {
                option.textContent += ' (Vision)';
            }
            modelSelect.appendChild(option);
        });

        onRewriterModelChange();

    } catch (err) {
        console.error('Failed to load rewriter models:', err);
    }
}

async function loadRewriterConfig() {
    try {
        const data = await ApiClient.getRewriterConfig();

        // Set default mode
        if (data.default_mode) {
            setRewriterMode(data.default_mode);
        }

        // Set default model
        const modelSelect = document.getElementById('rewriterModelSelect');
        if (data.default_model && modelSelect) {
            modelSelect.value = data.default_model;
        }

    } catch (err) {
        console.error('Failed to load rewriter config:', err);
    }
}

// =============================================================================
// Model Selection
// =============================================================================

function onRewriterModelChange() {
    const modelSelect = document.getElementById('rewriterModelSelect');
    const imageUploadArea = document.getElementById('rewriterImageUpload');

    if (!modelSelect || !imageUploadArea) return;

    const selectedModel = rewriterModels.find(m => m.id === modelSelect.value);
    const supportsVision = selectedModel?.supports_vision || false;

    imageUploadArea.classList.toggle('hidden', !supportsVision);
}

// =============================================================================
// Image Upload for Vision Models
// =============================================================================

function setupRewriterImageUpload() {
    const dropzone = document.getElementById('rewriterImageDropzone');
    const input = document.getElementById('rewriterImageInput');
    const preview = document.getElementById('rewriterImagePreview');

    if (!dropzone || !input) return;

    dropzone.addEventListener('click', () => input.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleRewriterImageFile(files[0]);
        }
    });

    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleRewriterImageFile(e.target.files[0]);
        }
    });
}

async function handleRewriterImageFile(file) {
    const validation = validateImageFile(file);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }

    try {
        const base64 = await fileToBase64(file);
        AppState.rewriterImage = base64;

        const preview = document.getElementById('rewriterImagePreview');
        const previewImg = document.getElementById('rewriterPreviewImg');

        if (preview && previewImg) {
            previewImg.src = base64;
            preview.classList.remove('hidden');
        }

    } catch (err) {
        console.error('Failed to load rewriter image:', err);
        showError('Failed to load image');
    }
}

function clearRewriterImagePreview() {
    AppState.rewriterImage = null;

    const preview = document.getElementById('rewriterImagePreview');
    const previewImg = document.getElementById('rewriterPreviewImg');

    if (preview) preview.classList.add('hidden');
    if (previewImg) previewImg.src = '';
}

// =============================================================================
// Rewrite Execution
// =============================================================================

async function rewritePrompt() {
    const promptEl = document.getElementById('prompt');
    const rewriteBtn = document.getElementById('rewriteBtn');

    if (!promptEl || !promptEl.value.trim()) {
        showError('Please enter a prompt to rewrite');
        return;
    }

    setButtonLoading(rewriteBtn, 'Rewriting...');

    try {
        const data = {
            prompt: promptEl.value,
            mode: rewriterMode,
        };

        if (rewriterMode === 'transformers') {
            const modelSelect = document.getElementById('rewriterModelSelect');
            if (modelSelect) data.model = modelSelect.value;

            // Add image if available
            if (AppState.rewriterImage) {
                data.image = AppState.rewriterImage;
            }
        } else {
            const rewriterSelect = document.getElementById('rewriterSelect');
            if (rewriterSelect) data.rewriter = rewriterSelect.value;
        }

        const result = await ApiClient.rewritePrompt(data);

        if (result.rewritten_prompt) {
            promptEl.value = result.rewritten_prompt;

            // Update token count
            if (typeof debouncedTokenCount === 'function') {
                debouncedTokenCount();
            }
        }

    } catch (err) {
        console.error('Failed to rewrite prompt:', err);
        showError('Failed to rewrite prompt');
    } finally {
        resetButton(rewriteBtn);
    }
}

// =============================================================================
// Event Binding
// =============================================================================

function initRewriterEvents() {
    setupRewriterModeToggle();
    setupRewriterImageUpload();

    const modelSelect = document.getElementById('rewriterModelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', onRewriterModelChange);
    }

    const rewriteBtn = document.getElementById('rewriteBtn');
    if (rewriteBtn) {
        rewriteBtn.addEventListener('click', rewritePrompt);
    }

    const clearBtn = document.getElementById('clearRewriterImage');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearRewriterImagePreview);
    }
}

// Export for use by other modules
window.setupRewriterModeToggle = setupRewriterModeToggle;
window.setRewriterMode = setRewriterMode;
window.loadRewriters = loadRewriters;
window.loadRewriterModels = loadRewriterModels;
window.loadRewriterConfig = loadRewriterConfig;
window.onRewriterModelChange = onRewriterModelChange;
window.handleRewriterImageFile = handleRewriterImageFile;
window.clearRewriterImagePreview = clearRewriterImagePreview;
window.rewritePrompt = rewritePrompt;
window.initRewriterEvents = initRewriterEvents;
