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
    const templateBtn = document.getElementById('rewriterModeTemplate');
    const customBtn = document.getElementById('rewriterModeCustom');

    if (templateBtn) {
        templateBtn.addEventListener('click', () => setRewriterMode('template'));
    }
    if (customBtn) {
        customBtn.addEventListener('click', () => setRewriterMode('custom'));
    }
}

function setRewriterMode(mode) {
    rewriterMode = mode;

    // Update toggle button styles
    const templateBtn = document.getElementById('rewriterModeTemplate');
    const customBtn = document.getElementById('rewriterModeCustom');

    if (templateBtn) {
        if (mode === 'template') {
            templateBtn.classList.add('bg-blue-600');
            templateBtn.classList.remove('bg-gray-700');
        } else {
            templateBtn.classList.remove('bg-blue-600');
            templateBtn.classList.add('bg-gray-700');
        }
    }
    if (customBtn) {
        if (mode === 'custom') {
            customBtn.classList.add('bg-blue-600');
            customBtn.classList.remove('bg-gray-700');
        } else {
            customBtn.classList.remove('bg-blue-600');
            customBtn.classList.add('bg-gray-700');
        }
    }

    // Show/hide relevant controls
    const templateMode = document.getElementById('rewriterTemplateMode');
    const customMode = document.getElementById('rewriterCustomMode');

    if (templateMode) {
        templateMode.classList.toggle('hidden', mode !== 'template');
    }
    if (customMode) {
        customMode.classList.toggle('hidden', mode !== 'custom');
    }

    // Load rewriter templates and models
    loadRewriters();
    loadRewriterModels();
}

// =============================================================================
// Rewriter Loading
// =============================================================================

async function loadRewriters() {
    const rewriterSelect = document.getElementById('rewriterTemplate');
    if (!rewriterSelect) return;

    try {
        const data = await ApiClient.getRewriters();
        const rewriters = data.rewriters || [];

        rewriterSelect.innerHTML = '<option value="">Select a rewriter...</option>';
        rewriters.forEach(rw => {
            const option = document.createElement('option');
            option.value = rw.name;
            option.textContent = rw.name;
            if (rw.description) option.title = rw.description;
            rewriterSelect.appendChild(option);
        });

    } catch (err) {
        console.error('Failed to load rewriters:', err);
    }
}

async function loadRewriterModels() {
    const modelSelect = document.getElementById('rewriterModel');
    if (!modelSelect) return;

    try {
        const data = await ApiClient.getRewriterModels();
        rewriterModels = data.models || [];

        modelSelect.innerHTML = '';
        rewriterModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            if (model.supports_image) {
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

        // Set default mode (use 'template' as default)
        setRewriterMode('template');

        // Set default model
        const modelSelect = document.getElementById('rewriterModel');
        if (data.default_model && modelSelect) {
            modelSelect.value = data.default_model;
        }

        // Set parameter defaults from config
        const maxTokensEl = document.getElementById('rewriterMaxTokens');
        const temperatureEl = document.getElementById('rewriterTemperature');
        const topPEl = document.getElementById('rewriterTopP');
        const topKEl = document.getElementById('rewriterTopK');
        const minPEl = document.getElementById('rewriterMinP');
        const presencePenaltyEl = document.getElementById('rewriterPresencePenalty');

        if (maxTokensEl && data.max_tokens) maxTokensEl.value = data.max_tokens;
        if (temperatureEl && data.temperature !== undefined) temperatureEl.value = data.temperature;
        if (topPEl && data.top_p !== undefined) topPEl.value = data.top_p;
        if (topKEl && data.top_k !== undefined) topKEl.value = data.top_k;
        if (minPEl && data.min_p !== undefined) minPEl.value = data.min_p;
        if (presencePenaltyEl && data.presence_penalty !== undefined) presencePenaltyEl.value = data.presence_penalty;

    } catch (err) {
        console.error('Failed to load rewriter config:', err);
    }
}

// =============================================================================
// Model Selection
// =============================================================================

function onRewriterModelChange() {
    const modelSelect = document.getElementById('rewriterModel');
    const imageUploadArea = document.getElementById('rewriterImageUpload');

    if (!modelSelect || !imageUploadArea) return;

    const selectedModel = rewriterModels.find(m => m.id === modelSelect.value);
    const supportsImage = selectedModel?.supports_image || false;

    imageUploadArea.classList.toggle('hidden', !supportsImage);
}

// =============================================================================
// Image Upload for Vision Models
// =============================================================================

function setupRewriterImageUpload() {
    const dropzone = document.getElementById('rewriterDropzone');
    const input = document.getElementById('rewriterImageInput');
    const preview = document.getElementById('rewriterImagePreview');

    if (!dropzone || !input) return;

    dropzone.addEventListener('click', (e) => {
        // Prevent triggering if clicking on clear button or preview
        if (e.target.id === 'rewriterClearImage' || e.target.closest('#rewriterImagePreview')) {
            return;
        }
        input.click();
    });

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
        };

        // Get model selection
        const modelSelect = document.getElementById('rewriterModel');
        if (modelSelect) {
            data.model = modelSelect.value;
        }

        // Add image if available (for VL models)
        if (AppState.rewriterImage) {
            data.image = AppState.rewriterImage;
        }

        // Get generation parameters
        const maxTokensEl = document.getElementById('rewriterMaxTokens');
        const temperatureEl = document.getElementById('rewriterTemperature');
        const topPEl = document.getElementById('rewriterTopP');
        const topKEl = document.getElementById('rewriterTopK');
        const minPEl = document.getElementById('rewriterMinP');
        const presencePenaltyEl = document.getElementById('rewriterPresencePenalty');

        if (maxTokensEl) data.max_tokens = parseInt(maxTokensEl.value, 10);
        if (temperatureEl) data.temperature = parseFloat(temperatureEl.value);
        if (topPEl) data.top_p = parseFloat(topPEl.value);
        if (topKEl) data.top_k = parseInt(topKEl.value, 10);
        if (minPEl) data.min_p = parseFloat(minPEl.value);
        if (presencePenaltyEl) data.presence_penalty = parseFloat(presencePenaltyEl.value);

        // Handle template vs custom mode
        if (rewriterMode === 'template') {
            const rewriterSelect = document.getElementById('rewriterTemplate');
            if (rewriterSelect && rewriterSelect.value) {
                data.rewriter = rewriterSelect.value;
            } else {
                showError('Please select a rewriter template');
                resetButton(rewriteBtn);
                return;
            }
        } else {
            // Custom mode - use custom system prompt
            const customPromptEl = document.getElementById('customRewriterPrompt');
            if (customPromptEl && customPromptEl.value.trim()) {
                data.custom_system_prompt = customPromptEl.value.trim();
            } else {
                showError('Please enter custom rewriting instructions');
                resetButton(rewriteBtn);
                return;
            }
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
        showError('Failed to rewrite prompt: ' + (err.message || 'Unknown error'));
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

    const modelSelect = document.getElementById('rewriterModel');
    if (modelSelect) {
        modelSelect.addEventListener('change', onRewriterModelChange);
    }

    const rewriteBtn = document.getElementById('rewriteBtn');
    if (rewriteBtn) {
        rewriteBtn.addEventListener('click', rewritePrompt);
    }

    const clearBtn = document.getElementById('rewriterClearImage');
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
