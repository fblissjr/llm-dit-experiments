/**
 * app.js - Application initialization and main form handling
 *
 * This is the main entry point that initializes all modules and
 * handles the generation form submission.
 */

// =============================================================================
// Model-Specific Defaults
// =============================================================================

function setModelDefaults(model) {
    const { stepsSlider, stepsValue, stepsMinLabel, stepsMaxLabel, guidanceScaleSlider, guidanceScaleValue, shiftContainer } = DOM;

    if (model === 'zimage') {
        // Z-Image defaults: turbo model with 9 steps, CFG baked in (0.0)
        if (stepsSlider) {
            stepsSlider.min = 1;
            stepsSlider.max = 50;
            stepsSlider.value = 9;
        }
        if (stepsMinLabel) stepsMinLabel.textContent = '1';
        if (stepsMaxLabel) stepsMaxLabel.textContent = '50';

        if (guidanceScaleSlider) guidanceScaleSlider.value = 0.0;

        // Show shift slider (Z-Image uses FlowMatch shift)
        if (shiftContainer) shiftContainer.classList.remove('hidden');
    } else if (model === 'qwenimage' || model === 'qwenimage2512') {
        // Qwen-Image defaults: non-turbo with 25-50 steps, CFG 4.0
        if (stepsSlider) {
            stepsSlider.min = 10;
            stepsSlider.max = 100;
            stepsSlider.value = 25;
        }
        if (stepsMinLabel) stepsMinLabel.textContent = '10';
        if (stepsMaxLabel) stepsMaxLabel.textContent = '100';

        if (guidanceScaleSlider) guidanceScaleSlider.value = 4.0;

        // Hide shift slider for Qwen-Image
        if (shiftContainer) shiftContainer.classList.add('hidden');
    }

    updateSliderDisplays();
}

// =============================================================================
// Token Count
// =============================================================================

let tokenCountTimeout = null;

async function updateTokenCount() {
    const tokenCountEl = document.getElementById('tokenCount');
    const promptEl = document.getElementById('prompt');

    if (!tokenCountEl || !promptEl) return;

    const prompt = promptEl.value;
    if (!prompt.trim()) {
        tokenCountEl.textContent = '-- / 1504 tokens';
        tokenCountEl.className = 'text-xs text-gray-400';
        return;
    }

    try {
        const params = {
            prompt: prompt,
            system_prompt: document.getElementById('systemPrompt')?.value || null,
            thinking_content: document.getElementById('thinkingContent')?.value || null,
            assistant_content: document.getElementById('assistantContent')?.value || null,
            force_think_block: document.getElementById('enableThinking')?.checked || false,
            template: document.getElementById('template')?.value || null,
        };

        const data = await ApiClient.formatPrompt(params);
        const count = data.token_count;
        const max = data.max_tokens || 1504;

        if (count !== null) {
            tokenCountEl.textContent = `${count} / ${max} tokens`;
            if (count > max) {
                tokenCountEl.className = 'text-xs text-red-400 font-medium';
            } else if (count > max * 0.9) {
                tokenCountEl.className = 'text-xs text-yellow-400';
            } else {
                tokenCountEl.className = 'text-xs text-gray-400';
            }
        }

    } catch (err) {
        console.error('Failed to update token count:', err);
    }
}

const debouncedTokenCount = debounce(updateTokenCount, 500);

// =============================================================================
// Form Submission
// =============================================================================

async function handleFormSubmit(e) {
    e.preventDefault();

    const { generateBtn, status, statusText, progressFill, error, result, resultImage, resultInfo, downloadBtn } = DOM;

    // Get model type
    const modelType = getSelectedModelType();

    // Validate
    const promptEl = document.getElementById('prompt');
    if (!promptEl?.value.trim()) {
        showError('Please enter a prompt');
        return;
    }

    // Check img2img
    const img2imgParams = getImg2ImgParams();
    if (document.getElementById('img2imgEnabled')?.checked && !img2imgParams) {
        showError('Img2Img enabled but no image uploaded');
        return;
    }

    // Hide previous results
    if (error) error.classList.add('hidden');
    if (result) result.classList.add('hidden');

    // Show status
    showStatus('Preparing generation...', 10);
    setButtonLoading(generateBtn, 'Generating...');

    try {
        // Build request params
        const resolution = getResolution();
        const params = {
            prompt: promptEl.value,
            width: resolution.width,
            height: resolution.height,
            steps: parseInt(DOM.stepsSlider?.value || 9),
            guidance_scale: parseFloat(DOM.guidanceScaleSlider?.value || 0),
            seed: document.getElementById('seed')?.value ? parseInt(document.getElementById('seed').value) : null,
            template: document.getElementById('template')?.value || null,
            system_prompt: document.getElementById('systemPrompt')?.value || null,
            thinking_content: document.getElementById('thinkingContent')?.value || null,
            assistant_content: document.getElementById('assistantContent')?.value || null,
            force_think_block: document.getElementById('enableThinking')?.checked || false,
            strip_quotes: document.getElementById('stripQuotes')?.checked || false,
        };

        // Add Z-Image specific params
        if (modelType === 'zimage') {
            params.shift = parseFloat(DOM.shiftSlider?.value || 3.0);
            params.long_prompt_mode = DOM.longPromptModeSelect?.value || 'interpolate';
            params.hidden_layer = parseInt(DOM.hiddenLayerSlider?.value || -2);

            // CFG enhancement
            const cfgConfig = getCfgEnhancementConfig();
            if (cfgConfig) {
                Object.assign(params, cfgConfig);
            }

            // Layer weights
            const layerWeights = getLayerWeights();
            if (layerWeights) {
                params.layer_weights = layerWeights;
            }

            // DyPE
            const dypeConfig = getDypeConfig();
            if (dypeConfig) {
                params.dype = dypeConfig;
            }

            // SLG
            const slgConfig = getSlgConfig();
            if (slgConfig) {
                params.slg = slgConfig;
            }

            // FMTT
            const fmttConfig = getFmttConfig();
            if (fmttConfig) {
                params.fmtt = fmttConfig;
            }
        }

        // Determine endpoint and add mode-specific params
        let endpoint;
        let statusMsg = 'Generating...';

        // VL conditioning
        const vlParams = getVLParams();
        if (vlParams && modelType === 'zimage') {
            Object.assign(params, { vl: vlParams });
            endpoint = 'vl';
            statusMsg = 'Generating with VL conditioning...';
        }
        // Img2Img
        else if (img2imgParams && modelType === 'zimage') {
            Object.assign(params, img2imgParams);
            endpoint = 'img2img';
            statusMsg = 'Generating with Img2Img...';
        }
        // Qwen-Image T2I
        else if (modelType === 'qwenimage2512') {
            endpoint = 'qwen2512';
            statusMsg = 'Generating with Qwen-Image T2I...';
        }
        // Standard Z-Image
        else {
            endpoint = 'generate';
        }

        showStatus(statusMsg, 30);

        // Make API call
        let data;
        switch (endpoint) {
            case 'vl':
                data = await ApiClient.generateWithVL(params);
                break;
            case 'img2img':
                data = await ApiClient.img2img(params);
                break;
            case 'qwen2512':
                data = await ApiClient.qwenImage2512Generate(params);
                break;
            default:
                data = await ApiClient.generate(params);
        }

        showStatus('Processing result...', 90);

        // Display result
        if (data.image) {
            const imageSrc = data.image.startsWith('data:') ? data.image : `data:image/png;base64,${data.image}`;

            if (resultImage) {
                resultImage.src = imageSrc;
                resultImage.onclick = () => openImageModal(imageSrc);
            }

            if (resultInfo) {
                const infoText = `${resolution.width}x${resolution.height} | ${params.steps} steps | seed: ${data.seed || 'random'}`;
                resultInfo.textContent = infoText;
            }

            if (downloadBtn) {
                downloadBtn.onclick = () => {
                    const link = document.createElement('a');
                    link.href = imageSrc;
                    link.download = `generated_${Date.now()}.png`;
                    link.click();
                };
            }

            if (result) result.classList.remove('hidden');

            // Store current params for reuse
            AppState.currentParams = { ...params, seed: data.seed };

            // Store current result for workflow actions (image-utils.js)
            if (typeof setCurrentResult === 'function') {
                setCurrentResult(imageSrc, resolution.width, resolution.height);
            }
        }

        hideStatus();

        // Reload history
        await loadHistory();

    } catch (err) {
        console.error('Generation failed:', err);
        showError(err.message || 'Generation failed');
        hideStatus();
    } finally {
        resetButton(generateBtn);
    }
}

// =============================================================================
// Settings Modal
// =============================================================================

async function refreshSystemStatus() {
    try {
        const data = await ApiClient.getSystemStatus();

        const vramUsed = document.getElementById('vramUsed');
        const vramTotal = document.getElementById('vramTotal');
        const loadedModels = document.getElementById('loadedModels');

        if (vramUsed && data.vram_used !== undefined) {
            vramUsed.textContent = formatBytes(data.vram_used * 1024 * 1024 * 1024);
        }
        if (vramTotal && data.vram_total !== undefined) {
            vramTotal.textContent = formatBytes(data.vram_total * 1024 * 1024 * 1024);
        }
        if (loadedModels && data.loaded_models) {
            loadedModels.textContent = data.loaded_models.join(', ') || 'None';
        }

    } catch (err) {
        console.error('Failed to refresh system status:', err);
    }
}

function setupSettingsModal() {
    const settingsToggle = document.getElementById('settingsToggle');
    const settingsModal = document.getElementById('settingsModal');
    const closeSettings = document.getElementById('closeSettings');

    if (settingsToggle && settingsModal) {
        settingsToggle.addEventListener('click', async () => {
            settingsModal.classList.remove('hidden');
            await refreshSystemStatus();
        });
    }

    if (closeSettings && settingsModal) {
        closeSettings.addEventListener('click', () => {
            settingsModal.classList.add('hidden');
        });
    }

    // Click outside to close
    if (settingsModal) {
        settingsModal.addEventListener('click', (e) => {
            if (e.target === settingsModal) {
                settingsModal.classList.add('hidden');
            }
        });
    }
}

// =============================================================================
// Keyboard Shortcuts
// =============================================================================

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Escape to close modals
        if (e.key === 'Escape') {
            closeImageModal();
            closeEditModal();
            document.getElementById('settingsModal')?.classList.add('hidden');
        }

        // Ctrl+Enter to generate
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            DOM.form?.requestSubmit();
        }
    });
}

// =============================================================================
// Result Action Buttons
// =============================================================================

function setupResultActionButtons() {
    // Use as Img2Img
    const useImg2Img = document.getElementById('resultUseImg2Img');
    if (useImg2Img) {
        useImg2Img.addEventListener('click', async () => {
            const result = getCurrentResult();
            if (result) {
                await useAsImg2Img(result.base64, result.width, result.height);
            }
        });
    }

    // Use as VL Reference
    const useVL = document.getElementById('resultUseVL');
    if (useVL) {
        useVL.addEventListener('click', async () => {
            const result = getCurrentResult();
            if (result) {
                await useAsVLReference(result.base64);
            }
        });
    }

    // Use in Qwen Edit
    const useQwenEdit = document.getElementById('resultUseQwenEdit');
    if (useQwenEdit) {
        useQwenEdit.addEventListener('click', async () => {
            const result = getCurrentResult();
            if (result) {
                await useInQwenEdit(result.base64);
            }
        });
    }

    // Add to Combine
    const addCombine = document.getElementById('resultAddCombine');
    if (addCombine) {
        addCombine.addEventListener('click', async () => {
            const result = getCurrentResult();
            if (result) {
                await addToCombine(result.base64);
            }
        });
    }
}

// =============================================================================
// Lightbox Action Buttons
// =============================================================================

function setupLightboxActionButtons() {
    // Store current lightbox image data
    let lightboxImageData = null;

    // Override openImageModal to track current image
    const originalOpenImageModal = window.openImageModal;
    window.openImageModal = function(src, width, height) {
        lightboxImageData = { base64: src, width: width, height: height };
        if (originalOpenImageModal) {
            originalOpenImageModal(src);
        } else {
            // Fallback implementation
            const modal = document.getElementById('imageModal');
            const modalImage = document.getElementById('modalImage');
            if (modal && modalImage) {
                modalImage.src = src;
                modal.classList.remove('hidden');
            }
        }
    };

    // Save button
    const saveBtn = document.getElementById('lightboxSave');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            if (lightboxImageData) {
                const link = document.createElement('a');
                link.href = lightboxImageData.base64;
                link.download = `image_${Date.now()}.png`;
                link.click();
            }
        });
    }

    // Use as Img2Img
    const useImg2Img = document.getElementById('lightboxUseImg2Img');
    if (useImg2Img) {
        useImg2Img.addEventListener('click', async () => {
            if (lightboxImageData) {
                closeImageModal();
                await useAsImg2Img(lightboxImageData.base64, lightboxImageData.width, lightboxImageData.height);
            }
        });
    }

    // Use as VL Reference
    const useVL = document.getElementById('lightboxUseVL');
    if (useVL) {
        useVL.addEventListener('click', async () => {
            if (lightboxImageData) {
                closeImageModal();
                await useAsVLReference(lightboxImageData.base64);
            }
        });
    }

    // Use in Qwen Edit
    const useQwenEdit = document.getElementById('lightboxUseQwenEdit');
    if (useQwenEdit) {
        useQwenEdit.addEventListener('click', async () => {
            if (lightboxImageData) {
                closeImageModal();
                await useInQwenEdit(lightboxImageData.base64);
            }
        });
    }

    // Add to Combine
    const addCombine = document.getElementById('lightboxAddCombine');
    if (addCombine) {
        addCombine.addEventListener('click', async () => {
            if (lightboxImageData) {
                closeImageModal();
                await addToCombine(lightboxImageData.base64);
            }
        });
    }
}

// =============================================================================
// Initialization
// =============================================================================

async function initializeApp() {
    // Initialize DOM references
    initDOMReferences();

    // Setup form submission
    if (DOM.form) {
        DOM.form.addEventListener('submit', handleFormSubmit);
    }

    // Initialize all feature modules
    initHistoryEvents();
    initTemplateEvents();
    initRewriterEvents();
    initVLEvents();
    initImg2ImgEvents();
    initLayerBlendEvents();
    initAdvancedEvents();
    initQwenImageEvents();

    // Initialize image utilities (workflow continuity)
    if (typeof initImageUtils === 'function') {
        initImageUtils();
    }

    // Setup result action buttons
    setupResultActionButtons();

    // Setup lightbox action buttons
    setupLightboxActionButtons();

    // Setup settings modal
    setupSettingsModal();

    // Setup keyboard shortcuts
    setupKeyboardShortcuts();

    // Setup slider value displays
    if (DOM.stepsSlider) {
        DOM.stepsSlider.addEventListener('input', updateSliderDisplays);
    }
    if (DOM.guidanceScaleSlider) {
        DOM.guidanceScaleSlider.addEventListener('input', updateSliderDisplays);
    }
    if (DOM.shiftSlider) {
        DOM.shiftSlider.addEventListener('input', updateSliderDisplays);
    }

    // Setup token count updates
    const promptEl = document.getElementById('prompt');
    if (promptEl) {
        promptEl.addEventListener('input', debouncedTokenCount);
    }

    // Load initial data
    await Promise.all([
        loadTemplates(),
        loadGenerationConfig(),
        loadResolutionPresets(),
        loadHistory(),
        checkVLStatus(),
        checkQwenImageStatus(),
        checkQwenImage2512Status(),
        loadRewriterConfig(),
    ]);

    // Set initial model defaults
    setModelDefaults('zimage');

    // Update UI states
    updateSliderDisplays();
    updateThinkingContentVisibility();

    console.log('LLM DiT Experiments UI initialized');
}

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);

// Export for use by other modules
window.setModelDefaults = setModelDefaults;
window.updateTokenCount = updateTokenCount;
window.debouncedTokenCount = debouncedTokenCount;
window.handleFormSubmit = handleFormSubmit;
window.refreshSystemStatus = refreshSystemStatus;
window.initializeApp = initializeApp;
