/**
 * templates.js - Template loading and selection
 *
 * Handles loading templates from the API and managing template selection.
 */

// =============================================================================
// Template Loading
// =============================================================================

async function loadTemplates() {
    const templateSelect = document.getElementById('template');
    if (!templateSelect) return;

    try {
        const data = await ApiClient.getTemplates();
        AppState.templatesCache = data.templates || [];

        // Clear existing options except first
        templateSelect.innerHTML = '<option value="">No template (raw prompt)</option>';

        // Group templates by category
        const byCategory = {};
        AppState.templatesCache.forEach(tpl => {
            const cat = tpl.category || 'general';
            if (!byCategory[cat]) byCategory[cat] = [];
            byCategory[cat].push(tpl);
        });

        // Add optgroups
        Object.keys(byCategory).sort().forEach(category => {
            const group = document.createElement('optgroup');
            group.label = category.charAt(0).toUpperCase() + category.slice(1);

            byCategory[category].forEach(tpl => {
                const option = document.createElement('option');
                option.value = tpl.name;
                option.textContent = tpl.name;
                if (tpl.description) {
                    option.title = tpl.description;
                }
                group.appendChild(option);
            });

            templateSelect.appendChild(group);
        });

    } catch (err) {
        console.error('Failed to load templates:', err);
    }
}

// =============================================================================
// Template Selection
// =============================================================================

function onTemplateChange(e) {
    const templateName = e.target.value;
    if (!templateName || !AppState.templatesCache) return;

    const template = AppState.templatesCache.find(t => t.name === templateName);
    if (!template) return;

    // Apply template settings
    const systemPromptEl = document.getElementById('systemPrompt');
    const thinkingContentEl = document.getElementById('thinkingContent');
    const assistantContentEl = document.getElementById('assistantContent');
    const enableThinkingEl = document.getElementById('enableThinking');

    if (template.system_prompt && systemPromptEl) {
        systemPromptEl.value = template.system_prompt;
    }

    if (template.thinking_content && thinkingContentEl) {
        thinkingContentEl.value = template.thinking_content;
    }

    if (template.assistant_content && assistantContentEl) {
        assistantContentEl.value = template.assistant_content;
    }

    if (template.add_think_block !== undefined && enableThinkingEl) {
        enableThinkingEl.checked = template.add_think_block;
        if (typeof updateThinkingContentVisibility === 'function') {
            updateThinkingContentVisibility();
        }
    }
}

// =============================================================================
// Generation Config Loading
// =============================================================================

async function loadGenerationConfig() {
    try {
        const data = await ApiClient.getGenerationConfig();

        // Apply defaults from config
        const { stepsSlider, stepsValue, guidanceScaleSlider, guidanceScaleValue, shiftSlider, shiftValue, longPromptModeSelect, hiddenLayerSlider } = DOM;

        if (data.steps && stepsSlider && stepsValue) {
            stepsSlider.value = data.steps;
            stepsValue.textContent = data.steps;
        }

        if (data.guidance_scale !== undefined && guidanceScaleSlider && guidanceScaleValue) {
            guidanceScaleSlider.value = data.guidance_scale;
            guidanceScaleValue.textContent = formatNumber(data.guidance_scale, 1);
        }

        if (data.shift !== undefined && shiftSlider && shiftValue) {
            shiftSlider.value = data.shift;
            shiftValue.textContent = formatNumber(data.shift, 1);
        }

        if (data.long_prompt_mode && longPromptModeSelect) {
            longPromptModeSelect.value = data.long_prompt_mode;
        }

        if (data.hidden_layer !== undefined && hiddenLayerSlider) {
            hiddenLayerSlider.value = data.hidden_layer;
            updateHiddenLayerUI(data.hidden_layer);
        }

        // Store feature flags
        if (data.features) {
            AppState.features = { ...AppState.features, ...data.features };
        }

    } catch (err) {
        console.error('Failed to load generation config:', err);
    }
}

// =============================================================================
// Resolution Presets
// =============================================================================

async function loadResolutionPresets() {
    const resolutionSelect = document.getElementById('resolution');
    if (!resolutionSelect) return;

    try {
        const data = await ApiClient.getResolutionConfig();

        // Clear and rebuild options
        resolutionSelect.innerHTML = '';

        // Add presets by category
        if (data.presets) {
            Object.keys(data.presets).forEach(category => {
                const group = document.createElement('optgroup');
                group.label = category;

                data.presets[category].forEach(preset => {
                    const option = document.createElement('option');
                    option.value = `${preset.width}x${preset.height}`;
                    option.textContent = preset.label || `${preset.width}x${preset.height}`;
                    if (preset.default) option.selected = true;
                    group.appendChild(option);
                });

                resolutionSelect.appendChild(group);
            });
        }

        // Add custom option
        const customOption = document.createElement('option');
        customOption.value = 'custom';
        customOption.textContent = 'Custom...';
        resolutionSelect.appendChild(customOption);

        // Update DyPE recommendation if function exists
        if (typeof updateDypeRecommendation === 'function') {
            updateDypeRecommendation();
        }

    } catch (err) {
        console.error('Failed to load resolution presets:', err);
    }
}

// =============================================================================
// Event Binding
// =============================================================================

function initTemplateEvents() {
    const templateSelect = document.getElementById('template');
    if (templateSelect) {
        templateSelect.addEventListener('change', onTemplateChange);
    }
}

// Export for use by other modules
window.loadTemplates = loadTemplates;
window.onTemplateChange = onTemplateChange;
window.loadGenerationConfig = loadGenerationConfig;
window.loadResolutionPresets = loadResolutionPresets;
window.initTemplateEvents = initTemplateEvents;
