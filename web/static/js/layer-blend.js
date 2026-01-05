/**
 * layer-blend.js - Layer blending controls
 *
 * Handles hidden layer selection and multi-layer blending configuration.
 */

// =============================================================================
// Layer Dropdown Population
// =============================================================================

function populateLayerDropdowns() {
    const layerLabels = {
        '-1': '-1 (Last)',
        '-2': '-2 (Default)',
        '-6': '-6 (VL optimal)',
        '-18': '-18 (Middle)',
        '-19': '-19 (Middle)',
        '-35': '-35 (First)'
    };

    const defaultValues = {
        'blendLayer1': '-2',
        'blendLayer2': '-6',
        'blendLayer3a': '-1',
        'blendLayer3b': '-2',
        'blendLayer3c': '-3'
    };

    document.querySelectorAll('.blend-layer-select').forEach(select => {
        select.innerHTML = '';
        for (let i = -1; i >= -35; i--) {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = layerLabels[String(i)] || String(i);
            if (String(i) === defaultValues[select.id]) {
                option.selected = true;
            }
            select.appendChild(option);
        }
    });
}

// =============================================================================
// Layer Blending Mode
// =============================================================================

function updateLayerBlendMode() {
    const { layerBlendMode, twoLayerControls, threeLayerControls } = DOM;

    if (!layerBlendMode) return;

    const mode = layerBlendMode.value;

    if (twoLayerControls) {
        twoLayerControls.classList.toggle('hidden', mode !== '2-layer');
    }

    if (threeLayerControls) {
        threeLayerControls.classList.toggle('hidden', mode !== '3-layer');
    }
}

function toggleLayerBlendControls() {
    const { layerBlendEnabled, layerBlendControls } = DOM;

    if (!layerBlendEnabled || !layerBlendControls) return;

    layerBlendControls.classList.toggle('hidden', !layerBlendEnabled.checked);
}

// =============================================================================
// Layer Weights
// =============================================================================

function getLayerWeights() {
    const { layerBlendEnabled, layerBlendMode, blendLayer1, blendLayer2, blendWeight1, blendLayer3a, blendLayer3b, blendLayer3c } = DOM;

    if (!layerBlendEnabled || !layerBlendEnabled.checked) {
        return null;
    }

    const mode = layerBlendMode ? layerBlendMode.value : '2-layer';

    if (mode === '2-layer') {
        const weight = blendWeight1 ? parseFloat(blendWeight1.value) : 0.5;
        return {
            mode: '2-layer',
            layers: [
                parseInt(blendLayer1?.value || -2),
                parseInt(blendLayer2?.value || -6),
            ],
            weights: [weight, 1 - weight],
        };
    } else {
        // 3-layer mode with equal weights
        return {
            mode: '3-layer',
            layers: [
                parseInt(blendLayer3a?.value || -1),
                parseInt(blendLayer3b?.value || -2),
                parseInt(blendLayer3c?.value || -3),
            ],
            weights: [1/3, 1/3, 1/3],
        };
    }
}

// =============================================================================
// Weight Slider Display
// =============================================================================

function updateBlendWeightDisplay() {
    const { blendWeight1, blendWeight1Value, blendWeight2Value } = DOM;

    if (!blendWeight1 || !blendWeight1Value || !blendWeight2Value) return;

    const weight = parseFloat(blendWeight1.value);
    blendWeight1Value.textContent = formatNumber(weight, 2);
    blendWeight2Value.textContent = formatNumber(1 - weight, 2);
}

// =============================================================================
// Layer Presets
// =============================================================================

function setupLayerPresets() {
    const { layerPresetButtons, hiddenLayerSlider } = DOM;

    if (!layerPresetButtons) return;

    layerPresetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const layer = parseInt(btn.dataset.layer);
            if (hiddenLayerSlider) {
                hiddenLayerSlider.value = layer;
                updateHiddenLayerUI(layer);
            }
            updateLayerPresetStyles(layer);
        });
    });
}

function updateLayerPresetStyles(selectedLayer) {
    const { layerPresetButtons } = DOM;

    if (!layerPresetButtons) return;

    layerPresetButtons.forEach(btn => {
        const layer = parseInt(btn.dataset.layer);
        if (layer === selectedLayer) {
            btn.classList.add('bg-purple-600', 'border-purple-500');
            btn.classList.remove('bg-gray-700', 'border-gray-600');
        } else {
            btn.classList.remove('bg-purple-600', 'border-purple-500');
            btn.classList.add('bg-gray-700', 'border-gray-600');
        }
    });
}

// =============================================================================
// Blend Presets
// =============================================================================

function setupBlendPresets() {
    const { blendPresetButtons, blendLayer1, blendLayer2, blendWeight1, layerBlendMode } = DOM;

    if (!blendPresetButtons) return;

    blendPresetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const preset = btn.dataset.preset;
            applyBlendPreset(preset);
            updateBlendPresetStyles(preset);
        });
    });
}

function applyBlendPreset(preset) {
    const { blendLayer1, blendLayer2, blendWeight1, layerBlendMode, layerBlendEnabled, layerBlendControls } = DOM;

    // Presets define layer1, layer2, and weight
    const presets = {
        'default': { l1: -2, l2: -6, w: 0.7 },
        'semantic': { l1: -6, l2: -18, w: 0.5 },
        'style': { l1: -1, l2: -2, w: 0.3 },
        'balanced': { l1: -2, l2: -19, w: 0.5 },
    };

    const p = presets[preset];
    if (!p) return;

    // Enable blending if not already
    if (layerBlendEnabled && !layerBlendEnabled.checked) {
        layerBlendEnabled.checked = true;
        if (layerBlendControls) {
            layerBlendControls.classList.remove('hidden');
        }
    }

    // Set to 2-layer mode
    if (layerBlendMode) {
        layerBlendMode.value = '2-layer';
        updateLayerBlendMode();
    }

    // Apply values
    if (blendLayer1) blendLayer1.value = p.l1;
    if (blendLayer2) blendLayer2.value = p.l2;
    if (blendWeight1) blendWeight1.value = p.w;

    updateBlendWeightDisplay();
}

function updateBlendPresetStyles(selectedPreset) {
    const { blendPresetButtons } = DOM;

    if (!blendPresetButtons) return;

    blendPresetButtons.forEach(btn => {
        const preset = btn.dataset.preset;
        if (preset === selectedPreset) {
            btn.classList.add('bg-purple-600');
            btn.classList.remove('bg-gray-700');
        } else {
            btn.classList.remove('bg-purple-600');
            btn.classList.add('bg-gray-700');
        }
    });
}

// =============================================================================
// Event Binding
// =============================================================================

function initLayerBlendEvents() {
    const { layerBlendEnabled, layerBlendMode, blendWeight1, hiddenLayerSlider } = DOM;

    if (layerBlendEnabled) {
        layerBlendEnabled.addEventListener('change', toggleLayerBlendControls);
    }

    if (layerBlendMode) {
        layerBlendMode.addEventListener('change', updateLayerBlendMode);
    }

    if (blendWeight1) {
        blendWeight1.addEventListener('input', updateBlendWeightDisplay);
    }

    if (hiddenLayerSlider) {
        hiddenLayerSlider.addEventListener('input', () => {
            const layer = parseInt(hiddenLayerSlider.value);
            updateHiddenLayerUI(layer);
            updateLayerPresetStyles(layer);
        });
    }

    setupLayerPresets();
    setupBlendPresets();
    populateLayerDropdowns();
}

// Export for use by other modules
window.populateLayerDropdowns = populateLayerDropdowns;
window.updateLayerBlendMode = updateLayerBlendMode;
window.toggleLayerBlendControls = toggleLayerBlendControls;
window.getLayerWeights = getLayerWeights;
window.updateBlendWeightDisplay = updateBlendWeightDisplay;
window.setupLayerPresets = setupLayerPresets;
window.setupBlendPresets = setupBlendPresets;
window.initLayerBlendEvents = initLayerBlendEvents;
