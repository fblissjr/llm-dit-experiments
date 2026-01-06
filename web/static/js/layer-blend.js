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
    const { layerBlendEnabled, layerBlendMode, blendLayer1, blendLayer2, blendWeight1,
            blendLayer3a, blendLayer3b, blendLayer3c,
            blendWeight3a, blendWeight3b, blendWeight3c } = DOM;

    if (!layerBlendEnabled || !layerBlendEnabled.checked) {
        return null;
    }

    const mode = layerBlendMode ? layerBlendMode.value : '2-layer';

    // Build dict mapping layer index to weight (backend format: Dict[int, float])
    const weights = {};

    if (mode === '2-layer') {
        // Slider is 0-100, convert to 0-1
        const weight = blendWeight1 ? parseFloat(blendWeight1.value) / 100 : 0.5;
        const layer1 = parseInt(blendLayer1?.value || -2);
        const layer2 = parseInt(blendLayer2?.value || -6);

        // If same layer selected twice, just use weight 1.0
        if (layer1 === layer2) {
            weights[layer1] = 1.0;
        } else {
            weights[layer1] = weight;
            weights[layer2] = 1 - weight;
        }
    } else {
        // 3-layer mode with per-layer weights (sliders are 0-100)
        const layer3a = parseInt(blendLayer3a?.value || -1);
        const layer3b = parseInt(blendLayer3b?.value || -2);
        const layer3c = parseInt(blendLayer3c?.value || -3);

        const weight3a = blendWeight3a ? parseFloat(blendWeight3a.value) / 100 : 0.33;
        const weight3b = blendWeight3b ? parseFloat(blendWeight3b.value) / 100 : 0.33;
        const weight3c = blendWeight3c ? parseFloat(blendWeight3c.value) / 100 : 0.34;

        // Handle duplicate layers by summing weights
        const layerWeightPairs = [
            [layer3a, weight3a],
            [layer3b, weight3b],
            [layer3c, weight3c]
        ];
        layerWeightPairs.forEach(([layer, w]) => {
            weights[layer] = (weights[layer] || 0) + w;
        });
    }

    return weights;
}

// =============================================================================
// Weight Slider Display
// =============================================================================

function updateBlendWeightDisplay() {
    const { blendWeight1, blendWeight1Value, blendWeight2Value } = DOM;

    if (!blendWeight1 || !blendWeight1Value || !blendWeight2Value) return;

    // Slider is 0-100, convert to 0-1 for display
    const weight = parseFloat(blendWeight1.value) / 100;
    blendWeight1Value.textContent = formatNumber(weight, 2);
    blendWeight2Value.textContent = formatNumber(1 - weight, 2);
}

function updateThreeLayerWeightDisplay() {
    const { blendWeight3a, blendWeight3b, blendWeight3c,
            blendWeight3aValue, blendWeight3bValue, blendWeight3cValue } = DOM;

    // Update each 3-layer weight display (sliders are 0-100, convert to 0-1)
    if (blendWeight3a && blendWeight3aValue) {
        blendWeight3aValue.textContent = formatNumber(parseFloat(blendWeight3a.value) / 100, 2);
    }
    if (blendWeight3b && blendWeight3bValue) {
        blendWeight3bValue.textContent = formatNumber(parseFloat(blendWeight3b.value) / 100, 2);
    }
    if (blendWeight3c && blendWeight3cValue) {
        blendWeight3cValue.textContent = formatNumber(parseFloat(blendWeight3c.value) / 100, 2);
    }
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
    const { blendLayer1, blendLayer2, blendWeight1, layerBlendMode, layerBlendEnabled, layerBlendControls,
            blendLayer3a, blendLayer3b, blendLayer3c,
            blendWeight3a, blendWeight3b, blendWeight3c } = DOM;

    // Presets define layers and weights (0-1 scale, converted to 0-100 for sliders)
    // 2-layer presets: l1, l2, w (weight for l1, l2 gets 1-w)
    // 3-layer presets: l1, l2, l3, w1, w2, w3 (individual weights)
    const presets = {
        // 2-layer presets
        'default': { mode: '2-layer', l1: -2, l2: -6, w: 0.7 },
        'semantic': { mode: '2-layer', l1: -2, l2: -6, w: 0.5 },
        'balanced': { mode: '2-layer', l1: -2, l2: -5, w: 0.5 },
        'deep': { mode: '2-layer', l1: -2, l2: -10, w: 0.6 },
        'middle': { mode: '2-layer', l1: -19, l2: -2, w: 0.5 },
        'structure': { mode: '2-layer', l1: -3, l2: -8, w: 0.5 },
        // 3-layer presets
        'top3': { mode: '3-layer', l1: -1, l2: -2, l3: -3, w1: 0.33, w2: 0.34, w3: 0.33 },
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

    // Set mode
    if (layerBlendMode) {
        layerBlendMode.value = p.mode;
        updateLayerBlendMode();
    }

    if (p.mode === '2-layer') {
        // Apply 2-layer values (w is 0-1, slider is 0-100)
        if (blendLayer1) blendLayer1.value = p.l1;
        if (blendLayer2) blendLayer2.value = p.l2;
        if (blendWeight1) blendWeight1.value = p.w * 100;
        updateBlendWeightDisplay();
    } else {
        // Apply 3-layer values (weights are 0-1, sliders are 0-100)
        if (blendLayer3a) blendLayer3a.value = p.l1;
        if (blendLayer3b) blendLayer3b.value = p.l2;
        if (blendLayer3c) blendLayer3c.value = p.l3;
        if (blendWeight3a) blendWeight3a.value = p.w1 * 100;
        if (blendWeight3b) blendWeight3b.value = p.w2 * 100;
        if (blendWeight3c) blendWeight3c.value = p.w3 * 100;
        updateThreeLayerWeightDisplay();
    }
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
    const { layerBlendEnabled, layerBlendMode, blendWeight1, hiddenLayerSlider,
            blendWeight3a, blendWeight3b, blendWeight3c } = DOM;

    if (layerBlendEnabled) {
        layerBlendEnabled.addEventListener('change', toggleLayerBlendControls);
    }

    if (layerBlendMode) {
        layerBlendMode.addEventListener('change', updateLayerBlendMode);
    }

    if (blendWeight1) {
        blendWeight1.addEventListener('input', updateBlendWeightDisplay);
    }

    // 3-layer weight sliders
    if (blendWeight3a) {
        blendWeight3a.addEventListener('input', updateThreeLayerWeightDisplay);
    }
    if (blendWeight3b) {
        blendWeight3b.addEventListener('input', updateThreeLayerWeightDisplay);
    }
    if (blendWeight3c) {
        blendWeight3c.addEventListener('input', updateThreeLayerWeightDisplay);
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
window.updateThreeLayerWeightDisplay = updateThreeLayerWeightDisplay;
window.setupLayerPresets = setupLayerPresets;
window.setupBlendPresets = setupBlendPresets;
window.initLayerBlendEvents = initLayerBlendEvents;
