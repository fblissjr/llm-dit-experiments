/**
 * advanced.js - Advanced generation features
 *
 * Handles DyPE (Dynamic Positional Encoding), SLG (Skip Layer Guidance),
 * FMTT (Flow Matching Trajectory Tilting), and CFG enhancement controls.
 */

// =============================================================================
// DyPE (Dynamic Positional Encoding)
// =============================================================================

function updateDypeStatus() {
    const dypeEnabled = document.getElementById('dypeEnabled');
    const dypeStatus = document.getElementById('dypeStatus');
    const dypeControls = document.getElementById('dypeControls');

    if (!dypeEnabled) return;

    const enabled = dypeEnabled.checked;

    if (dypeStatus) {
        if (enabled) {
            dypeStatus.textContent = 'Enabled';
            dypeStatus.classList.add('text-green-400');
            dypeStatus.classList.remove('text-gray-500');
        } else {
            dypeStatus.textContent = 'Disabled';
            dypeStatus.classList.remove('text-green-400');
            dypeStatus.classList.add('text-gray-500');
        }
    }

    if (dypeControls) {
        dypeControls.classList.toggle('hidden', !enabled);
    }

    updateMultipassVisibility();
}

function updateMultipassVisibility() {
    const dypeMethod = document.getElementById('dypeMethod');
    const multipassControls = document.getElementById('multipassControls');

    if (!dypeMethod || !multipassControls) return;

    const isMultipass = dypeMethod.value === 'multipass';
    multipassControls.classList.toggle('hidden', !isMultipass);
}

function updatePass3Visibility() {
    const multipassSelect = document.getElementById('dypeMultipass');
    const pass3Container = document.getElementById('dypePass3StrengthContainer');

    if (multipassSelect && pass3Container) {
        const isThreePass = multipassSelect.value === 'threepass';
        pass3Container.classList.toggle('hidden', !isThreePass);
    }
}

function updateDypeRecommendation() {
    const resolution = getResolution();
    const dypeRecommendation = document.getElementById('dypeRecommendation');

    if (!dypeRecommendation) return;

    const totalPixels = resolution.width * resolution.height;
    const megapixels = totalPixels / 1000000;

    if (megapixels > 2) {
        dypeRecommendation.textContent = 'Recommended for this resolution';
        dypeRecommendation.classList.add('text-yellow-400');
        dypeRecommendation.classList.remove('hidden');
    } else {
        dypeRecommendation.classList.add('hidden');
    }
}

function getDypeConfig() {
    const dypeEnabled = document.getElementById('dypeEnabled');
    if (!dypeEnabled || !dypeEnabled.checked) {
        return null;
    }

    return {
        enabled: true,
        method: document.getElementById('dypeMethod')?.value || 'vision_yarn',
        multipass: document.getElementById('dypeMultipass')?.value || 'twopass',
        dype_scale: parseFloat(document.getElementById('dypeScale')?.value) || 2.0,
        dype_exponent: parseFloat(document.getElementById('dypeExponent')?.value) || 2.0,
        base_shift: parseFloat(document.getElementById('dypeBaseShift')?.value) || 0.5,
        max_shift: parseFloat(document.getElementById('dypeMaxShift')?.value) || 1.15,
        pass2_strength: parseFloat(document.getElementById('dypeStrength')?.value) || 0.5,
        pass3_strength: parseFloat(document.getElementById('dypePass3Strength')?.value) || 0.4,
        frequency_modulation: document.getElementById('dypeFrequencyMod')?.checked || false,
    };
}

// =============================================================================
// SLG (Skip Layer Guidance)
// =============================================================================

function updateSlgStatus() {
    const slgEnabled = document.getElementById('slgEnabled');
    const slgStatus = document.getElementById('slgStatus');
    const slgControls = document.getElementById('slgControls');

    if (!slgEnabled) return;

    const enabled = slgEnabled.checked;

    if (slgStatus) {
        if (enabled) {
            slgStatus.textContent = 'Enabled';
            slgStatus.classList.add('text-green-400');
            slgStatus.classList.remove('text-gray-500');
        } else {
            slgStatus.textContent = 'Disabled';
            slgStatus.classList.remove('text-green-400');
            slgStatus.classList.add('text-gray-500');
        }
    }

    if (slgControls) {
        slgControls.classList.toggle('hidden', !enabled);
    }
}

function updateSlgLayersDisplay() {
    const slgLayers = document.getElementById('slgLayers');
    const slgLayersDisplay = document.getElementById('slgLayersDisplay');

    if (!slgLayers || !slgLayersDisplay) return;

    slgLayersDisplay.textContent = slgLayers.value || 'None selected';
}

async function loadSlgConfig() {
    try {
        const data = await ApiClient.getGenerationConfig();

        if (data.slg) {
            const slgScale = document.getElementById('slgScale');
            const slgStart = document.getElementById('slgStart');
            const slgEnd = document.getElementById('slgEnd');
            const slgLayers = document.getElementById('slgLayers');

            if (data.slg.scale !== undefined && slgScale) {
                slgScale.value = data.slg.scale;
            }
            if (data.slg.start !== undefined && slgStart) {
                slgStart.value = data.slg.start;
            }
            if (data.slg.end !== undefined && slgEnd) {
                slgEnd.value = data.slg.end;
            }
            if (data.slg.layers && slgLayers) {
                slgLayers.value = data.slg.layers.join(',');
                updateSlgLayersDisplay();
            }
        }

    } catch (err) {
        console.error('Failed to load SLG config:', err);
    }
}

function getSlgConfig() {
    const slgEnabled = document.getElementById('slgEnabled');
    if (!slgEnabled || !slgEnabled.checked) {
        return null;
    }

    const slgScale = document.getElementById('slgScale');
    const slgStart = document.getElementById('slgStart');
    const slgEnd = document.getElementById('slgEnd');
    const slgLayers = document.getElementById('slgLayers');

    return {
        enabled: true,
        scale: slgScale ? parseFloat(slgScale.value) : 2.5,
        start: slgStart ? parseFloat(slgStart.value) : 0.01,
        end: slgEnd ? parseFloat(slgEnd.value) : 0.2,
        layers: slgLayers ? slgLayers.value.split(',').map(l => parseInt(l.trim())).filter(l => !isNaN(l)) : [7, 8, 9],
    };
}

// =============================================================================
// FMTT (Flow Matching Trajectory Tilting)
// =============================================================================

function updateFmttStatus() {
    const fmttEnabled = document.getElementById('fmttEnabled');
    const fmttStatus = document.getElementById('fmttStatus');
    const fmttControls = document.getElementById('fmttControls');

    if (!fmttEnabled) return;

    const enabled = fmttEnabled.checked;

    if (fmttStatus) {
        if (enabled) {
            fmttStatus.textContent = 'Enabled';
            fmttStatus.classList.add('text-green-400');
            fmttStatus.classList.remove('text-gray-500');
        } else {
            fmttStatus.textContent = 'Disabled';
            fmttStatus.classList.remove('text-green-400');
            fmttStatus.classList.add('text-gray-500');
        }
    }

    if (fmttControls) {
        fmttControls.classList.toggle('hidden', !enabled);
    }
}

async function loadFmttConfig() {
    try {
        const data = await ApiClient.getGenerationConfig();

        if (data.fmtt) {
            const fmttStart = document.getElementById('fmttStart');
            const fmttEnd = document.getElementById('fmttEnd');

            if (data.fmtt.start !== undefined && fmttStart) {
                fmttStart.value = data.fmtt.start;
            }
            if (data.fmtt.end !== undefined && fmttEnd) {
                fmttEnd.value = data.fmtt.end;
            }
        }

    } catch (err) {
        console.error('Failed to load FMTT config:', err);
    }
}

function getFmttConfig() {
    const fmttEnabled = document.getElementById('fmttEnabled');
    if (!fmttEnabled || !fmttEnabled.checked) {
        return null;
    }

    const fmttScale = document.getElementById('fmttScale');
    const fmttStart = document.getElementById('fmttStart');
    const fmttEnd = document.getElementById('fmttEnd');

    return {
        enabled: true,
        scale: fmttScale ? parseFloat(fmttScale.value) : 1.0,
        start: fmttStart ? parseFloat(fmttStart.value) : 0.0,
        end: fmttEnd ? parseFloat(fmttEnd.value) : 0.5,
    };
}

// =============================================================================
// CFG Enhancement
// =============================================================================

function updateCfgStatus() {
    const cfgNormEnabled = document.getElementById('cfgNormEnabled');
    const cfgNormStatus = document.getElementById('cfgNormStatus');
    const cfgNormControls = document.getElementById('cfgNormControls');

    if (!cfgNormEnabled) return;

    const enabled = cfgNormEnabled.checked;

    if (cfgNormStatus) {
        if (enabled) {
            cfgNormStatus.textContent = 'Enabled';
            cfgNormStatus.classList.add('text-green-400');
            cfgNormStatus.classList.remove('text-gray-500');
        } else {
            cfgNormStatus.textContent = 'Disabled';
            cfgNormStatus.classList.remove('text-green-400');
            cfgNormStatus.classList.add('text-gray-500');
        }
    }

    if (cfgNormControls) {
        cfgNormControls.classList.toggle('hidden', !enabled);
    }
}

function getCfgEnhancementConfig() {
    const cfgNormEnabled = document.getElementById('cfgNormEnabled');
    const cfgNormValue = document.getElementById('cfgNormValue');
    const cfgNormMode = document.getElementById('cfgNormMode');
    const cfgTruncation = document.getElementById('cfgTruncation');

    const config = {};

    if (cfgNormEnabled && cfgNormEnabled.checked) {
        config.cfg_normalization = cfgNormValue ? parseFloat(cfgNormValue.value) : 0.0;
        config.cfg_norm_mode = cfgNormMode ? cfgNormMode.value : 'match';
    }

    if (cfgTruncation) {
        config.cfg_truncation = parseFloat(cfgTruncation.value);
    }

    return config;
}

// =============================================================================
// Thinking Content Visibility
// =============================================================================

function updateThinkingContentVisibility() {
    const enableThinking = document.getElementById('enableThinking');
    const thinkingContentSection = document.getElementById('thinkingContentSection');

    if (!enableThinking || !thinkingContentSection) return;

    thinkingContentSection.classList.toggle('hidden', !enableThinking.checked);
}

// =============================================================================
// Event Binding
// =============================================================================

function initAdvancedEvents() {
    // DyPE
    const dypeEnabled = document.getElementById('dypeEnabled');
    const dypeMethod = document.getElementById('dypeMethod');
    const dypeMultipass = document.getElementById('dypeMultipass');

    if (dypeEnabled) {
        dypeEnabled.addEventListener('change', updateDypeStatus);
    }
    if (dypeMethod) {
        dypeMethod.addEventListener('change', updateMultipassVisibility);
    }
    if (dypeMultipass) {
        dypeMultipass.addEventListener('change', updatePass3Visibility);
    }

    // DyPE slider value displays
    const dypeSliders = [
        { slider: 'dypeScale', display: 'dypeScaleValue', decimals: 1 },
        { slider: 'dypeExponent', display: 'dypeExponentValue', decimals: 1 },
        { slider: 'dypeBaseShift', display: 'dypeBaseShiftValue', decimals: 2 },
        { slider: 'dypeMaxShift', display: 'dypeMaxShiftValue', decimals: 2 },
        { slider: 'dypeStrength', display: 'dypeStrengthValue', decimals: 2 },
        { slider: 'dypePass3Strength', display: 'dypePass3StrengthValue', decimals: 2 },
    ];

    dypeSliders.forEach(({ slider, display, decimals }) => {
        const sliderEl = document.getElementById(slider);
        const displayEl = document.getElementById(display);
        if (sliderEl && displayEl) {
            sliderEl.addEventListener('input', () => {
                displayEl.textContent = parseFloat(sliderEl.value).toFixed(decimals);
            });
        }
    });

    // SLG
    const slgEnabled = document.getElementById('slgEnabled');
    const slgLayers = document.getElementById('slgLayers');

    if (slgEnabled) {
        slgEnabled.addEventListener('change', updateSlgStatus);
    }
    if (slgLayers) {
        slgLayers.addEventListener('input', updateSlgLayersDisplay);
    }

    // FMTT
    const fmttEnabled = document.getElementById('fmttEnabled');
    if (fmttEnabled) {
        fmttEnabled.addEventListener('change', updateFmttStatus);
    }

    // CFG
    const cfgNormEnabled = document.getElementById('cfgNormEnabled');
    if (cfgNormEnabled) {
        cfgNormEnabled.addEventListener('change', updateCfgStatus);
    }

    // Thinking content
    const enableThinking = document.getElementById('enableThinking');
    if (enableThinking) {
        enableThinking.addEventListener('change', updateThinkingContentVisibility);
    }

    // Resolution change listener for DyPE recommendation
    const resolutionSelect = document.getElementById('resolution');
    if (resolutionSelect) {
        resolutionSelect.addEventListener('change', updateDypeRecommendation);
    }
}

// Export for use by other modules
window.updateDypeStatus = updateDypeStatus;
window.updateMultipassVisibility = updateMultipassVisibility;
window.updatePass3Visibility = updatePass3Visibility;
window.updateDypeRecommendation = updateDypeRecommendation;
window.getDypeConfig = getDypeConfig;
window.updateSlgStatus = updateSlgStatus;
window.updateSlgLayersDisplay = updateSlgLayersDisplay;
window.loadSlgConfig = loadSlgConfig;
window.getSlgConfig = getSlgConfig;
window.updateFmttStatus = updateFmttStatus;
window.loadFmttConfig = loadFmttConfig;
window.getFmttConfig = getFmttConfig;
window.updateCfgStatus = updateCfgStatus;
window.getCfgEnhancementConfig = getCfgEnhancementConfig;
window.updateThinkingContentVisibility = updateThinkingContentVisibility;
window.initAdvancedEvents = initAdvancedEvents;
