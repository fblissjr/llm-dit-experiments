/**
 * state.js - Shared state and DOM element references
 *
 * This module provides centralized access to DOM elements and shared state
 * used across the web UI. All other modules should import elements from here.
 */

// API Configuration
const API_BASE = window.location.origin;

// =============================================================================
// DOM Element References - Core
// =============================================================================

const DOM = {
    // Form elements
    form: null,
    generateBtn: null,

    // Status and progress
    status: null,
    statusText: null,
    progressFill: null,
    error: null,
    errorText: null,

    // Result display
    result: null,
    resultImage: null,
    resultInfo: null,
    downloadBtn: null,
    reuseBtn: null,

    // History panel
    historyPanel: null,
    historyList: null,
    historyToggle: null,
    historyHandle: null,
    clearHistoryBtn: null,

    // Resolution and basic params
    resolutionSelect: null,
    stepsSlider: null,
    stepsValue: null,
    guidanceScaleSlider: null,
    guidanceScaleValue: null,
    shiftSlider: null,
    shiftValue: null,
    shiftContainer: null,
    stepsMinLabel: null,
    stepsMaxLabel: null,

    // Long prompt and hidden layer
    longPromptModeSelect: null,
    hiddenLayerSlider: null,
    hiddenLayerValue: null,
    hiddenLayerLabel: null,
    layerPositionIndicator: null,
    layerPresetButtons: null,

    // Layer blending
    layerBlendEnabled: null,
    layerBlendControls: null,
    layerBlendMode: null,
    twoLayerControls: null,
    threeLayerControls: null,
    blendLayer1: null,
    blendLayer2: null,
    blendWeight1: null,
    blendWeight1Value: null,
    blendWeight2Value: null,
    blendLayer3a: null,
    blendLayer3b: null,
    blendLayer3c: null,
    blendPresetButtons: null,

    // Modal
    imageModal: null,
    modalImage: null,
};

// =============================================================================
// Shared State
// =============================================================================

const AppState = {
    // Current generation parameters (for reuse)
    currentParams: null,

    // VL conditioning state
    vlEmbeddingsId: null,
    vlImagePreview: null,

    // Img2Img state
    img2imgImage: null,
    img2imgMaskCtx: null,
    isDrawing: false,
    lastX: 0,
    lastY: 0,

    // Qwen-Image state
    qiInputImage: null,
    qiDecomposedLayers: [],
    multiCombineImages: [],
    singleEditImage: null,

    // Rewriter state
    rewriterImage: null,

    // Template cache
    templatesCache: null,

    // Feature availability flags (loaded from server)
    features: {
        vlEnabled: false,
        qwenImageEnabled: false,
        qwenImage2512Enabled: false,
    },
};

// =============================================================================
// Initialize DOM References
// =============================================================================

function initDOMReferences() {
    // Core form elements
    DOM.form = document.getElementById('generateForm');
    DOM.generateBtn = document.getElementById('generateBtn');

    // Status and progress
    DOM.status = document.getElementById('status');
    DOM.statusText = document.getElementById('statusText');
    DOM.progressFill = document.getElementById('progressFill');
    DOM.error = document.getElementById('error');
    DOM.errorText = document.getElementById('errorText');

    // Result display
    DOM.result = document.getElementById('result');
    DOM.resultImage = document.getElementById('resultImage');
    DOM.resultInfo = document.getElementById('resultInfo');
    DOM.downloadBtn = document.getElementById('downloadBtn');
    DOM.reuseBtn = document.getElementById('reuseBtn');

    // History panel
    DOM.historyPanel = document.getElementById('historyPanel');
    DOM.historyList = document.getElementById('historyList');
    DOM.historyToggle = document.getElementById('historyToggle');
    DOM.historyHandle = document.getElementById('historyHandle');
    DOM.clearHistoryBtn = document.getElementById('clearHistoryBtn');

    // Resolution and basic params
    DOM.resolutionSelect = document.getElementById('resolution');
    DOM.stepsSlider = document.getElementById('steps');
    DOM.stepsValue = document.getElementById('stepsValue');
    DOM.guidanceScaleSlider = document.getElementById('guidanceScale');
    DOM.guidanceScaleValue = document.getElementById('guidanceScaleValue');
    DOM.shiftSlider = document.getElementById('shift');
    DOM.shiftValue = document.getElementById('shiftValue');
    DOM.shiftContainer = document.getElementById('shiftContainer');
    DOM.stepsMinLabel = document.getElementById('stepsMinLabel');
    DOM.stepsMaxLabel = document.getElementById('stepsMaxLabel');

    // Long prompt and hidden layer
    DOM.longPromptModeSelect = document.getElementById('longPromptMode');
    DOM.hiddenLayerSlider = document.getElementById('hiddenLayer');
    DOM.hiddenLayerValue = document.getElementById('hiddenLayerValue');
    DOM.hiddenLayerLabel = document.getElementById('hiddenLayerLabel');
    DOM.layerPositionIndicator = document.getElementById('layerPositionIndicator');
    DOM.layerPresetButtons = document.querySelectorAll('.layer-preset');

    // Layer blending
    DOM.layerBlendEnabled = document.getElementById('layerBlendEnabled');
    DOM.layerBlendControls = document.getElementById('layerBlendControls');
    DOM.layerBlendMode = document.getElementById('layerBlendMode');
    DOM.twoLayerControls = document.getElementById('twoLayerControls');
    DOM.threeLayerControls = document.getElementById('threeLayerControls');
    DOM.blendLayer1 = document.getElementById('blendLayer1');
    DOM.blendLayer2 = document.getElementById('blendLayer2');
    DOM.blendWeight1 = document.getElementById('blendWeight1');
    DOM.blendWeight1Value = document.getElementById('blendWeight1Value');
    DOM.blendWeight2Value = document.getElementById('blendWeight2Value');
    DOM.blendLayer3a = document.getElementById('blendLayer3a');
    DOM.blendLayer3b = document.getElementById('blendLayer3b');
    DOM.blendLayer3c = document.getElementById('blendLayer3c');
    DOM.blendPresetButtons = document.querySelectorAll('.blend-preset');

    // Modal
    DOM.imageModal = document.getElementById('imageModal');
    DOM.modalImage = document.getElementById('modalImage');
}

// Export for use by other modules
window.API_BASE = API_BASE;
window.DOM = DOM;
window.AppState = AppState;
window.initDOMReferences = initDOMReferences;
