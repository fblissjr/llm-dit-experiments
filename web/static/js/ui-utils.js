/**
 * ui-utils.js - UI helper functions
 *
 * Provides utility functions for DOM manipulation, formatting,
 * and common UI operations used across the application.
 */

// =============================================================================
// Text & Formatting Utilities
// =============================================================================

/**
 * Escape HTML special characters to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Format a number with fixed decimal places
 */
function formatNumber(value, decimals = 1) {
    return parseFloat(value).toFixed(decimals);
}

/**
 * Format bytes to human readable string
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Format duration in seconds to human readable string
 */
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(1)}s`;
    }
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins}m ${secs}s`;
}

// =============================================================================
// Slider Display Updates
// =============================================================================

/**
 * Update all slider value displays
 */
function updateSliderDisplays() {
    const { stepsSlider, stepsValue, guidanceScaleSlider, guidanceScaleValue, shiftSlider, shiftValue } = DOM;

    if (stepsValue && stepsSlider) {
        stepsValue.textContent = stepsSlider.value;
    }
    if (guidanceScaleValue && guidanceScaleSlider) {
        guidanceScaleValue.textContent = formatNumber(guidanceScaleSlider.value, 1);
    }
    if (shiftValue && shiftSlider) {
        shiftValue.textContent = formatNumber(shiftSlider.value, 1);
    }
}

/**
 * Update hidden layer UI with position indicator
 */
function updateHiddenLayerUI(layer) {
    const { hiddenLayerValue, hiddenLayerLabel, layerPositionIndicator } = DOM;

    if (hiddenLayerValue) {
        hiddenLayerValue.textContent = layer;
    }

    // Update label with semantic meaning
    if (hiddenLayerLabel) {
        const layerLabels = {
            '-1': 'Last (most processed)',
            '-2': 'Default',
            '-6': 'VL optimal',
            '-18': 'Middle',
            '-19': 'Middle',
            '-35': 'First (least processed)'
        };
        hiddenLayerLabel.textContent = layerLabels[String(layer)] || '';
    }

    // Update visual position indicator
    if (layerPositionIndicator) {
        // Layer goes from -1 (right) to -35 (left)
        const percent = ((layer + 35) / 34) * 100;
        layerPositionIndicator.style.left = `${percent}%`;
    }
}

// =============================================================================
// Message Notifications
// =============================================================================

/**
 * Show a settings modal message
 */
function showSettingsMessage(msg, type = 'success') {
    const settingsMessage = document.getElementById('settingsMessage');
    if (!settingsMessage) return;

    settingsMessage.textContent = msg;
    settingsMessage.classList.remove('hidden', 'bg-green-600/20', 'text-green-400', 'bg-red-600/20', 'text-red-400');

    if (type === 'success') {
        settingsMessage.classList.add('bg-green-600/20', 'text-green-400');
    } else {
        settingsMessage.classList.add('bg-red-600/20', 'text-red-400');
    }

    setTimeout(() => settingsMessage.classList.add('hidden'), 3000);
}

/**
 * Show error message in the error panel
 */
function showError(message) {
    const { error, errorText } = DOM;
    if (error && errorText) {
        errorText.textContent = message;
        error.classList.remove('hidden');
    }
}

/**
 * Hide error panel
 */
function hideError() {
    const { error } = DOM;
    if (error) {
        error.classList.add('hidden');
    }
}

/**
 * Show status message with optional progress
 */
function showStatus(message, progress = null) {
    const { status, statusText, progressFill } = DOM;
    if (status && statusText) {
        statusText.textContent = message;
        status.classList.remove('hidden');
    }
    if (progressFill && progress !== null) {
        progressFill.style.width = `${progress}%`;
    }
}

/**
 * Hide status panel
 */
function hideStatus() {
    const { status } = DOM;
    if (status) {
        status.classList.add('hidden');
    }
}

// =============================================================================
// Modal Helpers
// =============================================================================

/**
 * Open image in lightbox modal
 */
function openImageModal(imgSrc) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    if (modal && modalImage) {
        modalImage.src = imgSrc;
        modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

/**
 * Close lightbox modal
 */
function closeImageModal(event) {
    // Only close if clicking the backdrop or close button, not the image itself
    if (event && event.target.id === 'modalImage') return;

    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

// =============================================================================
// Button State Helpers
// =============================================================================

/**
 * Set button to loading state
 */
function setButtonLoading(button, loadingText = 'Loading...') {
    if (!button) return;
    button.disabled = true;
    button.dataset.originalText = button.textContent;
    button.textContent = loadingText;
    button.classList.add('opacity-50', 'cursor-not-allowed');
}

/**
 * Reset button from loading state
 */
function resetButton(button) {
    if (!button) return;
    button.disabled = false;
    if (button.dataset.originalText) {
        button.textContent = button.dataset.originalText;
    }
    button.classList.remove('opacity-50', 'cursor-not-allowed');
}

// =============================================================================
// Debounce Utility
// =============================================================================

/**
 * Create a debounced version of a function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// =============================================================================
// Resolution Helpers
// =============================================================================

/**
 * Get current resolution from ResolutionSelector or fallback to direct inputs
 */
function getResolution() {
    // Use ResolutionSelector if available (new UI)
    if (typeof ResolutionSelector !== 'undefined' && ResolutionSelector.getResolution) {
        return ResolutionSelector.getResolution();
    }

    // Fallback: read directly from new input fields
    const widthInput = document.getElementById('resWidth');
    const heightInput = document.getElementById('resHeight');
    if (widthInput && heightInput) {
        return {
            width: parseInt(widthInput.value) || 1024,
            height: parseInt(heightInput.value) || 1024
        };
    }

    // Legacy fallback: old select element
    const resolutionSelect = document.getElementById('resolution');
    const customWidth = document.getElementById('customWidth');
    const customHeight = document.getElementById('customHeight');

    if (!resolutionSelect) return { width: 1024, height: 1024 };

    const value = resolutionSelect.value;
    if (value === 'custom' && customWidth && customHeight) {
        return {
            width: parseInt(customWidth.value) || 1024,
            height: parseInt(customHeight.value) || 1024
        };
    }

    const parts = value.split('x');
    return {
        width: parseInt(parts[0]) || 1024,
        height: parseInt(parts[1]) || 1024
    };
}

// =============================================================================
// File Handling Helpers
// =============================================================================

/**
 * Convert file to base64 data URL
 */
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

/**
 * Validate image file
 */
function validateImageFile(file, maxSizeMB = 50) {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
    const maxBytes = maxSizeMB * 1024 * 1024;

    if (!validTypes.includes(file.type)) {
        return { valid: false, error: 'Invalid file type. Please use JPEG, PNG, WebP, or GIF.' };
    }

    if (file.size > maxBytes) {
        return { valid: false, error: `File too large. Maximum size is ${maxSizeMB}MB.` };
    }

    return { valid: true };
}

// Export for use by other modules
window.escapeHtml = escapeHtml;
window.formatNumber = formatNumber;
window.formatBytes = formatBytes;
window.formatDuration = formatDuration;
window.updateSliderDisplays = updateSliderDisplays;
window.updateHiddenLayerUI = updateHiddenLayerUI;
window.showSettingsMessage = showSettingsMessage;
window.showError = showError;
window.hideError = hideError;
window.showStatus = showStatus;
window.hideStatus = hideStatus;
window.openImageModal = openImageModal;
window.closeImageModal = closeImageModal;
window.setButtonLoading = setButtonLoading;
window.resetButton = resetButton;
window.debounce = debounce;
window.getResolution = getResolution;
window.fileToBase64 = fileToBase64;
window.validateImageFile = validateImageFile;
