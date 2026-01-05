/**
 * history.js - Generation history panel functionality
 *
 * Handles loading, displaying, and managing the generation history.
 */

// =============================================================================
// History Panel Toggle
// =============================================================================

function toggleHistoryPanel() {
    const { historyPanel } = DOM;
    if (!historyPanel) return;

    historyPanel.classList.toggle('collapsed');
    document.body.classList.toggle('panel-open', !historyPanel.classList.contains('collapsed'));
}

// =============================================================================
// History Loading & Rendering
// =============================================================================

async function loadHistory() {
    try {
        const data = await ApiClient.getHistory();
        renderHistory(data.history);
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

function renderHistory(history) {
    const { historyList } = DOM;
    if (!historyList) return;

    if (!history || history.length === 0) {
        historyList.innerHTML = '<div class="text-center text-gray-500 text-sm py-8">No generations yet</div>';
        return;
    }

    historyList.innerHTML = history.map((item, index) => `
        <div class="history-item relative rounded-lg overflow-hidden mb-2 cursor-pointer group" data-index="${index}">
            <img src="data:image/png;base64,${escapeHtml(item.image_b64)}" class="w-full aspect-square object-cover" alt="Generated">
            <div class="history-overlay absolute inset-0 bg-black/60 opacity-0 flex flex-col justify-end p-2">
                <p class="text-white text-xs line-clamp-2 mb-1">${escapeHtml(item.prompt.substring(0, 80))}${item.prompt.length > 80 ? '...' : ''}</p>
                <div class="flex gap-1">
                    <button class="history-reuse flex-1 px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors">
                        Use
                    </button>
                    <button class="history-delete px-2 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors">
                        X
                    </button>
                </div>
            </div>
            <div class="absolute top-1 right-1 text-xs text-white/70 bg-black/50 px-1 rounded">
                ${item.width}x${item.height}
            </div>
        </div>
    `).join('');

    // Add event listeners
    historyList.querySelectorAll('.history-reuse').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const index = parseInt(btn.closest('.history-item').dataset.index);
            reuseHistoryItem(history[index]);
        });
    });

    historyList.querySelectorAll('.history-delete').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const index = parseInt(btn.closest('.history-item').dataset.index);
            await deleteHistoryItem(index);
        });
    });

    historyList.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            const index = parseInt(item.dataset.index);
            showHistoryItemFull(history[index]);
        });
    });
}

// =============================================================================
// History Item Actions
// =============================================================================

function reuseHistoryItem(item) {
    // Set prompt fields
    const promptEl = document.getElementById('prompt');
    const systemPromptEl = document.getElementById('systemPrompt');
    const thinkingContentEl = document.getElementById('thinkingContent');
    const assistantContentEl = document.getElementById('assistantContent');
    const enableThinkingEl = document.getElementById('enableThinking');
    const stripQuotesEl = document.getElementById('stripQuotes');
    const seedEl = document.getElementById('seed');
    const templateEl = document.getElementById('template');

    if (promptEl) promptEl.value = item.prompt;
    if (systemPromptEl) systemPromptEl.value = item.system_prompt || '';
    if (thinkingContentEl) thinkingContentEl.value = item.thinking_content || '';
    if (assistantContentEl) assistantContentEl.value = item.assistant_content || '';
    if (enableThinkingEl) enableThinkingEl.checked = item.force_think_block || false;
    if (stripQuotesEl) stripQuotesEl.checked = item.strip_quotes || false;
    if (seedEl) seedEl.value = item.seed || '';
    if (templateEl) templateEl.value = item.template || '';

    // Update thinking visibility
    if (typeof updateThinkingContentVisibility === 'function') {
        updateThinkingContentVisibility();
    }

    // Set sliders
    const { stepsSlider, stepsValue, guidanceScaleSlider, guidanceScaleValue, shiftSlider, shiftValue, longPromptModeSelect, hiddenLayerSlider } = DOM;

    if (stepsSlider && stepsValue) {
        stepsSlider.value = item.steps || 9;
        stepsValue.textContent = stepsSlider.value;
    }

    if (item.guidance_scale !== undefined && guidanceScaleSlider && guidanceScaleValue) {
        guidanceScaleSlider.value = item.guidance_scale;
        guidanceScaleValue.textContent = formatNumber(item.guidance_scale, 1);
    }

    if (item.shift !== undefined && shiftSlider && shiftValue) {
        shiftSlider.value = item.shift;
        shiftValue.textContent = formatNumber(item.shift, 1);
    }

    if (item.long_prompt_mode && longPromptModeSelect) {
        longPromptModeSelect.value = item.long_prompt_mode;
    }

    if (item.hidden_layer !== undefined && hiddenLayerSlider) {
        hiddenLayerSlider.value = item.hidden_layer;
        updateHiddenLayerUI(item.hidden_layer);
    }

    // Set resolution
    setResolutionFromHistory(item.width, item.height);

    // Close history panel on mobile
    if (window.innerWidth < 768) {
        toggleHistoryPanel();
    }
}

function setResolutionFromHistory(width, height) {
    const resolutionSelect = document.getElementById('resolution');
    const customWidth = document.getElementById('customWidth');
    const customHeight = document.getElementById('customHeight');

    if (!resolutionSelect) return;

    const resValue = `${width}x${height}`;

    // Try to find matching preset
    const option = Array.from(resolutionSelect.options).find(opt => opt.value === resValue);
    if (option) {
        resolutionSelect.value = resValue;
    } else {
        // Use custom
        resolutionSelect.value = 'custom';
        if (customWidth) customWidth.value = width;
        if (customHeight) customHeight.value = height;
    }

    // Trigger resolution change handler if exists
    if (typeof handleResolutionChange === 'function') {
        handleResolutionChange();
    }
}

async function deleteHistoryItem(index) {
    try {
        await ApiClient.deleteHistoryItem(index);
        await loadHistory();
    } catch (err) {
        console.error('Failed to delete history item:', err);
    }
}

function showHistoryItemFull(item) {
    // Open full-size image in modal
    openImageModal(`data:image/png;base64,${item.image_b64}`);
}

async function clearAllHistory() {
    if (!confirm('Clear all generation history?')) return;

    try {
        await ApiClient.clearHistory();
        await loadHistory();
        showSettingsMessage('History cleared');
    } catch (err) {
        console.error('Failed to clear history:', err);
        showSettingsMessage('Failed to clear history', 'error');
    }
}

// =============================================================================
// Event Binding
// =============================================================================

function initHistoryEvents() {
    const { historyToggle, historyHandle, clearHistoryBtn } = DOM;

    if (historyToggle) {
        historyToggle.addEventListener('click', toggleHistoryPanel);
    }

    if (historyHandle) {
        historyHandle.addEventListener('click', toggleHistoryPanel);
    }

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', clearAllHistory);
    }
}

// Export for use by other modules
window.toggleHistoryPanel = toggleHistoryPanel;
window.loadHistory = loadHistory;
window.renderHistory = renderHistory;
window.reuseHistoryItem = reuseHistoryItem;
window.deleteHistoryItem = deleteHistoryItem;
window.showHistoryItemFull = showHistoryItemFull;
window.clearAllHistory = clearAllHistory;
window.initHistoryEvents = initHistoryEvents;
