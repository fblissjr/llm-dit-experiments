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
        <div class="history-item relative rounded-lg overflow-hidden mb-2 cursor-pointer group" data-index="${index}" draggable="true">
            <img src="data:image/png;base64,${escapeHtml(item.image_b64)}" class="w-full aspect-square object-cover pointer-events-none" alt="Generated">
            <div class="history-overlay absolute inset-0 bg-black/60 opacity-0 flex flex-col justify-end p-2">
                <p class="text-white text-xs line-clamp-2 mb-1">${escapeHtml(item.prompt.substring(0, 80))}${item.prompt.length > 80 ? '...' : ''}</p>
                <div class="flex gap-1">
                    <!-- Use As Dropdown -->
                    <div class="history-use-menu relative flex-1">
                        <button class="history-use-trigger w-full px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors flex items-center justify-center gap-1">
                            Use
                            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                            </svg>
                        </button>
                        <div class="history-use-dropdown hidden absolute bottom-full left-0 right-0 mb-1 bg-gray-800 border border-gray-700 rounded-lg shadow-xl overflow-hidden z-20">
                            <button class="history-use-params w-full px-3 py-2 text-left text-xs text-gray-200 hover:bg-gray-700 transition-colors border-b border-gray-700">
                                Use Parameters
                            </button>
                            <button class="history-use-img2img w-full px-3 py-2 text-left text-xs text-gray-200 hover:bg-purple-600 transition-colors">
                                Use as Img2Img
                            </button>
                            <button class="history-use-vl w-full px-3 py-2 text-left text-xs text-gray-200 hover:bg-blue-600 transition-colors">
                                Use as VL Reference
                            </button>
                            <button class="history-use-qwen w-full px-3 py-2 text-left text-xs text-gray-200 hover:bg-orange-600 transition-colors">
                                Use in Qwen Edit
                            </button>
                            <button class="history-use-combine w-full px-3 py-2 text-left text-xs text-gray-200 hover:bg-teal-600 transition-colors">
                                Add to Combine
                            </button>
                        </div>
                    </div>
                    <button class="history-delete px-2 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors">
                        X
                    </button>
                </div>
            </div>
            <div class="absolute top-1 right-1 flex gap-1">
                <span class="text-xs text-white/70 bg-black/50 px-1 rounded">${item.width}x${item.height}</span>
                ${item.seed ? `<span class="text-xs text-yellow-400/80 bg-black/50 px-1 rounded" title="Seed: ${item.seed}">#${item.seed}</span>` : ''}
            </div>
        </div>
    `).join('');

    // Add event listeners
    historyList.querySelectorAll('.history-item').forEach(item => {
        const index = parseInt(item.dataset.index);
        const historyItem = history[index];
        const imageBase64 = `data:image/png;base64,${historyItem.image_b64}`;

        // Click to view full size
        item.addEventListener('click', (e) => {
            // Don't trigger if clicking on buttons or dropdown
            if (e.target.closest('button') || e.target.closest('.history-use-dropdown')) return;
            showHistoryItemFull(historyItem);
        });

        // Drag and drop support
        item.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('application/x-image-transfer', JSON.stringify({
                base64: imageBase64,
                width: historyItem.width,
                height: historyItem.height,
            }));
            e.dataTransfer.effectAllowed = 'copy';
            item.classList.add('dragging');
            document.body.classList.add('dragging-image');

            // Highlight drop targets
            if (typeof highlightDropTargets === 'function') {
                highlightDropTargets(true);
            }
        });

        item.addEventListener('dragend', () => {
            item.classList.remove('dragging');
            document.body.classList.remove('dragging-image');
            if (typeof highlightDropTargets === 'function') {
                highlightDropTargets(false);
            }
        });

        // Use dropdown trigger
        const trigger = item.querySelector('.history-use-trigger');
        const dropdown = item.querySelector('.history-use-dropdown');

        if (trigger && dropdown) {
            trigger.addEventListener('click', (e) => {
                e.stopPropagation();
                // Close other dropdowns
                historyList.querySelectorAll('.history-use-dropdown').forEach(d => {
                    if (d !== dropdown) d.classList.add('hidden');
                });
                dropdown.classList.toggle('hidden');
            });
        }

        // Use Parameters (original behavior)
        const useParams = item.querySelector('.history-use-params');
        if (useParams) {
            useParams.addEventListener('click', (e) => {
                e.stopPropagation();
                dropdown?.classList.add('hidden');
                reuseHistoryItem(historyItem);
            });
        }

        // Use as Img2Img
        const useImg2Img = item.querySelector('.history-use-img2img');
        if (useImg2Img) {
            useImg2Img.addEventListener('click', async (e) => {
                e.stopPropagation();
                dropdown?.classList.add('hidden');
                if (typeof useAsImg2Img === 'function') {
                    await useAsImg2Img(imageBase64, historyItem.width, historyItem.height);
                    if (window.innerWidth < 768) toggleHistoryPanel();
                }
            });
        }

        // Use as VL Reference
        const useVL = item.querySelector('.history-use-vl');
        if (useVL) {
            useVL.addEventListener('click', async (e) => {
                e.stopPropagation();
                dropdown?.classList.add('hidden');
                if (typeof useAsVLReference === 'function') {
                    await useAsVLReference(imageBase64);
                    if (window.innerWidth < 768) toggleHistoryPanel();
                }
            });
        }

        // Use in Qwen Edit
        const useQwen = item.querySelector('.history-use-qwen');
        if (useQwen) {
            useQwen.addEventListener('click', async (e) => {
                e.stopPropagation();
                dropdown?.classList.add('hidden');
                if (typeof useInQwenEdit === 'function') {
                    await useInQwenEdit(imageBase64);
                    if (window.innerWidth < 768) toggleHistoryPanel();
                }
            });
        }

        // Add to Combine
        const useCombine = item.querySelector('.history-use-combine');
        if (useCombine) {
            useCombine.addEventListener('click', async (e) => {
                e.stopPropagation();
                dropdown?.classList.add('hidden');
                if (typeof addToCombine === 'function') {
                    await addToCombine(imageBase64);
                    if (window.innerWidth < 768) toggleHistoryPanel();
                }
            });
        }

        // Long-press for mobile (shows dropdown)
        if (typeof setupLongPress === 'function' && 'ontouchstart' in window) {
            setupLongPress(item, (e) => {
                e.preventDefault();
                if (dropdown) {
                    dropdown.classList.remove('hidden');
                }
            });
        }
    });

    // Delete buttons
    historyList.querySelectorAll('.history-delete').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const index = parseInt(btn.closest('.history-item').dataset.index);
            await deleteHistoryItem(index);
        });
    });

    // Close dropdowns when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.history-use-menu')) {
            historyList.querySelectorAll('.history-use-dropdown').forEach(d => {
                d.classList.add('hidden');
            });
        }
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
    // Open full-size image in modal with dimensions for lightbox actions
    openImageModal(`data:image/png;base64,${item.image_b64}`, item.width, item.height);
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
