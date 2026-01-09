/**
 * api-client.js - API communication layer
 *
 * Provides a centralized interface for all API calls to the backend.
 * Uses the global API_BASE from state.js.
 */

const ApiClient = {
    // =========================================================================
    // System & Status
    // =========================================================================

    async getSystemStatus() {
        const response = await fetch(`${API_BASE}/api/system/status`);
        return response.json();
    },

    async clearCache() {
        const response = await fetch(`${API_BASE}/api/system/clear-cache`, { method: 'POST' });
        return response.json();
    },

    async unloadFMTT() {
        const response = await fetch(`${API_BASE}/api/system/unload-fmtt`, { method: 'POST' });
        return response.json();
    },

    // =========================================================================
    // VRAM Management
    // =========================================================================

    async unloadZImage() {
        const response = await fetch(`${API_BASE}/api/vram/unload-zimage`, { method: 'POST' });
        return response.json();
    },

    async unloadQwenImage() {
        const response = await fetch(`${API_BASE}/api/vram/unload-qwen-image`, { method: 'POST' });
        return response.json();
    },

    async unloadQwenImageT2i() {
        const response = await fetch(`${API_BASE}/api/vram/unload-qwen-image-t2i`, { method: 'POST' });
        return response.json();
    },

    async unloadLtx2() {
        const response = await fetch(`${API_BASE}/api/vram/unload-ltx2`, { method: 'POST' });
        return response.json();
    },

    // =========================================================================
    // LTX-2 Video Generation
    // =========================================================================

    async getLtx2Status() {
        const response = await fetch(`${API_BASE}/api/ltx2/status`);
        return response.json();
    },

    /**
     * Generate video with LTX-2 using Server-Sent Events for progress.
     * @param {Object} params - Generation parameters
     * @param {Function} onProgress - Progress callback (step, total, elapsed, eta, its)
     * @param {Function} onStatus - Status callback (message)
     * @param {Function} onComplete - Completion callback (result)
     * @param {Function} onError - Error callback (error message)
     * @returns {EventSource} - The EventSource for manual control
     */
    ltx2GenerateStream(params, { onProgress, onStatus, onComplete, onError }) {
        const url = new URL(`${API_BASE}/api/ltx2/generate/stream`);

        // Use fetch with POST to get the SSE stream
        const eventSource = new EventSource(`${API_BASE}/api/ltx2/generate/stream?${new URLSearchParams({})}`);

        // Note: EventSource only supports GET, so we use fetch with ReadableStream
        return this._ltx2GenerateSSE(params, { onProgress, onStatus, onComplete, onError });
    },

    async _ltx2GenerateSSE(params, { onProgress, onStatus, onComplete, onError }) {
        try {
            const response = await fetch(`${API_BASE}/api/ltx2/generate/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            });

            if (!response.ok) {
                const error = await response.text();
                onError && onError(error);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.type === 'progress') {
                                onProgress && onProgress(data);
                            } else if (data.type === 'status') {
                                onStatus && onStatus(data.message);
                            } else if (data.type === 'complete') {
                                onComplete && onComplete(data);
                            } else if (data.type === 'error') {
                                onError && onError(data.message);
                            }
                        } catch (e) {
                            console.warn('Failed to parse SSE data:', line);
                        }
                    }
                }
            }
        } catch (err) {
            onError && onError(err.message);
        }
    },

    // =========================================================================
    // Configuration
    // =========================================================================

    async getAvailableConfigs() {
        const response = await fetch(`${API_BASE}/api/configs/available`);
        return response.json();
    },

    async loadConfig(filename, profile = null) {
        const response = await fetch(`${API_BASE}/api/configs/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename, profile }),
        });
        return response.json();
    },

    async getGenerationConfig() {
        const response = await fetch(`${API_BASE}/api/generation-config`);
        return response.json();
    },

    async getRewriterConfig() {
        const response = await fetch(`${API_BASE}/api/rewriter-config`);
        return response.json();
    },

    async getResolutionConfig() {
        const response = await fetch(`${API_BASE}/api/resolution-config`);
        return response.json();
    },

    // =========================================================================
    // Templates
    // =========================================================================

    async getTemplates() {
        const response = await fetch(`${API_BASE}/api/templates`);
        return response.json();
    },

    // =========================================================================
    // Prompt Formatting & Rewriting
    // =========================================================================

    async formatPrompt(data) {
        const response = await fetch(`${API_BASE}/api/format-prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return response.json();
    },

    async getRewriters() {
        const response = await fetch(`${API_BASE}/api/rewriters`);
        return response.json();
    },

    async getRewriterModels() {
        const response = await fetch(`${API_BASE}/api/rewriter-models`);
        return response.json();
    },

    async rewritePrompt(data) {
        const response = await fetch(`${API_BASE}/api/rewrite`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return response.json();
    },

    // =========================================================================
    // History
    // =========================================================================

    async getHistory() {
        const response = await fetch(`${API_BASE}/api/history`);
        return response.json();
    },

    async deleteHistoryItem(index) {
        const response = await fetch(`${API_BASE}/api/history/${index}`, { method: 'DELETE' });
        return response.json();
    },

    async clearHistory() {
        await fetch(`${API_BASE}/api/history`, { method: 'DELETE' });
    },

    // =========================================================================
    // VL (Vision-Language) Conditioning
    // =========================================================================

    async getVLStatus() {
        const response = await fetch(`${API_BASE}/api/vl/status`);
        return response.json();
    },

    async getVLConfig() {
        const response = await fetch(`${API_BASE}/api/vl/config`);
        return response.json();
    },

    async extractVLEmbeddings(data) {
        const response = await fetch(`${API_BASE}/api/vl/extract`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return response.json();
    },

    async deleteVLCache(embeddingsId) {
        await fetch(`${API_BASE}/api/vl/cache/${embeddingsId}`, { method: 'DELETE' });
    },

    async clearVLCache() {
        const response = await fetch(`${API_BASE}/api/system/vl-cache`, { method: 'DELETE' });
        return response.json();
    },

    async generateWithVL(data) {
        const response = await fetch(`${API_BASE}/api/vl/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return this._handleImageResponse(response);
    },

    // =========================================================================
    // Z-Image Generation
    // =========================================================================

    async generate(data) {
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return this._handleImageResponse(response);
    },

    async img2img(data) {
        const response = await fetch(`${API_BASE}/api/img2img`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return this._handleImageResponse(response);
    },

    /**
     * Handle binary PNG response from generation endpoints
     * Converts to { image: "data:image/png;base64,...", seed, gen_time, history_id }
     */
    async _handleImageResponse(response) {
        if (!response.ok) {
            // Try to parse error as JSON
            try {
                const error = await response.json();
                throw new Error(error.detail || 'Generation failed');
            } catch {
                throw new Error(`Generation failed: ${response.status} ${response.statusText}`);
            }
        }

        // Read binary PNG and convert to base64
        const blob = await response.blob();
        const base64 = await this._blobToBase64(blob);

        // Extract metadata from headers
        return {
            image: base64,
            seed: response.headers.get('X-Seed'),
            gen_time: parseFloat(response.headers.get('X-Generation-Time') || '0'),
            history_id: response.headers.get('X-History-Id'),
        };
    },

    /**
     * Convert a Blob to a base64 data URL
     */
    _blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    },

    // =========================================================================
    // Qwen-Image
    // =========================================================================

    async getQwenImageStatus() {
        const response = await fetch(`${API_BASE}/api/qwen-image/status`);
        return response.json();
    },

    async qwenImageDecompose(data) {
        const response = await fetch(`${API_BASE}/api/qwen-image/decompose`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return response.json();
    },

    async qwenImageEditLayer(data) {
        const response = await fetch(`${API_BASE}/api/qwen-image/edit-layer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return response.json();
    },

    async qwenImageEditMulti(data) {
        const response = await fetch(`${API_BASE}/api/qwen-image/edit-multi`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return response.json();
    },

    // =========================================================================
    // Qwen-Image T2I (2512)
    // =========================================================================

    async getQwenImage2512Status() {
        const response = await fetch(`${API_BASE}/api/qwen-image-2512/status`);
        return response.json();
    },

    async qwenImage2512Generate(data) {
        const response = await fetch(`${API_BASE}/api/qwen-image-2512/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return this._handleImageResponse(response);
    },
};

// Export for use by other modules
window.ApiClient = ApiClient;
