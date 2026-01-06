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
