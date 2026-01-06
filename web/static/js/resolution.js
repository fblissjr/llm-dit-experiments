/**
 * resolution.js - Resolution selector module
 *
 * Provides a smart resolution selector with:
 * - Width x Height numeric inputs with VAE multiple snapping
 * - Aspect ratio filter icons
 * - Quick preset chips
 * - Model-specific constraints
 * - DyPE recommendations
 */

const ResolutionSelector = {
    // ==========================================================================
    // State
    // ==========================================================================

    constraints: null,      // From /api/resolution-config
    presets: [],            // Current presets for selected model
    aspectLocked: false,    // Whether aspect ratio is locked
    aspectRatio: 1,         // Current locked ratio (width/height)
    activeFilter: 'all',    // Current aspect filter
    currentModel: 'zimage', // Current model type

    // DOM element references (cached on init)
    widthInput: null,
    heightInput: null,
    presetsContainer: null,
    dypeHint: null,
    warningEl: null,
    aspectLockBtn: null,

    // ==========================================================================
    // Constants
    // ==========================================================================

    VAE_MULTIPLE: 16,

    ASPECT_CATEGORIES: {
        'square': { minRatio: 0.95, maxRatio: 1.05 },
        'landscape': { minRatio: 1.05, maxRatio: 2.0 },
        'portrait': { minRatio: 0.5, maxRatio: 0.95 },
        'mobile-landscape': { minRatio: 2.0, maxRatio: Infinity },
        'mobile-portrait': { minRatio: 0, maxRatio: 0.5 },
    },

    // SVG icons for aspect ratio buttons
    ICONS: {
        square: `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="4" y="4" width="16" height="16" rx="2"/>
        </svg>`,
        landscape: `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="2" y="6" width="20" height="12" rx="2"/>
        </svg>`,
        portrait: `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="6" y="2" width="12" height="20" rx="2"/>
        </svg>`,
        'mobile-landscape': `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="1" y="6" width="22" height="12" rx="3"/>
            <circle cx="20" cy="12" r="1" fill="currentColor"/>
        </svg>`,
        'mobile-portrait': `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="6" y="1" width="12" height="22" rx="3"/>
            <circle cx="12" cy="20" r="1" fill="currentColor"/>
        </svg>`,
        all: `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="7" height="7" rx="1"/>
            <rect x="14" y="3" width="7" height="7" rx="1"/>
            <rect x="3" y="14" width="7" height="7" rx="1"/>
            <rect x="14" y="14" width="7" height="7" rx="1"/>
        </svg>`,
        lock: `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
            <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
        </svg>`,
        unlock: `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
            <path d="M7 11V7a5 5 0 0 1 9.9-1"/>
        </svg>`,
    },

    // ==========================================================================
    // Initialization
    // ==========================================================================

    async init() {
        // Cache DOM elements
        this.widthInput = document.getElementById('resWidth');
        this.heightInput = document.getElementById('resHeight');
        this.presetsContainer = document.getElementById('resolutionPresets');
        this.dypeHint = document.getElementById('dypeHint');
        this.warningEl = document.getElementById('resolutionWarning');
        this.aspectLockBtn = document.getElementById('aspectLockBtn');
        this.selectorContainer = document.getElementById('resolutionSelector');

        if (!this.widthInput || !this.heightInput) {
            console.warn('Resolution inputs not found');
            return;
        }

        // Setup event listeners
        this.widthInput.addEventListener('change', () => this.onWidthChange());
        this.widthInput.addEventListener('blur', () => this.onWidthBlur());
        this.heightInput.addEventListener('change', () => this.onHeightChange());
        this.heightInput.addEventListener('blur', () => this.onHeightBlur());

        if (this.aspectLockBtn) {
            this.aspectLockBtn.addEventListener('click', () => this.toggleAspectLock());
            this.updateLockIcon();
        }

        // Setup aspect filter buttons
        this.setupAspectFilters();

        // Load initial constraints
        await this.loadConstraints('zimage');
    },

    setupAspectFilters() {
        const container = document.querySelector('.aspect-filters');
        if (!container) return;

        container.querySelectorAll('.aspect-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const aspect = btn.dataset.aspect;
                this.setActiveFilter(aspect);
            });
        });
    },

    // ==========================================================================
    // Constraint Loading
    // ==========================================================================

    async loadConstraints(modelType) {
        this.currentModel = modelType || 'zimage';

        try {
            const response = await fetch(`/api/resolution-config?model=${this.currentModel}`);
            const data = await response.json();

            this.constraints = data.active_constraints || data;
            this.presets = data.presets || [];

            // Update input constraints
            this.updateInputConstraints();

            // Render presets
            this.renderPresets();

            // Check for fixed-only mode (Qwen-Image-Layered)
            this.updateFixedMode();

            // Update DyPE hint
            this.checkDypeRecommendation();

            // Set default resolution
            if (data.default_width && data.default_height) {
                this.setResolution(data.default_width, data.default_height);
            }

        } catch (err) {
            console.error('Failed to load resolution constraints:', err);
        }
    },

    updateInputConstraints() {
        if (!this.constraints) return;

        const min = this.constraints.min_resolution || 256;
        const max = this.constraints.max_resolution || 4096;

        if (this.widthInput) {
            this.widthInput.min = min;
            this.widthInput.max = max;
        }
        if (this.heightInput) {
            this.heightInput.min = min;
            this.heightInput.max = max;
        }
    },

    updateFixedMode() {
        if (!this.selectorContainer) return;

        const isFixed = this.constraints && this.constraints.flexible === false;
        this.selectorContainer.classList.toggle('fixed-only', isFixed);

        // Show fixed mode indicator
        let indicator = this.selectorContainer.querySelector('.fixed-indicator');
        if (isFixed) {
            if (!indicator) {
                indicator = document.createElement('div');
                indicator.className = 'fixed-indicator text-xs text-gray-400 mb-2';
                indicator.textContent = 'Fixed resolution model - select a preset';
                this.selectorContainer.insertBefore(indicator, this.selectorContainer.firstChild);
            }
        } else if (indicator) {
            indicator.remove();
        }
    },

    // ==========================================================================
    // Value Handling
    // ==========================================================================

    snapToMultiple(value) {
        const multiple = this.VAE_MULTIPLE;
        return Math.round(value / multiple) * multiple;
    },

    clampToConstraints(value) {
        if (!this.constraints) return value;
        const min = this.constraints.min_resolution || 256;
        const max = this.constraints.max_resolution || 4096;
        return Math.max(min, Math.min(max, value));
    },

    onWidthChange() {
        let width = parseInt(this.widthInput.value) || 1024;
        width = this.clampToConstraints(this.snapToMultiple(width));
        this.widthInput.value = width;

        if (this.aspectLocked && this.aspectRatio) {
            const height = this.snapToMultiple(width / this.aspectRatio);
            this.heightInput.value = this.clampToConstraints(height);
        }

        this.validate();
        this.updatePresetSelection();
        this.checkDypeRecommendation();
    },

    onWidthBlur() {
        // Snap on blur even if not changed
        let width = parseInt(this.widthInput.value) || 1024;
        width = this.clampToConstraints(this.snapToMultiple(width));
        this.widthInput.value = width;
    },

    onHeightChange() {
        let height = parseInt(this.heightInput.value) || 1024;
        height = this.clampToConstraints(this.snapToMultiple(height));
        this.heightInput.value = height;

        if (this.aspectLocked && this.aspectRatio) {
            const width = this.snapToMultiple(height * this.aspectRatio);
            this.widthInput.value = this.clampToConstraints(width);
        }

        this.validate();
        this.updatePresetSelection();
        this.checkDypeRecommendation();
    },

    onHeightBlur() {
        let height = parseInt(this.heightInput.value) || 1024;
        height = this.clampToConstraints(this.snapToMultiple(height));
        this.heightInput.value = height;
    },

    setResolution(width, height) {
        if (this.widthInput) this.widthInput.value = width;
        if (this.heightInput) this.heightInput.value = height;
        this.updatePresetSelection();
        this.checkDypeRecommendation();
    },

    getResolution() {
        return {
            width: parseInt(this.widthInput?.value) || 1024,
            height: parseInt(this.heightInput?.value) || 1024,
        };
    },

    // ==========================================================================
    // Aspect Lock
    // ==========================================================================

    toggleAspectLock() {
        this.aspectLocked = !this.aspectLocked;

        if (this.aspectLocked) {
            const width = parseInt(this.widthInput.value) || 1024;
            const height = parseInt(this.heightInput.value) || 1024;
            this.aspectRatio = width / height;
        }

        this.updateLockIcon();
    },

    updateLockIcon() {
        if (!this.aspectLockBtn) return;
        this.aspectLockBtn.innerHTML = this.aspectLocked ? this.ICONS.lock : this.ICONS.unlock;
        this.aspectLockBtn.title = this.aspectLocked ? 'Unlock aspect ratio' : 'Lock aspect ratio';
    },

    // ==========================================================================
    // Aspect Filters
    // ==========================================================================

    setActiveFilter(filter) {
        this.activeFilter = filter;

        // Update button states
        document.querySelectorAll('.aspect-filters .aspect-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.aspect === filter);
        });

        // Re-render presets with filter
        this.renderPresets();
    },

    getAspectCategory(width, height) {
        const ratio = width / height;

        if (ratio >= 0.95 && ratio <= 1.05) return 'square';
        if (ratio > 2.0) return 'mobile-landscape';
        if (ratio > 1.05) return 'landscape';
        if (ratio < 0.5) return 'mobile-portrait';
        return 'portrait';
    },

    // ==========================================================================
    // Presets
    // ==========================================================================

    renderPresets() {
        if (!this.presetsContainer) return;

        this.presetsContainer.innerHTML = '';

        const filtered = this.activeFilter === 'all'
            ? this.presets
            : this.presets.filter(p => {
                const cat = p.aspect_category || this.getAspectCategory(p.width, p.height);
                return cat === this.activeFilter;
            });

        filtered.forEach(preset => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'preset-chip';
            chip.dataset.width = preset.width;
            chip.dataset.height = preset.height;

            // Format label
            const label = preset.label || `${preset.width}x${preset.height}`;
            chip.textContent = label;

            // Add aspect ratio badge if not square
            if (preset.ratio && preset.ratio !== '1:1') {
                chip.textContent = `${label} (${preset.ratio})`;
            }

            chip.addEventListener('click', () => this.applyPreset(preset.width, preset.height));

            this.presetsContainer.appendChild(chip);
        });

        this.updatePresetSelection();
    },

    applyPreset(width, height) {
        this.setResolution(width, height);

        // Update aspect ratio if locked
        if (this.aspectLocked) {
            this.aspectRatio = width / height;
        }

        this.validate();
    },

    updatePresetSelection() {
        if (!this.presetsContainer) return;

        const currentWidth = parseInt(this.widthInput?.value) || 0;
        const currentHeight = parseInt(this.heightInput?.value) || 0;

        this.presetsContainer.querySelectorAll('.preset-chip').forEach(chip => {
            const w = parseInt(chip.dataset.width);
            const h = parseInt(chip.dataset.height);
            chip.classList.toggle('selected', w === currentWidth && h === currentHeight);
        });
    },

    // ==========================================================================
    // Validation
    // ==========================================================================

    validate() {
        if (!this.warningEl) return true;

        const width = parseInt(this.widthInput?.value) || 1024;
        const height = parseInt(this.heightInput?.value) || 1024;
        const errors = [];

        // Check VAE multiple
        if (width % this.VAE_MULTIPLE !== 0) {
            errors.push(`Width must be divisible by ${this.VAE_MULTIPLE}`);
        }
        if (height % this.VAE_MULTIPLE !== 0) {
            errors.push(`Height must be divisible by ${this.VAE_MULTIPLE}`);
        }

        // Check constraints
        if (this.constraints) {
            const min = this.constraints.min_resolution || 256;
            const max = this.constraints.max_resolution || 4096;

            if (width < min || width > max) {
                errors.push(`Width must be between ${min} and ${max}`);
            }
            if (height < min || height > max) {
                errors.push(`Height must be between ${min} and ${max}`);
            }

            // Check fixed resolutions (Qwen-Image-Layered)
            if (this.constraints.flexible === false && this.constraints.fixed_sizes) {
                const validSizes = this.constraints.fixed_sizes;
                if (!validSizes.includes(width) || !validSizes.includes(height)) {
                    errors.push(`This model only supports ${validSizes.join(' or ')} resolutions`);
                }
            }
        }

        // Display errors
        if (errors.length > 0) {
            this.warningEl.textContent = errors.join('. ');
            this.warningEl.classList.remove('hidden');
            return false;
        } else {
            this.warningEl.classList.add('hidden');
            return true;
        }
    },

    // ==========================================================================
    // DyPE Recommendation
    // ==========================================================================

    checkDypeRecommendation() {
        if (!this.dypeHint) return;

        // Only show for Z-Image
        if (this.currentModel !== 'zimage') {
            this.dypeHint.classList.add('hidden');
            return;
        }

        const width = parseInt(this.widthInput?.value) || 1024;
        const height = parseInt(this.heightInput?.value) || 1024;
        const maxDim = Math.max(width, height);
        const baseRes = 1024;

        if (maxDim > baseRes) {
            const scale = maxDim / baseRes;
            let exponent = 0.5;
            if (scale >= 3.0) exponent = 2.0;
            else if (scale >= 1.5) exponent = 1.0;

            this.dypeHint.innerHTML = `DyPE recommended for this resolution (exponent: ${exponent})`;
            this.dypeHint.classList.remove('hidden');
        } else {
            this.dypeHint.classList.add('hidden');
        }
    },
};

// Export
window.ResolutionSelector = ResolutionSelector;
