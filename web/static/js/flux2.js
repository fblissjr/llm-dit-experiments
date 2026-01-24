/**
 * FLUX.2 Klein Image Generation Module
 *
 * Handles:
 * - Image generation with progress tracking
 * - Form submission and validation
 * - Reference image uploads for editing mode
 * - Model variant switching with auto-defaults
 * - Image display and download
 */

// =============================================================================
// State
// =============================================================================

let flux2IsGenerating = false;
let flux2ReferenceImages = [];  // Array of base64 encoded images
let flux2CurrentImageUrl = null;

// Model defaults - distilled vs base models have different optimal parameters
const FLUX2_MODEL_DEFAULTS = {
    // Distilled models: fast, 4 steps, CFG=1.0 (baked in)
    'klein-9b': { steps: 4, guidance: 1.0, distilled: true },
    'klein-9b-fp8': { steps: 4, guidance: 1.0, distilled: true },
    'klein-4b': { steps: 4, guidance: 1.0, distilled: true },
    'klein-4b-fp8': { steps: 4, guidance: 1.0, distilled: true },
    // Base models: quality, 50 steps, CFG=4.0
    'klein-base-9b': { steps: 50, guidance: 4.0, distilled: false },
    'klein-base-9b-fp8': { steps: 50, guidance: 4.0, distilled: false },
    'klein-base-4b': { steps: 50, guidance: 4.0, distilled: false },
    'klein-base-4b-fp8': { steps: 50, guidance: 4.0, distilled: false },
};

// =============================================================================
// Initialization
// =============================================================================

function initFlux2() {
    setupFlux2Form();
    setupFlux2Controls();
    setupFlux2RefUpload();
    setupFlux2Downloads();
    checkFlux2Status();
}

// =============================================================================
// Status Check
// =============================================================================

async function checkFlux2Status() {
    try {
        const status = await ApiClient.getFlux2Status();
        const statusEl = document.getElementById('flux2Status');

        if (statusEl) {
            if (status.available) {
                statusEl.classList.add('hidden');
            } else {
                statusEl.classList.remove('hidden');
            }
        }

        // Update button state
        const flux2Btn = document.getElementById('modelTypeFLUX2');
        if (flux2Btn && !status.available) {
            flux2Btn.classList.add('opacity-50');
            flux2Btn.title = 'FLUX.2 not available. Configure flux2 settings.';
        }

        return status;
    } catch (err) {
        console.warn('Failed to check FLUX.2 status:', err);
        return { available: false };
    }
}

// =============================================================================
// Form Setup
// =============================================================================

function setupFlux2Form() {
    const form = document.getElementById('flux2Form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await generateFlux2Image();
    });
}

function setupFlux2Controls() {
    // Model variant selector - update defaults on change
    const modelSelect = document.getElementById('flux2Model');
    if (modelSelect) {
        modelSelect.addEventListener('change', () => {
            updateFlux2Defaults(modelSelect.value);
        });
    }

    // Steps slider
    const stepsSlider = document.getElementById('flux2Steps');
    const stepsValue = document.getElementById('flux2StepsValue');
    if (stepsSlider && stepsValue) {
        stepsSlider.addEventListener('input', () => {
            stepsValue.textContent = stepsSlider.value;
        });
    }

    // Guidance slider
    const guidanceSlider = document.getElementById('flux2Guidance');
    const guidanceValue = document.getElementById('flux2GuidanceValue');
    if (guidanceSlider && guidanceValue) {
        guidanceSlider.addEventListener('input', () => {
            guidanceValue.textContent = parseFloat(guidanceSlider.value).toFixed(1);
        });
    }

    // Random seed button
    const randomSeedBtn = document.getElementById('flux2RandomSeed');
    const seedInput = document.getElementById('flux2Seed');
    if (randomSeedBtn && seedInput) {
        randomSeedBtn.addEventListener('click', () => {
            seedInput.value = '';
            seedInput.placeholder = 'Random';
        });
    }
}

function updateFlux2Defaults(modelName) {
    const defaults = FLUX2_MODEL_DEFAULTS[modelName];
    if (!defaults) return;

    // Update steps slider and display
    const stepsSlider = document.getElementById('flux2Steps');
    const stepsValue = document.getElementById('flux2StepsValue');
    if (stepsSlider) {
        stepsSlider.value = defaults.steps;
        if (stepsValue) {
            stepsValue.textContent = defaults.steps;
        }
    }

    // Update guidance slider and display
    const guidanceSlider = document.getElementById('flux2Guidance');
    const guidanceValue = document.getElementById('flux2GuidanceValue');
    if (guidanceSlider) {
        guidanceSlider.value = defaults.guidance;
        if (guidanceValue) {
            guidanceValue.textContent = defaults.guidance.toFixed(1);
        }
    }

    // Update info text
    const infoEl = document.getElementById('flux2ModelInfo');
    if (infoEl) {
        if (defaults.distilled) {
            infoEl.textContent = 'Distilled model: Fast generation with built-in CFG';
        } else {
            infoEl.textContent = 'Base model: Higher quality, requires more steps';
        }
    }
}

// =============================================================================
// Reference Image Upload
// =============================================================================

function setupFlux2RefUpload() {
    const dropzone = document.getElementById('flux2RefDropzone');
    const input = document.getElementById('flux2RefInput');
    const clearBtn = document.getElementById('flux2ClearRefs');

    if (!dropzone || !input) return;

    // Click to open file picker
    dropzone.addEventListener('click', () => input.click());

    // Drag and drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('border-orange-500');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('border-orange-500');
    });

    dropzone.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropzone.classList.remove('border-orange-500');
        await handleFlux2RefFiles(e.dataTransfer.files);
    });

    // File input change
    input.addEventListener('change', async (e) => {
        await handleFlux2RefFiles(e.target.files);
    });

    // Clear button
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            flux2ReferenceImages = [];
            renderFlux2RefPreview();
        });
    }
}

async function handleFlux2RefFiles(files) {
    for (const file of files) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            console.warn('Skipping non-image file:', file.name);
            continue;
        }

        // Check limit (max 4 reference images)
        if (flux2ReferenceImages.length >= 4) {
            alert('Maximum 4 reference images allowed');
            break;
        }

        try {
            const base64 = await flux2FileToBase64(file);
            flux2ReferenceImages.push({
                name: file.name,
                data: base64,
            });
        } catch (err) {
            console.error('Failed to read file:', file.name, err);
        }
    }

    renderFlux2RefPreview();
}

function renderFlux2RefPreview() {
    const preview = document.getElementById('flux2RefPreview');
    const clearBtn = document.getElementById('flux2ClearRefs');

    if (!preview) return;

    // Clear existing content
    while (preview.firstChild) {
        preview.removeChild(preview.firstChild);
    }

    if (flux2ReferenceImages.length === 0) {
        preview.classList.add('hidden');
        if (clearBtn) clearBtn.classList.add('hidden');
        return;
    }

    preview.classList.remove('hidden');
    if (clearBtn) clearBtn.classList.remove('hidden');

    // Build preview using DOM methods
    flux2ReferenceImages.forEach((img, idx) => {
        const container = document.createElement('div');
        container.className = 'relative group';

        const imgEl = document.createElement('img');
        imgEl.src = img.data;
        imgEl.className = 'w-20 h-20 object-cover rounded-lg border border-gray-600';
        imgEl.alt = img.name;

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'absolute -top-2 -right-2 w-5 h-5 bg-red-600 hover:bg-red-700 rounded-full text-white text-xs flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity';
        removeBtn.textContent = '×';
        removeBtn.addEventListener('click', () => removeFlux2Ref(idx));

        const nameEl = document.createElement('p');
        nameEl.className = 'text-xs text-gray-500 mt-1 truncate max-w-[80px]';
        nameEl.textContent = img.name;

        container.appendChild(imgEl);
        container.appendChild(removeBtn);
        container.appendChild(nameEl);
        preview.appendChild(container);
    });
}

function removeFlux2Ref(index) {
    flux2ReferenceImages.splice(index, 1);
    renderFlux2RefPreview();
}

// Helper function to convert file to base64
function flux2FileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// =============================================================================
// Image Generation
// =============================================================================

async function generateFlux2Image() {
    if (flux2IsGenerating) return;

    const prompt = document.getElementById('flux2Prompt')?.value?.trim();
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    // Get parameters
    const modelName = document.getElementById('flux2Model')?.value || 'klein-9b-fp8';
    const resolution = document.getElementById('flux2Resolution')?.value || '1024x1024';
    const [width, height] = resolution.split('x').map(Number);
    const steps = parseInt(document.getElementById('flux2Steps')?.value || '4');
    const guidance = parseFloat(document.getElementById('flux2Guidance')?.value || '1.0');
    const seedInput = document.getElementById('flux2Seed')?.value;
    const seed = seedInput ? parseInt(seedInput) : null;
    const blockOffload = document.getElementById('flux2BlockOffload')?.checked || false;
    const modelPath = document.getElementById('flux2ModelPath')?.value?.trim() || null;
    const vaePath = document.getElementById('flux2VaePath')?.value?.trim() || null;

    const params = {
        prompt,
        model_name: modelName,
        width,
        height,
        num_steps: steps,
        guidance,
        seed,
        block_offload: blockOffload,
        model_path: modelPath,
        vae_path: vaePath,
        reference_images: flux2ReferenceImages.length > 0
            ? flux2ReferenceImages.map(img => img.data)
            : null,
    };

    // Update UI state
    flux2IsGenerating = true;
    setFlux2ButtonState(true);
    showFlux2Progress();
    hideFlux2Result();

    try {
        // Show initial progress
        updateFlux2Progress({ status: 'Loading model...', percent: 10 });

        const result = await ApiClient.flux2Generate(params);

        handleFlux2Complete(result);
    } catch (err) {
        handleFlux2Error(err.message || 'Generation failed');
    }
}

// =============================================================================
// Progress Handling
// =============================================================================

function updateFlux2Progress(data) {
    const progressBar = document.getElementById('flux2ProgressBar');
    const progressStatus = document.getElementById('flux2ProgressStatus');
    const progressDetail = document.getElementById('flux2ProgressDetail');

    if (progressBar && data.percent !== undefined) {
        progressBar.style.width = data.percent + '%';
    }

    if (progressStatus && data.status) {
        progressStatus.textContent = data.status;
    }

    if (progressDetail && data.detail) {
        progressDetail.textContent = data.detail;
    }
}

function handleFlux2Complete(data) {
    flux2IsGenerating = false;
    setFlux2ButtonState(false);
    hideFlux2Progress();

    // Store and display image
    flux2CurrentImageUrl = data.image;
    showFlux2Result(data);
}

function handleFlux2Error(message) {
    flux2IsGenerating = false;
    setFlux2ButtonState(false);
    hideFlux2Progress();

    alert('Image generation failed: ' + message);
}

// =============================================================================
// UI State Management
// =============================================================================

function setFlux2ButtonState(generating) {
    const btn = document.getElementById('flux2GenerateBtn');
    if (!btn) return;

    // Clear existing content
    while (btn.firstChild) {
        btn.removeChild(btn.firstChild);
    }

    if (generating) {
        btn.disabled = true;
        // Create spinner using DOM methods
        const spinner = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        spinner.setAttribute('class', 'animate-spin w-5 h-5');
        spinner.setAttribute('fill', 'none');
        spinner.setAttribute('viewBox', '0 0 24 24');

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('class', 'opacity-25');
        circle.setAttribute('cx', '12');
        circle.setAttribute('cy', '12');
        circle.setAttribute('r', '10');
        circle.setAttribute('stroke', 'currentColor');
        circle.setAttribute('stroke-width', '4');

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'opacity-75');
        path.setAttribute('fill', 'currentColor');
        path.setAttribute('d', 'M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z');

        spinner.appendChild(circle);
        spinner.appendChild(path);
        btn.appendChild(spinner);
        btn.appendChild(document.createTextNode(' Generating...'));
    } else {
        btn.disabled = false;
        // Create icon using DOM methods
        const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        icon.setAttribute('class', 'w-5 h-5');
        icon.setAttribute('fill', 'none');
        icon.setAttribute('stroke', 'currentColor');
        icon.setAttribute('viewBox', '0 0 24 24');

        const path1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path1.setAttribute('stroke-linecap', 'round');
        path1.setAttribute('stroke-linejoin', 'round');
        path1.setAttribute('stroke-width', '2');
        path1.setAttribute('d', 'M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z');

        icon.appendChild(path1);
        btn.appendChild(icon);
        btn.appendChild(document.createTextNode(' Generate Image'));
    }
}

function showFlux2Progress() {
    const progress = document.getElementById('flux2Progress');
    if (progress) {
        progress.classList.remove('hidden');
        // Reset progress bar
        const bar = document.getElementById('flux2ProgressBar');
        if (bar) bar.style.width = '0%';
    }
}

function hideFlux2Progress() {
    const progress = document.getElementById('flux2Progress');
    if (progress) {
        progress.classList.add('hidden');
    }
}

function showFlux2Result(data) {
    const result = document.getElementById('flux2Result');
    const img = document.getElementById('flux2ResultImg');
    const timeEl = document.getElementById('flux2ResultTime');
    const infoEl = document.getElementById('flux2ResultInfo');

    if (result) {
        result.classList.remove('hidden');
    }

    if (img && data.image) {
        img.src = data.image;
        // Click to view full size in lightbox
        img.onclick = () => {
            if (typeof openImageModal === 'function') {
                openImageModal(data.image);
            }
        };
    }

    if (timeEl && data.gen_time) {
        timeEl.textContent = 'Generated in ' + data.gen_time.toFixed(1) + 's';
    }

    if (infoEl) {
        const seedText = data.seed ? ('Seed: ' + data.seed) : 'Random seed';
        infoEl.textContent = seedText;
    }
}

function hideFlux2Result() {
    const result = document.getElementById('flux2Result');
    if (result) {
        result.classList.add('hidden');
    }
}

// =============================================================================
// Downloads
// =============================================================================

function setupFlux2Downloads() {
    const downloadBtn = document.getElementById('flux2Download');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            if (flux2CurrentImageUrl) {
                downloadFlux2Image(flux2CurrentImageUrl);
            }
        });
    }
}

function downloadFlux2Image(dataUrl) {
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = 'flux2_' + Date.now() + '.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// =============================================================================
// Exports
// =============================================================================

window.initFlux2 = initFlux2;
window.checkFlux2Status = checkFlux2Status;
window.generateFlux2Image = generateFlux2Image;
window.removeFlux2Ref = removeFlux2Ref;
window.updateFlux2Defaults = updateFlux2Defaults;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', initFlux2);
