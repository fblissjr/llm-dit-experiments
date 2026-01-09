/**
 * LTX-2 Video Generation Module
 *
 * Handles:
 * - Video generation with SSE progress tracking
 * - Form submission and validation
 * - Video playback and download
 * - Status checking
 */

// =============================================================================
// State
// =============================================================================

let isGenerating = false;
let currentVideoUrl = null;

// =============================================================================
// Initialization
// =============================================================================

function initLtx2() {
    setupLtx2Form();
    setupLtx2Controls();
    setupLtx2Downloads();
    checkLtx2Status();
}

// =============================================================================
// Status Check
// =============================================================================

async function checkLtx2Status() {
    try {
        const status = await ApiClient.getLtx2Status();
        const statusEl = document.getElementById('ltx2Status');

        if (statusEl) {
            if (status.available) {
                statusEl.classList.add('hidden');
            } else {
                statusEl.classList.remove('hidden');
            }
        }

        // Update button state
        const ltx2Btn = document.getElementById('modelTypeLTX2');
        if (ltx2Btn && !status.available) {
            ltx2Btn.classList.add('opacity-50');
            ltx2Btn.title = 'LTX-2 not available. Configure ltx2.model_path in config.toml';
        }

        return status;
    } catch (err) {
        console.warn('Failed to check LTX-2 status:', err);
        return { available: false, loaded: false };
    }
}

// =============================================================================
// Form Setup
// =============================================================================

function setupLtx2Form() {
    const form = document.getElementById('ltx2Form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await generateVideo();
    });
}

function setupLtx2Controls() {
    // Guidance slider
    const guidanceSlider = document.getElementById('ltx2Guidance');
    const guidanceValue = document.getElementById('ltx2GuidanceValue');
    if (guidanceSlider && guidanceValue) {
        guidanceSlider.addEventListener('input', () => {
            guidanceValue.textContent = guidanceSlider.value;
        });
    }

    // Random seed button
    const randomSeedBtn = document.getElementById('ltx2RandomSeed');
    const seedInput = document.getElementById('ltx2Seed');
    if (randomSeedBtn && seedInput) {
        randomSeedBtn.addEventListener('click', () => {
            seedInput.value = '';
            seedInput.placeholder = 'Random';
        });
    }
}

function setupLtx2Downloads() {
    const downloadBtn = document.getElementById('ltx2Download');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            if (currentVideoUrl) {
                downloadVideo(currentVideoUrl);
            }
        });
    }
}

// =============================================================================
// Video Generation
// =============================================================================

async function generateVideo() {
    if (isGenerating) return;

    const prompt = document.getElementById('ltx2Prompt')?.value?.trim();
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    // Get parameters
    const negativePrompt = document.getElementById('ltx2NegativePrompt')?.value || '';
    const resolution = document.getElementById('ltx2Resolution')?.value || '768x512';
    const [width, height] = resolution.split('x').map(Number);
    const numFrames = parseInt(document.getElementById('ltx2Frames')?.value || '33');
    const fps = parseFloat(document.getElementById('ltx2Fps')?.value || '24');
    const steps = parseInt(document.getElementById('ltx2Steps')?.value || '12');
    const guidance = parseFloat(document.getElementById('ltx2Guidance')?.value || '3.5');
    const seedInput = document.getElementById('ltx2Seed')?.value;
    const seed = seedInput ? parseInt(seedInput) : null;
    const enableAudio = document.getElementById('ltx2Audio')?.checked || false;

    const params = {
        prompt,
        negative_prompt: negativePrompt,
        width,
        height,
        num_frames: numFrames,
        fps,
        num_inference_steps: steps,
        guidance_scale: guidance,
        seed,
        enable_audio: enableAudio,
    };

    // Update UI state
    isGenerating = true;
    setGenerateButtonState(true);
    showProgress();
    hideResult();

    try {
        await ApiClient._ltx2GenerateSSE(params, {
            onProgress: handleProgress,
            onStatus: handleStatus,
            onComplete: handleComplete,
            onError: handleError,
        });
    } catch (err) {
        handleError(err.message);
    }
}

// =============================================================================
// Progress Handling
// =============================================================================

function handleProgress(data) {
    const { step, total, elapsed, eta, its } = data;

    const progressBar = document.getElementById('ltx2ProgressBar');
    const progressStatus = document.getElementById('ltx2ProgressStatus');
    const progressEta = document.getElementById('ltx2ProgressEta');
    const progressDetail = document.getElementById('ltx2ProgressDetail');

    if (progressBar) {
        const pct = (step / total) * 100;
        progressBar.style.width = `${pct}%`;
    }

    if (progressStatus) {
        progressStatus.textContent = `Step ${step}/${total}`;
    }

    if (progressEta && eta !== undefined) {
        progressEta.textContent = `ETA: ${formatTime(eta)}`;
    }

    if (progressDetail && its !== undefined) {
        progressDetail.textContent = `${its.toFixed(2)} it/s | Elapsed: ${formatTime(elapsed)}`;
    }
}

function handleStatus(message) {
    const progressStatus = document.getElementById('ltx2ProgressStatus');
    const progressDetail = document.getElementById('ltx2ProgressDetail');

    if (progressStatus) {
        progressStatus.textContent = message;
    }

    if (progressDetail) {
        progressDetail.textContent = '';
    }
}

function handleComplete(data) {
    isGenerating = false;
    setGenerateButtonState(false);
    hideProgress();

    // Display video
    currentVideoUrl = data.video_url;
    showResult(data);
}

function handleError(message) {
    isGenerating = false;
    setGenerateButtonState(false);
    hideProgress();

    alert(`Video generation failed: ${message}`);
}

// =============================================================================
// UI State Management
// =============================================================================

function setGenerateButtonState(generating) {
    const btn = document.getElementById('ltx2GenerateBtn');
    if (!btn) return;

    if (generating) {
        btn.disabled = true;
        // Clear existing content and add loading state using DOM methods
        btn.textContent = '';
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
        // Clear and reset to default state using DOM methods
        btn.textContent = '';
        const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        icon.setAttribute('class', 'w-5 h-5');
        icon.setAttribute('fill', 'none');
        icon.setAttribute('stroke', 'currentColor');
        icon.setAttribute('viewBox', '0 0 24 24');
        const path1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path1.setAttribute('stroke-linecap', 'round');
        path1.setAttribute('stroke-linejoin', 'round');
        path1.setAttribute('stroke-width', '2');
        path1.setAttribute('d', 'M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z');
        const path2 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path2.setAttribute('stroke-linecap', 'round');
        path2.setAttribute('stroke-linejoin', 'round');
        path2.setAttribute('stroke-width', '2');
        path2.setAttribute('d', 'M21 12a9 9 0 11-18 0 9 9 0 0118 0z');
        icon.appendChild(path1);
        icon.appendChild(path2);
        btn.appendChild(icon);
        btn.appendChild(document.createTextNode(' Generate Video'));
    }
}

function showProgress() {
    const progress = document.getElementById('ltx2Progress');
    if (progress) {
        progress.classList.remove('hidden');
        // Reset progress bar
        const bar = document.getElementById('ltx2ProgressBar');
        if (bar) bar.style.width = '0%';
    }
}

function hideProgress() {
    const progress = document.getElementById('ltx2Progress');
    if (progress) {
        progress.classList.add('hidden');
    }
}

function showResult(data) {
    const result = document.getElementById('ltx2Result');
    const video = document.getElementById('ltx2Video');
    const timeEl = document.getElementById('ltx2ResultTime');
    const infoEl = document.getElementById('ltx2ResultInfo');
    const audioBtn = document.getElementById('ltx2DownloadAudio');

    if (result) {
        result.classList.remove('hidden');
    }

    if (video) {
        video.src = data.video_url;
        video.load();
    }

    if (timeEl) {
        timeEl.textContent = `Generated in ${data.generation_time}s`;
    }

    if (infoEl) {
        infoEl.textContent = `${data.num_frames} frames @ ${data.fps}fps | Seed: ${data.seed}`;
    }

    if (audioBtn) {
        if (data.has_audio) {
            audioBtn.classList.remove('hidden');
        } else {
            audioBtn.classList.add('hidden');
        }
    }
}

function hideResult() {
    const result = document.getElementById('ltx2Result');
    if (result) {
        result.classList.add('hidden');
    }
}

// =============================================================================
// Utilities
// =============================================================================

function formatTime(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(1)}s`;
    } else {
        const mins = Math.floor(seconds / 60);
        const secs = Math.round(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
}

function downloadVideo(url) {
    const a = document.createElement('a');
    a.href = url;
    a.download = url.split('/').pop() || 'video.mp4';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// =============================================================================
// Exports
// =============================================================================

window.initLtx2 = initLtx2;
window.checkLtx2Status = checkLtx2Status;
window.generateVideo = generateVideo;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', initLtx2);
