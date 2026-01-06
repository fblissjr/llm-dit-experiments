/**
 * config-manager.js - Configuration management UI
 *
 * Handles:
 * - Settings modal tab switching
 * - Session config editing (hot-reload safe params)
 * - Profile listing and selection
 * - Server status and restart
 */

const ConfigManager = {
    // Current session values from server
    sessionValues: {},
    // Original values from file (for detecting changes)
    fileValues: {},
    // Fields modified in this session
    modifiedFields: new Set(),
    // Available profiles
    profiles: [],
    // Current profile name
    currentProfile: 'default',

    // Slider config: maps field names to DOM elements and formatting
    sliderConfig: {
        shift: { slider: 'configShiftSlider', display: 'configShiftValue', decimals: 1 },
        d_noise: { slider: 'configDNoiseSlider', display: 'configDNoiseValue', decimals: 2 },
        steps: { slider: 'configStepsSlider', display: 'configStepsValue', decimals: 0 },
        guidance_scale: { slider: 'configGuidanceSlider', display: 'configGuidanceValue', decimals: 1 },
    },

    /**
     * Initialize the config manager
     */
    async init() {
        console.log('[ConfigManager] Initializing...');
        this.setupTabs();
        this.setupSliderListeners();
        this.setupButtonListeners();

        // Load initial data
        await Promise.all([
            this.loadSessionConfig(),
            this.loadProfiles(),
            this.loadServerStatus(),
        ]);

        console.log('[ConfigManager] Initialized');
    },

    /**
     * Set up tab switching
     */
    setupTabs() {
        document.querySelectorAll('.settings-tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchTab(tab.dataset.tab));
        });
    },

    /**
     * Switch to a tab
     */
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.settings-tab').forEach(t => {
            const isActive = t.dataset.tab === tabName;
            t.classList.toggle('border-blue-500', isActive);
            t.classList.toggle('text-white', isActive);
            t.classList.toggle('border-transparent', !isActive);
            t.classList.toggle('text-gray-400', !isActive);
        });

        // Update tab content
        document.querySelectorAll('.settings-tab-content').forEach(c => {
            const tabId = c.id.replace('settingsTab', '').toLowerCase();
            c.classList.toggle('hidden', tabId !== tabName);
        });

        // Refresh data when switching to certain tabs
        if (tabName === 'profiles') {
            this.loadProfiles();
        } else if (tabName === 'server') {
            this.loadServerStatus();
        }
    },

    /**
     * Set up slider input listeners
     */
    setupSliderListeners() {
        Object.entries(this.sliderConfig).forEach(([field, config]) => {
            const slider = document.getElementById(config.slider);
            const display = document.getElementById(config.display);

            if (slider && display) {
                slider.addEventListener('input', () => {
                    const value = parseFloat(slider.value);
                    display.textContent = value.toFixed(config.decimals);
                    this.markModified(field, value);
                });
            }
        });
    },

    /**
     * Set up button click listeners
     */
    setupButtonListeners() {
        // Apply session button
        const applyBtn = document.getElementById('applySessionBtn');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => this.applyToSession());
        }

        // Reset button
        const resetBtn = document.getElementById('resetSessionBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetSession());
        }

        // Refresh profiles button
        const refreshBtn = document.getElementById('refreshProfilesBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadProfiles());
        }

        // Restart server button
        const restartBtn = document.getElementById('restartServerBtn');
        if (restartBtn) {
            restartBtn.addEventListener('click', () => this.restartServer());
        }
    },

    /**
     * Mark a field as modified
     */
    markModified(field, value) {
        this.sessionValues[field] = value;

        // Check if value differs from file value
        if (this.fileValues[field] !== value) {
            this.modifiedFields.add(field);
        } else {
            this.modifiedFields.delete(field);
        }

        this.updateModifiedIndicator();
    },

    /**
     * Update the modified indicator banner
     */
    updateModifiedIndicator() {
        const banner = document.getElementById('configModifiedBanner');
        if (banner) {
            banner.classList.toggle('hidden', this.modifiedFields.size === 0);
        }
    },

    /**
     * Load session config from server
     */
    async loadSessionConfig() {
        try {
            const response = await fetch('/api/config/session');
            const data = await response.json();

            this.sessionValues = data.values || {};
            this.fileValues = { ...this.sessionValues };
            this.currentProfile = data.profile || 'default';
            this.modifiedFields = new Set(data.modified || []);

            // Update UI
            this.updateConfigUI();
            this.updateModifiedIndicator();

            // Update profile badge
            const profileBadge = document.getElementById('sessionProfile');
            if (profileBadge) {
                profileBadge.textContent = this.currentProfile;
            }

            console.log('[ConfigManager] Session config loaded:', this.currentProfile);
        } catch (error) {
            console.error('[ConfigManager] Failed to load session config:', error);
        }
    },

    /**
     * Update config UI sliders with current values
     */
    updateConfigUI() {
        Object.entries(this.sliderConfig).forEach(([field, config]) => {
            const slider = document.getElementById(config.slider);
            const display = document.getElementById(config.display);
            const value = this.sessionValues[field];

            if (slider && display && value !== undefined) {
                slider.value = value;
                display.textContent = parseFloat(value).toFixed(config.decimals);
            }
        });
    },

    /**
     * Apply current slider values to session
     */
    async applyToSession() {
        if (this.modifiedFields.size === 0) {
            console.log('[ConfigManager] No changes to apply');
            return;
        }

        const updates = {};
        this.modifiedFields.forEach(field => {
            updates[field] = this.sessionValues[field];
        });

        try {
            const response = await fetch('/api/config/session', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updates),
            });

            const result = await response.json();

            if (result.success) {
                // Update file values to match (session is now synced)
                result.updated.forEach(field => {
                    this.fileValues[field] = this.sessionValues[field];
                    this.modifiedFields.delete(field);
                });

                this.updateModifiedIndicator();
                console.log('[ConfigManager] Applied to session:', result.updated);

                // Show success message
                this.showMessage('Session updated successfully', 'success');

                // Also update the main UI sliders if they exist
                this.syncMainUISliders();
            }

            if (result.pending_restart && result.pending_restart.length > 0) {
                console.log('[ConfigManager] Changes pending restart:', result.pending_restart);
            }
        } catch (error) {
            console.error('[ConfigManager] Failed to apply session:', error);
            this.showMessage('Failed to apply changes', 'error');
        }
    },

    /**
     * Sync config changes to main UI sliders
     */
    syncMainUISliders() {
        // Map config fields to main UI slider IDs
        const mainUIMap = {
            shift: 'shiftSlider',
            d_noise: 'dNoiseSlider',
            steps: 'stepsSlider',
            guidance_scale: 'guidanceScaleSlider',
        };

        Object.entries(mainUIMap).forEach(([field, sliderId]) => {
            const slider = document.getElementById(sliderId);
            const value = this.sessionValues[field];

            if (slider && value !== undefined) {
                slider.value = value;
                // Trigger input event to update display
                slider.dispatchEvent(new Event('input'));
            }
        });
    },

    /**
     * Reset session to file values
     */
    resetSession() {
        this.sessionValues = { ...this.fileValues };
        this.modifiedFields.clear();
        this.updateConfigUI();
        this.updateModifiedIndicator();
        console.log('[ConfigManager] Reset to file values');
    },

    /**
     * Load profiles list from server
     */
    async loadProfiles() {
        try {
            const response = await fetch('/api/config/profiles');
            const data = await response.json();

            this.profiles = data.profiles || [];
            this.currentProfile = data.current || 'default';

            // Update profile list UI
            this.renderProfileList();

            // Update config file path
            const configPath = document.getElementById('configFilePath');
            if (configPath) {
                configPath.textContent = data.config_file || 'Not loaded';
            }

            // Update restart profile selector
            this.updateRestartProfileSelector();

            console.log('[ConfigManager] Profiles loaded:', this.profiles);
        } catch (error) {
            console.error('[ConfigManager] Failed to load profiles:', error);
        }
    },

    /**
     * Render the profile list UI
     */
    renderProfileList() {
        const container = document.getElementById('profileList');
        if (!container) return;

        if (this.profiles.length === 0) {
            container.innerHTML = '<div class="text-sm text-gray-400 text-center py-4">No profiles found</div>';
            return;
        }

        container.innerHTML = this.profiles.map(profile => `
            <div class="p-3 bg-gray-900 rounded-lg flex items-center justify-between">
                <div class="flex items-center gap-2">
                    ${profile === this.currentProfile ?
                        '<span class="w-2 h-2 rounded-full bg-green-500"></span>' :
                        '<span class="w-2 h-2 rounded-full bg-gray-600"></span>'}
                    <span class="text-sm text-gray-300">${profile}</span>
                    ${profile === this.currentProfile ?
                        '<span class="text-xs text-green-400">(active)</span>' : ''}
                </div>
                <div class="flex gap-2">
                    ${profile !== this.currentProfile ? `
                        <button class="profile-load-btn text-xs px-2 py-1 rounded bg-blue-600/20 text-blue-400 hover:bg-blue-600/30"
                                data-profile="${profile}">
                            Load
                        </button>
                    ` : ''}
                </div>
            </div>
        `).join('');

        // Add click handlers for load buttons
        container.querySelectorAll('.profile-load-btn').forEach(btn => {
            btn.addEventListener('click', () => this.loadProfile(btn.dataset.profile));
        });
    },

    /**
     * Load a profile (requires restart)
     */
    async loadProfile(profileName) {
        if (confirm(`Load profile "${profileName}"? This will restart the server.`)) {
            await this.restartServer(profileName);
        }
    },

    /**
     * Update the restart profile selector
     */
    updateRestartProfileSelector() {
        const select = document.getElementById('restartProfileSelect');
        if (!select) return;

        // Keep first option (Current profile)
        const firstOption = select.options[0];
        select.innerHTML = '';
        select.appendChild(firstOption);

        // Add profile options
        this.profiles.forEach(profile => {
            const option = document.createElement('option');
            option.value = profile;
            option.textContent = profile + (profile === this.currentProfile ? ' (current)' : '');
            select.appendChild(option);
        });
    },

    /**
     * Load server status
     */
    async loadServerStatus() {
        try {
            const response = await fetch('/api/server/status');
            const data = await response.json();

            // Update status display
            const statusText = document.getElementById('serverStatusText');
            if (statusText) {
                statusText.textContent = data.status || 'Unknown';
                statusText.className = data.status === 'running' ?
                    'text-green-400' : 'text-yellow-400';
            }

            // Update uptime
            const uptimeEl = document.getElementById('serverUptime');
            if (uptimeEl && data.uptime_seconds !== null) {
                uptimeEl.textContent = this.formatUptime(data.uptime_seconds);
            }

            // Update profile
            const profileEl = document.getElementById('serverProfile');
            if (profileEl) {
                profileEl.textContent = data.profile || 'default';
            }

            // Update pending changes
            const pendingContainer = document.getElementById('pendingChanges');
            const pendingList = document.getElementById('pendingChangesList');
            if (pendingContainer && pendingList) {
                const pending = Object.entries(data.pending_restart || {});
                if (pending.length > 0) {
                    pendingContainer.classList.remove('hidden');
                    pendingList.innerHTML = pending.map(([k, v]) =>
                        `<li>${k}: ${v}</li>`
                    ).join('');
                } else {
                    pendingContainer.classList.add('hidden');
                }
            }

            console.log('[ConfigManager] Server status loaded');
        } catch (error) {
            console.error('[ConfigManager] Failed to load server status:', error);
        }
    },

    /**
     * Format uptime seconds as human-readable string
     */
    formatUptime(seconds) {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    },

    /**
     * Restart the server
     */
    async restartServer(newProfile = null) {
        const profileSelect = document.getElementById('restartProfileSelect');
        const selectedProfile = newProfile || (profileSelect ? profileSelect.value : null);

        const confirmMsg = selectedProfile && selectedProfile !== this.currentProfile ?
            `Restart server with profile "${selectedProfile}"?` :
            'Restart the server?';

        if (!confirm(confirmMsg + '\n\nThis will interrupt any active generation.')) {
            return;
        }

        try {
            this.showMessage('Restarting server...', 'info');

            const response = await fetch('/api/server/restart', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    reason: 'user_request',
                    new_profile: selectedProfile || null,
                }),
            });

            const result = await response.json();

            if (result.success) {
                this.showMessage('Server restarting, please wait...', 'success');

                // Poll for server to come back
                setTimeout(() => this.waitForServer(), 2000);
            }
        } catch (error) {
            console.error('[ConfigManager] Failed to restart server:', error);
            this.showMessage('Failed to restart server', 'error');
        }
    },

    /**
     * Wait for server to come back after restart
     */
    async waitForServer(attempts = 0) {
        if (attempts > 30) {
            this.showMessage('Server did not come back. Please refresh.', 'error');
            return;
        }

        try {
            const response = await fetch('/health');
            if (response.ok) {
                this.showMessage('Server restarted successfully!', 'success');
                // Reload data
                await Promise.all([
                    this.loadSessionConfig(),
                    this.loadProfiles(),
                    this.loadServerStatus(),
                ]);
                return;
            }
        } catch (e) {
            // Server not ready yet
        }

        setTimeout(() => this.waitForServer(attempts + 1), 1000);
    },

    /**
     * Show a temporary message
     */
    showMessage(text, type = 'info') {
        const msgEl = document.getElementById('settingsMessage');
        if (!msgEl) return;

        msgEl.textContent = text;
        msgEl.className = 'text-sm text-center py-2 px-3 rounded-lg';

        if (type === 'success') {
            msgEl.classList.add('bg-green-600/20', 'text-green-400');
        } else if (type === 'error') {
            msgEl.classList.add('bg-red-600/20', 'text-red-400');
        } else {
            msgEl.classList.add('bg-blue-600/20', 'text-blue-400');
        }

        msgEl.classList.remove('hidden');

        // Auto-hide after 3 seconds
        setTimeout(() => {
            msgEl.classList.add('hidden');
        }, 3000);
    },
};
