// frontend/js/tabs/editing.js
import { sessionManager } from '../utils/session.js';

export class EditingTab {
    constructor() {
        this.sessionId = null;
        this.entries = [];
        this.filteredEntries = [];
        this.videoPlayer = null;
        this.wavesurfer = null;
        this.currentEntryIndex = null;
        this.filterGender = 'all';
        this.filterSpeaker = 'all';
    }

    setSession(sessionId) {
        this.sessionId = sessionId;
    }

    render() {
        return `
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-2xl font-bold mb-6 text-yellow-400">
                    <i class="fas fa-edit mr-2"></i>Editing
                </h2>

                <!-- Controls Header -->
                <div class="bg-gray-700 rounded-lg p-4 mb-6">
                    <div class="flex flex-wrap items-center justify-between gap-4">
                        <div class="flex flex-wrap items-center gap-4">
                            <button id="load-season" class="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded flex items-center">
                                <i class="fas fa-folder-open mr-2"></i>Load Season Folder
                            </button>
                            <div class="flex items-center space-x-2">
                                <label class="text-sm">Filter Gender:</label>
                                <select id="filter-gender" class="px-3 py-1 bg-gray-600 rounded border border-gray-500">
                                    <option value="all">All</option>
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                    <option value="Unknown">Unknown</option>
                                </select>
                            </div>
                            <div class="flex items-center space-x-2">
                                <label class="text-sm">Speaker:</label>
                                <select id="filter-speaker" class="px-3 py-1 bg-gray-600 rounded border border-gray-500">
                                    <option value="all">All</option>
                                </select>
                            </div>
                        </div>
                        <div class="flex items-center space-x-2">
                            <input type="checkbox" id="autosave" checked class="mr-2">
                            <label for="autosave" class="text-sm">Autosave</label>
                            <button id="save-changes" class="bg-green-500 hover:bg-green-600 px-4 py-2 rounded flex items-center">
                                <i class="fas fa-save mr-2"></i>Save
                            </button>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 xl:grid-cols-4 gap-6">
                    <!-- Left Column: Episode List -->
                    <div class="xl:col-span-1">
                        <div class="bg-gray-700 rounded-lg p-4 mb-6">
                            <h3 class="text-lg font-semibold mb-3">Episodes / Workdirs</h3>
                            <div id="episode-list" class="space-y-2 max-h-60 overflow-y-auto">
                                <div class="text-gray-400 text-center py-4">No episodes loaded</div>
                            </div>
                            <div id="video-source" class="mt-3 text-sm text-gray-400">
                                Video: (unknown)
                            </div>
                        </div>
                    </div>

                    <!-- Middle Column: Editing Table -->
                    <div class="xl:col-span-2">
                        <div class="bg-gray-700 rounded-lg p-4 mb-6">
                            <h3 class="text-lg font-semibold mb-3">Subtitle Entries</h3>
                            <div class="overflow-x-auto">
                                <table id="editing-table" class="w-full">
                                    <thead>
                                        <tr class="bg-gray-600">
                                            <th class="px-3 py-2 text-left">#</th>
                                            <th class="px-3 py-2 text-left">Start</th>
                                            <th class="px-3 py-2 text-left">End</th>
                                            <th class="px-3 py-2 text-left">Speaker</th>
                                            <th class="px-3 py-2 text-left">Gender</th>
                                            <th class="px-3 py-2 text-left">Text</th>
                                            <th class="px-3 py-2 text-left">Warning</th>
                                        </tr>
                                    </thead>
                                    <tbody id="editing-table-body">
                                        <tr>
                                            <td colspan="7" class="px-3 py-4 text-center text-gray-400">
                                                Load episode to see entries
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    <!-- Right Column: Editor -->
                    <div class="xl:col-span-1">
                        <div class="bg-gray-700 rounded-lg p-4 mb-6">
                            <h3 class="text-lg font-semibold mb-3">Editor</h3>
                            
                            <!-- Gender Controls -->
                            <div class="mb-4">
                                <label class="block text-sm font-medium mb-2">Gender:</label>
                                <div class="flex flex-wrap gap-2 mb-3">
                                    <select id="edit-gender" class="flex-1 px-3 py-2 bg-gray-600 rounded border border-gray-500">
                                        <option value="Male">Male</option>
                                        <option value="Female">Female</option>
                                        <option value="Unknown">Unknown</option>
                                    </select>
                                    <button id="mass-set-gender" class="bg-purple-500 hover:bg-purple-600 px-3 py-2 rounded">
                                        Set Selected
                                    </button>
                                </div>
                            </div>

                            <!-- Text Editor -->
                            <div class="mb-4">
                                <label class="block text-sm font-medium mb-2">Text:</label>
                                <textarea id="edit-text" 
                                          class="w-full h-32 px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500 resize-none"
                                          placeholder="Edit subtitle text..."></textarea>
                                <div id="multi-line-warning" class="text-yellow-400 text-sm mt-1 hidden">
                                    ⚠️ Multiple lines detected
                                </div>
                            </div>

                            <!-- Entry Info -->
                            <div class="bg-gray-600 rounded p-3">
                                <div id="entry-info" class="text-sm space-y-1">
                                    <div>#: -</div>
                                    <div>Time: -</div>
                                    <div>Speaker: -</div>
                                </div>
                                <button id="play-segment" class="w-full mt-3 bg-blue-500 hover:bg-blue-600 px-3 py-2 rounded flex items-center justify-center">
                                    <i class="fas fa-play mr-2"></i>Play Segment
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Video Player Section -->
                <div class="bg-gray-700 rounded-lg p-4">
                    <h3 class="text-lg font-semibold mb-3">Video Preview</h3>
                    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <!-- Video Player -->
                        <div class="lg:col-span-2">
                            <div class="bg-black rounded overflow-hidden">
                                <video id="video-player" class="w-full" controls>
                                    Your browser does not support the video tag.
                                </video>
                            </div>
                            <div class="flex justify-between items-center mt-2">
                                <div class="text-sm text-gray-400" id="video-time">00:00 / 00:00</div>
                                <div class="flex space-x-2">
                                    <button id="play-pause" class="bg-blue-500 hover:bg-blue-600 px-3 py-1 rounded">
                                        <i class="fas fa-play"></i>
                                    </button>
                                    <button id="skip-back" class="bg-gray-600 hover:bg-gray-500 px-3 py-1 rounded">
                                        <i class="fas fa-backward"></i>
                                    </button>
                                    <button id="skip-forward" class="bg-gray-600 hover:bg-gray-500 px-3 py-1 rounded">
                                        <i class="fas fa-forward"></i>
                                    </button>
                                </div>
                            </div>
                        </div>

                        <!-- Waveform -->
                        <div class="lg:col-span-1">
                            <div class="bg-gray-800 rounded p-3 h-full">
                                <h4 class="text-sm font-medium mb-2">Audio Waveform</h4>
                                <div id="waveform" class="bg-gray-900 rounded h-32"></div>
                                <div class="text-xs text-gray-400 mt-2 text-center">
                                    Visual representation of audio
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    async afterRender() {
        this.setupEventListeners();
        await this.initializeVideoPlayer();
        await this.initializeWaveform();
        await this.loadEpisodes();
    }

    setupEventListeners() {
        document.getElementById('load-season').addEventListener('click', () => this.loadSeasonFolder());
        document.getElementById('filter-gender').addEventListener('change', () => this.applyFilters());
        document.getElementById('filter-speaker').addEventListener('change', () => this.applyFilters());
        document.getElementById('save-changes').addEventListener('click', () => this.saveChanges());
        document.getElementById('mass-set-gender').addEventListener('click', () => this.massSetGender());
        document.getElementById('play-segment').addEventListener('click', () => this.playSegment());
        document.getElementById('edit-text').addEventListener('input', () => this.onTextEdit());
        document.getElementById('edit-gender').addEventListener('change', () => this.onGenderEdit());

        // Video controls
        document.getElementById('play-pause').addEventListener('click', () => this.togglePlayPause());
        document.getElementById('skip-back').addEventListener('click', () => this.skipBack());
        document.getElementById('skip-forward').addEventListener('click', () => this.skipForward());
    }

    async initializeVideoPlayer() {
        const video = document.getElementById('video-player');
        if (!video) return;

        this.videoPlayer = video;

        video.addEventListener('loadedmetadata', () => {
            this.updateVideoTime();
        });

        video.addEventListener('timeupdate', () => {
            this.updateVideoTime();
            this.updateWaveformCursor();
        });

        video.addEventListener('play', () => {
            document.getElementById('play-pause').innerHTML = '<i class="fas fa-pause"></i>';
        });

        video.addEventListener('pause', () => {
            document.getElementById('play-pause').innerHTML = '<i class="fas fa-play"></i>';
        });
    }

    async initializeWaveform() {
        if (typeof WaveSurfer === 'undefined') {
            console.warn('WaveSurfer not loaded');
            return;
        }

        this.wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#4B5563',
            progressColor: '#3B82F6',
            cursorColor: '#FFFFFF',
            barWidth: 2,
            barGap: 1,
            height: 128,
            normalize: true
        });

        this.wavesurfer.on('ready', () => {
            console.log('Waveform ready');
        });

        this.wavesurfer.on('seek', (progress) => {
            if (this.videoPlayer) {
                this.videoPlayer.currentTime = progress * this.videoPlayer.duration;
            }
        });
    }

    async loadEpisodes() {
        try {
            const sessions = await sessionManager.listSessions();
            this.renderEpisodeList(sessions);
        } catch (error) {
            console.error('Error loading episodes:', error);
        }
    }

    renderEpisodeList(sessions) {
        const container = document.getElementById('episode-list');
        if (!container) return;

        if (sessions.length === 0) {
            container.innerHTML = '<div class="text-gray-400 text-center py-4">No sessions found</div>';
            return;
        }

        container.innerHTML = sessions.map(session => `
            <div class="episode-item p-2 rounded cursor-pointer hover:bg-gray-600 border border-gray-600"
                 data-session-id="${session.id}">
                <div class="font-medium text-sm">${session.video_name || 'Unknown'}</div>
                <div class="text-xs text-gray-400">${new Date(session.created_at * 1000).toLocaleDateString()}</div>
            </div>
        `).join('');

        // Add click handlers
        container.querySelectorAll('.episode-item').forEach(item => {
            item.addEventListener('click', () => {
                this.loadEpisode(item.dataset.sessionId);
            });
        });
    }

    async loadEpisode(sessionId) {
        try {
            this.sessionId = sessionId;
            this.log(`Loading episode: ${sessionId}`);

            const editingData = await sessionManager.getEditingData(sessionId);
            this.entries = editingData.entries;
            this.filteredEntries = [...this.entries];

            this.updateSpeakerFilter();
            this.renderTable();
            this.loadVideo(editingData.video_url);

            // Update video source display
            document.getElementById('video-source').textContent = `Video: ${sessionId}`;

        } catch (error) {
            this.log(`Error loading episode: ${error.message}`, 'error');
        }
    }

    async loadVideo(videoUrl) {
        if (!this.videoPlayer) return;

        try {
            this.videoPlayer.src = videoUrl;
            await this.videoPlayer.load();

            // Load audio for waveform when video is ready
            this.videoPlayer.addEventListener('canplay', () => {
                if (this.wavesurfer) {
                    this.wavesurfer.load(videoUrl);
                }
            }, { once: true });

        } catch (error) {
            console.error('Error loading video:', error);
        }
    }

    updateSpeakerFilter() {
        const speakers = [...new Set(this.entries.map(entry => entry.speaker).filter(Boolean))];
        const filterSelect = document.getElementById('filter-speaker');
        
        filterSelect.innerHTML = '<option value="all">All</option>' +
            speakers.map(speaker => `<option value="${speaker}">${speaker}</option>`).join('');
    }

    renderTable() {
        const tbody = document.getElementById('editing-table-body');
        if (!tbody) return;

        tbody.innerHTML = this.filteredEntries.map(entry => `
            <tr class="border-b border-gray-600 hover:bg-gray-600 cursor-pointer"
                data-index="${entry.index}">
                <td class="px-3 py-2">${entry.index}</td>
                <td class="px-3 py-2">${this.formatTime(entry.start)}</td>
                <td class="px-3 py-2">${this.formatTime(entry.end)}</td>
                <td class="px-3 py-2">${entry.speaker || '-'}</td>
                <td class="px-3 py-2">
                    <span class="px-2 py-1 rounded text-xs ${
                        entry.gender === 'Male' ? 'bg-blue-500' :
                        entry.gender === 'Female' ? 'bg-pink-500' : 'bg-gray-500'
                    }">${entry.gender}</span>
                </td>
                <td class="px-3 py-2 max-w-xs truncate">${this.escapeHtml(entry.text)}</td>
                <td class="px-3 py-2">
                    ${this.hasMultipleLines(entry.text) ? 
                      '<span class="text-yellow-400 text-xs">⚠ Multiple lines</span>' : ''}
                </td>
            </tr>
        `).join('');

        // Add click handlers
        tbody.querySelectorAll('tr').forEach(row => {
            row.addEventListener('click', () => {
                this.selectEntry(parseInt(row.dataset.index));
            });
        });
    }

    selectEntry(index) {
        this.currentEntryIndex = index;
        const entry = this.entries.find(e => e.index === index);
        if (!entry) return;

        // Update editor
        document.getElementById('edit-text').value = entry.text;
        document.getElementById('edit-gender').value = entry.gender;
        
        // Update entry info
        document.getElementById('entry-info').innerHTML = `
            <div><strong>#:</strong> ${entry.index}</div>
            <div><strong>Time:</strong> ${this.formatTime(entry.start)} → ${this.formatTime(entry.end)}</div>
            <div><strong>Speaker:</strong> ${entry.speaker || 'Unknown'}</div>
            <div><strong>Duration:</strong> ${(entry.end - entry.start).toFixed(2)}s</div>
        `;

        // Update multi-line warning
        const warning = document.getElementById('multi-line-warning');
        warning.classList.toggle('hidden', !this.hasMultipleLines(entry.text));

        // Highlight selected row
        document.querySelectorAll('#editing-table-body tr').forEach(row => {
            row.classList.toggle('bg-blue-600', parseInt(row.dataset.index) === index);
        });
    }

    applyFilters() {
        const genderFilter = document.getElementById('filter-gender').value;
        const speakerFilter = document.getElementById('filter-speaker').value;

        this.filteredEntries = this.entries.filter(entry => {
            const genderMatch = genderFilter === 'all' || entry.gender === genderFilter;
            const speakerMatch = speakerFilter === 'all' || entry.speaker === speakerFilter;
            return genderMatch && speakerMatch;
        });

        this.renderTable();
    }

    onTextEdit() {
        const text = document.getElementById('edit-text').value;
        const warning = document.getElementById('multi-line-warning');
        warning.classList.toggle('hidden', !this.hasMultipleLines(text));

        if (this.currentEntryIndex !== null) {
            const entry = this.entries.find(e => e.index === this.currentEntryIndex);
            if (entry) {
                entry.text = text;
                this.markDirty();
            }
        }
    }

    onGenderEdit() {
        const gender = document.getElementById('edit-gender').value;
        
        if (this.currentEntryIndex !== null) {
            const entry = this.entries.find(e => e.index === this.currentEntryIndex);
            if (entry) {
                entry.gender = gender;
                this.markDirty();
            }
        }
    }

    async massSetGender() {
        const selectedRows = document.querySelectorAll('#editing-table-body tr.bg-blue-600');
        if (selectedRows.length < 2) {
            alert('Please select multiple rows first (Ctrl+Click or Shift+Click)');
            return;
        }

        const newGender = document.getElementById('edit-gender').value;
        const confirmed = confirm(`Set gender to "${newGender}" for ${selectedRows.length} selected rows?`);
        
        if (!confirmed) return;

        selectedRows.forEach(row => {
            const index = parseInt(row.dataset.index);
            const entry = this.entries.find(e => e.index === index);
            if (entry) {
                entry.gender = newGender;
            }
        });

        this.renderTable();
        this.markDirty();
    }

    playSegment() {
        if (this.currentEntryIndex === null || !this.videoPlayer) return;

        const entry = this.entries.find(e => e.index === this.currentEntryIndex);
        if (!entry) return;

        this.videoPlayer.currentTime = entry.start;
        this.videoPlayer.play().catch(console.error);

        // Highlight the segment in waveform
        if (this.wavesurfer) {
            this.wavesurfer.seekTo(entry.start / this.videoPlayer.duration);
        }
    }

    togglePlayPause() {
        if (!this.videoPlayer) return;

        if (this.videoPlayer.paused) {
            this.videoPlayer.play();
        } else {
            this.videoPlayer.pause();
        }
    }

    skipBack() {
        if (!this.videoPlayer) return;
        this.videoPlayer.currentTime = Math.max(0, this.videoPlayer.currentTime - 5);
    }

    skipForward() {
        if (!this.videoPlayer) return;
        this.videoPlayer.currentTime = Math.min(
            this.videoPlayer.duration, 
            this.videoPlayer.currentTime + 5
        );
    }

    async saveChanges() {
        if (!this.sessionId) return;

        try {
            // Save all changes to backend
            for (const entry of this.entries) {
                await sessionManager.editEntry(
                    this.sessionId,
                    entry.index,
                    entry.text,
                    entry.gender
                );
            }

            this.log('Changes saved successfully');
            this.markClean();
        } catch (error) {
            this.log(`Error saving changes: ${error.message}`, 'error');
        }
    }

    markDirty() {
        if (document.getElementById('autosave').checked) {
            // Auto-save after delay
            clearTimeout(this.autoSaveTimeout);
            this.autoSaveTimeout = setTimeout(() => this.saveChanges(), 1000);
        }
    }

    markClean() {
        // Clear any dirty state indicators
    }

    updateVideoTime() {
        if (!this.videoPlayer) return;

        const current = this.formatTime(this.videoPlayer.currentTime);
        const duration = this.formatTime(this.videoPlayer.duration);
        document.getElementById('video-time').textContent = `${current} / ${duration}`;
    }

    updateWaveformCursor() {
        if (!this.wavesurfer || !this.videoPlayer) return;

        const progress = this.videoPlayer.currentTime / this.videoPlayer.duration;
        this.wavesurfer.seekTo(progress);
    }

    formatTime(seconds) {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);

        if (hrs > 0) {
            return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
        } else {
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
        }
    }

    hasMultipleLines(text) {
        return text && text.split('\n').filter(line => line.trim()).length > 1;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    log(message, type = 'info') {
        console.log(`[Editing] ${message}`);
        // Could be integrated with a global logging system
    }

    update(data) {
        // Handle updates from WebSocket
        if (data.entries) {
            this.entries = data.entries;
            this.applyFilters();
        }
    }
}