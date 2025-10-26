// frontend/js/tabs/tts_export.js
import { sessionManager } from '../utils/session.js';

export class TTSExportTab {
    constructor() {
        this.sessionId = null;
        this.progress = 0;
        this.currentStep = '';
    }

    setSession(sessionId) {
        this.sessionId = sessionId;
    }

    render() {
        return `
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-2xl font-bold mb-6 text-purple-400">
                    <i class="fas fa-file-export mr-2"></i>TTS & Export
                </h2>

                <!-- TTS Engine Selection -->
                <div class="bg-gray-700 rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-semibold mb-4">TTS Engine Configuration</h3>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <!-- Engine Selection -->
                        <div>
                            <label class="block text-sm font-medium mb-2">TTS Engine</label>
                            <select id="tts-engine" class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                                <option value="edge">Edge TTS</option>
                                <option value="elevenlabs">ElevenLabs</option>
                            </select>
                        </div>

                        <!-- Common Parameters -->
                        <div class="grid grid-cols-3 gap-2">
                            <div>
                                <label class="block text-sm font-medium mb-1">Rate</label>
                                <input type="text" id="tts-rate" value="+0%" 
                                       class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-1">Volume</label>
                                <input type="text" id="tts-volume" value="+0%" 
                                       class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-1">Pitch</label>
                                <input type="text" id="tts-pitch" value="+0Hz" 
                                       class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Voice Configuration -->
                <div class="bg-gray-700 rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-semibold mb-4">Voice Configuration</h3>
                    
                    <div id="edge-tts-config">
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div>
                                <label class="block text-sm font-medium mb-1">Male Voice</label>
                                <input type="text" id="edge-voice-male" value="id-ID-ArdiNeural" 
                                       class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-1">Female Voice</label>
                                <input type="text" id="edge-voice-female" value="id-ID-GadisNeural" 
                                       class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-1">Unknown Voice</label>
                                <input type="text" id="edge-voice-unknown" value="id-ID-ArdiNeural" 
                                       class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                            </div>
                        </div>
                    </div>

                    <div id="elevenlabs-config" class="hidden">
                        <div class="space-y-4">
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label class="block text-sm font-medium mb-1">API Key</label>
                                    <input type="password" id="el-api-key" 
                                           class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium mb-1">Model</label>
                                    <input type="text" id="el-model" value="eleven_multilingual_v2" 
                                           class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                                </div>
                            </div>
                            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div>
                                    <label class="block text-sm font-medium mb-1">Male Voice ID</label>
                                    <input type="text" id="el-voice-male" 
                                           class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium mb-1">Female Voice ID</label>
                                    <input type="text" id="el-voice-female" 
                                           class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium mb-1">Unknown Voice ID</label>
                                    <input type="text" id="el-voice-unknown" 
                                           class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                                </div>
                            </div>
                            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div>
                                    <label class="block text-sm font-medium mb-1">Stability</label>
                                    <input type="number" id="el-stability" value="0.3" step="0.1" min="0" max="1"
                                           class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium mb-1">Similarity</label>
                                    <input type="number" id="el-similarity" value="0.8" step="0.1" min="0" max="1"
                                           class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium mb-1">Style</label>
                                    <input type="number" id="el-style" value="0.0" step="0.1" min="0" max="1"
                                           class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                                </div>
                                <div class="flex items-end">
                                    <div class="flex items-center">
                                        <input type="checkbox" id="el-boost" checked class="mr-2">
                                        <label for="el-boost" class="text-sm">Speaker Boost</label>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Timing and Export Settings -->
                <div class="bg-gray-700 rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-semibold mb-4">Timing and Export Settings</h3>
                    
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div>
                            <label class="block text-sm font-medium mb-1">Max Atempo</label>
                            <input type="number" id="max-atempo" value="1.8" step="0.1"
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">Fade Seconds</label>
                            <input type="number" id="fade-sec" value="0.02" step="0.01"
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">Mix Chunk Size</label>
                            <input type="number" id="mix-chunk" value="400"
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">TTS Timeout (s)</label>
                            <input type="number" id="tts-timeout" value="25"
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                        </div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                        <div class="flex items-center">
                            <input type="checkbox" id="replace-audio" class="mr-2">
                            <label for="replace-audio">Replace audio (no BGM)</label>
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">BGM Mode</label>
                            <select id="bg-mode" class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500">
                                <option value="center_cut">Center Cut</option>
                                <option value="mix_low">Mix Low</option>
                                <option value="off">Off</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Progress Section -->
                <div class="bg-gray-700 rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-semibold mb-3">Export Progress</h3>
                    <div class="space-y-4">
                        <div>
                            <div class="flex justify-between mb-1">
                                <span class="text-sm" id="export-step">Ready</span>
                                <span class="text-sm" id="export-progress">0%</span>
                            </div>
                            <div class="w-full bg-gray-600 rounded-full h-2">
                                <div id="export-progress-bar" class="bg-purple-500 h-2 rounded-full transition-all duration-300" 
                                     style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="grid grid-cols-4 gap-4 text-sm text-center">
                            <div>
                                <div class="text-2xl font-bold text-blue-400" id="tts-generated">0</div>
                                <div class="text-gray-400">TTS Generated</div>
                            </div>
                            <div>
                                <div class="text-2xl font-bold text-green-400" id="segments-mixed">0</div>
                                <div class="text-gray-400">Segments Mixed</div>
                            </div>
                            <div>
                                <div class="text-2xl font-bold text-yellow-400" id="export-status">Idle</div>
                                <div class="text-gray-400">Status</div>
                            </div>
                            <div>
                                <div class="text-2xl font-bold text-purple-400" id="estimated-time">-</div>
                                <div class="text-gray-400">Estimated Time</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="flex flex-wrap gap-3 mb-6">
                    <button id="start-export" class="bg-green-500 hover:bg-green-600 px-6 py-3 rounded-lg flex items-center text-lg">
                        <i class="fas fa-play-circle mr-2"></i>Start TTS & Export
                    </button>
                    <button id="stop-export" class="bg-red-500 hover:bg-red-600 px-6 py-3 rounded-lg flex items-center text-lg" disabled>
                        <i class="fas fa-stop-circle mr-2"></i>Stop Export
                    </button>
                    <button id="test-tts" class="bg-blue-500 hover:bg-blue-600 px-4 py-3 rounded-lg flex items-center">
                        <i class="fas fa-volume-up mr-2"></i>Test TTS
                    </button>
                </div>

                <!-- Log Output -->
                <div class="bg-gray-700 rounded-lg p-4">
                    <h3 class="text-lg font-semibold mb-3">Export Log</h3>
                    <div id="export-log" class="bg-gray-900 rounded p-4 h-48 overflow-y-auto font-mono text-sm">
                        <div class="text-gray-400">Export log will appear here...</div>
                    </div>
                </div>
            </div>
        `;
    }

    async afterRender() {
        this.setupEventListeners();
        await this.loadConfig();
        this.updateEngineConfig();
    }

    setupEventListeners() {
        document.getElementById('tts-engine').addEventListener('change', () => this.updateEngineConfig());
        document.getElementById('start-export').addEventListener('click', () => this.startExport());
        document.getElementById('stop-export').addEventListener('click', () => this.stopExport());
        document.getElementById('test-tts').addEventListener('click', () => this.testTTS());
    }

    updateEngineConfig() {
        const engine = document.getElementById('tts-engine').value;
        const edgeConfig = document.getElementById('edge-tts-config');
        const elevenConfig = document.getElementById('elevenlabs-config');

        if (engine === 'edge') {
            edgeConfig.classList.remove('hidden');
            elevenConfig.classList.add('hidden');
        } else {
            edgeConfig.classList.add('hidden');
            elevenConfig.classList.remove('hidden');
        }
    }

    async startExport() {
        if (!this.sessionId) {
            this.log('Error: No active session. Please load a session first.', 'error');
            return;
        }

        this.isExporting = true;
        this.updateExportButtonStates();
        this.updateProgress(0, 'Starting export...');

        try {
            const config = this.getExportConfig();
            await sessionManager.runTTSExport(this.sessionId, config);
            
        } catch (error) {
            this.log(`Error starting export: ${error.message}`, 'error');
            this.isExporting = false;
            this.updateExportButtonStates();
        }
    }

    stopExport() {
        this.isExporting = false;
        this.updateExportButtonStates();
        this.log('Export stopped by user');
    }

    async testTTS() {
        const testText = prompt('Enter text for TTS test:', 'Hello, this is a test of the TTS system.');
        if (!testText) return;

        this.log(`Testing TTS with: "${testText}"`);
        
        // Implementation for TTS testing would go here
        // This could involve calling a separate test endpoint
    }

    getExportConfig() {
        const engine = document.getElementById('tts-engine').value;
        
        const baseConfig = {
            tts_engine: engine,
            voice_male: document.getElementById('edge-voice-male').value,
            voice_female: document.getElementById('edge-voice-female').value,
            voice_unknown: document.getElementById('edge-voice-unknown').value,
            replace_audio: document.getElementById('replace-audio').checked,
            bg_mode: document.getElementById('bg-mode').value,
            max_atempo: parseFloat(document.getElementById('max-atempo').value),
            fade_sec: parseFloat(document.getElementById('fade-sec').value),
            mix_chunk: parseInt(document.getElementById('mix-chunk').value),
            tts_timeout: parseInt(document.getElementById('tts-timeout').value)
        };

        if (engine === 'elevenlabs') {
            baseConfig.el_api_key = document.getElementById('el-api-key').value;
            baseConfig.el_model = document.getElementById('el-model').value;
            baseConfig.el_voice_male = document.getElementById('el-voice-male').value;
            baseConfig.el_voice_female = document.getElementById('el-voice-female').value;
            baseConfig.el_voice_unknown = document.getElementById('el-voice-unknown').value;
            baseConfig.el_stability = parseFloat(document.getElementById('el-stability').value);
            baseConfig.el_similarity = parseFloat(document.getElementById('el-similarity').value);
            baseConfig.el_style = parseFloat(document.getElementById('el-style').value);
            baseConfig.el_boost = document.getElementById('el-boost').checked;
        }

        return baseConfig;
    }

    updateProgress(percent, message) {
        this.progress = percent;
        this.currentStep = message;

        const progressBar = document.getElementById('export-progress-bar');
        const progressText = document.getElementById('export-progress');
        const stepText = document.getElementById('export-step');

        if (progressBar) {
            progressBar.style.width = `${percent}%`;
        }
        if (progressText) {
            progressText.textContent = `${percent}%`;
        }
        if (stepText) {
            stepText.textContent = message;
        }
    }

    updateExportButtonStates() {
        const startButton = document.getElementById('start-export');
        const stopButton = document.getElementById('stop-export');

        if (startButton) startButton.disabled = this.isExporting;
        if (stopButton) stopButton.disabled = !this.isExporting;
    }

    log(message, type = 'info') {
        const logOutput = document.getElementById('export-log');
        if (!logOutput) return;

        const timestamp = new Date().toLocaleTimeString();
        const color = type === 'error' ? 'text-red-400' : 
                     type === 'warning' ? 'text-yellow-400' : 'text-gray-300';
        const icon = type === 'error' ? 'fa-exclamation-circle' : 
                    type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle';

        const logEntry = document.createElement('div');
        logEntry.className = `flex items-start ${color} mb-1`;
        logEntry.innerHTML = `
            <i class="fas ${icon} mr-2 mt-1 flex-shrink-0"></i>
            <span class="flex-1">
                <span class="text-gray-500 text-xs">[${timestamp}]</span> ${message}
            </span>
        `;

        logOutput.appendChild(logEntry);
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    async loadConfig() {
        const savedConfig = localStorage.getItem('dracin-tts-config');
        if (savedConfig) {
            const config = JSON.parse(savedConfig);
            
            document.getElementById('tts-engine').value = config.tts_engine || 'edge';
            document.getElementById('tts-rate').value = config.tts_rate || '+0%';
            document.getElementById('tts-volume').value = config.tts_volume || '+0%';
            document.getElementById('tts-pitch').value = config.tts_pitch || '+0Hz';
            
            document.getElementById('edge-voice-male').value = config.edge_voice_male || 'id-ID-ArdiNeural';
            document.getElementById('edge-voice-female').value = config.edge_voice_female || 'id-ID-GadisNeural';
            document.getElementById('edge-voice-unknown').value = config.edge_voice_unknown || 'id-ID-ArdiNeural';
            
            document.getElementById('el-api-key').value = config.el_api_key || '';
            document.getElementById('el-model').value = config.el_model || 'eleven_multilingual_v2';
            document.getElementById('el-voice-male').value = config.el_voice_male || '';
            document.getElementById('el-voice-female').value = config.el_voice_female || '';
            document.getElementById('el-voice-unknown').value = config.el_voice_unknown || '';
            document.getElementById('el-stability').value = config.el_stability || 0.3;
            document.getElementById('el-similarity').value = config.el_similarity || 0.8;
            document.getElementById('el-style').value = config.el_style || 0.0;
            document.getElementById('el-boost').checked = config.el_boost !== false;
            
            document.getElementById('max-atempo').value = config.max_atempo || 1.8;
            document.getElementById('fade-sec').value = config.fade_sec || 0.02;
            document.getElementById('mix-chunk').value = config.mix_chunk || 400;
            document.getElementById('tts-timeout').value = config.tts_timeout || 25;
            document.getElementById('replace-audio').checked = config.replace_audio || false;
            document.getElementById('bg-mode').value = config.bg_mode || 'center_cut';
        }
    }

    saveConfig() {
        const config = this.getExportConfig();
        localStorage.setItem('dracin-tts-config', JSON.stringify(config));
    }

    update(data) {
        if (data.progress !== undefined) {
            this.updateProgress(data.progress, data.current_step || '');
        }
        if (data.status) {
            document.getElementById('export-status').textContent = data.status;
        }
        if (data.tts_generated !== undefined) {
            document.getElementById('tts-generated').textContent = data.tts_generated;
        }
        if (data.segments_mixed !== undefined) {
            document.getElementById('segments-mixed').textContent = data.segments_mixed;
        }
        if (data.estimated_time) {
            document.getElementById('estimated-time').textContent = data.estimated_time;
        }
    }
}