// frontend/js/tabs/srt_processing.js
import { sessionManager } from '../utils/session.js';

export class SRTProcessingTab {
    constructor() {
        this.sessionId = null;
        this.videoFile = null;
        this.srtFile = null;
        this.progress = 0;
        this.currentStep = '';
    }

    setSession(sessionId) {
        this.sessionId = sessionId;
    }

    render() {
        return `
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-2xl font-bold mb-6 text-blue-400">
                    <i class="fas fa-file-alt mr-2"></i>SRT Processing
                </h2>

                <!-- File Upload Section -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                    <!-- Video Upload -->
                    <div class="bg-gray-700 rounded-lg p-4">
                        <h3 class="text-lg font-semibold mb-3">Video File</h3>
                        <div class="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center"
                             id="video-drop-zone">
                            <i class="fas fa-video text-4xl text-gray-400 mb-3"></i>
                            <p class="text-gray-300 mb-2">Drop video file here or click to browse</p>
                            <input type="file" id="video-file" accept="video/*" class="hidden">
                            <button type="button" onclick="document.getElementById('video-file').click()" 
                                    class="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded">
                                Browse Files
                            </button>
                            <div id="video-file-info" class="mt-2 text-sm text-gray-400 hidden"></div>
                        </div>
                    </div>

                    <!-- SRT Upload -->
                    <div class="bg-gray-700 rounded-lg p-4">
                        <h3 class="text-lg font-semibold mb-3">SRT File</h3>
                        <div class="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center"
                             id="srt-drop-zone">
                            <i class="fas fa-file-text text-4xl text-gray-400 mb-3"></i>
                            <p class="text-gray-300 mb-2">Drop SRT file here or click to browse</p>
                            <input type="file" id="srt-file" accept=".srt" class="hidden">
                            <button type="button" onclick="document.getElementById('srt-file').click()" 
                                    class="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded">
                                Browse Files
                            </button>
                            <div id="srt-file-info" class="mt-2 text-sm text-gray-400 hidden"></div>
                        </div>
                    </div>
                </div>

                <!-- Diarization Configuration -->
                <div class="bg-gray-700 rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-semibold mb-4">Diarization Configuration</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium mb-1">Male Reference</label>
                            <input type="text" id="male-ref" value="D:\\dubdracin\\samples\\male" 
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">Female Reference</label>
                            <input type="text" id="female-ref" value="D:\\dubdracin\\samples\\female" 
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">HF Token</label>
                            <input type="password" id="hf-token" 
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">Top N Segments</label>
                            <input type="number" id="top-n" value="5" 
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500">
                        </div>
                        <div class="flex items-center">
                            <input type="checkbox" id="use-gpu" checked class="mr-2">
                            <label for="use-gpu">Use GPU for Diarization</label>
                        </div>
                    </div>
                </div>

                <!-- Progress Section -->
                <div class="bg-gray-700 rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-semibold mb-3">Processing Progress</h3>
                    <div class="space-y-4">
                        <div>
                            <div class="flex justify-between mb-1">
                                <span class="text-sm" id="current-step">Ready</span>
                                <span class="text-sm" id="progress-percent">0%</span>
                            </div>
                            <div class="w-full bg-gray-600 rounded-full h-2">
                                <div id="progress-bar" class="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                                     style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="flex flex-wrap gap-3">
                    <button id="create-session" class="bg-green-500 hover:bg-green-600 px-4 py-2 rounded flex items-center">
                        <i class="fas fa-plus mr-2"></i>Create Session
                    </button>
                    <button id="generate-workdir" class="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded flex items-center" disabled>
                        <i class="fas fa-folder-plus mr-2"></i>Generate Workdir
                    </button>
                    <button id="extract-audio" class="bg-purple-500 hover:bg-purple-600 px-4 py-2 rounded flex items-center" disabled>
                        <i class="fas fa-volume-up mr-2"></i>Extract Audio
                    </button>
                    <button id="run-diarization" class="bg-orange-500 hover:bg-orange-600 px-4 py-2 rounded flex items-center" disabled>
                        <i class="fas fa-users mr-2"></i>Run Diarization
                    </button>
                    <button id="load-to-translate" class="bg-indigo-500 hover:bg-indigo-600 px-4 py-2 rounded flex items-center" disabled>
                        <i class="fas fa-arrow-right mr-2"></i>Load to Translate Tab
                    </button>
                </div>

                <!-- Log Output -->
                <div class="mt-6">
                    <h3 class="text-lg font-semibold mb-3">Processing Log</h3>
                    <div id="log-output" class="bg-gray-900 rounded p-4 h-48 overflow-y-auto font-mono text-sm">
                        <div class="text-gray-400">Log output will appear here...</div>
                    </div>
                </div>
            </div>
        `;
    }

    async afterRender() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        await this.loadConfig();
    }

    setupEventListeners() {
        // File input handlers
        document.getElementById('video-file').addEventListener('change', (e) => this.handleFileSelect(e, 'video'));
        document.getElementById('srt-file').addEventListener('change', (e) => this.handleFileSelect(e, 'srt'));

        // Button handlers
        document.getElementById('create-session').addEventListener('click', () => this.createSession());
        document.getElementById('generate-workdir').addEventListener('click', () => this.generateWorkdir());
        document.getElementById('extract-audio').addEventListener('click', () => this.extractAudio());
        document.getElementById('run-diarization').addEventListener('click', () => this.runDiarization());
        document.getElementById('load-to-translate').addEventListener('click', () => this.loadToTranslate());
    }

    setupDragAndDrop() {
        const videoDropZone = document.getElementById('video-drop-zone');
        const srtDropZone = document.getElementById('srt-drop-zone');

        this.setupDropZone(videoDropZone, 'video');
        this.setupDropZone(srtDropZone, 'srt');
    }

    setupDropZone(dropZone, type) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-blue-400', 'bg-gray-600');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-blue-400', 'bg-gray-600');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-blue-400', 'bg-gray-600');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleDroppedFile(files[0], type);
            }
        });
    }

    handleFileSelect(event, type) {
        const file = event.target.files[0];
        if (file) {
            this.handleDroppedFile(file, type);
        }
    }

    handleDroppedFile(file, type) {
        const fileInfo = document.getElementById(`${type}-file-info`);
        const fileName = file.name;
        const fileSize = (file.size / (1024 * 1024)).toFixed(2);

        fileInfo.innerHTML = `
            <div class="text-green-400">
                <i class="fas fa-check-circle mr-1"></i>
                ${fileName} (${fileSize} MB)
            </div>
        `;
        fileInfo.classList.remove('hidden');

        if (type === 'video') {
            this.videoFile = file;
        } else {
            this.srtFile = file;
        }

        this.updateButtonStates();
    }

    updateButtonStates() {
        const hasFiles = this.videoFile && this.srtFile;
        document.getElementById('create-session').disabled = !hasFiles;
        document.getElementById('generate-workdir').disabled = !this.sessionId;
        document.getElementById('extract-audio').disabled = !this.sessionId;
        document.getElementById('run-diarization').disabled = !this.sessionId;
        document.getElementById('load-to-translate').disabled = !this.sessionId;
    }

    async createSession() {
        if (!this.videoFile || !this.srtFile) {
            this.log('Error: Please select both video and SRT files');
            return;
        }

        try {
            this.log('Creating new session...');
            
            const session = await sessionManager.createSession(
                this.videoFile.name,
                this.srtFile.name
            );

            this.sessionId = session.session_id;
            this.log(`Session created: ${this.sessionId}`);

            // Upload files
            this.log('Uploading files...');
            await sessionManager.uploadFiles(
                this.sessionId,
                this.videoFile,
                this.srtFile
            );

            this.log('Files uploaded successfully');
            this.updateButtonStates();
            this.updateProgress(10, 'Files uploaded');

            // Store session ID globally
            if (window.dracinApp) {
                window.dracinApp.setCurrentSession(this.sessionId);
            }

        } catch (error) {
            this.log(`Error creating session: ${error.message}`, 'error');
        }
    }

    async generateWorkdir() {
        if (!this.sessionId) return;

        try {
            this.log('Generating workdir...');
            await sessionManager.generateWorkdir(this.sessionId);
            this.log('Workdir generated successfully');
            this.updateProgress(20, 'Workdir ready');
        } catch (error) {
            this.log(`Error generating workdir: ${error.message}`, 'error');
        }
    }

    async extractAudio() {
        if (!this.sessionId) return;

        try {
            this.log('Extracting 16kHz audio...');
            await sessionManager.extractAudio(this.sessionId);
            this.log('Audio extraction completed');
            this.updateProgress(40, 'Audio extracted');
        } catch (error) {
            this.log(`Error extracting audio: ${error.message}`, 'error');
        }
    }

    async runDiarization() {
        if (!this.sessionId) return;

        const config = {
            male_ref: document.getElementById('male-ref').value,
            female_ref: document.getElementById('female-ref').value,
            hf_token: document.getElementById('hf-token').value,
            use_gpu: document.getElementById('use-gpu').checked,
            top_n: parseInt(document.getElementById('top-n').value)
        };

        try {
            this.log('Starting diarization and gender detection...');
            await sessionManager.runDiarization(this.sessionId, config);
            this.log('Diarization completed successfully');
            this.updateProgress(70, 'Diarization complete');
        } catch (error) {
            this.log(`Error running diarization: ${error.message}`, 'error');
        }
    }

    async loadToTranslate() {
        if (!this.sessionId) return;

        try {
            this.log('Loading data to Translate tab...');
            // Switch to translate tab
            if (window.dracinApp) {
                window.dracinApp.loadTab('translate');
            }
        } catch (error) {
            this.log(`Error loading to translate: ${error.message}`, 'error');
        }
    }

    updateProgress(percent, message) {
        this.progress = percent;
        this.currentStep = message;

        const progressBar = document.getElementById('progress-bar');
        const progressPercent = document.getElementById('progress-percent');
        const currentStep = document.getElementById('current-step');

        if (progressBar) {
            progressBar.style.width = `${percent}%`;
        }
        if (progressPercent) {
            progressPercent.textContent = `${percent}%`;
        }
        if (currentStep) {
            currentStep.textContent = message;
        }
    }

    log(message, type = 'info') {
        const logOutput = document.getElementById('log-output');
        if (!logOutput) return;

        const timestamp = new Date().toLocaleTimeString();
        const color = type === 'error' ? 'text-red-400' : 'text-gray-300';
        const icon = type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle';

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
        // Load saved configuration
        const savedConfig = localStorage.getItem('dracin-srt-config');
        if (savedConfig) {
            const config = JSON.parse(savedConfig);
            document.getElementById('male-ref').value = config.male_ref || '';
            document.getElementById('female-ref').value = config.female_ref || '';
            document.getElementById('hf-token').value = config.hf_token || '';
            document.getElementById('top-n').value = config.top_n || 5;
            document.getElementById('use-gpu').checked = config.use_gpu !== false;
        }
    }

    saveConfig() {
        const config = {
            male_ref: document.getElementById('male-ref').value,
            female_ref: document.getElementById('female-ref').value,
            hf_token: document.getElementById('hf-token').value,
            top_n: parseInt(document.getElementById('top-n').value),
            use_gpu: document.getElementById('use-gpu').checked
        };
        localStorage.setItem('dracin-srt-config', JSON.stringify(config));
    }

    update(data) {
        if (data.progress !== undefined) {
            this.updateProgress(data.progress, data.current_step || '');
        }
        if (data.status) {
            this.log(`Status: ${data.status}`);
        }
    }
}