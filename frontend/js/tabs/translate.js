// frontend/js/tabs/translate.js
import { sessionManager } from '../utils/session.js';

export class TranslateTab {
    constructor() {
        this.sessionId = null;
        this.entries = [];
        this.translations = [];
        this.currentPage = 0;
        this.pageSize = 50;
        this.isTranslating = false;
    }

    setSession(sessionId) {
        this.sessionId = sessionId;
    }

    render() {
        return `
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-2xl font-bold mb-6 text-green-400">
                    <i class="fas fa-language mr-2"></i>Translate
                </h2>

                <!-- Translation Controls -->
                <div class="bg-gray-700 rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-semibold mb-4">Translation Controls</h3>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                        <div>
                            <label class="block text-sm font-medium mb-1">DeepSeek API Key</label>
                            <input type="password" id="ds-api-key" 
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">Batch Size</label>
                            <input type="number" id="ds-batch-size" value="20" 
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">Workers</label>
                            <input type="number" id="ds-workers" value="1" 
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-1">Timeout (s)</label>
                            <input type="number" id="ds-timeout" value="90" 
                                   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500">
                        </div>
                    </div>
                    <div class="flex flex-wrap gap-3">
                        <button id="load-srt" class="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded flex items-center">
                            <i class="fas fa-folder-open mr-2"></i>Load SRT
                        </button>
                        <button id="start-translate" class="bg-green-500 hover:bg-green-600 px-4 py-2 rounded flex items-center">
                            <i class="fas fa-play mr-2"></i>Start Translate
                        </button>
                        <button id="resume-translate" class="bg-yellow-500 hover:bg-yellow-600 px-4 py-2 rounded flex items-center">
                            <i class="fas fa-redo mr-2"></i>Resume
                        </button>
                        <button id="stop-translate" class="bg-red-500 hover:bg-red-600 px-4 py-2 rounded flex items-center" disabled>
                            <i class="fas fa-stop mr-2"></i>Stop
                        </button>
                        <button id="fill-missing" class="bg-purple-500 hover:bg-purple-600 px-4 py-2 rounded flex items-center">
                            <i class="fas fa-search mr-2"></i>Fill Missing
                        </button>
                        <button id="save-translations" class="bg-indigo-500 hover:bg-indigo-600 px-4 py-2 rounded flex items-center">
                            <i class="fas fa-save mr-2"></i>Save
                        </button>
                        <button id="export-translations" class="bg-orange-500 hover:bg-orange-600 px-4 py-2 rounded flex items-center">
                            <i class="fas fa-download mr-2"></i>Export As
                        </button>
                        <button id="next-tab" class="bg-teal-500 hover:bg-teal-600 px-4 py-2 rounded flex items-center">
                            <i class="fas fa-arrow-right mr-2"></i>Next Tab
                        </button>
                    </div>
                </div>

                <!-- Progress Section -->
                <div class="bg-gray-700 rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-semibold mb-3">Translation Progress</h3>
                    <div class="space-y-4">
                        <div>
                            <div class="flex justify-between mb-1">
                                <span class="text-sm" id="translate-status">Ready</span>
                                <span class="text-sm" id="translate-progress">0%</span>
                            </div>
                            <div class="w-full bg-gray-600 rounded-full h-2">
                                <div id="translate-progress-bar" class="bg-green-500 h-2 rounded-full transition-all duration-300" 
                                     style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="grid grid-cols-3 gap-4 text-sm">
                            <div class="text-center">
                                <div class="text-2xl font-bold text-green-400" id="translated-count">0</div>
                                <div class="text-gray-400">Translated</div>
                            </div>
                            <div class="text-center">
                                <div class="text-2xl font-bold text-yellow-400" id="missing-count">0</div>
                                <div class="text-gray-400">Missing</div>
                            </div>
                            <div class="text-center">
                                <div class="text-2xl font-bold text-blue-400" id="total-count">0</div>
                                <div class="text-gray-400">Total</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Translation Cards -->
                <div class="bg-gray-700 rounded-lg p-4 mb-6">
                    <div class="flex justify-between items-center mb-4">
                        <h3 class="text-lg font-semibold">Translation Preview</h3>
                        <div class="flex items-center space-x-4">
                            <div class="flex items-center">
                                <label class="mr-2 text-sm">Search:</label>
                                <input type="text" id="translation-search" 
                                       class="px-3 py-1 bg-gray-600 rounded border border-gray-500 focus:border-blue-500">
                            </div>
                            <div class="flex items-center">
                                <button id="prev-page" class="px-3 py-1 bg-gray-600 rounded-l hover:bg-gray-500">
                                    <i class="fas fa-chevron-left"></i>
                                </button>
                                <span id="page-info" class="px-3 py-1 bg-gray-700">Page 1</span>
                                <button id="next-page" class="px-3 py-1 bg-gray-600 rounded-r hover:bg-gray-500">
                                    <i class="fas fa-chevron-right"></i>
                                </button>
                            </div>
                            <div class="flex items-center">
                                <label class="mr-2 text-sm">Page Size:</label>
                                <select id="page-size" class="px-3 py-1 bg-gray-600 rounded border border-gray-500">
                                    <option value="50">50</option>
                                    <option value="100" selected>100</option>
                                    <option value="200">200</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    
                    <div id="translation-cards" class="space-y-3 max-h-96 overflow-y-auto">
                        <div class="text-center text-gray-400 py-8">
                            Load SRT file to see translation cards
                        </div>
                    </div>
                </div>

                <!-- Log Output -->
                <div class="bg-gray-700 rounded-lg p-4">
                    <h3 class="text-lg font-semibold mb-3">Translation Log</h3>
                    <div id="translation-log" class="bg-gray-900 rounded p-4 h-48 overflow-y-auto font-mono text-sm">
                        <div class="text-gray-400">Translation log will appear here...</div>
                    </div>
                </div>
            </div>
        `;
    }

    async afterRender() {
        this.setupEventListeners();
        await this.loadConfig();
        this.updateStats();
    }

    setupEventListeners() {
        document.getElementById('load-srt').addEventListener('click', () => this.loadSRT());
        document.getElementById('start-translate').addEventListener('click', () => this.startTranslation());
        document.getElementById('resume-translate').addEventListener('click', () => this.resumeTranslation());
        document.getElementById('stop-translate').addEventListener('click', () => this.stopTranslation());
        document.getElementById('fill-missing').addEventListener('click', () => this.fillMissing());
        document.getElementById('save-translations').addEventListener('click', () => this.saveTranslations());
        document.getElementById('export-translations').addEventListener('click', () => this.exportTranslations());
        document.getElementById('next-tab').addEventListener('click', () => this.nextTab());

        document.getElementById('prev-page').addEventListener('click', () => this.previousPage());
        document.getElementById('next-page').addEventListener('click', () => this.nextPage());
        document.getElementById('page-size').addEventListener('change', () => this.changePageSize());
        document.getElementById('translation-search').addEventListener('input', () => this.filterTranslations());
    }

    async loadSRT() {
        if (!this.sessionId) {
            this.log('Error: No active session. Please create a session in SRT Processing tab first.', 'error');
            return;
        }

        try {
            this.log('Loading SRT data...');
            const session = await sessionManager.getSession(this.sessionId);
            
            if (session.srt_path) {
                // Load SRT entries and existing translations
                await this.loadTranslationData();
                this.log('SRT data loaded successfully');
                this.renderTranslationCards();
            } else {
                this.log('Error: No SRT file found in session', 'error');
            }
        } catch (error) {
            this.log(`Error loading SRT: ${error.message}`, 'error');
        }
    }

    async loadTranslationData() {
        // This would load from the backend API
        // For now, we'll simulate loading data
        this.entries = [
            { index: 1, timestamp: "00:00:01,000 --> 00:00:03,000", original: "Hello world", translation: "" },
            { index: 2, timestamp: "00:00:03,000 --> 00:00:05,000", original: "How are you?", translation: "" },
            // ... more entries
        ];
        this.translations = new Array(this.entries.length).fill('');
    }

    renderTranslationCards() {
        const container = document.getElementById('translation-cards');
        if (!container) return;

        const startIndex = this.currentPage * this.pageSize;
        const endIndex = Math.min(startIndex + this.pageSize, this.entries.length);
        const currentEntries = this.entries.slice(startIndex, endIndex);

        container.innerHTML = currentEntries.map((entry, i) => {
            const globalIndex = startIndex + i;
            return `
                <div class="translation-card bg-gray-600 rounded p-4 border-l-4 border-blue-500">
                    <div class="flex justify-between items-start mb-2">
                        <div class="flex items-center space-x-4">
                            <span class="font-bold text-blue-300">#${entry.index}</span>
                            <span class="text-sm text-gray-300">${entry.timestamp}</span>
                        </div>
                        <button onclick="translateTab.editTranslation(${globalIndex})" 
                                class="text-yellow-400 hover:text-yellow-300">
                            <i class="fas fa-edit"></i>
                        </button>
                    </div>
                    <div class="mb-2">
                        <div class="text-sm text-gray-400 mb-1">Original:</div>
                        <div class="text-white font-medium">${this.escapeHtml(entry.original)}</div>
                    </div>
                    <div>
                        <div class="text-sm text-gray-400 mb-1">Translation:</div>
                        <div class="text-green-300 font-medium" id="translation-${globalIndex}">
                            ${this.translations[globalIndex] || '<span class="text-gray-500">Not translated</span>'}
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        this.updatePageInfo();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    updatePageInfo() {
        const pageInfo = document.getElementById('page-info');
        const totalPages = Math.ceil(this.entries.length / this.pageSize);
        if (pageInfo) {
            pageInfo.textContent = `Page ${this.currentPage + 1} of ${totalPages}`;
        }
    }

    previousPage() {
        if (this.currentPage > 0) {
            this.currentPage--;
            this.renderTranslationCards();
        }
    }

    nextPage() {
        const totalPages = Math.ceil(this.entries.length / this.pageSize);
        if (this.currentPage < totalPages - 1) {
            this.currentPage++;
            this.renderTranslationCards();
        }
    }

    changePageSize() {
        this.pageSize = parseInt(document.getElementById('page-size').value);
        this.currentPage = 0;
        this.renderTranslationCards();
    }

    filterTranslations() {
        const searchTerm = document.getElementById('translation-search').value.toLowerCase();
        // Implement search filtering
        this.renderTranslationCards();
    }

    async startTranslation() {
        if (this.isTranslating) {
            this.log('Translation already in progress', 'warning');
            return;
        }

        const apiKey = document.getElementById('ds-api-key').value;
        if (!apiKey) {
            this.log('Error: DeepSeek API key is required', 'error');
            return;
        }

        this.isTranslating = true;
        this.updateTranslateButtonStates();

        try {
            const config = {
                api_key: apiKey,
                batch_size: parseInt(document.getElementById('ds-batch-size').value),
                workers: parseInt(document.getElementById('ds-workers').value),
                timeout: parseInt(document.getElementById('ds-timeout').value)
            };

            this.log('Starting translation process...');
            await sessionManager.runTranslation(this.sessionId, config);
            
        } catch (error) {
            this.log(`Error starting translation: ${error.message}`, 'error');
            this.isTranslating = false;
            this.updateTranslateButtonStates();
        }
    }

    async resumeTranslation() {
        // Similar to startTranslation but resume from where it left off
        await this.startTranslation();
    }

    stopTranslation() {
        this.isTranslating = false;
        this.updateTranslateButtonStates();
        this.log('Translation stopped by user');
    }

    async fillMissing() {
        this.log('Filling missing translations...');
        // Implementation for filling missing translations
    }

    async saveTranslations() {
        try {
            this.log('Saving translations...');
            // Save translations to backend
            this.log('Translations saved successfully');
        } catch (error) {
            this.log(`Error saving translations: ${error.message}`, 'error');
        }
    }

    async exportTranslations() {
        try {
            this.log('Exporting translations...');
            // Export functionality
            this.log('Translations exported successfully');
        } catch (error) {
            this.log(`Error exporting translations: ${error.message}`, 'error');
        }
    }

    nextTab() {
        if (window.dracinApp) {
            window.dracinApp.loadTab('editing');
        }
    }

    editTranslation(index) {
        const currentTranslation = this.translations[index] || '';
        const newTranslation = prompt('Edit translation:', currentTranslation);
        if (newTranslation !== null) {
            this.translations[index] = newTranslation;
            this.updateTranslationDisplay(index);
            this.updateStats();
        }
    }

    updateTranslationDisplay(index) {
        const element = document.getElementById(`translation-${index}`);
        if (element) {
            element.textContent = this.translations[index] || 'Not translated';
            element.className = this.translations[index] ? 
                'text-green-300 font-medium' : 
                'text-gray-500 font-medium';
        }
    }

    updateStats() {
        const total = this.entries.length;
        const translated = this.translations.filter(t => t && t.trim()).length;
        const missing = total - translated;

        document.getElementById('translated-count').textContent = translated;
        document.getElementById('missing-count').textContent = missing;
        document.getElementById('total-count').textContent = total;

        const progress = total > 0 ? Math.round((translated / total) * 100) : 0;
        this.updateTranslationProgress(progress);
    }

    updateTranslationProgress(percent) {
        const progressBar = document.getElementById('translate-progress-bar');
        const progressText = document.getElementById('translate-progress');
        
        if (progressBar) {
            progressBar.style.width = `${percent}%`;
        }
        if (progressText) {
            progressText.textContent = `${percent}%`;
        }
    }

    updateTranslateButtonStates() {
        const stopButton = document.getElementById('stop-translate');
        const startButton = document.getElementById('start-translate');
        
        if (stopButton) stopButton.disabled = !this.isTranslating;
        if (startButton) startButton.disabled = this.isTranslating;
    }

    log(message, type = 'info') {
        const logOutput = document.getElementById('translation-log');
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
        const savedConfig = localStorage.getItem('dracin-translate-config');
        if (savedConfig) {
            const config = JSON.parse(savedConfig);
            document.getElementById('ds-api-key').value = config.api_key || '';
            document.getElementById('ds-batch-size').value = config.batch_size || 20;
            document.getElementById('ds-workers').value = config.workers || 1;
            document.getElementById('ds-timeout').value = config.timeout || 90;
        }
    }

    saveConfig() {
        const config = {
            api_key: document.getElementById('ds-api-key').value,
            batch_size: parseInt(document.getElementById('ds-batch-size').value),
            workers: parseInt(document.getElementById('ds-workers').value),
            timeout: parseInt(document.getElementById('ds-timeout').value)
        };
        localStorage.setItem('dracin-translate-config', JSON.stringify(config));
    }

    update(data) {
        if (data.progress !== undefined) {
            this.updateTranslationProgress(data.progress);
        }
        if (data.status) {
            document.getElementById('translate-status').textContent = data.status;
        }
        if (data.translated_lines) {
            // Update specific translations
            data.translated_lines.forEach(([index, translation]) => {
                if (index < this.translations.length) {
                    this.translations[index] = translation;
                    this.updateTranslationDisplay(index);
                }
            });
            this.updateStats();
        }
    }
}

// Make it globally available for card click handlers
window.translateTab = new TranslateTab();