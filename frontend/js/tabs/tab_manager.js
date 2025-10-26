// frontend/js/tabs/tab_manager.js
import { SRTProcessingTab } from './srt_processing.js';
import { TranslateTab } from './translate.js';
import { EditingTab } from './editing.js';
import { TTSExportTab } from './tts_export.js';

export class TabManager {
    constructor() {
        this.tabs = {
            'srt-processing': new SRTProcessingTab(),
            'translate': new TranslateTab(),
            'editing': new EditingTab(),
            'tts-export': new TTSExportTab()
        };
        this.currentSessionId = null;
    }

    async loadTab(tabName, sessionId = null) {
        if (sessionId) {
            this.currentSessionId = sessionId;
        }

        const tab = this.tabs[tabName];
        if (tab && typeof tab.load === 'function') {
            await tab.load(this.currentSessionId);
        }

        // Update tab content container
        const tabContent = document.getElementById(tabName);
        if (tabContent && typeof tab.render === 'function') {
            tabContent.innerHTML = tab.render();
            if (typeof tab.afterRender === 'function') {
                await tab.afterRender();
            }
        }
    }

    updateTabContent(tabName, data) {
        const tab = this.tabs[tabName];
        if (tab && typeof tab.update === 'function') {
            tab.update(data);
        }
    }

    updateProgress(data) {
        // Update progress across all tabs if needed
        Object.values(this.tabs).forEach(tab => {
            if (tab && typeof tab.updateProgress === 'function') {
                tab.updateProgress(data);
            }
        });
    }

    setCurrentSession(sessionId) {
        this.currentSessionId = sessionId;
        Object.values(this.tabs).forEach(tab => {
            if (tab && typeof tab.setSession === 'function') {
                tab.setSession(sessionId);
            }
        });
    }
}