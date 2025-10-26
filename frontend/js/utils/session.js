// frontend/js/utils/session.js
class SessionManager {
    constructor() {
        this.baseUrl = '/api';
        this.currentSessionId = null;
    }

    async createSession(videoName, srtName) {
        const formData = new FormData();
        formData.append('video_name', videoName);
        formData.append('srt_name', srtName);

        const response = await fetch(`${this.baseUrl}/session/create`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Failed to create session: ${response.statusText}`);
        }

        return await response.json();
    }

    async uploadFiles(sessionId, videoFile, srtFile) {
        const formData = new FormData();
        if (videoFile) {
            formData.append('video', videoFile);
        }
        formData.append('srt', srtFile);

        const response = await fetch(`${this.baseUrl}/session/${sessionId}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Failed to upload files: ${response.statusText}`);
        }

        return await response.json();
    }

    async generateWorkdir(sessionId) {
        const response = await fetch(`${this.baseUrl}/session/${sessionId}/generate-workdir`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error(`Failed to generate workdir: ${response.statusText}`);
        }

        return await response.json();
    }

    async extractAudio(sessionId) {
        const response = await fetch(`${this.baseUrl}/session/${sessionId}/extract-audio`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error(`Failed to extract audio: ${response.statusText}`);
        }

        return await response.json();
    }

    async runDiarization(sessionId, config) {
        const formData = new FormData();
        formData.append('male_ref', config.male_ref);
        formData.append('female_ref', config.female_ref);
        formData.append('hf_token', config.hf_token);
        formData.append('use_gpu', config.use_gpu);
        formData.append('top_n', config.top_n.toString());

        const response = await fetch(`${this.baseUrl}/session/${sessionId}/diarization`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Failed to run diarization: ${response.statusText}`);
        }

        return await response.json();
    }

    async runTranslation(sessionId, config) {
        const formData = new FormData();
        formData.append('api_key', config.api_key);
        formData.append('batch_size', config.batch_size.toString());
        formData.append('workers', config.workers.toString());
        formData.append('timeout', config.timeout.toString());

        const response = await fetch(`${this.baseUrl}/session/${sessionId}/translate`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Failed to start translation: ${response.statusText}`);
        }

        return await response.json();
    }

    async runTTSExport(sessionId, config) {
        const formData = new FormData();
        
        // Add all config fields to form data
        for (const [key, value] of Object.entries(config)) {
            if (typeof value === 'boolean') {
                formData.append(key, value.toString());
            } else {
                formData.append(key, value);
            }
        }

        const response = await fetch(`${this.baseUrl}/session/${sessionId}/tts-export`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Failed to start TTS export: ${response.statusText}`);
        }

        return await response.json();
    }

    async getSession(sessionId) {
        const response = await fetch(`${this.baseUrl}/session/${sessionId}`);
        
        if (!response.ok) {
            throw new Error(`Failed to get session: ${response.statusText}`);
        }

        return await response.json();
    }

    async getEditingData(sessionId) {
        const response = await fetch(`${this.baseUrl}/session/${sessionId}/editing-data`);
        
        if (!response.ok) {
            throw new Error(`Failed to get editing data: ${response.statusText}`);
        }

        return await response.json();
    }

    async editEntry(sessionId, index, text, gender) {
        const formData = new FormData();
        formData.append('index', index.toString());
        formData.append('text', text);
        formData.append('gender', gender);

        const response = await fetch(`${this.baseUrl}/session/${sessionId}/edit-entry`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Failed to edit entry: ${response.statusText}`);
        }

        return await response.json();
    }

    async listSessions() {
        const response = await fetch(`${this.baseUrl}/sessions`);
        
        if (!response.ok) {
            throw new Error(`Failed to list sessions: ${response.statusText}`);
        }

        return await response.json();
    }

    setCurrentSession(sessionId) {
        this.currentSessionId = sessionId;
    }

    getCurrentSession() {
        return this.currentSessionId;
    }
}

// Export singleton instance
export const sessionManager = new SessionManager();