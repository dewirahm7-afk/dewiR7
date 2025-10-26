// frontend/js/utils/websocket.js
class WebSocketManager {
    constructor() {
        this.socket = null;
        this.clientId = null;
        this.sessionId = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.messageHandlers = new Map();
    }

    connect(sessionId = null) {
        this.clientId = this.generateClientId();
        this.sessionId = sessionId;

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.clientId}`;

        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            
            // Join session if provided
            if (this.sessionId) {
                this.joinSession(this.sessionId);
            }
        };

        this.socket.onmessage = (event) => {
            this.handleMessage(JSON.parse(event.data));
        };

        this.socket.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.handleReconnection();
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    handleReconnection() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            
            console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
            
            setTimeout(() => {
                this.connect(this.sessionId);
            }, delay);
        } else {
            console.error('Max reconnection attempts reached');
            this.emit('connection_lost');
        }
    }

    handleMessage(message) {
        const { type, data } = message;

        // Call registered handlers for this message type
        if (this.messageHandlers.has(type)) {
            this.messageHandlers.get(type).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in message handler for ${type}:`, error);
                }
            });
        }

        // Also emit event for global listeners
        this.emit(type, data);
    }

    joinSession(sessionId) {
        this.sessionId = sessionId;
        this.send({
            type: 'join_session',
            data: { session_id: sessionId }
        });
    }

    send(message) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected, message not sent:', message);
        }
    }

    on(messageType, handler) {
        if (!this.messageHandlers.has(messageType)) {
            this.messageHandlers.set(messageType, []);
        }
        this.messageHandlers.get(messageType).push(handler);
    }

    off(messageType, handler) {
        if (this.messageHandlers.has(messageType)) {
            const handlers = this.messageHandlers.get(messageType);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    emit(eventType, data) {
        // Dispatch custom event
        const event = new CustomEvent(`websocket:${eventType}`, { detail: data });
        window.dispatchEvent(event);
    }

    disconnect() {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
    }

    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }

    isConnected() {
        return this.socket && this.socket.readyState === WebSocket.OPEN;
    }
}

// Export singleton instance
export const websocketManager = new WebSocketManager();