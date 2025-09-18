/**
 * Chat Interface Component
 * Handles communication between user and the raven familiar
 */

export class ChatInterface {
    constructor(options = {}) {
        this.familiar = options.familiar;
        this.ragBackend = options.ragBackend || 'http://localhost:3001/api';
        this.element = null;
        this.isOpen = false;
        this.conversationHistory = [];
        this.socket = null;
    }

    /**
     * Initialize the chat interface
     */
    async initialize() {
        this.createChatElement();
        this.attachEventListeners();
        this.connectToBackend();

        console.log("Chat Interface | Initialized successfully");
    }

    /**
     * Create the chat interface DOM element
     */
    createChatElement() {
        this.element = document.createElement('div');
        this.element.className = 'familiar-chat-interface';
        this.element.style.display = 'none';

        this.element.innerHTML = `
            <div class="chat-container">
                <div class="chat-header">
                    <div class="chat-title">
                        <span class="chat-raven-icon">🐦‍⬛</span>
                        <h3>Familiar Assistant</h3>
                    </div>
                    <div class="chat-controls">
                        <button class="chat-minimize" title="Minimize">—</button>
                        <button class="chat-close" title="Close">×</button>
                    </div>
                </div>

                <div class="chat-messages">
                    <div class="welcome-message">
                        <div class="message familiar-message">
                            <div class="message-avatar">🐦‍⬛</div>
                            <div class="message-content">
                                <p>Greetings! I'm your raven familiar, here to help with:</p>
                                <ul>
                                    <li>Pathfinder 2e rules clarification</li>
                                    <li>Combat mechanics</li>
                                    <li>Spell descriptions</li>
                                    <li>Character abilities</li>
                                    <li>GM assistance</li>
                                </ul>
                                <p>Ask me anything about your campaign!</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="chat-input-area">
                    <div class="chat-context-info">
                        <span class="context-scene">Scene: Unknown</span>
                        <span class="context-mode">Mode: GM</span>
                    </div>
                    <div class="chat-input-container">
                        <textarea class="chat-input" placeholder="Ask me about Pathfinder 2e rules..." rows="2"></textarea>
                        <button class="chat-send" disabled>
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                    <div class="chat-quick-actions">
                        <button class="quick-action" data-action="rules">Rules Question</button>
                        <button class="quick-action" data-action="combat">Combat Help</button>
                        <button class="quick-action" data-action="spells">Spell Lookup</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(this.element);
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Listen for familiar chat open event
        document.addEventListener('familiar-chat-open', (e) => {
            this.openChat();
        });

        // Chat controls
        this.element.querySelector('.chat-close').addEventListener('click', () => {
            this.closeChat();
        });

        this.element.querySelector('.chat-minimize').addEventListener('click', () => {
            this.minimizeChat();
        });

        // Input handling
        const chatInput = this.element.querySelector('.chat-input');
        const sendButton = this.element.querySelector('.chat-send');

        chatInput.addEventListener('input', (e) => {
            sendButton.disabled = e.target.value.trim() === '';
        });

        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        sendButton.addEventListener('click', () => {
            this.sendMessage();
        });

        // Quick actions
        this.element.querySelectorAll('.quick-action').forEach(button => {
            button.addEventListener('click', (e) => {
                this.handleQuickAction(e.target.dataset.action);
            });
        });

        // Auto-resize textarea
        chatInput.addEventListener('input', this.autoResizeTextarea.bind(this));
    }

    /**
     * Connect to RAG backend
     */
    async connectToBackend() {
        try {
            // Test backend connection
            const response = await fetch(`${this.ragBackend}/health`);
            if (response.ok) {
                console.log("Chat Interface | Connected to RAG backend");

                // Initialize WebSocket connection for real-time responses
                if (typeof io !== 'undefined') {
                    this.socket = io(this.ragBackend.replace('/api', ''));
                    this.setupSocketListeners();
                }
            }
        } catch (error) {
            console.warn("Chat Interface | RAG backend unavailable, using fallback mode");
            this.showOfflineMessage();
        }
    }

    /**
     * Setup WebSocket listeners
     */
    setupSocketListeners() {
        if (!this.socket) return;

        this.socket.on('familiar-response', (data) => {
            this.receiveFamiliarMessage(data.message, data.type || 'response');
        });

        this.socket.on('familiar-thinking', () => {
            this.showThinkingIndicator();
        });

        this.socket.on('familiar-context-update', (context) => {
            this.updateContextInfo(context);
        });
    }

    /**
     * Open chat interface
     */
    openChat() {
        if (this.isOpen) return;

        this.element.style.display = 'block';
        this.element.classList.add('chat-opening');

        // Position chat interface
        this.positionChat();

        // Update context
        this.updateContextInfo(this.familiar.getContext());

        // Focus input
        setTimeout(() => {
            this.element.querySelector('.chat-input').focus();
            this.element.classList.remove('chat-opening');
        }, 300);

        this.isOpen = true;
        this.familiar.setState('listening');
    }

    /**
     * Close chat interface
     */
    closeChat() {
        this.element.classList.add('chat-closing');

        setTimeout(() => {
            this.element.style.display = 'none';
            this.element.classList.remove('chat-closing');
            this.isOpen = false;
            this.familiar.setState('idle');
        }, 300);
    }

    /**
     * Minimize chat interface
     */
    minimizeChat() {
        this.element.classList.toggle('chat-minimized');
    }

    /**
     * Position chat interface relative to familiar
     */
    positionChat() {
        const familiarRect = this.familiar.element.getBoundingClientRect();
        const chatWidth = 400;
        const chatHeight = 500;

        let left = familiarRect.left - chatWidth - 10;
        let top = familiarRect.top - chatHeight + 80;

        // Ensure chat stays within viewport
        if (left < 10) {
            left = familiarRect.right + 10;
        }
        if (top < 10) {
            top = 10;
        }
        if (top + chatHeight > window.innerHeight - 10) {
            top = window.innerHeight - chatHeight - 10;
        }

        this.element.style.left = `${left}px`;
        this.element.style.top = `${top}px`;
    }

    /**
     * Send user message
     */
    async sendMessage() {
        const input = this.element.querySelector('.chat-input');
        const message = input.value.trim();

        if (!message) return;

        // Add user message to chat
        this.addUserMessage(message);

        // Clear input
        input.value = '';
        this.element.querySelector('.chat-send').disabled = true;

        // Show thinking indicator
        this.showThinkingIndicator();
        this.familiar.setState('thinking');

        // Process message
        await this.processMessage(message);
    }

    /**
     * Process user message and get response
     */
    async processMessage(message) {
        try {
            const context = this.familiar.getContext();
            const payload = {
                message: message,
                context: context,
                conversationHistory: this.conversationHistory.slice(-10) // Last 10 messages
            };

            const response = await fetch(`${this.ragBackend}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const data = await response.json();
                this.receiveFamiliarMessage(data.response, data.type);

                // Update conversation history
                this.conversationHistory.push(
                    { role: 'user', content: message, timestamp: Date.now() },
                    { role: 'assistant', content: data.response, timestamp: Date.now() }
                );
            } else {
                this.handleFallbackResponse(message);
            }
        } catch (error) {
            console.error('Chat Interface | Error processing message:', error);
            this.handleFallbackResponse(message);
        } finally {
            this.hideThinkingIndicator();
            this.familiar.setState('idle');
        }
    }

    /**
     * Handle fallback responses when backend is unavailable
     */
    handleFallbackResponse(message) {
        const fallbackResponses = {
            rules: "I'd love to help with rules questions! However, I need my backend to access the Pathfinder 2e database. Please check that the backend server is running.",
            combat: "For combat assistance, I typically reference real-time rules data. Please ensure the RAG backend is connected.",
            default: "I'm having trouble accessing my knowledge base right now. Please make sure the backend server is running at localhost:3001."
        };

        const messageType = this.classifyMessage(message);
        const response = fallbackResponses[messageType] || fallbackResponses.default;

        this.receiveFamiliarMessage(response, 'fallback');
    }

    /**
     * Classify message type for fallback responses
     */
    classifyMessage(message) {
        const lower = message.toLowerCase();
        if (lower.includes('rule') || lower.includes('how does')) return 'rules';
        if (lower.includes('combat') || lower.includes('attack')) return 'combat';
        return 'default';
    }

    /**
     * Add user message to chat
     */
    addUserMessage(message) {
        const messagesContainer = this.element.querySelector('.chat-messages');
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';

        messageElement.innerHTML = `
            <div class="message-content">
                <p>${this.escapeHtml(message)}</p>
                <span class="message-time">${this.formatTime(new Date())}</span>
            </div>
            <div class="message-avatar">${game.user.name.charAt(0).toUpperCase()}</div>
        `;

        messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }

    /**
     * Receive familiar message
     */
    receiveFamiliarMessage(message, type = 'response') {
        const messagesContainer = this.element.querySelector('.chat-messages');
        const messageElement = document.createElement('div');
        messageElement.className = `message familiar-message message-${type}`;

        messageElement.innerHTML = `
            <div class="message-avatar">🐦‍⬛</div>
            <div class="message-content">
                <div class="message-text">${this.formatMessage(message)}</div>
                <span class="message-time">${this.formatTime(new Date())}</span>
            </div>
        `;

        messagesContainer.appendChild(messageElement);
        this.scrollToBottom();

        // Make familiar speak
        this.familiar.speak(this.truncateForSpeech(message));
        this.familiar.setState('speaking');
    }

    /**
     * Show thinking indicator
     */
    showThinkingIndicator() {
        const messagesContainer = this.element.querySelector('.chat-messages');

        // Remove existing thinking indicator
        const existing = messagesContainer.querySelector('.thinking-indicator');
        if (existing) existing.remove();

        const thinkingElement = document.createElement('div');
        thinkingElement.className = 'message familiar-message thinking-indicator';
        thinkingElement.innerHTML = `
            <div class="message-avatar">🐦‍⬛</div>
            <div class="message-content">
                <div class="thinking-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        messagesContainer.appendChild(thinkingElement);
        this.scrollToBottom();
    }

    /**
     * Hide thinking indicator
     */
    hideThinkingIndicator() {
        const thinking = this.element.querySelector('.thinking-indicator');
        if (thinking) thinking.remove();
    }

    /**
     * Handle quick actions
     */
    handleQuickAction(action) {
        const quickPrompts = {
            rules: "Can you help me understand a Pathfinder 2e rule?",
            combat: "I need help with combat mechanics",
            spells: "Can you look up a spell for me?"
        };

        const input = this.element.querySelector('.chat-input');
        input.value = quickPrompts[action] || '';
        input.focus();
        this.element.querySelector('.chat-send').disabled = false;
    }

    /**
     * Update context information display
     */
    updateContextInfo(context) {
        const sceneSpan = this.element.querySelector('.context-scene');
        const modeSpan = this.element.querySelector('.context-mode');

        sceneSpan.textContent = `Scene: ${context.scene || 'Unknown'}`;
        modeSpan.textContent = `Mode: ${context.isGM ? 'GM' : 'Player'}`;
    }

    /**
     * Show offline message
     */
    showOfflineMessage() {
        const message = "Backend unavailable - running in limited mode. Some features may not work properly.";
        this.receiveFamiliarMessage(message, 'warning');
    }

    /**
     * Auto-resize textarea
     */
    autoResizeTextarea(e) {
        const textarea = e.target;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    /**
     * Scroll messages to bottom
     */
    scrollToBottom() {
        const messagesContainer = this.element.querySelector('.chat-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    /**
     * Format message content
     */
    formatMessage(message) {
        // Convert markdown-like formatting
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    /**
     * Escape HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Format time
     */
    formatTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    /**
     * Truncate message for speech bubble
     */
    truncateForSpeech(message, maxLength = 60) {
        if (message.length <= maxLength) return message;
        return message.substring(0, maxLength) + '...';
    }

    /**
     * Cleanup when module is disabled
     */
    destroy() {
        if (this.socket) {
            this.socket.disconnect();
        }
        if (this.element) {
            this.element.remove();
        }
    }
}