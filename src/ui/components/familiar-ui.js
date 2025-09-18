/**
 * Familiar UI Component
 * Main UI coordinator for the familiar system
 */

export class FamiliarUI {
    constructor(options = {}) {
        this.familiar = options.familiar;
        this.container = options.container || document.body;
        this.initialized = false;
        this.theme = 'dark'; // Default Foundry theme
    }

    /**
     * Initialize the familiar UI system
     */
    async initialize() {
        if (this.initialized) return;

        try {
            // Add global styles
            this.injectStyles();

            // Initialize chat interface styles
            this.initializeChatStyles();

            // Set up global event listeners
            this.setupGlobalEventListeners();

            // Set up theme detection
            this.detectTheme();

            this.initialized = true;
            console.log('Familiar UI | Initialized successfully');

        } catch (error) {
            console.error('Familiar UI | Initialization failed:', error);
        }
    }

    /**
     * Inject global CSS styles
     */
    injectStyles() {
        const existingStyles = document.getElementById('familiar-ui-styles');
        if (existingStyles) {
            existingStyles.remove();
        }

        const styleSheet = document.createElement('style');
        styleSheet.id = 'familiar-ui-styles';
        styleSheet.textContent = this.getGlobalStyles();

        document.head.appendChild(styleSheet);
    }

    /**
     * Get global CSS styles
     */
    getGlobalStyles() {
        return `
            /* Chat Interface Styles */
            .familiar-chat-interface {
                position: fixed;
                width: 400px;
                height: 500px;
                background: rgba(20, 20, 20, 0.95);
                border: 2px solid #333;
                border-radius: 10px;
                font-family: "Signika", sans-serif;
                font-size: 14px;
                z-index: 1001;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(10px);
                overflow: hidden;
                transition: all 0.3s ease;
            }

            .familiar-chat-interface.chat-opening {
                transform: scale(0.9);
                opacity: 0;
                animation: chat-open 0.3s ease forwards;
            }

            .familiar-chat-interface.chat-closing {
                animation: chat-close 0.3s ease forwards;
            }

            .familiar-chat-interface.chat-minimized {
                height: 60px;
                overflow: hidden;
            }

            .familiar-chat-interface.chat-minimized .chat-messages,
            .familiar-chat-interface.chat-minimized .chat-input-area {
                display: none;
            }

            @keyframes chat-open {
                from { transform: scale(0.9); opacity: 0; }
                to { transform: scale(1); opacity: 1; }
            }

            @keyframes chat-close {
                from { transform: scale(1); opacity: 1; }
                to { transform: scale(0.9); opacity: 0; }
            }

            /* Chat Header */
            .chat-container {
                display: flex;
                flex-direction: column;
                height: 100%;
            }

            .chat-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px 20px;
                background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
                border-bottom: 1px solid #444;
            }

            .chat-title {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .chat-raven-icon {
                font-size: 20px;
                filter: drop-shadow(0 0 5px rgba(255, 255, 255, 0.3));
            }

            .chat-title h3 {
                margin: 0;
                color: #fff;
                font-size: 16px;
                font-weight: 600;
            }

            .chat-controls {
                display: flex;
                gap: 5px;
            }

            .chat-controls button {
                background: none;
                border: none;
                color: #ccc;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                padding: 5px 10px;
                border-radius: 3px;
                transition: all 0.2s ease;
            }

            .chat-controls button:hover {
                background: rgba(255, 255, 255, 0.1);
                color: #fff;
            }

            /* Chat Messages */
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 15px;
                background: rgba(10, 10, 10, 0.3);
            }

            .message {
                display: flex;
                margin-bottom: 15px;
                animation: message-appear 0.3s ease;
            }

            @keyframes message-appear {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .message-avatar {
                width: 36px;
                height: 36px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                font-weight: bold;
                margin-right: 12px;
                flex-shrink: 0;
            }

            .familiar-message .message-avatar {
                background: linear-gradient(135deg, #333 0%, #111 100%);
                color: #fff;
                border: 2px solid #555;
            }

            .user-message {
                flex-direction: row-reverse;
            }

            .user-message .message-avatar {
                background: linear-gradient(135deg, #4a4a4a 0%, #2a2a2a 100%);
                color: #fff;
                border: 2px solid #666;
                margin-right: 0;
                margin-left: 12px;
            }

            .message-content {
                flex: 1;
                background: rgba(40, 40, 40, 0.8);
                padding: 12px 15px;
                border-radius: 15px;
                position: relative;
                max-width: 280px;
            }

            .user-message .message-content {
                background: rgba(60, 60, 60, 0.8);
                text-align: right;
            }

            .message-text {
                line-height: 1.4;
                color: #fff;
                margin-bottom: 5px;
            }

            .message-text p {
                margin: 0 0 8px 0;
            }

            .message-text strong {
                color: #ff6b35;
                font-weight: 600;
            }

            .message-text em {
                color: #ccc;
                font-style: italic;
            }

            .message-text code {
                background: rgba(0, 0, 0, 0.3);
                padding: 2px 6px;
                border-radius: 3px;
                font-family: monospace;
                font-size: 12px;
            }

            .message-text ul {
                margin: 8px 0;
                padding-left: 20px;
            }

            .message-text li {
                margin: 4px 0;
            }

            .message-time {
                font-size: 11px;
                color: #888;
                display: block;
            }

            /* Thinking indicator */
            .thinking-indicator .message-content {
                padding: 15px;
            }

            .thinking-dots {
                display: flex;
                gap: 4px;
                justify-content: center;
            }

            .thinking-dots span {
                width: 8px;
                height: 8px;
                background: #666;
                border-radius: 50%;
                animation: thinking-pulse 1.5s ease-in-out infinite;
            }

            .thinking-dots span:nth-child(2) {
                animation-delay: 0.2s;
            }

            .thinking-dots span:nth-child(3) {
                animation-delay: 0.4s;
            }

            @keyframes thinking-pulse {
                0%, 60%, 100% { opacity: 0.3; transform: scale(1); }
                30% { opacity: 1; transform: scale(1.2); }
            }

            /* Chat Input Area */
            .chat-input-area {
                border-top: 1px solid #444;
                background: rgba(20, 20, 20, 0.9);
            }

            .chat-context-info {
                display: flex;
                justify-content: space-between;
                padding: 8px 15px;
                font-size: 11px;
                color: #888;
                border-bottom: 1px solid #333;
            }

            .chat-input-container {
                display: flex;
                padding: 15px;
                gap: 10px;
                align-items: flex-end;
            }

            .chat-input {
                flex: 1;
                background: rgba(40, 40, 40, 0.8);
                border: 1px solid #555;
                border-radius: 8px;
                padding: 10px 12px;
                color: #fff;
                font-family: inherit;
                font-size: 14px;
                resize: none;
                min-height: 40px;
                max-height: 120px;
                transition: all 0.2s ease;
            }

            .chat-input:focus {
                outline: none;
                border-color: #ff6b35;
                box-shadow: 0 0 0 2px rgba(255, 107, 53, 0.2);
            }

            .chat-input::placeholder {
                color: #888;
            }

            .chat-send {
                background: linear-gradient(135deg, #ff6b35 0%, #e55a2b 100%);
                border: none;
                border-radius: 8px;
                padding: 10px 15px;
                color: #fff;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 44px;
                height: 44px;
            }

            .chat-send:hover:not(:disabled) {
                background: linear-gradient(135deg, #e55a2b 0%, #cc4d24 100%);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
            }

            .chat-send:disabled {
                background: #444;
                cursor: not-allowed;
                opacity: 0.5;
            }

            .chat-quick-actions {
                display: flex;
                gap: 8px;
                padding: 0 15px 15px 15px;
                flex-wrap: wrap;
            }

            .quick-action {
                background: rgba(60, 60, 60, 0.8);
                border: 1px solid #666;
                border-radius: 15px;
                padding: 6px 12px;
                font-size: 12px;
                color: #ccc;
                cursor: pointer;
                transition: all 0.2s ease;
                white-space: nowrap;
            }

            .quick-action:hover {
                background: rgba(80, 80, 80, 0.8);
                color: #fff;
                border-color: #888;
            }

            /* Welcome message styling */
            .welcome-message {
                margin-bottom: 20px;
            }

            .welcome-message .message-content {
                background: linear-gradient(135deg, rgba(255, 107, 53, 0.2) 0%, rgba(255, 107, 53, 0.1) 100%);
                border: 1px solid rgba(255, 107, 53, 0.3);
                max-width: none;
            }

            /* Message type variations */
            .message-fallback .message-content {
                background: linear-gradient(135deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 193, 7, 0.1) 100%);
                border: 1px solid rgba(255, 193, 7, 0.3);
            }

            .message-error .message-content {
                background: linear-gradient(135deg, rgba(220, 53, 69, 0.2) 0%, rgba(220, 53, 69, 0.1) 100%);
                border: 1px solid rgba(220, 53, 69, 0.3);
            }

            .message-warning .message-content {
                background: linear-gradient(135deg, rgba(255, 152, 0, 0.2) 0%, rgba(255, 152, 0, 0.1) 100%);
                border: 1px solid rgba(255, 152, 0, 0.3);
            }

            /* Scrollbar styling */
            .chat-messages::-webkit-scrollbar {
                width: 6px;
            }

            .chat-messages::-webkit-scrollbar-track {
                background: rgba(0, 0, 0, 0.2);
                border-radius: 3px;
            }

            .chat-messages::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 3px;
            }

            .chat-messages::-webkit-scrollbar-thumb:hover {
                background: rgba(255, 255, 255, 0.5);
            }

            /* Responsive adjustments */
            @media (max-width: 1024px) {
                .familiar-chat-interface {
                    width: 350px;
                    height: 450px;
                }
            }

            @media (max-width: 768px) {
                .familiar-chat-interface {
                    width: 300px;
                    height: 400px;
                }

                .message-content {
                    max-width: 220px;
                }
            }
        `;
    }

    /**
     * Initialize chat-specific styles
     */
    initializeChatStyles() {
        // Any additional dynamic styling for chat interface
        console.log('Familiar UI | Chat styles initialized');
    }

    /**
     * Set up global event listeners
     */
    setupGlobalEventListeners() {
        // Listen for theme changes
        document.addEventListener('foundry-theme-change', (e) => {
            this.handleThemeChange(e.detail.theme);
        });

        // Listen for window resize
        window.addEventListener('resize', this.handleWindowResize.bind(this));

        // Listen for Foundry ready state changes
        document.addEventListener('foundry-ready', () => {
            this.detectTheme();
        });
    }

    /**
     * Detect current Foundry theme
     */
    detectTheme() {
        if (typeof getComputedStyle !== 'undefined') {
            const bodyStyles = getComputedStyle(document.body);
            const bgColor = bodyStyles.backgroundColor;

            // Simple detection based on background color
            if (bgColor && bgColor.includes('rgb')) {
                const rgb = bgColor.match(/\d+/g);
                if (rgb && rgb.length >= 3) {
                    const brightness = (parseInt(rgb[0]) + parseInt(rgb[1]) + parseInt(rgb[2])) / 3;
                    this.theme = brightness > 128 ? 'light' : 'dark';
                }
            }
        }

        this.updateThemeColors();
    }

    /**
     * Handle theme changes
     */
    handleThemeChange(newTheme) {
        this.theme = newTheme;
        this.updateThemeColors();
    }

    /**
     * Update theme-specific colors
     */
    updateThemeColors() {
        const root = document.documentElement;

        if (this.theme === 'light') {
            root.style.setProperty('--familiar-bg', 'rgba(240, 240, 240, 0.95)');
            root.style.setProperty('--familiar-text', '#333');
            root.style.setProperty('--familiar-border', '#ccc');
        } else {
            root.style.setProperty('--familiar-bg', 'rgba(20, 20, 20, 0.95)');
            root.style.setProperty('--familiar-text', '#fff');
            root.style.setProperty('--familiar-border', '#444');
        }
    }

    /**
     * Handle window resize events
     */
    handleWindowResize() {
        // Reposition chat interface if it's open
        const chatInterface = document.querySelector('.familiar-chat-interface');
        if (chatInterface && chatInterface.style.display !== 'none') {
            // Trigger reposition
            setTimeout(() => {
                const event = new CustomEvent('familiar-chat-reposition');
                document.dispatchEvent(event);
            }, 100);
        }
    }

    /**
     * Create notification popup
     */
    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `familiar-notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${this.getNotificationIcon(type)}</span>
                <span class="notification-text">${message}</span>
            </div>
        `;

        // Add notification styles if not already present
        this.ensureNotificationStyles();

        document.body.appendChild(notification);

        // Show notification
        setTimeout(() => notification.classList.add('show'), 10);

        // Remove notification
        setTimeout(() => {
            notification.classList.add('hide');
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }

    /**
     * Get notification icon based on type
     */
    getNotificationIcon(type) {
        const icons = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };
        return icons[type] || icons.info;
    }

    /**
     * Ensure notification styles are loaded
     */
    ensureNotificationStyles() {
        if (document.getElementById('familiar-notification-styles')) return;

        const styleSheet = document.createElement('style');
        styleSheet.id = 'familiar-notification-styles';
        styleSheet.textContent = `
            .familiar-notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                z-index: 10000;
                transform: translateX(400px);
                transition: transform 0.3s ease;
                max-width: 300px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            }

            .familiar-notification.show {
                transform: translateX(0);
            }

            .familiar-notification.hide {
                transform: translateX(400px);
            }

            .notification-content {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .notification-icon {
                font-size: 18px;
                flex-shrink: 0;
            }

            .notification-text {
                flex: 1;
                font-size: 14px;
                line-height: 1.4;
            }

            .notification-success {
                background: rgba(40, 167, 69, 0.9);
            }

            .notification-warning {
                background: rgba(255, 193, 7, 0.9);
                color: #333;
            }

            .notification-error {
                background: rgba(220, 53, 69, 0.9);
            }
        `;

        document.head.appendChild(styleSheet);
    }

    /**
     * Get current UI state
     */
    getState() {
        return {
            initialized: this.initialized,
            theme: this.theme,
            chatInterfaceOpen: !!document.querySelector('.familiar-chat-interface[style*="block"]')
        };
    }

    /**
     * Cleanup and destroy UI
     */
    destroy() {
        // Remove styles
        const styles = document.getElementById('familiar-ui-styles');
        if (styles) styles.remove();

        const notificationStyles = document.getElementById('familiar-notification-styles');
        if (notificationStyles) notificationStyles.remove();

        // Remove event listeners
        window.removeEventListener('resize', this.handleWindowResize);

        this.initialized = false;
        console.log('Familiar UI | Destroyed');
    }
}