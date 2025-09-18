/**
 * Raven Familiar Component
 * Interactive raven that provides GM assistance and Pathfinder 2e rules help
 */

export class RavenFamiliar {
    constructor(options = {}) {
        this.position = options.position || 'bottom-right';
        this.pathfinderRules = options.pathfinderRules;
        this.element = null;
        this.isVisible = true;
        this.isAnimating = false;
        this.currentState = 'idle'; // idle, listening, speaking, thinking
        this.currentContext = {};
    }

    /**
     * Initialize the raven familiar
     */
    async initialize() {
        this.createElement();
        this.attachEventListeners();
        this.positionElement();
        this.startIdleAnimation();

        console.log("Raven Familiar | Initialized successfully");
    }

    /**
     * Create the raven familiar DOM element
     */
    createElement() {
        // Create container
        this.element = document.createElement('div');
        this.element.className = 'familiar-container';
        this.element.innerHTML = `
            <div class="familiar-raven" data-state="idle">
                <div class="raven-body">
                    <div class="raven-head">
                        <div class="raven-eye left-eye"></div>
                        <div class="raven-eye right-eye"></div>
                        <div class="raven-beak"></div>
                    </div>
                    <div class="raven-wing left-wing"></div>
                    <div class="raven-wing right-wing"></div>
                    <div class="raven-tail"></div>
                </div>
                <div class="familiar-speech-bubble" style="display: none;">
                    <div class="speech-content"></div>
                    <div class="speech-arrow"></div>
                </div>
                <div class="familiar-tooltip">
                    Click me for GM assistance!
                </div>
            </div>
        `;

        // Add to DOM
        document.body.appendChild(this.element);
    }

    /**
     * Position the familiar element
     */
    positionElement() {
        const positions = {
            'bottom-right': { bottom: '20px', right: '20px' },
            'bottom-left': { bottom: '20px', left: '20px' },
            'top-right': { top: '20px', right: '20px' },
            'top-left': { top: '20px', left: '20px' }
        };

        const pos = positions[this.position];
        Object.assign(this.element.style, {
            position: 'fixed',
            zIndex: '1000',
            ...pos
        });
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        const raven = this.element.querySelector('.familiar-raven');

        // Click to open chat
        raven.addEventListener('click', () => {
            this.openChat();
        });

        // Hover effects
        raven.addEventListener('mouseenter', () => {
            this.setState('listening');
            this.showTooltip();
        });

        raven.addEventListener('mouseleave', () => {
            this.setState('idle');
            this.hideTooltip();
        });

        // Context menu for settings
        raven.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.showContextMenu(e);
        });
    }

    /**
     * Start idle animation cycle
     */
    startIdleAnimation() {
        setInterval(() => {
            if (this.currentState === 'idle' && !this.isAnimating) {
                this.performIdleAction();
            }
        }, 3000 + Math.random() * 5000); // Random interval 3-8 seconds
    }

    /**
     * Perform random idle actions
     */
    performIdleAction() {
        const actions = ['blink', 'head-tilt', 'wing-flutter', 'look-around'];
        const action = actions[Math.floor(Math.random() * actions.length)];

        this.performAnimation(action);
    }

    /**
     * Perform specific animation
     */
    performAnimation(animation) {
        const raven = this.element.querySelector('.familiar-raven');

        this.isAnimating = true;
        raven.classList.add(`animation-${animation}`);

        setTimeout(() => {
            raven.classList.remove(`animation-${animation}`);
            this.isAnimating = false;
        }, 1000);
    }

    /**
     * Set familiar state
     */
    setState(state) {
        if (this.currentState === state) return;

        const raven = this.element.querySelector('.familiar-raven');
        raven.setAttribute('data-state', state);
        this.currentState = state;

        // State-specific behaviors
        switch (state) {
            case 'listening':
                this.performAnimation('alert');
                break;
            case 'speaking':
                this.performAnimation('speaking');
                break;
            case 'thinking':
                this.performAnimation('thinking');
                break;
        }
    }

    /**
     * Show tooltip
     */
    showTooltip() {
        const tooltip = this.element.querySelector('.familiar-tooltip');
        tooltip.style.display = 'block';
        tooltip.style.opacity = '1';
    }

    /**
     * Hide tooltip
     */
    hideTooltip() {
        const tooltip = this.element.querySelector('.familiar-tooltip');
        tooltip.style.opacity = '0';
        setTimeout(() => tooltip.style.display = 'none', 200);
    }

    /**
     * Open chat interface
     */
    openChat() {
        this.setState('listening');

        // Trigger chat interface
        const chatEvent = new CustomEvent('familiar-chat-open', {
            detail: { familiar: this }
        });
        document.dispatchEvent(chatEvent);
    }

    /**
     * Speak a message
     */
    speak(message, duration = 5000) {
        this.setState('speaking');

        const speechBubble = this.element.querySelector('.familiar-speech-bubble');
        const speechContent = this.element.querySelector('.speech-content');

        speechContent.textContent = message;
        speechBubble.style.display = 'block';
        speechBubble.style.opacity = '1';

        setTimeout(() => {
            speechBubble.style.opacity = '0';
            setTimeout(() => {
                speechBubble.style.display = 'none';
                this.setState('idle');
            }, 300);
        }, duration);
    }

    /**
     * Suggest rules assistance
     */
    suggestRulesAssistance(originalMessage) {
        this.setState('thinking');

        setTimeout(() => {
            this.speak("I heard a rules question! Click me to get Pathfinder 2e assistance.", 6000);
            this.performAnimation('alert');
        }, 1000);
    }

    /**
     * Update context based on current scene/actors
     */
    updateContext() {
        if (!canvas.ready) return;

        this.currentContext = {
            scene: canvas.scene?.name || 'Unknown Scene',
            actors: canvas.tokens.objects.children.map(token => ({
                name: token.name,
                actorType: token.actor?.type,
                level: token.actor?.system?.details?.level?.value
            })),
            timestamp: Date.now()
        };
    }

    /**
     * Show context menu
     */
    showContextMenu(event) {
        const menu = new ContextMenu(this.element, '.familiar-raven', [
            {
                name: "Settings",
                icon: '<i class="fas fa-cog"></i>',
                callback: () => this.openSettings()
            },
            {
                name: "Toggle Visibility",
                icon: '<i class="fas fa-eye"></i>',
                callback: () => this.toggleVisibility()
            },
            {
                name: "Reset Position",
                icon: '<i class="fas fa-crosshairs"></i>',
                callback: () => this.resetPosition()
            }
        ]);

        menu.render(event);
    }

    /**
     * Open settings dialog
     */
    openSettings() {
        // Implementation for settings dialog
        console.log("Opening familiar settings...");
    }

    /**
     * Toggle visibility
     */
    toggleVisibility() {
        this.isVisible = !this.isVisible;
        this.element.style.display = this.isVisible ? 'block' : 'none';
    }

    /**
     * Reset position
     */
    resetPosition() {
        this.position = 'bottom-right';
        this.positionElement();
    }

    /**
     * Get current context for AI assistance
     */
    getContext() {
        return {
            ...this.currentContext,
            currentUser: game.user.name,
            isGM: game.user.isGM,
            system: game.system.id
        };
    }

    /**
     * Cleanup when module is disabled
     */
    destroy() {
        if (this.element) {
            this.element.remove();
        }
    }
}