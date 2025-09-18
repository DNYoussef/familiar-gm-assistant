/**
 * Familiar GM Assistant - Foundry VTT Module
 * Raven familiar with Pathfinder 2e rules assistance
 */

import { FamiliarUI } from './components/familiar-ui.js';
import { RavenFamiliar } from './components/raven-familiar.js';
import { ChatInterface } from './components/chat-interface.js';
import { PathfinderIntegration } from '../core/pathfinder-integration.js';

class FamiliarGMAssistant {
    constructor() {
        this.familiar = null;
        this.ui = null;
        this.chat = null;
        this.pathfinder = null;
        this.initialized = false;
    }

    /**
     * Initialize the Familiar GM Assistant
     */
    async initialize() {
        if (this.initialized) return;

        console.log("Familiar GM Assistant | Initializing...");

        try {
            // Initialize core components
            this.pathfinder = new PathfinderIntegration();
            await this.pathfinder.initialize();

            // Initialize UI components
            this.familiar = new RavenFamiliar({
                position: 'bottom-right',
                pathfinderRules: this.pathfinder
            });

            this.ui = new FamiliarUI({
                familiar: this.familiar,
                container: document.body
            });

            this.chat = new ChatInterface({
                familiar: this.familiar,
                ragBackend: 'http://localhost:3001/api'
            });

            // Set up event listeners
            this.setupEventListeners();

            // Initialize components
            await this.familiar.initialize();
            await this.ui.initialize();
            await this.chat.initialize();

            this.initialized = true;
            console.log("Familiar GM Assistant | Successfully initialized");

            // Show welcome message
            this.showWelcomeMessage();

        } catch (error) {
            console.error("Familiar GM Assistant | Initialization failed:", error);
            ui.notifications.error("Failed to initialize Familiar GM Assistant");
        }
    }

    /**
     * Set up event listeners for Foundry hooks
     */
    setupEventListeners() {
        // Initialize when Foundry is ready
        Hooks.once('ready', () => {
            this.initialize();
        });

        // Update familiar when scene changes
        Hooks.on('canvasReady', () => {
            this.familiar?.updateContext();
        });

        // Respond to chat messages for rules questions
        Hooks.on('createChatMessage', (message) => {
            this.handleChatMessage(message);
        });

        // Update context on actor changes
        Hooks.on('updateActor', () => {
            this.familiar?.updateContext();
        });
    }

    /**
     * Handle chat messages for potential rules assistance
     */
    async handleChatMessage(message) {
        if (!this.initialized || !message.content) return;

        const content = message.content.toLowerCase();

        // Detect rules-related questions
        const rulesKeywords = ['rule', 'ruling', 'how does', 'can i', 'pathfinder', 'pf2e'];
        const hasRulesKeyword = rulesKeywords.some(keyword => content.includes(keyword));

        if (hasRulesKeyword && content.includes('?')) {
            // Trigger familiar response
            setTimeout(() => {
                this.familiar.suggestRulesAssistance(message.content);
            }, 1000);
        }
    }

    /**
     * Show welcome message when module loads
     */
    showWelcomeMessage() {
        ChatMessage.create({
            content: `
                <div class="familiar-welcome">
                    <h3>🐦‍⬛ Your Familiar Has Arrived!</h3>
                    <p>I'm your raven familiar, here to assist with Pathfinder 2e rules and GM tasks.</p>
                    <p><em>Click on me in the bottom-right corner to start chatting!</em></p>
                </div>
            `,
            whisper: [game.user.id]
        });
    }

    /**
     * Get the familiar instance
     */
    getFamiliar() {
        return this.familiar;
    }
}

// Initialize the module
const familiarAssistant = new FamiliarGMAssistant();

// Export for global access
window.FamiliarGMAssistant = familiarAssistant;

// Foundry module hooks
Hooks.once('init', () => {
    console.log("Familiar GM Assistant | Module registered");
});

export { FamiliarGMAssistant };