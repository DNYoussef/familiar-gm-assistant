/**
 * Chat Service
 * Orchestrates responses using AI models and Pathfinder knowledge
 */

class ChatService {
    constructor(options = {}) {
        this.rulesService = options.rulesService;
        this.vectorService = options.vectorService;
        this.contextService = options.contextService;
        this.conversationMemory = new Map();
        this.stats = {
            messagesProcessed: 0,
            averageResponseTime: 0,
            lastProcessedMessage: null
        };
    }

    /**
     * Initialize chat service
     */
    async initialize() {
        console.log('Chat Service | Initialized');
    }

    /**
     * Process incoming message and generate response
     */
    async processMessage(options = {}) {
        const startTime = Date.now();
        const { message, context = {}, history = [] } = options;

        try {
            this.stats.messagesProcessed++;
            this.stats.lastProcessedMessage = message;

            // Analyze message intent
            const intent = await this.analyzeMessageIntent(message);

            // Generate response based on intent
            let response;
            switch (intent.type) {
                case 'rules_query':
                    response = await this.handleRulesQuery(message, context, intent);
                    break;
                case 'spell_lookup':
                    response = await this.handleSpellLookup(message, context, intent);
                    break;
                case 'combat_help':
                    response = await this.handleCombatHelp(message, context, intent);
                    break;
                case 'general_chat':
                    response = await this.handleGeneralChat(message, context, history);
                    break;
                default:
                    response = await this.handleDefaultResponse(message, context);
            }

            // Update conversation memory
            this.updateConversationMemory(context.currentUser || 'anonymous', {
                message,
                response: response.response,
                intent,
                timestamp: Date.now()
            });

            // Update stats
            const responseTime = Date.now() - startTime;
            this.updateResponseTimeStats(responseTime);

            return {
                ...response,
                responseTime,
                intent: intent.type
            };

        } catch (error) {
            console.error('Chat Service | Error processing message:', error);
            return this.createErrorResponse(error);
        }
    }

    /**
     * Analyze message intent using pattern matching
     */
    async analyzeMessageIntent(message) {
        const lowerMessage = message.toLowerCase();

        // Rules query patterns
        if (this.matchesPatterns(lowerMessage, [
            /how do(?:es)? .* work/,
            /what (?:is|are) the rules? for/,
            /can i .* when/,
            /rule(?:s)? (?:about|for|on)/,
            /mechanics? (?:of|for)/
        ])) {
            return {
                type: 'rules_query',
                confidence: 0.9,
                keywords: this.extractKeywords(message)
            };
        }

        // Spell lookup patterns
        if (this.matchesPatterns(lowerMessage, [
            /spell.*(?:cast|work|do)/,
            /(?:what|how) does .* spell/,
            /magic missile|fireball|heal|cure/,
            /spell(?:s)? (?:called|named)/,
            /cast(?:ing)? (?:a )?spell/
        ])) {
            return {
                type: 'spell_lookup',
                confidence: 0.8,
                spellName: this.extractSpellName(message)
            };
        }

        // Combat help patterns
        if (this.matchesPatterns(lowerMessage, [
            /combat|attack|strike|hit/,
            /initiative|turn order/,
            /damage|ac|armor class/,
            /critical hit|crit/,
            /multiple attack penalty/
        ])) {
            return {
                type: 'combat_help',
                confidence: 0.8,
                combatTopic: this.extractCombatTopic(message)
            };
        }

        // General chat (greetings, thanks, etc.)
        if (this.matchesPatterns(lowerMessage, [
            /^(?:hi|hello|hey|greetings)/,
            /thank(?:s| you)/,
            /(?:good )?(?:morning|afternoon|evening)/,
            /how are you/,
            /^(?:bye|goodbye|see you)/
        ])) {
            return {
                type: 'general_chat',
                confidence: 0.7,
                subtype: this.getGeneralChatSubtype(lowerMessage)
            };
        }

        return {
            type: 'default',
            confidence: 0.3,
            keywords: this.extractKeywords(message)
        };
    }

    /**
     * Check if message matches any of the given patterns
     */
    matchesPatterns(message, patterns) {
        return patterns.some(pattern => pattern.test(message));
    }

    /**
     * Extract keywords from message
     */
    extractKeywords(message) {
        const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why', 'can', 'could', 'would', 'should', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had']);

        return message
            .toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 2 && !stopWords.has(word))
            .slice(0, 10); // Limit to 10 keywords
    }

    /**
     * Extract spell name from message
     */
    extractSpellName(message) {
        // Common spell names to look for
        const spellNames = ['magic missile', 'fireball', 'heal', 'cure wounds', 'shield', 'mage armor', 'burning hands'];

        for (const spellName of spellNames) {
            if (message.toLowerCase().includes(spellName)) {
                return spellName;
            }
        }

        // Try to extract quoted spell names
        const quoted = message.match(/["']([^"']+)["']/);
        if (quoted) {
            return quoted[1];
        }

        return null;
    }

    /**
     * Extract combat topic from message
     */
    extractCombatTopic(message) {
        const lowerMessage = message.toLowerCase();

        if (lowerMessage.includes('initiative')) return 'initiative';
        if (lowerMessage.includes('attack') || lowerMessage.includes('strike')) return 'attack';
        if (lowerMessage.includes('damage')) return 'damage';
        if (lowerMessage.includes('critical') || lowerMessage.includes('crit')) return 'critical';
        if (lowerMessage.includes('penalty')) return 'penalty';

        return 'general';
    }

    /**
     * Get general chat subtype
     */
    getGeneralChatSubtype(message) {
        if (/^(?:hi|hello|hey|greetings)/.test(message)) return 'greeting';
        if (/thank(?:s| you)/.test(message)) return 'thanks';
        if (/^(?:bye|goodbye|see you)/.test(message)) return 'farewell';
        return 'chat';
    }

    /**
     * Handle rules queries
     */
    async handleRulesQuery(message, context, intent) {
        try {
            const searchResults = await this.rulesService.searchRules({
                query: message,
                limit: 3
            });

            if (searchResults.length === 0) {
                return {
                    response: "I couldn't find specific rules matching your query. Could you be more specific? For example, ask about 'Strike action' or 'spellcasting rules'.",
                    type: 'rules_not_found',
                    suggestions: ['Strike action', 'Cast a spell', 'Move action', 'Initiative rules']
                };
            }

            // Format the best matching rule
            const bestMatch = searchResults[0];
            let response = `**${bestMatch.name}**\n\n${bestMatch.description}`;

            if (bestMatch.requirements) {
                response += `\n\n*Requirements:* ${bestMatch.requirements}`;
            }

            if (bestMatch.mechanics) {
                response += `\n\n*Mechanics:* ${bestMatch.mechanics}`;
            }

            // Add related rules if available
            if (searchResults.length > 1) {
                response += '\n\n*Related rules:* ';
                response += searchResults.slice(1).map(rule => rule.name).join(', ');
            }

            return {
                response,
                type: 'rules_found',
                rules: searchResults,
                confidence: bestMatch.score
            };

        } catch (error) {
            return this.createErrorResponse(error, 'rules lookup');
        }
    }

    /**
     * Handle spell lookups
     */
    async handleSpellLookup(message, context, intent) {
        try {
            const spellName = intent.spellName || this.extractSpellName(message);

            if (!spellName) {
                return {
                    response: "Which spell would you like to look up? Please specify the spell name, for example: 'Tell me about Magic Missile' or 'How does Fireball work?'",
                    type: 'spell_name_needed',
                    suggestions: ['Magic Missile', 'Fireball', 'Heal', 'Shield']
                };
            }

            const spell = await this.rulesService.getSpell(spellName);

            if (!spell) {
                return {
                    response: `I couldn't find a spell called "${spellName}". Make sure you're spelling it correctly, or try searching for a similar spell name.`,
                    type: 'spell_not_found',
                    searchedFor: spellName
                };
            }

            let response = `**${spell.name}** (Level ${spell.level})\n\n`;
            response += `${spell.description}\n\n`;

            if (spell.actions) {
                response += `*Actions:* ${spell.actions}\n`;
            }
            if (spell.components) {
                response += `*Components:* ${spell.components.join(', ')}\n`;
            }
            if (spell.range) {
                response += `*Range:* ${spell.range}\n`;
            }
            if (spell.area) {
                response += `*Area:* ${spell.area}\n`;
            }
            if (spell.savingThrow) {
                response += `*Saving Throw:* ${spell.savingThrow}\n`;
            }

            if (spell.variants) {
                response += '\n*Action Variants:*\n';
                for (const [actions, effect] of Object.entries(spell.variants)) {
                    response += `• ${actions}: ${effect}\n`;
                }
            }

            if (spell.heightened) {
                response += `\n*Heightened:* ${spell.heightened}`;
            }

            return {
                response,
                type: 'spell_found',
                spell,
                confidence: 0.9
            };

        } catch (error) {
            return this.createErrorResponse(error, 'spell lookup');
        }
    }

    /**
     * Handle combat help requests
     */
    async handleCombatHelp(message, context, intent) {
        try {
            const combatAction = intent.combatTopic;

            const resolution = await this.rulesService.resolveCombatAction({
                action: combatAction,
                context
            });

            let response = `**${combatAction.toUpperCase()} Help**\n\n`;

            if (resolution.mechanics) {
                for (const [key, value] of Object.entries(resolution.mechanics)) {
                    response += `*${key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}:* ${value}\n`;
                }
            }

            if (resolution.suggestions && resolution.suggestions.length > 0) {
                response += `\n*Need more help with:* ${resolution.suggestions.join(', ')}`;
            }

            return {
                response,
                type: 'combat_help',
                resolution,
                confidence: 0.8
            };

        } catch (error) {
            return this.createErrorResponse(error, 'combat help');
        }
    }

    /**
     * Handle general chat messages
     */
    async handleGeneralChat(message, context, history) {
        const responses = {
            greeting: [
                "Greetings, adventurer! I'm your raven familiar, ready to assist with Pathfinder 2e rules and GM tasks.",
                "Hello there! What can I help you with in your campaign today?",
                "Welcome! I'm here to help with any Pathfinder 2e questions you might have."
            ],
            thanks: [
                "You're welcome! Happy to help with your campaign.",
                "My pleasure! Feel free to ask if you need more assistance.",
                "Glad I could help! Let me know if you have other questions."
            ],
            farewell: [
                "Farewell, and may your adventures be legendary!",
                "Until next time, adventurer! *caw*",
                "Safe travels! I'll be here when you need rules assistance."
            ],
            chat: [
                "I'm doing well, thank you! Ready to help with any Pathfinder 2e questions.",
                "As your faithful familiar, I'm always ready to assist with rules and campaign guidance!",
                "I'm here and ready to help! What would you like to know about Pathfinder 2e?"
            ]
        };

        const intent = await this.analyzeMessageIntent(message);
        const responseArray = responses[intent.subtype] || responses.chat;
        const response = responseArray[Math.floor(Math.random() * responseArray.length)];

        return {
            response,
            type: 'general_chat',
            subtype: intent.subtype,
            confidence: 0.7
        };
    }

    /**
     * Handle default responses
     */
    async handleDefaultResponse(message, context) {
        return {
            response: "I'm not quite sure what you're asking about. I can help with:\n\n• Pathfinder 2e rules questions\n• Spell descriptions and mechanics\n• Combat assistance\n• General GM guidance\n\nTry being more specific, like 'How does the Strike action work?' or 'Tell me about the Fireball spell.'",
            type: 'default',
            suggestions: [
                'How does [action] work?',
                'Tell me about [spell name]',
                'Combat help with [topic]',
                'What are the rules for [situation]?'
            ],
            confidence: 0.3
        };
    }

    /**
     * Create error response
     */
    createErrorResponse(error, context = 'general') {
        console.error(`Chat Service | Error in ${context}:`, error);

        return {
            response: `I encountered an error while processing your request. Please try rephrasing your question or ask something simpler. I'm here to help with Pathfinder 2e rules!`,
            type: 'error',
            error: error.message,
            confidence: 0.0
        };
    }

    /**
     * Update conversation memory
     */
    updateConversationMemory(userId, interaction) {
        if (!this.conversationMemory.has(userId)) {
            this.conversationMemory.set(userId, []);
        }

        const userHistory = this.conversationMemory.get(userId);
        userHistory.push(interaction);

        // Keep only last 20 interactions per user
        if (userHistory.length > 20) {
            userHistory.shift();
        }
    }

    /**
     * Update response time statistics
     */
    updateResponseTimeStats(responseTime) {
        const currentAverage = this.stats.averageResponseTime;
        const totalMessages = this.stats.messagesProcessed;

        this.stats.averageResponseTime = currentAverage === 0
            ? responseTime
            : ((currentAverage * (totalMessages - 1)) + responseTime) / totalMessages;
    }

    /**
     * Get service statistics
     */
    getStats() {
        return {
            ...this.stats,
            conversationsActive: this.conversationMemory.size,
            totalInteractions: Array.from(this.conversationMemory.values())
                .reduce((total, history) => total + history.length, 0)
        };
    }

    /**
     * Clear conversation memory for a user
     */
    clearUserMemory(userId) {
        this.conversationMemory.delete(userId);
    }

    /**
     * Get user conversation history
     */
    getUserHistory(userId, limit = 10) {
        const history = this.conversationMemory.get(userId) || [];
        return history.slice(-limit);
    }
}

module.exports = { ChatService };