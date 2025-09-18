/**
 * Context Service
 * Analyzes Foundry VTT context to provide relevant assistance
 */

class ContextService {
    constructor(options = {}) {
        this.contextHistory = new Map();
        this.sceneAnalytics = new Map();
        this.playerPatterns = new Map();
        this.stats = {
            contextsAnalyzed: 0,
            scenesTracked: 0,
            averageAnalysisTime: 0
        };
    }

    /**
     * Initialize context service
     */
    async initialize() {
        console.log('Context Service | Initialized');
    }

    /**
     * Analyze current Foundry context
     */
    async analyzeContext(context) {
        const startTime = Date.now();

        try {
            this.stats.contextsAnalyzed++;

            const analysis = {
                timestamp: Date.now(),
                scene: await this.analyzeScene(context),
                actors: await this.analyzeActors(context.actors || []),
                user: await this.analyzeUser(context),
                gameState: await this.analyzeGameState(context),
                suggestions: await this.generateSuggestions(context),
                priority: this.calculatePriority(context)
            };

            // Store context history
            this.storeContextHistory(context.currentUser || 'anonymous', analysis);

            // Update scene analytics
            this.updateSceneAnalytics(context.scene, analysis);

            // Update stats
            const analysisTime = Date.now() - startTime;
            this.updateAnalysisTimeStats(analysisTime);

            return analysis;

        } catch (error) {
            console.error('Context Service | Error analyzing context:', error);
            return this.createFallbackAnalysis(context);
        }
    }

    /**
     * Analyze scene context
     */
    async analyzeScene(context) {
        const sceneInfo = {
            name: context.scene || 'Unknown Scene',
            type: this.classifySceneType(context.scene),
            complexity: 'medium',
            suggestedRules: []
        };

        // Classify scene type based on name patterns
        const sceneName = (context.scene || '').toLowerCase();

        if (this.matchesPatterns(sceneName, [/combat|battle|fight|arena/])) {
            sceneInfo.type = 'combat';
            sceneInfo.complexity = 'high';
            sceneInfo.suggestedRules = ['initiative', 'attacks', 'movement', 'conditions'];
        } else if (this.matchesPatterns(sceneName, [/dungeon|cave|ruins|tomb/])) {
            sceneInfo.type = 'exploration';
            sceneInfo.complexity = 'medium';
            sceneInfo.suggestedRules = ['perception', 'stealth', 'traps', 'skills'];
        } else if (this.matchesPatterns(sceneName, [/town|city|village|tavern|shop/])) {
            sceneInfo.type = 'social';
            sceneInfo.complexity = 'low';
            sceneInfo.suggestedRules = ['diplomacy', 'deception', 'intimidation', 'gather information'];
        } else if (this.matchesPatterns(sceneName, [/wilderness|forest|mountain|plains/])) {
            sceneInfo.type = 'travel';
            sceneInfo.complexity = 'low';
            sceneInfo.suggestedRules = ['survival', 'navigation', 'encounters', 'weather'];
        }

        return sceneInfo;
    }

    /**
     * Analyze actors in the scene
     */
    async analyzeActors(actors) {
        const actorAnalysis = {
            totalCount: actors.length,
            playerCharacters: 0,
            npcs: 0,
            averageLevel: 0,
            partyComposition: {
                martialClasses: 0,
                spellcasters: 0,
                skillExperts: 0
            },
            threatLevel: 'low',
            recommendations: []
        };

        if (actors.length === 0) {
            return actorAnalysis;
        }

        let totalLevel = 0;
        let validLevels = 0;

        actors.forEach(actor => {
            // Count character types
            if (actor.actorType === 'character') {
                actorAnalysis.playerCharacters++;
            } else {
                actorAnalysis.npcs++;
            }

            // Calculate average level
            if (actor.level && typeof actor.level === 'number') {
                totalLevel += actor.level;
                validLevels++;

                // Classify threat level
                if (actor.level > 10) {
                    actorAnalysis.threatLevel = 'high';
                } else if (actor.level > 5 && actorAnalysis.threatLevel !== 'high') {
                    actorAnalysis.threatLevel = 'medium';
                }
            }

            // Analyze party composition (simplified)
            if (actor.name && actor.actorType === 'character') {
                const name = actor.name.toLowerCase();
                if (this.matchesPatterns(name, [/fighter|ranger|barbarian|champion/])) {
                    actorAnalysis.partyComposition.martialClasses++;
                } else if (this.matchesPatterns(name, [/wizard|sorcerer|cleric|druid/])) {
                    actorAnalysis.partyComposition.spellcasters++;
                } else if (this.matchesPatterns(name, [/rogue|bard|investigator/])) {
                    actorAnalysis.partyComposition.skillExperts++;
                }
            }
        });

        if (validLevels > 0) {
            actorAnalysis.averageLevel = Math.round(totalLevel / validLevels);
        }

        // Generate recommendations based on analysis
        this.generateActorRecommendations(actorAnalysis);

        return actorAnalysis;
    }

    /**
     * Analyze user context
     */
    async analyzeUser(context) {
        const userAnalysis = {
            name: context.currentUser || 'Unknown User',
            isGM: context.isGM || false,
            role: context.isGM ? 'Game Master' : 'Player',
            experience: this.estimateUserExperience(context.currentUser),
            preferredAssistance: this.getUserPreferences(context.currentUser),
            recentQueries: this.getRecentUserQueries(context.currentUser)
        };

        return userAnalysis;
    }

    /**
     * Analyze game state
     */
    async analyzeGameState(context) {
        return {
            system: context.system || 'pathfinder2e',
            timestamp: Date.now(),
            sessionPhase: this.determineSessionPhase(context),
            urgency: this.calculateUrgency(context),
            activeElements: this.identifyActiveElements(context)
        };
    }

    /**
     * Generate contextual suggestions
     */
    async generateSuggestions(context) {
        const suggestions = [];

        // Scene-based suggestions
        const sceneAnalysis = await this.analyzeScene(context);
        suggestions.push(...this.getSceneSuggestions(sceneAnalysis));

        // Actor-based suggestions
        const actorAnalysis = await this.analyzeActors(context.actors || []);
        suggestions.push(...this.getActorSuggestions(actorAnalysis));

        // User-based suggestions
        if (context.isGM) {
            suggestions.push(...this.getGMSuggestions(context));
        } else {
            suggestions.push(...this.getPlayerSuggestions(context));
        }

        // Remove duplicates and limit
        return [...new Set(suggestions)].slice(0, 5);
    }

    /**
     * Get scene-based suggestions
     */
    getSceneSuggestions(sceneAnalysis) {
        const suggestions = [];

        switch (sceneAnalysis.type) {
            case 'combat':
                suggestions.push(
                    'Ask about initiative rules',
                    'Get help with attack bonuses',
                    'Learn about critical hits'
                );
                break;
            case 'exploration':
                suggestions.push(
                    'Ask about Perception checks',
                    'Learn about skill challenges',
                    'Get help with trap mechanics'
                );
                break;
            case 'social':
                suggestions.push(
                    'Ask about Diplomacy rules',
                    'Learn about social skills',
                    'Get help with roleplay mechanics'
                );
                break;
            case 'travel':
                suggestions.push(
                    'Ask about Survival checks',
                    'Learn about travel rules',
                    'Get help with random encounters'
                );
                break;
        }

        return suggestions;
    }

    /**
     * Get actor-based suggestions
     */
    getActorSuggestions(actorAnalysis) {
        const suggestions = [];

        if (actorAnalysis.averageLevel < 3) {
            suggestions.push('Ask about basic action rules');
        } else if (actorAnalysis.averageLevel > 10) {
            suggestions.push('Ask about high-level mechanics');
        }

        if (actorAnalysis.partyComposition.spellcasters > 0) {
            suggestions.push('Get spell descriptions');
        }

        if (actorAnalysis.threatLevel === 'high') {
            suggestions.push('Ask about advanced combat tactics');
        }

        return suggestions;
    }

    /**
     * Get GM-specific suggestions
     */
    getGMSuggestions(context) {
        return [
            'Ask about encounter balancing',
            'Get help with NPC stat blocks',
            'Learn about scene transitions',
            'Ask about rules adjudication'
        ];
    }

    /**
     * Get player-specific suggestions
     */
    getPlayerSuggestions(context) {
        return [
            'Ask about character abilities',
            'Get help with action economy',
            'Learn about skill uses',
            'Ask about spell effects'
        ];
    }

    /**
     * Helper methods
     */
    matchesPatterns(text, patterns) {
        return patterns.some(pattern => pattern.test(text));
    }

    classifySceneType(sceneName) {
        if (!sceneName) return 'unknown';

        const name = sceneName.toLowerCase();
        if (this.matchesPatterns(name, [/combat|battle|fight/])) return 'combat';
        if (this.matchesPatterns(name, [/dungeon|exploration/])) return 'exploration';
        if (this.matchesPatterns(name, [/town|social|tavern/])) return 'social';
        if (this.matchesPatterns(name, [/travel|wilderness/])) return 'travel';

        return 'general';
    }

    calculatePriority(context) {
        let priority = 1; // Default priority

        if (context.isGM) priority += 1;
        if (context.actors && context.actors.length > 4) priority += 1;
        if (context.scene && context.scene.toLowerCase().includes('combat')) priority += 2;

        return Math.min(priority, 5);
    }

    determineSessionPhase(context) {
        const hour = new Date().getHours();
        if (hour < 2) return 'late-session';
        if (hour < 12) return 'early-session';
        if (hour < 18) return 'mid-session';
        return 'evening-session';
    }

    calculateUrgency(context) {
        if (context.scene && context.scene.toLowerCase().includes('combat')) return 'high';
        if (context.actors && context.actors.length > 6) return 'medium';
        return 'low';
    }

    identifyActiveElements(context) {
        const elements = [];
        if (context.scene) elements.push('scene');
        if (context.actors && context.actors.length > 0) elements.push('actors');
        if (context.isGM) elements.push('gm-tools');
        return elements;
    }

    estimateUserExperience(username) {
        const history = this.contextHistory.get(username);
        if (!history || history.length < 5) return 'beginner';
        if (history.length < 20) return 'intermediate';
        return 'experienced';
    }

    getUserPreferences(username) {
        // This would be based on historical data
        return ['rules-help', 'quick-references'];
    }

    getRecentUserQueries(username) {
        const history = this.contextHistory.get(username) || [];
        return history.slice(-3).map(entry => entry.query || 'general').reverse();
    }

    generateActorRecommendations(actorAnalysis) {
        if (actorAnalysis.partyComposition.spellcasters === 0) {
            actorAnalysis.recommendations.push('Consider spell assistance');
        }
        if (actorAnalysis.averageLevel > 8) {
            actorAnalysis.recommendations.push('Advanced rules may be needed');
        }
        if (actorAnalysis.npcs > actorAnalysis.playerCharacters * 2) {
            actorAnalysis.recommendations.push('Complex encounter - initiative help available');
        }
    }

    storeContextHistory(username, analysis) {
        if (!this.contextHistory.has(username)) {
            this.contextHistory.set(username, []);
        }

        const history = this.contextHistory.get(username);
        history.push({
            timestamp: analysis.timestamp,
            scene: analysis.scene.name,
            priority: analysis.priority,
            suggestions: analysis.suggestions
        });

        // Keep only last 50 entries
        if (history.length > 50) {
            history.shift();
        }
    }

    updateSceneAnalytics(sceneName, analysis) {
        if (!sceneName) return;

        if (!this.sceneAnalytics.has(sceneName)) {
            this.sceneAnalytics.set(sceneName, {
                visits: 0,
                totalTime: 0,
                lastVisit: null,
                commonQueries: new Map()
            });
            this.stats.scenesTracked++;
        }

        const analytics = this.sceneAnalytics.get(sceneName);
        analytics.visits++;
        analytics.lastVisit = analysis.timestamp;
    }

    updateAnalysisTimeStats(analysisTime) {
        const currentAverage = this.stats.averageAnalysisTime;
        const totalAnalyses = this.stats.contextsAnalyzed;

        this.stats.averageAnalysisTime = currentAverage === 0
            ? analysisTime
            : ((currentAverage * (totalAnalyses - 1)) + analysisTime) / totalAnalyses;
    }

    createFallbackAnalysis(context) {
        return {
            timestamp: Date.now(),
            scene: { name: context.scene || 'Unknown', type: 'general' },
            actors: { totalCount: 0, playerCharacters: 0, npcs: 0 },
            user: { name: context.currentUser || 'Unknown', isGM: false },
            gameState: { system: 'pathfinder2e', urgency: 'low' },
            suggestions: ['Ask about basic rules', 'Get spell help', 'Learn combat mechanics'],
            priority: 1,
            error: 'Fallback analysis used'
        };
    }

    /**
     * Get service statistics
     */
    getStats() {
        return {
            ...this.stats,
            activeUsers: this.contextHistory.size,
            totalContextEntries: Array.from(this.contextHistory.values())
                .reduce((total, history) => total + history.length, 0)
        };
    }

    /**
     * Clear user context history
     */
    clearUserHistory(username) {
        this.contextHistory.delete(username);
    }

    /**
     * Get scene analytics
     */
    getSceneAnalytics(sceneName) {
        return this.sceneAnalytics.get(sceneName) || null;
    }
}

module.exports = { ContextService };