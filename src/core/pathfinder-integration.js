/**
 * Pathfinder Integration Module
 * Integrates with Foundry VTT's Pathfinder 2e system
 */

export class PathfinderIntegration {
    constructor(options = {}) {
        this.system = null;
        this.gameData = null;
        this.initialized = false;
        this.cache = new Map();
    }

    /**
     * Initialize Pathfinder integration
     */
    async initialize() {
        if (this.initialized) return;

        try {
            // Verify we're in Foundry with Pathfinder 2e system
            if (typeof game === 'undefined') {
                throw new Error('Foundry VTT not available');
            }

            if (game.system.id !== 'pf2e') {
                console.warn('Pathfinder Integration | Not using Pathfinder 2e system');
            }

            this.system = game.system;
            this.gameData = game.data;

            // Cache commonly used data
            await this.cacheSystemData();

            this.initialized = true;
            console.log('Pathfinder Integration | Initialized successfully');

        } catch (error) {
            console.error('Pathfinder Integration | Failed to initialize:', error);
            // Initialize in fallback mode
            this.initializeFallbackMode();
        }
    }

    /**
     * Initialize fallback mode when Foundry isn't available
     */
    initializeFallbackMode() {
        console.log('Pathfinder Integration | Running in fallback mode');

        this.system = {
            id: 'pf2e',
            title: 'Pathfinder Second Edition (Fallback)',
            version: '1.0.0'
        };

        this.gameData = {
            actions: this.getFallbackActions(),
            spells: this.getFallbackSpells(),
            conditions: this.getFallbackConditions()
        };

        this.initialized = true;
    }

    /**
     * Cache frequently accessed system data
     */
    async cacheSystemData() {
        if (typeof game === 'undefined') return;

        try {
            // Cache actions
            const actions = game.pf2e?.actions || {};
            this.cache.set('actions', actions);

            // Cache conditions
            const conditions = game.pf2e?.conditions || {};
            this.cache.set('conditions', conditions);

            // Cache basic rules
            const rules = {
                actionEconomy: {
                    actionsPerTurn: 3,
                    freeActionsUnlimited: true,
                    reactionsPerRound: 1
                },
                attackRolls: {
                    formula: '1d20 + ability + proficiency + item + other',
                    criticalSuccess: 'Natural 20 or exceed DC by 10+',
                    criticalFailure: 'Natural 1 or fail DC by 10+'
                }
            };
            this.cache.set('rules', rules);

        } catch (error) {
            console.warn('Pathfinder Integration | Error caching system data:', error);
        }
    }

    /**
     * Get action information
     */
    getAction(actionName) {
        const actions = this.cache.get('actions') || this.getFallbackActions();

        const normalizedName = actionName.toLowerCase().replace(/\s+/g, '-');
        return actions[normalizedName] || null;
    }

    /**
     * Get condition information
     */
    getCondition(conditionName) {
        const conditions = this.cache.get('conditions') || this.getFallbackConditions();

        const normalizedName = conditionName.toLowerCase();
        return conditions[normalizedName] || null;
    }

    /**
     * Get spell information from system
     */
    getSpell(spellName) {
        if (typeof game !== 'undefined' && game.items) {
            // Search through system spells
            const spells = game.items.filter(item =>
                item.type === 'spell' &&
                item.name.toLowerCase() === spellName.toLowerCase()
            );

            if (spells.length > 0) {
                return this.formatSpellData(spells[0]);
            }
        }

        // Fallback to cached spell data
        const fallbackSpells = this.getFallbackSpells();
        const normalizedName = spellName.toLowerCase().replace(/\s+/g, '-');
        return fallbackSpells[normalizedName] || null;
    }

    /**
     * Format spell data for consistent output
     */
    formatSpellData(spellItem) {
        return {
            name: spellItem.name,
            level: spellItem.system?.level?.value || 0,
            school: spellItem.system?.school?.value || 'unknown',
            actions: spellItem.system?.time?.value || '2',
            components: this.extractComponents(spellItem.system?.components),
            range: spellItem.system?.range?.value || 'unknown',
            description: spellItem.system?.description?.value || 'No description available',
            source: spellItem.system?.source || 'Core Rulebook'
        };
    }

    /**
     * Extract components from spell system data
     */
    extractComponents(components) {
        if (!components) return [];

        const componentList = [];
        if (components.somatic) componentList.push('somatic');
        if (components.verbal) componentList.push('verbal');
        if (components.material) componentList.push('material');
        if (components.focus) componentList.push('focus');

        return componentList;
    }

    /**
     * Get current scene context
     */
    getSceneContext() {
        if (typeof canvas === 'undefined' || !canvas.ready) {
            return { name: 'Unknown Scene', type: 'general' };
        }

        const scene = canvas.scene;
        return {
            name: scene.name,
            type: this.classifyScene(scene),
            dimensions: {
                width: scene.width,
                height: scene.height
            },
            tokens: canvas.tokens.objects.children.length,
            lighting: scene.darkness > 0.5 ? 'dark' : 'bright'
        };
    }

    /**
     * Classify scene type based on properties
     */
    classifyScene(scene) {
        if (!scene) return 'general';

        const name = scene.name.toLowerCase();
        if (name.includes('combat') || name.includes('battle')) return 'combat';
        if (name.includes('dungeon') || name.includes('cave')) return 'exploration';
        if (name.includes('town') || name.includes('city')) return 'social';
        if (name.includes('wilderness') || name.includes('travel')) return 'travel';

        return 'general';
    }

    /**
     * Get actor information
     */
    getActors() {
        if (typeof game === 'undefined') {
            return [];
        }

        return game.actors.map(actor => ({
            name: actor.name,
            type: actor.type,
            level: actor.system?.details?.level?.value || 0,
            class: actor.system?.details?.class?.name || 'Unknown',
            ancestry: actor.system?.details?.ancestry?.name || 'Unknown',
            isPC: actor.hasPlayerOwner,
            hp: {
                current: actor.system?.attributes?.hp?.value || 0,
                max: actor.system?.attributes?.hp?.max || 0
            },
            ac: actor.system?.attributes?.ac?.value || 10
        }));
    }

    /**
     * Get party information
     */
    getPartyInfo() {
        const actors = this.getActors();
        const playerCharacters = actors.filter(actor => actor.isPC);

        return {
            size: playerCharacters.length,
            averageLevel: this.calculateAverageLevel(playerCharacters),
            classes: this.getUniqueClasses(playerCharacters),
            totalHp: playerCharacters.reduce((sum, pc) => sum + pc.hp.current, 0),
            maxHp: playerCharacters.reduce((sum, pc) => sum + pc.hp.max, 0)
        };
    }

    /**
     * Calculate average party level
     */
    calculateAverageLevel(characters) {
        if (characters.length === 0) return 0;

        const totalLevels = characters.reduce((sum, char) => sum + char.level, 0);
        return Math.round(totalLevels / characters.length);
    }

    /**
     * Get unique classes in party
     */
    getUniqueClasses(characters) {
        const classes = new Set(characters.map(char => char.class));
        return Array.from(classes);
    }

    /**
     * Check if user is GM
     */
    isGameMaster() {
        return typeof game !== 'undefined' && game.user && game.user.isGM;
    }

    /**
     * Get current user information
     */
    getCurrentUser() {
        if (typeof game === 'undefined') {
            return { name: 'Unknown User', isGM: false };
        }

        return {
            name: game.user.name,
            id: game.user.id,
            isGM: game.user.isGM,
            character: game.user.character?.name || null
        };
    }

    /**
     * Fallback data when system isn't available
     */
    getFallbackActions() {
        return {
            'strike': {
                name: 'Strike',
                actions: 1,
                traits: ['attack'],
                description: 'Make a melee or ranged attack roll against a target within your reach or range.',
                requirements: 'You have a weapon or are capable of making an unarmed attack.'
            },
            'stride': {
                name: 'Stride',
                actions: 1,
                traits: ['move'],
                description: 'Move up to your Speed.',
                requirements: 'None'
            },
            'cast-a-spell': {
                name: 'Cast a Spell',
                actions: 'varies',
                traits: ['concentrate'],
                description: 'Cast a spell you know or have prepared.',
                requirements: 'You can Cast a Spell as long as you have spell slots remaining for spells of the appropriate spell level.'
            }
        };
    }

    getFallbackSpells() {
        return {
            'magic-missile': {
                name: 'Magic Missile',
                level: 1,
                school: 'evocation',
                actions: 2,
                components: ['somatic', 'verbal'],
                range: '120 feet',
                description: 'You send a dart of force streaking toward a creature that you can see. It automatically hits and deals 1d4+1 force damage.'
            },
            'heal': {
                name: 'Heal',
                level: 1,
                school: 'necromancy',
                actions: 'varies',
                components: ['somatic'],
                range: 'varies',
                description: 'You channel positive energy to heal the living or damage the undead.'
            }
        };
    }

    getFallbackConditions() {
        return {
            'frightened': {
                name: 'Frightened',
                description: 'You take a status penalty equal to this condition value to all your checks and DCs.',
                effects: 'Status penalty to all checks and DCs'
            },
            'prone': {
                name: 'Prone',
                description: 'You are lying on the ground. You take a -2 circumstance penalty to attack rolls.',
                effects: '-2 circumstance penalty to attack rolls, +2 circumstance bonus to AC against ranged attacks'
            }
        };
    }

    /**
     * Get system statistics
     */
    getStats() {
        return {
            initialized: this.initialized,
            system: this.system?.id || 'unknown',
            cacheSize: this.cache.size,
            fallbackMode: typeof game === 'undefined'
        };
    }

    /**
     * Cleanup and close integration
     */
    destroy() {
        this.cache.clear();
        this.initialized = false;
        console.log('Pathfinder Integration | Destroyed');
    }
}