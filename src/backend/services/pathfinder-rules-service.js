/**
 * Pathfinder Rules Service
 * Handles Pathfinder 2e rules queries, spell lookups, and combat resolution
 */

const fs = require('fs').promises;
const path = require('path');

class PathfinderRulesService {
    constructor(options = {}) {
        this.dataPath = options.dataPath || path.join(__dirname, '../data/pathfinder');
        this.rulesDatabase = new Map();
        this.spellsDatabase = new Map();
        this.combatRules = new Map();
        this.initialized = false;
        this.stats = {
            queriesProcessed: 0,
            rulesLoaded: 0,
            spellsLoaded: 0,
            lastQuery: null
        };
    }

    /**
     * Initialize the Pathfinder rules service
     */
    async initialize() {
        if (this.initialized) return;

        console.log('Pathfinder Rules Service | Initializing...');

        try {
            await this.ensureDataDirectory();
            await this.loadRulesData();
            await this.loadSpellsData();
            await this.loadCombatRules();

            this.initialized = true;
            console.log(`Pathfinder Rules Service | Loaded ${this.stats.rulesLoaded} rules, ${this.stats.spellsLoaded} spells`);
        } catch (error) {
            console.error('Pathfinder Rules Service | Initialization failed:', error);
            // Initialize with minimal fallback data
            await this.initializeFallbackData();
            this.initialized = true;
        }
    }

    /**
     * Ensure data directory exists and create sample data
     */
    async ensureDataDirectory() {
        try {
            await fs.mkdir(this.dataPath, { recursive: true });
        } catch (error) {
            // Directory already exists or creation failed
        }

        // Create sample data files if they don't exist
        const rulesFile = path.join(this.dataPath, 'core-rules.json');
        const spellsFile = path.join(this.dataPath, 'spells.json');
        const combatFile = path.join(this.dataPath, 'combat-rules.json');

        // Check if files exist, create samples if not
        try {
            await fs.access(rulesFile);
        } catch {
            await this.createSampleRulesFile(rulesFile);
        }

        try {
            await fs.access(spellsFile);
        } catch {
            await this.createSampleSpellsFile(spellsFile);
        }

        try {
            await fs.access(combatFile);
        } catch {
            await this.createSampleCombatFile(combatFile);
        }
    }

    /**
     * Create sample rules file
     */
    async createSampleRulesFile(filePath) {
        const sampleRules = {
            "actions": {
                "strike": {
                    "name": "Strike",
                    "actions": 1,
                    "traits": ["attack"],
                    "description": "Make a melee or ranged attack against one target within reach or range.",
                    "requirements": "You have a weapon or unarmed attack available."
                },
                "cast-spell": {
                    "name": "Cast a Spell",
                    "actions": "varies",
                    "traits": ["concentrate"],
                    "description": "Cast a spell you know or have prepared.",
                    "requirements": "You must be able to speak or have a free hand for somatic components."
                }
            },
            "conditions": {
                "frightened": {
                    "name": "Frightened",
                    "description": "You're gripped by fear. You take a status penalty equal to the condition value to all checks and DCs.",
                    "mechanics": "Status penalty to all checks and DCs equal to condition value"
                },
                "prone": {
                    "name": "Prone",
                    "description": "You're lying on the ground. You take a -2 circumstance penalty to attack rolls.",
                    "mechanics": "-2 circumstance penalty to attack rolls, +2 circumstance bonus to AC against ranged attacks"
                }
            },
            "skills": {
                "perception": {
                    "name": "Perception",
                    "keyAbility": "Wisdom",
                    "description": "Measure how well you notice things around you.",
                    "uses": ["Notice creatures", "Find hidden objects", "Initiative"]
                }
            }
        };

        await fs.writeFile(filePath, JSON.stringify(sampleRules, null, 2));
    }

    /**
     * Create sample spells file
     */
    async createSampleSpellsFile(filePath) {
        const sampleSpells = {
            "magic-missile": {
                "name": "Magic Missile",
                "level": 1,
                "school": "evocation",
                "actions": 2,
                "components": ["somatic", "verbal"],
                "range": "120 feet",
                "targets": "1-3 creatures",
                "description": "You send a dart of force streaking toward a creature that you can see. It automatically hits and deals 1d4+1 force damage.",
                "heightened": "When you cast this spell using a spell slot of 2nd level or higher, the spell creates one more dart for each slot level above 1st."
            },
            "heal": {
                "name": "Heal",
                "level": 1,
                "school": "necromancy",
                "actions": "1-3",
                "components": ["somatic"],
                "range": "varies",
                "description": "You channel positive energy to heal the living or damage the undead.",
                "variants": {
                    "1-action": "Range touch, restore 1d8 HP",
                    "2-action": "Range 30 feet, restore 1d8+8 HP",
                    "3-action": "30-foot emanation, restore 1d8 HP to all living creatures"
                }
            },
            "fireball": {
                "name": "Fireball",
                "level": 3,
                "school": "evocation",
                "actions": 2,
                "components": ["somatic", "verbal"],
                "range": "500 feet",
                "area": "20-foot burst",
                "savingThrow": "Reflex",
                "description": "A roaring blast of fire appears at a spot you designate, dealing 6d6 fire damage.",
                "criticalSuccess": "No damage",
                "success": "Half damage",
                "failure": "Full damage",
                "criticalFailure": "Double damage"
            }
        };

        await fs.writeFile(filePath, JSON.stringify(sampleSpells, null, 2));
    }

    /**
     * Create sample combat rules file
     */
    async createSampleCombatFile(filePath) {
        const sampleCombat = {
            "initiative": {
                "description": "Roll Perception for initiative. Act in descending order of results.",
                "ties": "Higher initiative modifier wins. If still tied, players go first."
            },
            "actions": {
                "perTurn": 3,
                "types": {
                    "single": "Most actions, like Strike or Cast a Spell",
                    "free": "Simple actions that don't require significant time",
                    "reaction": "Immediate response to a trigger"
                }
            },
            "attackRolls": {
                "formula": "1d20 + ability modifier + proficiency + item bonus + other bonuses",
                "criticalHit": "Natural 20 or beat AC by 10+",
                "criticalMiss": "Natural 1 or miss AC by 10+"
            },
            "multipleAttacks": {
                "penalty": "Second attack -5, third attack -10",
                "agileWeapons": "Second attack -4, third attack -8"
            }
        };

        await fs.writeFile(filePath, JSON.stringify(sampleCombat, null, 2));
    }

    /**
     * Load rules data from files
     */
    async loadRulesData() {
        try {
            const rulesFile = path.join(this.dataPath, 'core-rules.json');
            const data = await fs.readFile(rulesFile, 'utf8');
            const rules = JSON.parse(data);

            // Load rules into database
            for (const [category, items] of Object.entries(rules)) {
                for (const [key, rule] of Object.entries(items)) {
                    this.rulesDatabase.set(`${category}.${key}`, {
                        ...rule,
                        category,
                        key,
                        searchTerms: this.generateSearchTerms(rule.name, rule.description)
                    });
                    this.stats.rulesLoaded++;
                }
            }
        } catch (error) {
            console.warn('Failed to load rules data:', error.message);
        }
    }

    /**
     * Load spells data from files
     */
    async loadSpellsData() {
        try {
            const spellsFile = path.join(this.dataPath, 'spells.json');
            const data = await fs.readFile(spellsFile, 'utf8');
            const spells = JSON.parse(data);

            // Load spells into database
            for (const [key, spell] of Object.entries(spells)) {
                this.spellsDatabase.set(key, {
                    ...spell,
                    key,
                    searchTerms: this.generateSearchTerms(spell.name, spell.description)
                });
                this.stats.spellsLoaded++;
            }
        } catch (error) {
            console.warn('Failed to load spells data:', error.message);
        }
    }

    /**
     * Load combat rules
     */
    async loadCombatRules() {
        try {
            const combatFile = path.join(this.dataPath, 'combat-rules.json');
            const data = await fs.readFile(combatFile, 'utf8');
            const combat = JSON.parse(data);

            // Load combat rules
            for (const [key, rule] of Object.entries(combat)) {
                this.combatRules.set(key, rule);
            }
        } catch (error) {
            console.warn('Failed to load combat rules:', error.message);
        }
    }

    /**
     * Initialize fallback data when files can't be loaded
     */
    async initializeFallbackData() {
        console.log('Pathfinder Rules Service | Using fallback data');

        // Add basic fallback rules
        const fallbackRules = [
            {
                key: 'basic.strike',
                name: 'Strike',
                category: 'actions',
                description: 'Make a melee or ranged attack against a target.',
                searchTerms: ['strike', 'attack', 'melee', 'ranged']
            },
            {
                key: 'basic.move',
                name: 'Stride',
                category: 'actions',
                description: 'Move up to your Speed.',
                searchTerms: ['stride', 'move', 'movement', 'speed']
            }
        ];

        fallbackRules.forEach(rule => {
            this.rulesDatabase.set(rule.key, rule);
            this.stats.rulesLoaded++;
        });

        // Add basic spells
        this.spellsDatabase.set('magic-missile', {
            name: 'Magic Missile',
            level: 1,
            description: 'Force dart that automatically hits for 1d4+1 damage.',
            searchTerms: ['magic', 'missile', 'force', 'automatic', 'hit']
        });
        this.stats.spellsLoaded++;
    }

    /**
     * Generate search terms for better matching
     */
    generateSearchTerms(name, description) {
        const terms = [];

        if (name) {
            terms.push(...name.toLowerCase().split(/\s+/));
        }

        if (description) {
            const words = description.toLowerCase()
                .replace(/[^\w\s]/g, ' ')
                .split(/\s+/)
                .filter(word => word.length > 2);
            terms.push(...words);
        }

        return [...new Set(terms)]; // Remove duplicates
    }

    /**
     * Search rules by query
     */
    async searchRules(options = {}) {
        const { query, category, limit = 5 } = options;
        this.stats.queriesProcessed++;
        this.stats.lastQuery = query;

        const results = [];
        const queryTerms = query.toLowerCase().split(/\s+/);

        for (const [key, rule] of this.rulesDatabase) {
            if (category && rule.category !== category) continue;

            let score = 0;

            // Check name match
            if (rule.name && rule.name.toLowerCase().includes(query.toLowerCase())) {
                score += 10;
            }

            // Check search terms
            for (const term of queryTerms) {
                if (rule.searchTerms && rule.searchTerms.some(searchTerm =>
                    searchTerm.includes(term) || term.includes(searchTerm))) {
                    score += 1;
                }
            }

            if (score > 0) {
                results.push({ ...rule, score });
            }
        }

        return results
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);
    }

    /**
     * Get specific spell by name
     */
    async getSpell(spellName) {
        const normalizedName = spellName.toLowerCase().replace(/\s+/g, '-');

        // Try direct lookup first
        if (this.spellsDatabase.has(normalizedName)) {
            return this.spellsDatabase.get(normalizedName);
        }

        // Try fuzzy search
        for (const [key, spell] of this.spellsDatabase) {
            if (spell.name.toLowerCase() === spellName.toLowerCase()) {
                return spell;
            }
        }

        // Search by partial name
        for (const [key, spell] of this.spellsDatabase) {
            if (spell.name.toLowerCase().includes(spellName.toLowerCase())) {
                return spell;
            }
        }

        return null;
    }

    /**
     * Resolve combat action
     */
    async resolveCombatAction(options = {}) {
        const { action, context } = options;

        const resolution = {
            action: action,
            success: true,
            message: '',
            mechanics: {},
            suggestions: []
        };

        try {
            switch (action.toLowerCase()) {
                case 'strike':
                case 'attack':
                    resolution.mechanics = this.resolveMeleeAttack(context);
                    break;

                case 'spell':
                case 'cast':
                    resolution.mechanics = this.resolveSpellCasting(context);
                    break;

                case 'initiative':
                    resolution.mechanics = this.resolveInitiative(context);
                    break;

                default:
                    resolution.message = `I can help with strike, spell casting, or initiative. What would you like to resolve?`;
                    resolution.suggestions = ['Strike attack', 'Cast spell', 'Roll initiative'];
            }
        } catch (error) {
            resolution.success = false;
            resolution.message = `Error resolving ${action}: ${error.message}`;
        }

        return resolution;
    }

    /**
     * Resolve melee attack mechanics
     */
    resolveMeleeAttack(context) {
        return {
            formula: "1d20 + ability modifier + proficiency + item bonus",
            criticalHit: "Natural 20 or exceed AC by 10+",
            criticalMiss: "Natural 1 or miss AC by 10+",
            multipleAttackPenalty: context.attackNumber > 1 ? (context.attackNumber - 1) * 5 : 0,
            damage: "Weapon damage + ability modifier"
        };
    }

    /**
     * Resolve spell casting mechanics
     */
    resolveSpellCasting(context) {
        return {
            actions: "Varies by spell (usually 2 actions)",
            components: "Check spell for somatic, verbal, material requirements",
            spellAttack: "1d20 + spellcasting ability + proficiency",
            savingThrow: "DC = 10 + spell level + spellcasting ability modifier",
            concentration: "Required for spells with duration > instantaneous"
        };
    }

    /**
     * Resolve initiative mechanics
     */
    resolveInitiative(context) {
        return {
            roll: "1d20 + Perception modifier",
            order: "Highest result goes first",
            ties: "Higher initiative modifier wins, players beat NPCs",
            exploration: "Activities before combat may change initiative skill"
        };
    }

    /**
     * Get service statistics
     */
    getStats() {
        return {
            ...this.stats,
            rulesInMemory: this.rulesDatabase.size,
            spellsInMemory: this.spellsDatabase.size,
            combatRulesLoaded: this.combatRules.size,
            initialized: this.initialized
        };
    }

    /**
     * Close service and cleanup resources
     */
    async close() {
        this.rulesDatabase.clear();
        this.spellsDatabase.clear();
        this.combatRules.clear();
        this.initialized = false;
        console.log('Pathfinder Rules Service | Closed');
    }
}

module.exports = { PathfinderRulesService };