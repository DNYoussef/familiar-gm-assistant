/**
 * Query Classifier for Pathfinder 2e RAG System
 * Determines optimal RAG approach based on query characteristics
 */

class QueryClassifier {
    constructor() {
        this.patterns = this.initializePatterns();
        this.cache = new Map();
    }

    /**
     * Classify user query to determine optimal RAG approach
     */
    async classify(query) {
        const cacheKey = query.toLowerCase().trim();

        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        const classification = {
            type: this.determineQueryType(query),
            complexity: this.determineComplexity(query),
            entities: this.extractMentionedEntities(query),
            intent: this.determineIntent(query),
            confidence: 0.8
        };

        // Enhance classification with pattern matching
        this.enhanceWithPatterns(query, classification);

        // Cache result
        this.cache.set(cacheKey, classification);

        return classification;
    }

    /**
     * Determine primary query type
     */
    determineQueryType(query) {
        const types = {
            rule_lookup: /what is|how does|explain|define|tell me about/i,
            rule_interaction: /interact|affect|combine|stack|work together|conflict/i,
            comparison: /compare|difference|better|versus|vs|which is/i,
            calculation: /damage|bonus|total|calculate|how much|what's the/i,
            prerequisite: /prerequisite|require|need|must have|qualify/i,
            spell_compatibility: /(spell|magic).*(combine|stack|work)/i,
            class_feature: /(class|feature|ability).*(work|function|use)/i,
            stat_block: /stat block|stats|creature|monster|ac|hp|abilities/i,
            item_lookup: /item|equipment|weapon|armor|gear|magic item/i,
            character_build: /build|character|feat|class progression/i
        };

        for (const [type, pattern] of Object.entries(types)) {
            if (pattern.test(query)) {
                return type;
            }
        }

        return 'general';
    }

    /**
     * Determine query complexity
     */
    determineComplexity(query) {
        let complexityScore = 0;

        // Length factor
        if (query.length > 100) complexityScore += 1;
        if (query.length > 200) complexityScore += 1;

        // Multiple entities
        const entityCount = this.countEntities(query);
        if (entityCount > 2) complexityScore += 1;
        if (entityCount > 4) complexityScore += 1;

        // Complex relationship words
        const complexWords = [
            'interact', 'combine', 'stack', 'conflict', 'modify',
            'prerequisite', 'requirement', 'condition', 'multiclass',
            'archetype', 'variant', 'optional', 'alternative'
        ];

        complexWords.forEach(word => {
            if (query.toLowerCase().includes(word)) {
                complexityScore += 1;
            }
        });

        // Question complexity indicators
        const complexPatterns = [
            /if.*then/i,           // Conditional
            /both.*and/i,          // Multiple conditions
            /either.*or/i,         // Alternatives
            /when.*while/i,        // Temporal complexity
            /instead of/i,         // Substitution
            /as long as/i,         // Duration conditions
            /unless/i              // Exceptions
        ];

        complexPatterns.forEach(pattern => {
            if (pattern.test(query)) {
                complexityScore += 1;
            }
        });

        // Classify based on score
        if (complexityScore <= 1) return 'simple';
        if (complexityScore <= 3) return 'medium';
        return 'complex';
    }

    /**
     * Extract mentioned entities from query
     */
    extractMentionedEntities(query) {
        const entities = {
            spells: this.extractByPattern(query, this.patterns.spells),
            classes: this.extractByPattern(query, this.patterns.classes),
            creatures: this.extractByPattern(query, this.patterns.creatures),
            items: this.extractByPattern(query, this.patterns.items),
            conditions: this.extractByPattern(query, this.patterns.conditions),
            actions: this.extractByPattern(query, this.patterns.actions),
            feats: this.extractByPattern(query, this.patterns.feats),
            skills: this.extractByPattern(query, this.patterns.skills),
            schools: this.extractByPattern(query, this.patterns.schools),
            traits: this.extractByPattern(query, this.patterns.traits)
        };

        return entities;
    }

    /**
     * Determine user intent
     */
    determineIntent(query) {
        const intents = {
            learn: /learn|understand|explain|what is|how does/i,
            apply: /use|apply|cast|activate|trigger/i,
            optimize: /best|optimal|maximize|efficient|better/i,
            compare: /compare|versus|difference|which/i,
            build: /build|create|design|plan|choose/i,
            resolve: /resolve|clarify|conflict|contradiction/i,
            calculate: /calculate|compute|total|sum|damage/i
        };

        for (const [intent, pattern] of Object.entries(intents)) {
            if (pattern.test(query)) {
                return intent;
            }
        }

        return 'general';
    }

    /**
     * Enhance classification with advanced pattern matching
     */
    enhanceWithPatterns(query, classification) {
        // Multi-hop reasoning indicators
        const multiHopPatterns = [
            /then.*can I/i,
            /if I have.*what about/i,
            /after.*do I/i,
            /chain.*with/i,
            /follow up.*with/i
        ];

        if (multiHopPatterns.some(pattern => pattern.test(query))) {
            classification.requiresMultiHop = true;
            if (classification.complexity === 'simple') {
                classification.complexity = 'medium';
            }
        }

        // Context dependency indicators
        const contextPatterns = [
            /at level/i,
            /with my/i,
            /as a.*character/i,
            /in my campaign/i,
            /house rule/i
        ];

        if (contextPatterns.some(pattern => pattern.test(query))) {
            classification.requiresContext = true;
        }

        // Source citation needs
        const citationPatterns = [
            /according to/i,
            /official/i,
            /rule/i,
            /page/i,
            /source/i,
            /legal/i
        ];

        if (citationPatterns.some(pattern => pattern.test(query))) {
            classification.requiresCitation = true;
        }

        // Mathematical calculation needs
        const mathPatterns = [
            /\d+.*\+.*\d+/,      // Addition
            /\d+.*-.*\d+/,       // Subtraction
            /\d+.*\*.*\d+/,      // Multiplication
            /\d+d\d+/,           // Dice notation
            /DC\s*\d+/i,         // Difficulty Class
            /level\s*\d+/i       // Level references
        ];

        if (mathPatterns.some(pattern => pattern.test(query))) {
            classification.requiresCalculation = true;
        }

        // Adjust confidence based on pattern matches
        const totalPatterns = multiHopPatterns.length + contextPatterns.length +
                            citationPatterns.length + mathPatterns.length;
        const matchedPatterns = [
            classification.requiresMultiHop,
            classification.requiresContext,
            classification.requiresCitation,
            classification.requiresCalculation
        ].filter(Boolean).length;

        classification.confidence = Math.min(0.95, 0.7 + (matchedPatterns / totalPatterns) * 0.25);
    }

    /**
     * Count entities mentioned in query
     */
    countEntities(query) {
        let count = 0;

        Object.values(this.patterns).forEach(pattern => {
            const matches = query.match(pattern);
            if (matches) {
                count += matches.length;
            }
        });

        return count;
    }

    /**
     * Extract entities by pattern
     */
    extractByPattern(query, pattern) {
        const matches = query.match(pattern);
        return matches ? [...new Set(matches.map(m => m.toLowerCase()))] : [];
    }

    /**
     * Initialize entity patterns
     */
    initializePatterns() {
        return {
            spells: /\b(fireball|magic missile|heal|shield|cure wounds|bless|haste|slow|teleport|fly|invisibility|charm person|sleep|burning hands|cone of cold|lightning bolt|wall of fire|dispel magic|counterspell|silence|darkness|light|detect magic|mage armor|false life|mirror image|blur|web|grease|entangle|faerie fire|color spray|comprehend languages|identify|alarm|feather fall|jump|expeditious retreat|misty step|thunder wave|acid splash|ray of frost|fire bolt|eldritch blast|sacred flame|guidance|thaumaturgy|prestidigitation|mending|minor illusion|dancing lights|druidcraft|resistance|spare the dying|toll the dead|word of radiance|vicious mockery|minor image|suggestion|hold person|spiritual weapon|scorching ray|shatter|moonbeam|call lightning|fireball|counterspell|hypnotic pattern|slow|haste|polymorph|greater invisibility|confusion|wall of fire|cone of cold|dominate person|disintegrate|chain lightning|sunbeam|reverse gravity|meteor swarm|wish|power word kill)\b/gi,

            classes: /\b(alchemist|barbarian|bard|champion|cleric|druid|fighter|gunslinger|inventor|investigator|kineticist|magus|monk|oracle|psychic|ranger|rogue|sorcerer|summoner|swashbuckler|thaumaturge|witch|wizard)\b/gi,

            creatures: /\b(goblin|orc|dragon|troll|giant|elemental|demon|devil|angel|undead|zombie|skeleton|ghost|vampire|werewolf|bear|wolf|lion|tiger|eagle|hawk|spider|scorpion|snake|basilisk|chimera|griffon|pegasus|unicorn|centaur|minotaur|harpy|siren|medusa|cyclops|ogre|ettin|hill giant|stone giant|fire giant|frost giant|cloud giant|storm giant|ancient dragon|young dragon|wyrmling|kobold|lizardfolk|troglodyte|drow|elf|dwarf|halfling|gnome|human|tiefling|aasimar|genasi|dragonborn|warforged|changeling|shifter|kalashtar|githyanki|githzerai|mind flayer|beholder|lich|tarrasque|kraken|aboleth|owlbear|bulezau|succubus|incubus|balor|pit fiend|solar|planetar|deva)\b/gi,

            items: /\b(sword|axe|hammer|bow|crossbow|staff|wand|rod|armor|shield|helmet|boots|gloves|cloak|ring|amulet|necklace|bracers|belt|potion|scroll|wand|bag of holding|portable hole|rope|torch|lantern|oil|rations|bedroll|tent|backpack|waterskin|grappling hook|crowbar|hammer|piton|spike|rope|chain|manacles|lock|key|thieves tools|disguise kit|forgery kit|herbalism kit|poisoners kit|artisan tools|musical instrument|holy symbol|spellbook|component pouch|focus|crystal|orb|tome|grimoire|dagger|shortsword|longsword|greatsword|scimitar|rapier|club|mace|warhammer|battleaxe|handaxe|greataxe|spear|javelin|trident|pike|glaive|halberd|quarterstaff|sling|dart|blowgun|net|whip|flail|morningstar|war pick|maul|lance|shortbow|longbow|light crossbow|heavy crossbow|hand crossbow)\b/gi,

            conditions: /\b(blinded|charmed|deafened|frightened|grappled|incapacitated|invisible|paralyzed|petrified|poisoned|prone|restrained|stunned|unconscious|exhausted|dying|doomed|drained|enfeebled|clumsy|stupefied|sickened|slowed|fascinated|confused|controlled|grabbed|immobilized|off-guard|flat-footed|hidden|concealed|undetected|observed|broken|fatigued|persistent damage|bleeding|burning|acid|cold|electricity|fire|force|mental|poison|sonic|positive|negative|good|evil|lawful|chaotic)\b/gi,

            actions: /\b(strike|cast|move|step|stride|crawl|climb|swim|fly|interact|activate|ready|delay|aid|escape|grapple|shove|trip|disarm|feint|demoralize|recall knowledge|seek|hide|sneak|avoid notice|balance|tumble|jump|squeeze|force open|disable|pick lock|steal|palm|sleight of hand|perform|impersonate|lie|diplomacy|intimidate|coerce|gather information|make impression|request|command|treat wounds|battle medicine|craft|earn income|subsist|long-term rest|refocus|sustain|dismiss)\b/gi,

            feats: /\b(power attack|combat expertise|weapon finesse|two-weapon fighting|archery|defense|dueling|great weapon fighting|protection|blessed warrior|eldritch sight|pact of blade|pact of tome|pact of chain|invocation|metamagic|careful spell|distant spell|empowered spell|extended spell|heightened spell|quickened spell|subtle spell|twinned spell|wild shape|combat wild shape|elemental wild shape|rage|reckless attack|danger sense|feral instinct|brutal critical|relentless rage|persistent rage|indomitable might|primal champion|action surge|second wind|fighting style|extra attack|indomitable|survivor|sneak attack|thieves cant|cunning action|uncanny dodge|evasion|reliable talent|blindsense|slippery mind|elusive|stroke of luck|spellcasting|ritual casting|spellcasting focus|cantrips|spell preparation|spell recovery|arcane recovery|spell mastery|signature spells)\b/gi,

            skills: /\b(acrobatics|athletics|crafting|deception|diplomacy|intimidation|lore|medicine|nature|occultism|performance|religion|society|stealth|survival|thievery|arcana|history|investigation|insight|perception|persuasion|animal handling|sleight of hand)\b/gi,

            schools: /\b(abjuration|conjuration|divination|enchantment|evocation|illusion|necromancy|transmutation)\b/gi,

            traits: /\b(acid|air|chaotic|cold|darkness|death|earth|evil|fear|fire|force|good|healing|lawful|light|mental|poison|sonic|water|attack|cantrip|concentrate|exploration|fortune|incapacitation|manipulate|metamagic|misfortune|move|open|secret|visual|auditory|linguistic|emotion|magical|divine|occult|primal|arcane|rare|uncommon|unique|aura|polymorph|scrying|teleportation|prediction)\b/gi
        };
    }

    /**
     * Get routing recommendation based on classification
     */
    getRoutingRecommendation(classification) {
        const { type, complexity, requiresMultiHop, requiresContext } = classification;

        // GraphRAG recommendations
        if (requiresMultiHop ||
            type === 'rule_interaction' ||
            type === 'prerequisite' ||
            complexity === 'complex') {
            return {
                primary: 'graphrag',
                secondary: 'vector',
                reason: 'Complex relationships require graph traversal'
            };
        }

        // Vector RAG recommendations
        if (type === 'rule_lookup' ||
            type === 'stat_block' ||
            type === 'item_lookup' ||
            complexity === 'simple') {
            return {
                primary: 'vector',
                secondary: 'graph',
                reason: 'Simple lookups work well with semantic search'
            };
        }

        // Hybrid recommendations
        return {
            primary: 'hybrid',
            secondary: null,
            reason: 'Medium complexity benefits from combined approach'
        };
    }

    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
    }

    /**
     * Get cache statistics
     */
    getCacheStats() {
        return {
            size: this.cache.size,
            hitRate: this.cacheHits / (this.cacheHits + this.cacheMisses),
            entries: Array.from(this.cache.keys()).slice(0, 10)
        };
    }
}

module.exports = { QueryClassifier };