/**
 * Archives of Nethys Connector
 * Handles data ingestion from Archives of Nethys for Pathfinder 2e
 */

const axios = require('axios');
const cheerio = require('cheerio');
const { RateLimiter } = require('limiter');

class ArchivesOfNethysConnector {
    constructor(config = {}) {
        this.baseUrl = 'https://2e.aonprd.com';
        this.rateLimiter = new RateLimiter({
            tokensPerInterval: config.requestsPerSecond || 2,
            interval: 'second'
        });

        this.cache = new Map();
        this.cacheExpiry = config.cacheExpiry || 3600000; // 1 hour

        // Common endpoints
        this.endpoints = {
            spells: '/Spells.aspx',
            creatures: '/Monsters.aspx',
            items: '/Equipment.aspx',
            classes: '/Classes.aspx',
            feats: '/Feats.aspx',
            rules: '/Rules.aspx',
            search: '/Search.aspx'
        };
    }

    /**
     * Get detailed source information for a rule/item
     */
    async getSourceDetails(sourceId) {
        const cacheKey = `source_${sourceId}`;

        // Check cache first
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheExpiry) {
                return cached.data;
            }
        }

        try {
            await this.rateLimiter.removeTokens(1);

            const response = await axios.get(`${this.baseUrl}${sourceId}`, {
                timeout: 10000,
                headers: {
                    'User-Agent': 'Familiar-Bot/1.0 (GM Assistant)'
                }
            });

            const details = this.parseSourcePage(response.data, sourceId);

            // Cache the result
            this.cache.set(cacheKey, {
                data: details,
                timestamp: Date.now()
            });

            return details;

        } catch (error) {
            console.warn(`Failed to fetch source details for ${sourceId}:`, error.message);
            return null;
        }
    }

    /**
     * Search Archives of Nethys
     */
    async search(query, filters = {}) {
        await this.rateLimiter.removeTokens(1);

        try {
            const searchParams = new URLSearchParams({
                q: query,
                ...filters
            });

            const response = await axios.get(`${this.baseUrl}${this.endpoints.search}?${searchParams}`, {
                timeout: 15000,
                headers: {
                    'User-Agent': 'Familiar-Bot/1.0 (GM Assistant)'
                }
            });

            return this.parseSearchResults(response.data);

        } catch (error) {
            console.error('Archives search failed:', error);
            throw new Error(`Archives search failed: ${error.message}`);
        }
    }

    /**
     * Scrape all spells for knowledge graph building
     */
    async scrapeAllSpells() {
        const spells = [];

        try {
            await this.rateLimiter.removeTokens(1);

            const response = await axios.get(`${this.baseUrl}${this.endpoints.spells}`, {
                timeout: 20000
            });

            const $ = cheerio.load(response.data);

            // Find spell links
            const spellLinks = [];
            $('a[href*="Spells.aspx?ID="]').each((i, el) => {
                const href = $(el).attr('href');
                const name = $(el).text().trim();
                if (href && name && !spellLinks.find(s => s.name === name)) {
                    spellLinks.push({ name, url: href });
                }
            });

            console.log(`Found ${spellLinks.length} spells to scrape`);

            // Scrape each spell (with rate limiting)
            for (const spell of spellLinks.slice(0, 50)) { // Limit for testing
                try {
                    await this.rateLimiter.removeTokens(1);

                    const spellData = await this.scrapeSpellDetails(spell.url);
                    if (spellData) {
                        spells.push(spellData);
                    }

                    // Small delay to be respectful
                    await this.delay(500);

                } catch (error) {
                    console.warn(`Failed to scrape spell ${spell.name}:`, error.message);
                }
            }

            return spells;

        } catch (error) {
            console.error('Failed to scrape spells:', error);
            throw error;
        }
    }

    /**
     * Scrape individual spell details
     */
    async scrapeSpellDetails(spellUrl) {
        try {
            const response = await axios.get(`${this.baseUrl}${spellUrl}`, {
                timeout: 10000
            });

            const $ = cheerio.load(response.data);

            const spell = {
                name: $('h1').first().text().trim(),
                url: spellUrl,
                source: this.extractSource($),
                level: this.extractLevel($),
                school: this.extractSchool($),
                traditions: this.extractTraditions($),
                castTime: this.extractCastTime($),
                range: this.extractRange($),
                area: this.extractArea($),
                targets: this.extractTargets($),
                duration: this.extractDuration($),
                savingThrow: this.extractSavingThrow($),
                description: this.extractDescription($),
                heightened: this.extractHeightened($)
            };

            return spell;

        } catch (error) {
            console.warn(`Failed to scrape spell details from ${spellUrl}:`, error.message);
            return null;
        }
    }

    /**
     * Scrape creature data for knowledge graph
     */
    async scrapeCreatures(limit = 20) {
        const creatures = [];

        try {
            await this.rateLimiter.removeTokens(1);

            const response = await axios.get(`${this.baseUrl}${this.endpoints.creatures}`, {
                timeout: 20000
            });

            const $ = cheerio.load(response.data);

            // Find creature links
            const creatureLinks = [];
            $('a[href*="Monsters.aspx?ID="]').each((i, el) => {
                const href = $(el).attr('href');
                const name = $(el).text().trim();
                if (href && name && !creatureLinks.find(c => c.name === name)) {
                    creatureLinks.push({ name, url: href });
                }
            });

            // Scrape subset of creatures
            for (const creature of creatureLinks.slice(0, limit)) {
                try {
                    await this.rateLimiter.removeTokens(1);

                    const creatureData = await this.scrapeCreatureDetails(creature.url);
                    if (creatureData) {
                        creatures.push(creatureData);
                    }

                    await this.delay(1000); // Longer delay for larger pages

                } catch (error) {
                    console.warn(`Failed to scrape creature ${creature.name}:`, error.message);
                }
            }

            return creatures;

        } catch (error) {
            console.error('Failed to scrape creatures:', error);
            throw error;
        }
    }

    /**
     * Scrape individual creature details
     */
    async scrapeCreatureDetails(creatureUrl) {
        try {
            const response = await axios.get(`${this.baseUrl}${creatureUrl}`, {
                timeout: 15000
            });

            const $ = cheerio.load(response.data);

            return {
                name: $('h1').first().text().trim(),
                url: creatureUrl,
                level: this.extractCreatureLevel($),
                type: this.extractCreatureType($),
                traits: this.extractTraits($),
                ac: this.extractAC($),
                hp: this.extractHP($),
                speed: this.extractSpeed($),
                abilities: this.extractAbilities($),
                skills: this.extractSkills($),
                attacks: this.extractAttacks($),
                spells: this.extractCreatureSpells($),
                description: this.extractCreatureDescription($)
            };

        } catch (error) {
            console.warn(`Failed to scrape creature from ${creatureUrl}:`, error.message);
            return null;
        }
    }

    /**
     * Parse source page to extract metadata
     */
    parseSourcePage(html, sourceId) {
        const $ = cheerio.load(html);

        return {
            title: $('h1').first().text().trim(),
            bookReference: this.extractBookReference($),
            pageNumber: this.extractPageNumber($),
            url: `${this.baseUrl}${sourceId}`,
            lastUpdated: this.extractLastUpdated($),
            traits: this.extractTraits($),
            level: this.extractLevel($) || this.extractCreatureLevel($),
            type: this.extractType($)
        };
    }

    /**
     * Parse search results
     */
    parseSearchResults(html) {
        const $ = cheerio.load(html);
        const results = [];

        $('.searchresult').each((i, el) => {
            const $result = $(el);
            results.push({
                title: $result.find('.title').text().trim(),
                url: $result.find('a').attr('href'),
                description: $result.find('.description').text().trim(),
                type: $result.find('.type').text().trim()
            });
        });

        return results;
    }

    // Extraction helper methods
    extractSource($) {
        return $('b:contains("Source")').next().text().trim();
    }

    extractLevel($) {
        const levelText = $('b:contains("Level")').next().text();
        const match = levelText.match(/(\d+)/);
        return match ? parseInt(match[1]) : null;
    }

    extractSchool($) {
        return $('b:contains("School")').next().text().trim();
    }

    extractTraditions($) {
        const traditionsText = $('b:contains("Traditions")').next().text();
        return traditionsText.split(',').map(t => t.trim()).filter(t => t);
    }

    extractCastTime($) {
        return $('b:contains("Cast")').next().text().trim();
    }

    extractRange($) {
        return $('b:contains("Range")').next().text().trim();
    }

    extractArea($) {
        return $('b:contains("Area")').next().text().trim();
    }

    extractTargets($) {
        return $('b:contains("Targets")').next().text().trim();
    }

    extractDuration($) {
        return $('b:contains("Duration")').next().text().trim();
    }

    extractSavingThrow($) {
        return $('b:contains("Saving Throw")').next().text().trim();
    }

    extractDescription($) {
        // Find the main description paragraph
        let description = '';
        $('hr').each((i, el) => {
            const nextP = $(el).next('p');
            if (nextP.length && !description) {
                description = nextP.text().trim();
            }
        });
        return description;
    }

    extractHeightened($) {
        const heightenedSections = [];
        $('b:contains("Heightened")').each((i, el) => {
            const text = $(el).parent().text();
            heightenedSections.push(text.trim());
        });
        return heightenedSections;
    }

    extractCreatureLevel($) {
        const text = $('span:contains("Creature")').text();
        const match = text.match(/Creature (\d+)/);
        return match ? parseInt(match[1]) : null;
    }

    extractCreatureType($) {
        return $('b:contains("Creature")').next().text().trim();
    }

    extractTraits($) {
        const traits = [];
        $('span.trait').each((i, el) => {
            traits.push($(el).text().trim());
        });
        return traits;
    }

    extractAC($) {
        const acText = $('b:contains("AC")').next().text();
        const match = acText.match(/(\d+)/);
        return match ? parseInt(match[1]) : null;
    }

    extractHP($) {
        const hpText = $('b:contains("HP")').next().text();
        const match = hpText.match(/(\d+)/);
        return match ? parseInt(match[1]) : null;
    }

    extractSpeed($) {
        return $('b:contains("Speed")').next().text().trim();
    }

    extractAbilities($) {
        const abilities = {};
        $('b:contains("Str")').parent().text().split(',').forEach(stat => {
            const match = stat.match(/(\w+)\s*([+-]?\d+)/);
            if (match) {
                abilities[match[1].toLowerCase()] = parseInt(match[2]);
            }
        });
        return abilities;
    }

    extractSkills($) {
        return $('b:contains("Skills")').next().text().trim();
    }

    extractAttacks($) {
        const attacks = [];
        $('b:contains("Melee"), b:contains("Ranged")').each((i, el) => {
            attacks.push($(el).parent().text().trim());
        });
        return attacks;
    }

    extractCreatureSpells($) {
        const spells = [];
        $('b:contains("Spells")').each((i, el) => {
            spells.push($(el).parent().text().trim());
        });
        return spells;
    }

    extractCreatureDescription($) {
        return $('.description').first().text().trim();
    }

    extractBookReference($) {
        const sourceText = $('b:contains("Source")').next().text();
        const match = sourceText.match(/([A-Z]+)\s*(\d+)/);
        return match ? `${match[1]} ${match[2]}` : null;
    }

    extractPageNumber($) {
        const sourceText = $('b:contains("Source")').next().text();
        const match = sourceText.match(/pg\.\s*(\d+)/i);
        return match ? parseInt(match[1]) : null;
    }

    extractLastUpdated($) {
        const dateText = $('span:contains("Updated")').text();
        const match = dateText.match(/(\d{1,2}\/\d{1,2}\/\d{4})/);
        return match ? match[1] : null;
    }

    extractType($) {
        // Generic type extraction
        return $('b:contains("Type")').next().text().trim();
    }

    /**
     * Build knowledge graph from scraped data
     */
    async buildKnowledgeGraph(spells, creatures) {
        const graph = {
            nodes: [],
            relationships: []
        };

        // Add spell nodes
        spells.forEach(spell => {
            graph.nodes.push({
                id: `spell_${spell.name.toLowerCase().replace(/\s+/g, '_')}`,
                type: 'Spell',
                properties: {
                    name: spell.name,
                    level: spell.level,
                    school: spell.school,
                    traditions: spell.traditions,
                    description: spell.description,
                    url: spell.url
                }
            });

            // Add school relationship
            if (spell.school) {
                graph.relationships.push({
                    from: `spell_${spell.name.toLowerCase().replace(/\s+/g, '_')}`,
                    to: `school_${spell.school.toLowerCase()}`,
                    type: 'BELONGS_TO_SCHOOL'
                });
            }

            // Add tradition relationships
            spell.traditions?.forEach(tradition => {
                graph.relationships.push({
                    from: `spell_${spell.name.toLowerCase().replace(/\s+/g, '_')}`,
                    to: `tradition_${tradition.toLowerCase()}`,
                    type: 'AVAILABLE_TO_TRADITION'
                });
            });
        });

        // Add creature nodes
        creatures.forEach(creature => {
            graph.nodes.push({
                id: `creature_${creature.name.toLowerCase().replace(/\s+/g, '_')}`,
                type: 'Creature',
                properties: {
                    name: creature.name,
                    level: creature.level,
                    type: creature.type,
                    traits: creature.traits,
                    ac: creature.ac,
                    hp: creature.hp,
                    url: creature.url
                }
            });

            // Add trait relationships
            creature.traits?.forEach(trait => {
                graph.relationships.push({
                    from: `creature_${creature.name.toLowerCase().replace(/\s+/g, '_')}`,
                    to: `trait_${trait.toLowerCase()}`,
                    type: 'HAS_TRAIT'
                });
            });
        });

        return graph;
    }

    /**
     * Utility methods
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    clearCache() {
        this.cache.clear();
    }

    getCacheStats() {
        return {
            size: this.cache.size,
            items: Array.from(this.cache.keys())
        };
    }
}

module.exports = { ArchivesOfNethysConnector };