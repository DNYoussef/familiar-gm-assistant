/**
 * GraphRAG Engine Implementation for Pathfinder 2e
 * Handles complex rule relationships and multi-hop reasoning
 */

const neo4j = require('neo4j-driver');
const { OpenAI } = require('openai');

class GraphRAGEngine {
    constructor(config) {
        this.driver = neo4j.driver(
            config.neo4jUri,
            neo4j.auth.basic(config.neo4jUser, config.neo4jPassword)
        );

        this.openai = new OpenAI({
            apiKey: config.openaiApiKey || process.env.OPENAI_API_KEY
        });

        this.queryTemplates = this.initializeQueryTemplates();
    }

    /**
     * Process query using GraphRAG approach
     */
    async query(query, context = {}) {
        const session = this.driver.session();

        try {
            // Step 1: Extract entities from query
            const entities = await this.extractEntities(query);

            // Step 2: Generate Cypher query
            const cypherQuery = await this.generateCypherQuery(query, entities, context);

            // Step 3: Execute graph traversal
            const graphResults = await session.run(cypherQuery.query, cypherQuery.parameters);

            // Step 4: Process graph results into structured context
            const structuredContext = this.processGraphResults(graphResults);

            // Step 5: Generate final answer using graph context
            const answer = await this.generateAnswerFromGraph(query, structuredContext, context);

            return {
                answer: answer.text,
                confidence: answer.confidence,
                sources: structuredContext.sources,
                reasoning: {
                    entities: entities,
                    cypherQuery: cypherQuery.query,
                    graphPath: structuredContext.reasoning,
                    method: 'GraphRAG'
                }
            };

        } finally {
            await session.close();
        }
    }

    /**
     * Extract named entities from user query
     */
    async extractEntities(query) {
        const prompt = `
Extract Pathfinder 2e game entities from this query. Return JSON format:

Query: "${query}"

Extract these entity types:
- spells: Spell names
- classes: Character classes
- creatures: Monster/creature names
- items: Equipment, weapons, armor
- conditions: Game conditions/states
- actions: Combat actions
- feats: Character feats
- skills: Skill names

Return only JSON, no explanation:
{
  "spells": [],
  "classes": [],
  "creatures": [],
  "items": [],
  "conditions": [],
  "actions": [],
  "feats": [],
  "skills": []
}`;

        try {
            const response = await this.openai.chat.completions.create({
                model: 'gpt-3.5-turbo',
                messages: [{ role: 'user', content: prompt }],
                temperature: 0.1,
                max_tokens: 500
            });

            return JSON.parse(response.choices[0].message.content);
        } catch (error) {
            console.warn('Entity extraction failed, using fallback:', error);
            return this.fallbackEntityExtraction(query);
        }
    }

    /**
     * Generate Cypher query based on natural language query and entities
     */
    async generateCypherQuery(query, entities, context) {
        // Determine query pattern based on query type
        const queryPattern = this.classifyQueryPattern(query, entities);

        switch (queryPattern.type) {
            case 'rule_interaction':
                return this.buildRuleInteractionQuery(entities, queryPattern);
            case 'prerequisite_chain':
                return this.buildPrerequisiteQuery(entities, queryPattern);
            case 'spell_compatibility':
                return this.buildSpellCompatibilityQuery(entities, queryPattern);
            case 'class_feature_lookup':
                return this.buildClassFeatureQuery(entities, queryPattern);
            default:
                return this.buildGenericTraversalQuery(entities, queryPattern);
        }
    }

    /**
     * Query pattern classification
     */
    classifyQueryPattern(query, entities) {
        const patterns = {
            rule_interaction: /interact|affect|modify|change|combine/i,
            prerequisite_chain: /prerequisite|require|need|must have/i,
            spell_compatibility: /(spell|magic).*(stack|combine|work together)/i,
            class_feature_lookup: /(class|feature|ability).*(work|function)/i,
            comparison: /compare|difference|better|versus|vs/i
        };

        for (const [type, regex] of Object.entries(patterns)) {
            if (regex.test(query)) {
                return { type, confidence: 0.8 };
            }
        }

        return { type: 'generic', confidence: 0.6 };
    }

    /**
     * Build specific query patterns
     */
    buildRuleInteractionQuery(entities, pattern) {
        let query = `
        MATCH (r1:Rule)-[:INTERACTS_WITH|MODIFIES|AFFECTS*1..3]-(r2:Rule)
        WHERE `;

        const conditions = [];
        const parameters = {};

        // Add entity conditions
        Object.entries(entities).forEach(([type, names]) => {
            if (names.length > 0) {
                names.forEach((name, index) => {
                    const paramKey = `${type}_${index}`;
                    conditions.push(`r1.name CONTAINS $${paramKey} OR r2.name CONTAINS $${paramKey}`);
                    parameters[paramKey] = name;
                });
            }
        });

        if (conditions.length === 0) {
            // Fallback to text search
            conditions.push(`r1.text CONTAINS $searchTerm OR r2.text CONTAINS $searchTerm`);
            parameters.searchTerm = entities.spells[0] || entities.classes[0] || 'rules';
        }

        query += conditions.join(' OR ');
        query += `
        RETURN r1, r2,
               [rel in relationships(path(r1, r2)) | type(rel)] as relationship_types,
               path(r1, r2) as interaction_path
        LIMIT 10`;

        return { query, parameters };
    }

    buildPrerequisiteQuery(entities, pattern) {
        const query = `
        MATCH path = (start)-[:PREREQUISITE*1..5]->(target)
        WHERE target.name IN $targetNames
        RETURN path, start, target,
               [node in nodes(path) | node.name] as prerequisite_chain,
               length(path) as chain_length
        ORDER BY chain_length
        LIMIT 20`;

        const targetNames = Object.values(entities).flat().filter(name => name);
        return {
            query,
            parameters: { targetNames: targetNames.length > 0 ? targetNames : ['feat'] }
        };
    }

    buildSpellCompatibilityQuery(entities, pattern) {
        const query = `
        MATCH (s1:Spell)-[:COMPATIBLE_WITH|STACKS_WITH|CONFLICTS_WITH]-(s2:Spell)
        WHERE s1.name IN $spellNames OR s2.name IN $spellNames
        RETURN s1, s2,
               type(r) as relationship_type,
               s1.school as school1, s2.school as school2
        LIMIT 15`;

        return {
            query,
            parameters: { spellNames: entities.spells.length > 0 ? entities.spells : ['fireball'] }
        };
    }

    buildClassFeatureQuery(entities, pattern) {
        const query = `
        MATCH (c:Class)-[:HAS_FEATURE]->(f:Feature)
        WHERE c.name IN $classNames
        OPTIONAL MATCH (f)-[:MODIFIES|ENHANCES]->(r:Rule)
        RETURN c, f, r,
               f.level as feature_level,
               f.description as feature_description
        ORDER BY f.level
        LIMIT 20`;

        return {
            query,
            parameters: { classNames: entities.classes.length > 0 ? entities.classes : ['wizard'] }
        };
    }

    buildGenericTraversalQuery(entities, pattern) {
        const query = `
        MATCH (n)
        WHERE ANY(prop in keys(n) WHERE toString(n[prop]) CONTAINS $searchTerm)
        OPTIONAL MATCH (n)-[r*1..2]-(connected)
        RETURN n, collect(DISTINCT connected) as related_nodes,
               collect(DISTINCT type(r)) as relationship_types
        LIMIT 10`;

        const searchTerm = Object.values(entities).flat()[0] || 'rules';
        return { query, parameters: { searchTerm } };
    }

    /**
     * Process Neo4j results into structured context
     */
    processGraphResults(results) {
        const context = {
            entities: [],
            relationships: [],
            paths: [],
            sources: [],
            reasoning: []
        };

        results.records.forEach(record => {
            // Extract nodes
            record.keys.forEach(key => {
                const value = record.get(key);

                if (value && value.labels) {
                    // It's a Neo4j node
                    context.entities.push({
                        id: value.identity.toString(),
                        labels: value.labels,
                        properties: value.properties,
                        type: value.labels[0]
                    });

                    // Add as source
                    context.sources.push({
                        id: value.identity.toString(),
                        title: value.properties.name || value.properties.title,
                        type: value.labels[0],
                        relevance: 0.9
                    });
                }

                if (Array.isArray(value)) {
                    // Could be relationship types or paths
                    if (key.includes('relationship')) {
                        context.relationships.push(...value);
                    } else if (key.includes('path') || key.includes('chain')) {
                        context.paths.push({
                            type: key,
                            nodes: value
                        });
                    }
                }
            });
        });

        // Build reasoning explanation
        context.reasoning = this.buildReasoningExplanation(context);

        return context;
    }

    buildReasoningExplanation(context) {
        const reasoning = [];

        if (context.entities.length > 0) {
            reasoning.push(`Found ${context.entities.length} relevant entities in the knowledge graph`);
        }

        if (context.relationships.length > 0) {
            const uniqueRels = [...new Set(context.relationships)];
            reasoning.push(`Discovered relationships: ${uniqueRels.join(', ')}`);
        }

        if (context.paths.length > 0) {
            reasoning.push(`Traced ${context.paths.length} logical paths through rule connections`);
        }

        return reasoning;
    }

    /**
     * Generate final answer using graph context
     */
    async generateAnswerFromGraph(query, context, userContext) {
        const prompt = `
You are a Pathfinder 2e rules expert. Answer the user's question using the provided graph context.

User Question: "${query}"

Graph Context:
${JSON.stringify(context, null, 2)}

Instructions:
1. Use the graph relationships to provide accurate rule interactions
2. Cite specific sources from the context
3. Explain any rule chains or prerequisites found
4. Be precise about mechanical interactions
5. If multiple paths exist, explain the different possibilities

Provide a comprehensive answer that leverages the relationship data:`;

        try {
            const response = await this.openai.chat.completions.create({
                model: 'gpt-4',
                messages: [{ role: 'user', content: prompt }],
                temperature: 0.3,
                max_tokens: 800
            });

            return {
                text: response.choices[0].message.content,
                confidence: 0.9
            };
        } catch (error) {
            console.error('Answer generation failed:', error);
            return {
                text: 'Unable to generate answer from graph context.',
                confidence: 0.1
            };
        }
    }

    /**
     * Fallback entity extraction using simple regex
     */
    fallbackEntityExtraction(query) {
        const entities = {
            spells: [],
            classes: [],
            creatures: [],
            items: [],
            conditions: [],
            actions: [],
            feats: [],
            skills: []
        };

        // Simple pattern matching for common terms
        const patterns = {
            spells: /\b(fireball|magic missile|heal|shield|fireball)\b/gi,
            classes: /\b(wizard|fighter|rogue|cleric|ranger|barbarian)\b/gi,
            conditions: /\b(prone|stunned|paralyzed|confused|frightened)\b/gi,
            actions: /\b(attack|move|cast|defend|ready)\b/gi
        };

        Object.entries(patterns).forEach(([type, regex]) => {
            const matches = query.match(regex);
            if (matches) {
                entities[type] = [...new Set(matches.map(m => m.toLowerCase()))];
            }
        });

        return entities;
    }

    /**
     * Initialize query templates for common patterns
     */
    initializeQueryTemplates() {
        return {
            // Add more templates as needed
        };
    }

    async close() {
        await this.driver.close();
    }
}

module.exports = { GraphRAGEngine };