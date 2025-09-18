/**
 * Hybrid RAG System Core Implementation for Pathfinder 2e
 * Research Princess Implementation - Familiar Project
 * Combines GraphRAG and Vector RAG for optimal rule querying
 */

const { GraphRAGEngine } = require('./graph-rag-engine');
const { VectorRAGEngine } = require('./vector-rag-engine');
const { ArchivesOfNethysConnector } = require('./archives-connector');
const { CostOptimizer } = require('./cost-optimizer');
const { QueryClassifier } = require('./query-classifier');

class HybridRAGSystem {
    constructor(config = {}) {
        this.config = {
            costLimit: 0.015, // $0.015 per session target
            cacheEnabled: true,
            fallbackToVector: true,
            performanceThreshold: 2000, // 2 seconds max response time
            ...config
        };

        this.graphRAG = new GraphRAGEngine({
            neo4jUri: config.neo4jUri || process.env.NEO4J_URI,
            neo4jUser: config.neo4jUser || process.env.NEO4J_USER,
            neo4jPassword: config.neo4jPassword || process.env.NEO4J_PASSWORD
        });

        this.vectorRAG = new VectorRAGEngine({
            pineconeApiKey: config.pineconeApiKey || process.env.PINECONE_API_KEY,
            indexName: config.indexName || 'pathfinder-2e-rules'
        });

        this.nethysConnector = new ArchivesOfNethysConnector();
        this.costOptimizer = new CostOptimizer(this.config.costLimit);
        this.queryClassifier = new QueryClassifier();

        this.performanceMetrics = {
            queriesProcessed: 0,
            averageResponseTime: 0,
            costPerSession: 0,
            cacheHitRate: 0
        };
    }

    /**
     * Main query processing pipeline
     * @param {string} query - User query about Pathfinder 2e rules
     * @param {Object} context - Additional context (character level, campaign, etc.)
     * @returns {Promise<Object>} - Structured response with sources
     */
    async processQuery(query, context = {}) {
        const startTime = Date.now();

        try {
            // Step 1: Classify query type for optimal routing
            const queryClass = await this.queryClassifier.classify(query);

            // Step 2: Check cache first
            const cacheKey = this.generateCacheKey(query, context);
            const cachedResult = await this.checkCache(cacheKey);
            if (cachedResult) {
                this.updateMetrics(Date.now() - startTime, true);
                return this.formatResponse(cachedResult, queryClass);
            }

            // Step 3: Route to appropriate RAG system
            let result;
            if (queryClass.complexity === 'simple' && queryClass.type === 'rule_lookup') {
                // Vector RAG for simple rule lookups
                result = await this.vectorRAG.query(query, context);
            } else if (queryClass.complexity === 'complex' || queryClass.type === 'rule_interaction') {
                // GraphRAG for complex multi-hop reasoning
                result = await this.graphRAG.query(query, context);
            } else {
                // Hybrid approach for medium complexity
                result = await this.hybridQuery(query, context, queryClass);
            }

            // Step 4: Post-process and enrich response
            const enrichedResult = await this.enrichResponse(result, query, context);

            // Step 5: Cache result for future queries
            await this.cacheResult(cacheKey, enrichedResult);

            // Step 6: Update performance metrics
            this.updateMetrics(Date.now() - startTime, false);

            return this.formatResponse(enrichedResult, queryClass);

        } catch (error) {
            console.error('HybridRAG query error:', error);

            // Fallback to vector RAG if graph fails
            if (this.config.fallbackToVector && !error.isVectorError) {
                try {
                    const fallbackResult = await this.vectorRAG.query(query, context);
                    return this.formatResponse(fallbackResult, { type: 'fallback' });
                } catch (fallbackError) {
                    throw new Error(`Both RAG systems failed: ${error.message}, ${fallbackError.message}`);
                }
            }

            throw error;
        }
    }

    /**
     * Hybrid query approach combining both systems
     */
    async hybridQuery(query, context, queryClass) {
        const [vectorResults, graphResults] = await Promise.allSettled([
            this.vectorRAG.query(query, context),
            this.graphRAG.query(query, context)
        ]);

        // Combine and rank results
        const combinedResults = this.combineResults(
            vectorResults.status === 'fulfilled' ? vectorResults.value : null,
            graphResults.status === 'fulfilled' ? graphResults.value : null,
            queryClass
        );

        return combinedResults;
    }

    /**
     * Combine results from vector and graph RAG systems
     */
    combineResults(vectorResults, graphResults, queryClass) {
        if (!vectorResults && !graphResults) {
            throw new Error('No results from either RAG system');
        }

        if (!vectorResults) return graphResults;
        if (!graphResults) return vectorResults;

        // Weight results based on query type
        const weights = this.getResultWeights(queryClass);

        return {
            answer: this.fuseAnswers(vectorResults.answer, graphResults.answer, weights),
            sources: this.mergeSources(vectorResults.sources, graphResults.sources),
            confidence: this.calculateCombinedConfidence(vectorResults, graphResults, weights),
            reasoning: {
                vector: vectorResults.reasoning,
                graph: graphResults.reasoning,
                fusion: 'Results combined using weighted fusion based on query complexity'
            }
        };
    }

    /**
     * Get weights for combining results based on query classification
     */
    getResultWeights(queryClass) {
        const weights = {
            rule_lookup: { vector: 0.7, graph: 0.3 },
            rule_interaction: { vector: 0.3, graph: 0.7 },
            stat_block: { vector: 0.8, graph: 0.2 },
            comparison: { vector: 0.4, graph: 0.6 },
            calculation: { vector: 0.2, graph: 0.8 }
        };

        return weights[queryClass.type] || { vector: 0.5, graph: 0.5 };
    }

    /**
     * Enrich response with additional context from Archives of Nethys
     */
    async enrichResponse(result, originalQuery, context) {
        try {
            // Add source citations and page references
            const enrichedSources = await Promise.all(
                result.sources.map(async (source) => {
                    const nethysData = await this.nethysConnector.getSourceDetails(source.id);
                    return {
                        ...source,
                        bookReference: nethysData?.bookReference,
                        pageNumber: nethysData?.pageNumber,
                        url: nethysData?.url,
                        lastUpdated: nethysData?.lastUpdated
                    };
                })
            );

            return {
                ...result,
                sources: enrichedSources,
                metadata: {
                    queryProcessedAt: new Date().toISOString(),
                    systemVersion: '1.0.0',
                    costEstimate: this.costOptimizer.getLastQueryCost()
                }
            };
        } catch (error) {
            console.warn('Failed to enrich response:', error);
            return result;
        }
    }

    /**
     * Format final response for API
     */
    formatResponse(result, queryClass) {
        return {
            answer: result.answer,
            sources: result.sources || [],
            confidence: result.confidence || 0.85,
            queryType: queryClass.type,
            complexity: queryClass.complexity,
            reasoning: result.reasoning,
            metadata: result.metadata,
            performance: {
                responseTime: this.performanceMetrics.averageResponseTime,
                cacheHit: result.fromCache || false,
                costEstimate: result.metadata?.costEstimate || 0
            }
        };
    }

    /**
     * Cache management
     */
    generateCacheKey(query, context) {
        const normalizedQuery = query.toLowerCase().trim();
        const contextHash = this.hashObject(context);
        return `rag_${this.hashString(normalizedQuery)}_${contextHash}`;
    }

    async checkCache(key) {
        // Implementation depends on cache backend (Redis, etc.)
        // Placeholder for now
        return null;
    }

    async cacheResult(key, result, ttl = 3600) {
        // Cache implementation
    }

    /**
     * Performance monitoring
     */
    updateMetrics(responseTime, fromCache) {
        this.performanceMetrics.queriesProcessed++;
        this.performanceMetrics.averageResponseTime =
            (this.performanceMetrics.averageResponseTime + responseTime) / 2;

        if (fromCache) {
            this.performanceMetrics.cacheHitRate =
                (this.performanceMetrics.cacheHitRate + 1) / 2;
        }
    }

    getPerformanceMetrics() {
        return {
            ...this.performanceMetrics,
            costEfficiency: this.performanceMetrics.costPerSession < this.config.costLimit,
            performance: this.performanceMetrics.averageResponseTime < this.config.performanceThreshold
        };
    }

    /**
     * Utility methods
     */
    hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return hash.toString(36);
    }

    hashObject(obj) {
        return this.hashString(JSON.stringify(obj));
    }

    fuseAnswers(vectorAnswer, graphAnswer, weights) {
        // Simple fusion - in production, use more sophisticated NLP techniques
        if (weights.graph > weights.vector) {
            return graphAnswer + '\n\nAdditional context: ' + vectorAnswer;
        } else {
            return vectorAnswer + '\n\nDetailed relationships: ' + graphAnswer;
        }
    }

    mergeSources(vectorSources, graphSources) {
        const allSources = [...(vectorSources || []), ...(graphSources || [])];
        const uniqueSources = allSources.filter((source, index, self) =>
            index === self.findIndex(s => s.id === source.id)
        );
        return uniqueSources.sort((a, b) => (b.relevance || 0) - (a.relevance || 0));
    }

    calculateCombinedConfidence(vectorResults, graphResults, weights) {
        const vectorConf = vectorResults.confidence || 0.8;
        const graphConf = graphResults.confidence || 0.8;
        return (vectorConf * weights.vector) + (graphConf * weights.graph);
    }
}

module.exports = { HybridRAGSystem };