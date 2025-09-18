/**
 * Vector RAG Engine Implementation for Pathfinder 2e
 * Handles semantic similarity search and fast rule lookups
 */

const { Pinecone } = require('@pinecone-database/pinecone');
const { OpenAI } = require('openai');

class VectorRAGEngine {
    constructor(config) {
        this.pinecone = new Pinecone({
            apiKey: config.pineconeApiKey,
            environment: config.environment || 'us-west1-gcp'
        });

        this.openai = new OpenAI({
            apiKey: config.openaiApiKey || process.env.OPENAI_API_KEY
        });

        this.indexName = config.indexName;
        this.embeddingModel = config.embeddingModel || 'text-embedding-ada-002';
        this.topK = config.topK || 10;
        this.threshold = config.threshold || 0.7;
    }

    /**
     * Process query using Vector RAG approach
     */
    async query(query, context = {}) {
        try {
            // Step 1: Generate query embedding
            const queryEmbedding = await this.generateEmbedding(query);

            // Step 2: Search vector database
            const searchResults = await this.searchVectors(queryEmbedding, context);

            // Step 3: Filter by relevance threshold
            const relevantResults = this.filterByRelevance(searchResults);

            // Step 4: Prepare context for generation
            const structuredContext = this.prepareContext(relevantResults, query);

            // Step 5: Generate answer
            const answer = await this.generateAnswerFromContext(query, structuredContext, context);

            return {
                answer: answer.text,
                confidence: answer.confidence,
                sources: structuredContext.sources,
                reasoning: {
                    searchResults: searchResults.matches?.length || 0,
                    relevantResults: relevantResults.length,
                    averageScore: this.calculateAverageScore(relevantResults),
                    method: 'VectorRAG'
                }
            };

        } catch (error) {
            console.error('VectorRAG query error:', error);
            error.isVectorError = true;
            throw error;
        }
    }

    /**
     * Generate embedding for query text
     */
    async generateEmbedding(text) {
        try {
            const response = await this.openai.embeddings.create({
                model: this.embeddingModel,
                input: text.trim()
            });

            return response.data[0].embedding;
        } catch (error) {
            console.error('Embedding generation failed:', error);
            throw new Error(`Failed to generate embedding: ${error.message}`);
        }
    }

    /**
     * Search vector database for similar content
     */
    async searchVectors(embedding, context) {
        try {
            const index = this.pinecone.Index(this.indexName);

            const searchParams = {
                vector: embedding,
                topK: this.topK,
                includeMetadata: true,
                includeValues: false
            };

            // Add filters based on context
            if (context.sourceType) {
                searchParams.filter = { source_type: context.sourceType };
            }

            if (context.level) {
                searchParams.filter = {
                    ...searchParams.filter,
                    level: { $lte: context.level }
                };
            }

            const results = await index.query(searchParams);
            return results;

        } catch (error) {
            console.error('Vector search failed:', error);
            throw new Error(`Vector search failed: ${error.message}`);
        }
    }

    /**
     * Filter results by relevance threshold
     */
    filterByRelevance(searchResults) {
        if (!searchResults.matches) {
            return [];
        }

        return searchResults.matches
            .filter(match => match.score >= this.threshold)
            .sort((a, b) => b.score - a.score);
    }

    /**
     * Prepare context for answer generation
     */
    prepareContext(results, query) {
        const context = {
            sources: [],
            content: [],
            metadata: {
                totalResults: results.length,
                averageRelevance: this.calculateAverageScore(results)
            }
        };

        results.forEach((result, index) => {
            const metadata = result.metadata || {};

            // Add to sources
            context.sources.push({
                id: result.id,
                title: metadata.title || `Rule ${index + 1}`,
                type: metadata.source_type || 'rule',
                relevance: result.score,
                book: metadata.book,
                page: metadata.page,
                url: metadata.url,
                section: metadata.section
            });

            // Add content for generation
            context.content.push({
                text: metadata.text || metadata.content || '',
                title: metadata.title || '',
                relevance: result.score,
                type: metadata.source_type || 'rule'
            });
        });

        return context;
    }

    /**
     * Generate answer from retrieved context
     */
    async generateAnswerFromContext(query, context, userContext) {
        if (context.content.length === 0) {
            return {
                text: "I couldn't find relevant information in the rules database for your query.",
                confidence: 0.1
            };
        }

        // Prepare content for prompt
        const contentText = context.content
            .map((item, index) => `[${index + 1}] ${item.title}: ${item.text}`)
            .join('\n\n');

        const prompt = `
You are a Pathfinder 2e rules expert. Answer the user's question using only the provided rule context.

User Question: "${query}"

Rule Context:
${contentText}

Instructions:
1. Answer based ONLY on the provided rules
2. Be precise and cite specific rules by their numbers [1], [2], etc.
3. If the rules conflict, explain the conflicts
4. If information is insufficient, say so clearly
5. Include relevant mechanical details (DCs, bonuses, etc.)
6. Keep answers concise but complete

Answer:`;

        try {
            const response = await this.openai.chat.completions.create({
                model: 'gpt-4',
                messages: [{ role: 'user', content: prompt }],
                temperature: 0.2,
                max_tokens: 600
            });

            // Calculate confidence based on content relevance
            const confidence = this.calculateAnswerConfidence(context, query);

            return {
                text: response.choices[0].message.content,
                confidence: confidence
            };

        } catch (error) {
            console.error('Answer generation failed:', error);
            return {
                text: 'Unable to generate answer from the available context.',
                confidence: 0.1
            };
        }
    }

    /**
     * Calculate confidence score for generated answer
     */
    calculateAnswerConfidence(context, query) {
        if (context.content.length === 0) return 0.1;

        const avgRelevance = context.metadata.averageRelevance;
        const resultCount = Math.min(context.content.length / this.topK, 1);
        const queryLength = Math.min(query.length / 100, 1); // Normalize query length

        // Weighted combination
        const confidence = (avgRelevance * 0.6) + (resultCount * 0.3) + (queryLength * 0.1);

        return Math.min(Math.max(confidence, 0.1), 0.95);
    }

    /**
     * Calculate average relevance score
     */
    calculateAverageScore(results) {
        if (results.length === 0) return 0;

        const totalScore = results.reduce((sum, result) => sum + result.score, 0);
        return totalScore / results.length;
    }

    /**
     * Index new content to vector database
     */
    async indexContent(content) {
        try {
            const index = this.pinecone.Index(this.indexName);

            const vectors = await Promise.all(
                content.map(async (item) => {
                    const embedding = await this.generateEmbedding(item.text);

                    return {
                        id: item.id,
                        values: embedding,
                        metadata: {
                            title: item.title,
                            text: item.text,
                            source_type: item.type,
                            book: item.book,
                            page: item.page,
                            section: item.section,
                            level: item.level,
                            url: item.url,
                            indexed_at: new Date().toISOString()
                        }
                    };
                })
            );

            // Batch upsert to Pinecone
            const batchSize = 100;
            for (let i = 0; i < vectors.length; i += batchSize) {
                const batch = vectors.slice(i, i + batchSize);
                await index.upsert(batch);
            }

            return {
                success: true,
                indexed: vectors.length,
                message: `Successfully indexed ${vectors.length} items`
            };

        } catch (error) {
            console.error('Content indexing failed:', error);
            throw new Error(`Indexing failed: ${error.message}`);
        }
    }

    /**
     * Update vector database with new content
     */
    async updateContent(updates) {
        try {
            const index = this.pinecone.Index(this.indexName);

            for (const update of updates) {
                if (update.operation === 'delete') {
                    await index.deleteOne(update.id);
                } else if (update.operation === 'update') {
                    const embedding = await this.generateEmbedding(update.text);
                    await index.upsert([{
                        id: update.id,
                        values: embedding,
                        metadata: update.metadata
                    }]);
                }
            }

            return { success: true, updates: updates.length };

        } catch (error) {
            console.error('Content update failed:', error);
            throw error;
        }
    }

    /**
     * Get database statistics
     */
    async getStats() {
        try {
            const index = this.pinecone.Index(this.indexName);
            const stats = await index.describeIndexStats();

            return {
                totalVectors: stats.totalVectorCount,
                dimension: stats.dimension,
                namespaces: stats.namespaces
            };

        } catch (error) {
            console.error('Failed to get stats:', error);
            return null;
        }
    }
}

module.exports = { VectorRAGEngine };