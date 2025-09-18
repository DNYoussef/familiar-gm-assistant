/**
 * Vector Search Service
 * Provides semantic search capabilities for Pathfinder content
 * Simplified implementation without external vector database dependencies
 */

class VectorSearchService {
    constructor(options = {}) {
        this.embeddings = new Map();
        this.documents = new Map();
        this.stats = {
            documentsIndexed: 0,
            queriesProcessed: 0,
            averageQueryTime: 0
        };
    }

    /**
     * Initialize vector search service
     */
    async initialize() {
        console.log('Vector Search Service | Initialized with simplified text matching');
        await this.indexDefaultDocuments();
    }

    /**
     * Index default Pathfinder documents
     */
    async indexDefaultDocuments() {
        const defaultDocs = [
            {
                id: 'pf2e-combat-basics',
                title: 'Combat Basics',
                content: 'Combat in Pathfinder 2e uses a three-action system. Each turn you get three actions to spend on strikes, movement, spells, or other activities. Initiative determines turn order using Perception checks.',
                category: 'combat',
                keywords: ['combat', 'actions', 'initiative', 'turns', 'strikes']
            },
            {
                id: 'pf2e-spellcasting',
                title: 'Spellcasting Rules',
                content: 'Spellcasting requires spell slots and meeting component requirements. Most spells take 2 actions to cast. Spell attack rolls use your spellcasting ability modifier plus proficiency.',
                category: 'spells',
                keywords: ['spellcasting', 'spell slots', 'components', 'spell attack', 'proficiency']
            },
            {
                id: 'pf2e-conditions',
                title: 'Common Conditions',
                content: 'Conditions affect characters in various ways. Frightened reduces all checks and DCs. Prone gives -2 to attack rolls but +2 AC against ranged attacks. Unconscious makes you helpless.',
                category: 'conditions',
                keywords: ['conditions', 'frightened', 'prone', 'unconscious', 'penalties', 'bonuses']
            },
            {
                id: 'pf2e-skills',
                title: 'Skill System',
                content: 'Skills use the same proficiency system as attacks and saves. Skill checks are d20 + ability modifier + proficiency bonus + item bonuses + circumstance bonuses.',
                category: 'skills',
                keywords: ['skills', 'proficiency', 'ability modifier', 'bonuses', 'circumstance']
            },
            {
                id: 'pf2e-critical-hits',
                title: 'Critical Success and Failure',
                content: 'Critical success occurs when you roll a natural 20 or exceed the DC by 10 or more. Critical failure happens on a natural 1 or when you fail the DC by 10 or more.',
                category: 'mechanics',
                keywords: ['critical', 'success', 'failure', 'natural 20', 'natural 1', 'DC']
            }
        ];

        for (const doc of defaultDocs) {
            await this.indexDocument(doc);
        }

        console.log(`Vector Search Service | Indexed ${this.stats.documentsIndexed} default documents`);
    }

    /**
     * Index a document for search
     */
    async indexDocument(document) {
        const docId = document.id;

        // Store document
        this.documents.set(docId, {
            ...document,
            indexed: Date.now()
        });

        // Create simple text-based "embedding" (just processed keywords and content)
        const embedding = this.createTextEmbedding(document);
        this.embeddings.set(docId, embedding);

        this.stats.documentsIndexed++;
    }

    /**
     * Create text-based embedding from document
     */
    createTextEmbedding(document) {
        const allText = [
            document.title,
            document.content,
            ...(document.keywords || [])
        ].join(' ').toLowerCase();

        // Extract meaningful terms
        const terms = allText
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(term => term.length > 2)
            .filter(term => !this.isStopWord(term));

        // Count term frequencies
        const termFrequency = new Map();
        terms.forEach(term => {
            termFrequency.set(term, (termFrequency.get(term) || 0) + 1);
        });

        return {
            terms: Array.from(termFrequency.keys()),
            frequencies: termFrequency,
            totalTerms: terms.length,
            uniqueTerms: termFrequency.size
        };
    }

    /**
     * Check if word is a stop word
     */
    isStopWord(word) {
        const stopWords = new Set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        ]);
        return stopWords.has(word);
    }

    /**
     * Find similar documents to a query
     */
    async findSimilar(query, limit = 5) {
        const startTime = Date.now();

        try {
            this.stats.queriesProcessed++;

            // Create query embedding
            const queryEmbedding = this.createTextEmbedding({
                title: query,
                content: query,
                keywords: []
            });

            // Calculate similarities
            const similarities = [];
            for (const [docId, docEmbedding] of this.embeddings) {
                const similarity = this.calculateSimilarity(queryEmbedding, docEmbedding);
                if (similarity > 0) {
                    similarities.push({
                        document: this.documents.get(docId),
                        similarity,
                        docId
                    });
                }
            }

            // Sort by similarity and limit results
            const results = similarities
                .sort((a, b) => b.similarity - a.similarity)
                .slice(0, limit);

            // Update stats
            const queryTime = Date.now() - startTime;
            this.updateQueryTimeStats(queryTime);

            return results;

        } catch (error) {
            console.error('Vector Search Service | Error finding similar documents:', error);
            return [];
        }
    }

    /**
     * Calculate similarity between two embeddings using Jaccard similarity
     */
    calculateSimilarity(embedding1, embedding2) {
        const terms1 = new Set(embedding1.terms);
        const terms2 = new Set(embedding2.terms);

        const intersection = new Set([...terms1].filter(term => terms2.has(term)));
        const union = new Set([...terms1, ...terms2]);

        if (union.size === 0) return 0;

        // Jaccard similarity
        const jaccardSimilarity = intersection.size / union.size;

        // Boost similarity based on term frequency overlap
        let frequencyBoost = 0;
        for (const term of intersection) {
            const freq1 = embedding1.frequencies.get(term) || 0;
            const freq2 = embedding2.frequencies.get(term) || 0;
            frequencyBoost += Math.min(freq1, freq2) / Math.max(freq1, freq2);
        }

        frequencyBoost = frequencyBoost / Math.max(intersection.size, 1);

        // Combine Jaccard similarity with frequency boost
        return (jaccardSimilarity * 0.7) + (frequencyBoost * 0.3);
    }

    /**
     * Search documents by category
     */
    async searchByCategory(category, query = null, limit = 5) {
        const categoryDocs = [];

        for (const [docId, doc] of this.documents) {
            if (doc.category === category) {
                if (query) {
                    // If query provided, calculate relevance
                    const queryEmbedding = this.createTextEmbedding({
                        title: query,
                        content: query,
                        keywords: []
                    });
                    const docEmbedding = this.embeddings.get(docId);
                    const similarity = this.calculateSimilarity(queryEmbedding, docEmbedding);

                    categoryDocs.push({
                        document: doc,
                        similarity,
                        docId
                    });
                } else {
                    categoryDocs.push({
                        document: doc,
                        similarity: 1.0,
                        docId
                    });
                }
            }
        }

        return categoryDocs
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit);
    }

    /**
     * Add new content to index
     */
    async addContent(content) {
        const docId = content.id || `doc-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        await this.indexDocument({
            id: docId,
            ...content
        });

        return docId;
    }

    /**
     * Remove content from index
     */
    async removeContent(docId) {
        this.documents.delete(docId);
        this.embeddings.delete(docId);
        this.stats.documentsIndexed = Math.max(0, this.stats.documentsIndexed - 1);
    }

    /**
     * Get document by ID
     */
    getDocument(docId) {
        return this.documents.get(docId);
    }

    /**
     * Update query time statistics
     */
    updateQueryTimeStats(queryTime) {
        const currentAverage = this.stats.averageQueryTime;
        const totalQueries = this.stats.queriesProcessed;

        this.stats.averageQueryTime = currentAverage === 0
            ? queryTime
            : ((currentAverage * (totalQueries - 1)) + queryTime) / totalQueries;
    }

    /**
     * Get all categories
     */
    getCategories() {
        const categories = new Set();
        for (const doc of this.documents.values()) {
            if (doc.category) {
                categories.add(doc.category);
            }
        }
        return Array.from(categories);
    }

    /**
     * Get search suggestions based on indexed content
     */
    getSearchSuggestions(partialQuery = '', limit = 5) {
        const suggestions = new Set();

        // Get common terms from all documents
        const allTerms = new Map();
        for (const embedding of this.embeddings.values()) {
            for (const [term, frequency] of embedding.frequencies) {
                allTerms.set(term, (allTerms.get(term) || 0) + frequency);
            }
        }

        // Filter terms that start with partial query
        const query = partialQuery.toLowerCase();
        for (const [term, frequency] of allTerms) {
            if (term.startsWith(query) && term !== query) {
                suggestions.add(term);
            }
        }

        // Also add document titles that match
        for (const doc of this.documents.values()) {
            if (doc.title.toLowerCase().includes(query)) {
                suggestions.add(doc.title);
            }
        }

        return Array.from(suggestions)
            .sort()
            .slice(0, limit);
    }

    /**
     * Get service statistics
     */
    getStats() {
        return {
            ...this.stats,
            documentsInIndex: this.documents.size,
            embeddingsInMemory: this.embeddings.size,
            categories: this.getCategories().length
        };
    }

    /**
     * Close service and cleanup
     */
    async close() {
        this.documents.clear();
        this.embeddings.clear();
        console.log('Vector Search Service | Closed');
    }
}

module.exports = { VectorSearchService };