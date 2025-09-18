/**
 * Cost Optimizer for Pathfinder 2e RAG System
 * Manages API costs and optimizes for <$0.015 per session target
 */

class CostOptimizer {
    constructor(sessionLimit = 0.015) {
        this.sessionLimit = sessionLimit; // $0.015 target
        this.currentSessionCost = 0;
        this.totalCost = 0;
        this.queryCount = 0;
        this.sessionStart = Date.now();

        // API pricing (per 1K tokens)
        this.pricing = {
            'gpt-4': { input: 0.03, output: 0.06 },
            'gpt-3.5-turbo': { input: 0.0015, output: 0.002 },
            'text-embedding-ada-002': { input: 0.0001, output: 0 },
            'gemini-pro': { input: 0.00125, output: 0.00375 },
            'gemini-flash': { input: 0.000075, output: 0.0003 },
            'claude-haiku': { input: 0.00025, output: 0.00125 },
            'claude-sonnet': { input: 0.003, output: 0.015 }
        };

        // Cost tracking
        this.costs = {
            embedding: 0,
            graphQuery: 0,
            vectorQuery: 0,
            answerGeneration: 0,
            total: 0
        };

        this.optimizations = {
            cacheEnabled: true,
            preferCheaperModels: true,
            maxContextLength: 4000,
            maxAnswerLength: 600,
            batchEmbeddings: true
        };
    }

    /**
     * Track cost for an API call
     */
    trackCost(operation, model, inputTokens, outputTokens = 0) {
        const pricing = this.pricing[model];
        if (!pricing) {
            console.warn(`No pricing data for model: ${model}`);
            return 0;
        }

        const inputCost = (inputTokens / 1000) * pricing.input;
        const outputCost = (outputTokens / 1000) * pricing.output;
        const totalCost = inputCost + outputCost;

        // Track by operation type
        this.costs[operation] = (this.costs[operation] || 0) + totalCost;
        this.costs.total += totalCost;
        this.currentSessionCost += totalCost;
        this.totalCost += totalCost;

        this.queryCount++;

        console.log(`Cost: ${operation} (${model}) - $${totalCost.toFixed(6)} (Input: ${inputTokens}, Output: ${outputTokens})`);

        return totalCost;
    }

    /**
     * Get optimal model selection based on cost constraints
     */
    getOptimalModel(operation, complexity = 'medium') {
        const remainingBudget = this.sessionLimit - this.currentSessionCost;

        if (remainingBudget < 0.001) {
            // Very low budget - use cheapest options
            return this.getCheapestModel(operation);
        }

        if (remainingBudget < 0.005) {
            // Limited budget - prefer cost-effective models
            return this.getCostEffectiveModel(operation, complexity);
        }

        // Sufficient budget - use optimal models
        return this.getOptimalQualityModel(operation, complexity);
    }

    /**
     * Get cheapest model for operation
     */
    getCheapestModel(operation) {
        switch (operation) {
            case 'embedding':
                return 'text-embedding-ada-002';
            case 'graphQuery':
                return 'gemini-flash';
            case 'vectorQuery':
                return 'gemini-flash';
            case 'answerGeneration':
                return 'gemini-flash';
            default:
                return 'gemini-flash';
        }
    }

    /**
     * Get cost-effective model balancing quality and cost
     */
    getCostEffectiveModel(operation, complexity) {
        switch (operation) {
            case 'embedding':
                return 'text-embedding-ada-002';
            case 'graphQuery':
                return complexity === 'complex' ? 'gpt-3.5-turbo' : 'gemini-flash';
            case 'vectorQuery':
                return 'gemini-flash';
            case 'answerGeneration':
                return complexity === 'complex' ? 'claude-haiku' : 'gemini-flash';
            default:
                return 'gemini-flash';
        }
    }

    /**
     * Get optimal quality model regardless of cost
     */
    getOptimalQualityModel(operation, complexity) {
        switch (operation) {
            case 'embedding':
                return 'text-embedding-ada-002';
            case 'graphQuery':
                return complexity === 'complex' ? 'gpt-4' : 'gpt-3.5-turbo';
            case 'vectorQuery':
                return 'gpt-3.5-turbo';
            case 'answerGeneration':
                return complexity === 'complex' ? 'claude-sonnet' : 'gpt-4';
            default:
                return 'gpt-3.5-turbo';
        }
    }

    /**
     * Check if query should proceed given cost constraints
     */
    shouldProceedWithQuery() {
        if (this.currentSessionCost >= this.sessionLimit) {
            console.warn(`Session cost limit reached: $${this.currentSessionCost.toFixed(6)}`);
            return false;
        }

        const averageCostPerQuery = this.currentSessionCost / Math.max(this.queryCount, 1);
        const projectedCost = this.currentSessionCost + averageCostPerQuery;

        if (projectedCost > this.sessionLimit * 1.1) { // 10% buffer
            console.warn(`Projected cost exceeds limit: $${projectedCost.toFixed(6)}`);
            return false;
        }

        return true;
    }

    /**
     * Optimize context length based on remaining budget
     */
    optimizeContextLength(originalLength) {
        const remainingBudget = this.sessionLimit - this.currentSessionCost;
        const tokenCostEstimate = 0.00003; // Rough estimate per token

        const maxAffordableTokens = remainingBudget / tokenCostEstimate;
        const optimizedLength = Math.min(originalLength, maxAffordableTokens, this.optimizations.maxContextLength);

        return Math.max(optimizedLength, 500); // Minimum context for quality
    }

    /**
     * Get cost-optimized query strategy
     */
    getQueryStrategy(queryClassification) {
        const remainingBudget = this.sessionLimit - this.currentSessionCost;

        if (remainingBudget < 0.002) {
            // Emergency mode - cache only
            return {
                approach: 'cache_only',
                fallback: 'vector_minimal',
                maxTokens: 200,
                skipEnrichment: true
            };
        }

        if (remainingBudget < 0.005) {
            // Budget mode - prefer vector, minimal graph
            return {
                approach: queryClassification.complexity === 'complex' ? 'hybrid_minimal' : 'vector_only',
                fallback: 'vector_only',
                maxTokens: 400,
                skipEnrichment: false
            };
        }

        // Normal mode - full capabilities
        return {
            approach: this.getOptimalApproach(queryClassification),
            fallback: 'vector_fallback',
            maxTokens: 800,
            skipEnrichment: false
        };
    }

    /**
     * Get optimal approach based on query classification
     */
    getOptimalApproach(classification) {
        if (classification.requiresMultiHop || classification.complexity === 'complex') {
            return 'graph_primary';
        }

        if (classification.type === 'rule_lookup' || classification.complexity === 'simple') {
            return 'vector_primary';
        }

        return 'hybrid_balanced';
    }

    /**
     * Estimate cost for a planned operation
     */
    estimateCost(operation, model, estimatedTokens) {
        const pricing = this.pricing[model];
        if (!pricing) return 0;

        // Rough estimate assuming 70% input, 30% output
        const inputTokens = Math.floor(estimatedTokens * 0.7);
        const outputTokens = Math.floor(estimatedTokens * 0.3);

        const inputCost = (inputTokens / 1000) * pricing.input;
        const outputCost = (outputTokens / 1000) * pricing.output;

        return inputCost + outputCost;
    }

    /**
     * Get recommendations for cost optimization
     */
    getOptimizationRecommendations() {
        const recommendations = [];
        const costPerQuery = this.currentSessionCost / Math.max(this.queryCount, 1);

        if (costPerQuery > 0.003) {
            recommendations.push({
                type: 'model_optimization',
                message: 'Consider using more cost-effective models',
                impact: 'high',
                suggestion: 'Switch to Gemini Flash for simple queries'
            });
        }

        if (this.costs.answerGeneration > this.costs.total * 0.6) {
            recommendations.push({
                type: 'context_optimization',
                message: 'Answer generation is the largest cost component',
                impact: 'medium',
                suggestion: 'Reduce context length and answer complexity'
            });
        }

        if (this.costs.embedding > this.costs.total * 0.3) {
            recommendations.push({
                type: 'embedding_optimization',
                message: 'Embedding costs are high',
                impact: 'medium',
                suggestion: 'Enable embedding caching and batching'
            });
        }

        if (!this.optimizations.cacheEnabled) {
            recommendations.push({
                type: 'cache_optimization',
                message: 'Caching is disabled',
                impact: 'high',
                suggestion: 'Enable caching for repeated queries'
            });
        }

        return recommendations;
    }

    /**
     * Get session cost summary
     */
    getSessionSummary() {
        const sessionDuration = (Date.now() - this.sessionStart) / 1000 / 60; // minutes

        return {
            costs: {
                current: this.currentSessionCost,
                limit: this.sessionLimit,
                remaining: Math.max(0, this.sessionLimit - this.currentSessionCost),
                breakdown: { ...this.costs }
            },
            usage: {
                queries: this.queryCount,
                avgCostPerQuery: this.currentSessionCost / Math.max(this.queryCount, 1),
                sessionDuration: sessionDuration,
                costEfficiency: this.currentSessionCost < this.sessionLimit
            },
            projections: {
                budgetUtilization: (this.currentSessionCost / this.sessionLimit) * 100,
                estimatedRemainingQueries: this.estimateRemainingQueries(),
                targetAchieved: this.currentSessionCost <= this.sessionLimit
            }
        };
    }

    /**
     * Estimate how many more queries can fit in budget
     */
    estimateRemainingQueries() {
        if (this.queryCount === 0) return 10; // Default estimate

        const avgCost = this.currentSessionCost / this.queryCount;
        const remainingBudget = this.sessionLimit - this.currentSessionCost;

        return Math.floor(remainingBudget / avgCost);
    }

    /**
     * Reset session cost tracking
     */
    resetSession() {
        this.currentSessionCost = 0;
        this.queryCount = 0;
        this.sessionStart = Date.now();
        this.costs = {
            embedding: 0,
            graphQuery: 0,
            vectorQuery: 0,
            answerGeneration: 0,
            total: 0
        };
    }

    /**
     * Get last query cost
     */
    getLastQueryCost() {
        return this.queryCount > 0 ? this.currentSessionCost / this.queryCount : 0;
    }

    /**
     * Enable/disable optimizations
     */
    setOptimization(key, value) {
        if (key in this.optimizations) {
            this.optimizations[key] = value;
        }
    }

    /**
     * Get cost breakdown by operation
     */
    getCostBreakdown() {
        return {
            ...this.costs,
            percentages: {
                embedding: (this.costs.embedding / this.costs.total) * 100,
                graphQuery: (this.costs.graphQuery / this.costs.total) * 100,
                vectorQuery: (this.costs.vectorQuery / this.costs.total) * 100,
                answerGeneration: (this.costs.answerGeneration / this.costs.total) * 100
            }
        };
    }
}

module.exports = { CostOptimizer };