"use strict";
/**
 * Kelly Criterion Engine
 * Phase 2 Integration - Optimal position sizing and risk management
 * Advanced Kelly implementation with fractional Kelly and ensemble methods
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.KellyCriterionEngine = void 0;
const events_1 = require("events");
const perf_hooks_1 = require("perf_hooks");
/**
 * Advanced Kelly Criterion Engine with ensemble methods
 */
class KellyCriterionEngine extends events_1.EventEmitter {
    constructor() {
        super();
        this.kellyCalculations = new Map();
        this.marketOpportunities = [];
        this.isRunning = false;
        this.updateInterval = null;
        // Configuration
        this.config = {
            fractionalKellyRatio: 0.25, // Use 25% of full Kelly
            maxSinglePosition: 0.15, // Maximum 15% in single position
            minSampleSize: 30, // Minimum observations for calculation
            confidenceThreshold: 0.6, // Minimum confidence to act
            rebalanceThreshold: 0.05, // 5% drift triggers rebalance
            lookbackPeriod: 252, // Trading days to analyze
            ensembleMethods: ['historical', 'montecarlo', 'bayesian'] // Calculation methods
        };
        this.currentPortfolio = this.initializePortfolio();
        this.kellyMetrics = this.initializeMetrics();
        console.log(' Kelly Criterion Engine initialized - optimizing position sizes...');
    }
    /**
     * Start the Kelly Engine
     */
    start() {
        if (this.isRunning) {
            console.log(' Kelly Criterion Engine already running');
            return;
        }
        this.isRunning = true;
        // Start real-time Kelly calculations
        this.updateInterval = setInterval(() => {
            this.updateKellyAnalysis();
        }, 2000); // Update every 2 seconds
        console.log(' Kelly Criterion Engine started - calculating optimal positions...');
        this.emit('started');
    }
    /**
     * Stop the Kelly Engine
     */
    stop() {
        if (!this.isRunning)
            return;
        this.isRunning = false;
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        console.log(' Kelly Criterion Engine stopped');
        this.emit('stopped');
    }
    /**
     * Initialize empty portfolio
     */
    initializePortfolio() {
        return {
            totalKellyPercent: 0,
            diversificationFactor: 1.0,
            adjustedKellyPercent: 0,
            portfolioVolatility: 0,
            sharpeRatio: 0,
            maxDrawdown: 0,
            positions: [],
            rebalanceRequired: false,
            confidenceScore: 0,
            timestamp: Date.now()
        };
    }
    /**
     * Initialize Kelly metrics
     */
    initializeMetrics() {
        return {
            totalPositions: 0,
            averageKelly: 0,
            maxSinglePosition: 0,
            diversificationBenefit: 0,
            riskBudgetUtilization: 0,
            expectedReturn: 0,
            expectedVolatility: 0,
            informationRatio: 0,
            calmarRatio: 0,
            timestamp: Date.now()
        };
    }
    /**
     * Update Kelly analysis and calculations
     */
    updateKellyAnalysis() {
        const startTime = perf_hooks_1.performance.now();
        try {
            // Generate or update market data for analysis
            this.generateMarketData();
            // Calculate Kelly percentages for each asset
            this.calculateKellyPositions();
            // Identify market opportunities
            this.identifyMarketOpportunities();
            // Update portfolio construction
            this.constructKellyPortfolio();
            // Update metrics
            this.updateKellyMetrics();
            // Emit updates
            this.emit('kellyUpdate', {
                portfolio: this.currentPortfolio,
                metrics: this.kellyMetrics,
                opportunities: this.marketOpportunities.slice(-5),
                processingTime: perf_hooks_1.performance.now() - startTime
            });
        }
        catch (error) {
            console.error(' Kelly Engine analysis error:', error);
            this.emit('error', error);
        }
    }
    /**
     * Generate synthetic market data for Kelly calculations
     */
    generateMarketData() {
        const assets = ['SPY', 'QQQ', 'BTC', 'MSFT', 'GOOGL', 'TSLA', 'GLD', 'TLT'];
        assets.forEach(asset => {
            if (!this.kellyCalculations.has(asset)) {
                // Initialize with historical simulation
                const winRate = 0.45 + Math.random() * 0.20; // 45-65% win rate
                const averageWin = 0.02 + Math.random() * 0.08; // 2-10% average win
                const averageLoss = 0.01 + Math.random() * 0.04; // 1-5% average loss
                const sampleSize = Math.floor(100 + Math.random() * 200); // 100-300 observations
                const kellyPercent = this.calculateKelly(winRate, averageWin, averageLoss);
                const fractionalKelly = kellyPercent * this.config.fractionalKellyRatio;
                this.kellyCalculations.set(asset, {
                    asset,
                    winRate,
                    averageWin,
                    averageLoss,
                    kellyPercent,
                    fractionalKelly,
                    expectedValue: (winRate * averageWin) - ((1 - winRate) * averageLoss),
                    maxDrawdown: this.calculateMaxDrawdown(averageWin, averageLoss, winRate),
                    confidence: Math.min(1.0, sampleSize / 100),
                    sampleSize,
                    timestamp: Date.now()
                });
            }
            else {
                // Update existing calculation with new data
                const calc = this.kellyCalculations.get(asset);
                // Simulate parameter drift
                calc.winRate += (Math.random() - 0.5) * 0.01; // 1% drift
                calc.winRate = Math.max(0.1, Math.min(0.9, calc.winRate));
                calc.averageWin += (Math.random() - 0.5) * 0.002; // 0.2% drift
                calc.averageWin = Math.max(0.005, calc.averageWin);
                calc.averageLoss += (Math.random() - 0.5) * 0.001; // 0.1% drift
                calc.averageLoss = Math.max(0.002, calc.averageLoss);
                // Recalculate Kelly
                calc.kellyPercent = this.calculateKelly(calc.winRate, calc.averageWin, calc.averageLoss);
                calc.fractionalKelly = calc.kellyPercent * this.config.fractionalKellyRatio;
                calc.expectedValue = (calc.winRate * calc.averageWin) - ((1 - calc.winRate) * calc.averageLoss);
                calc.timestamp = Date.now();
                // Improve confidence with more samples
                calc.sampleSize += Math.floor(Math.random() * 3);
                calc.confidence = Math.min(1.0, calc.sampleSize / 100);
            }
        });
    }
    /**
     * Calculate optimal Kelly percentage
     */
    calculateKelly(winRate, averageWin, averageLoss) {
        if (averageLoss <= 0 || winRate <= 0 || winRate >= 1)
            return 0;
        // Kelly formula: f* = (bp - q) / b
        // Where b = odds (averageWin/averageLoss), p = win rate, q = loss rate
        const b = averageWin / averageLoss;
        const p = winRate;
        const q = 1 - winRate;
        const kelly = (b * p - q) / b;
        // Cap Kelly at reasonable levels
        return Math.max(0, Math.min(0.5, kelly));
    }
    /**
     * Calculate expected maximum drawdown
     */
    calculateMaxDrawdown(averageWin, averageLoss, winRate) {
        // Simplified drawdown calculation based on Kelly theory
        const kelly = this.calculateKelly(winRate, averageWin, averageLoss);
        if (kelly <= 0)
            return 0;
        // Expected max drawdown approximation
        const expectedDrawdown = kelly * averageLoss * Math.sqrt(1 / winRate - 1);
        return Math.min(0.5, expectedDrawdown);
    }
    /**
     * Calculate Kelly positions for all assets
     */
    calculateKellyPositions() {
        const positions = [];
        for (const [asset, calc] of this.kellyCalculations.entries()) {
            if (calc.confidence < this.config.confidenceThreshold)
                continue;
            const currentWeight = Math.random() * 0.1; // Simulate current position
            const kellyWeight = Math.min(calc.kellyPercent, this.config.maxSinglePosition);
            const fractionalWeight = Math.min(calc.fractionalKelly, this.config.maxSinglePosition);
            let recommendedAction = 'MAINTAIN';
            let urgency = 'LOW';
            const reasoning = [];
            const weightDiff = fractionalWeight - currentWeight;
            if (Math.abs(weightDiff) > this.config.rebalanceThreshold) {
                if (weightDiff > 0) {
                    recommendedAction = 'INCREASE';
                    reasoning.push(`Kelly suggests ${(fractionalWeight * 100).toFixed(1)}% allocation`);
                    reasoning.push(`Currently underweight by ${(Math.abs(weightDiff) * 100).toFixed(1)}%`);
                }
                else {
                    recommendedAction = 'DECREASE';
                    reasoning.push(`Current allocation exceeds Kelly optimal`);
                    reasoning.push(`Overweight by ${(Math.abs(weightDiff) * 100).toFixed(1)}%`);
                }
                if (Math.abs(weightDiff) > 0.10) {
                    urgency = 'CRITICAL';
                }
                else if (Math.abs(weightDiff) > 0.05) {
                    urgency = 'HIGH';
                }
                else {
                    urgency = 'MEDIUM';
                }
            }
            if (calc.kellyPercent <= 0 || calc.expectedValue <= 0) {
                recommendedAction = 'EXIT';
                urgency = 'HIGH';
                reasoning.push('Negative expected value - no edge detected');
            }
            positions.push({
                asset,
                currentWeight,
                kellyWeight,
                fractionalWeight,
                recommendedAction,
                urgency,
                expectedReturn: calc.expectedValue,
                risk: calc.maxDrawdown,
                kelly: calc.kellyPercent,
                reasoning
            });
        }
        this.currentPortfolio.positions = positions;
    }
    /**
     * Identify market opportunities using Kelly framework
     */
    identifyMarketOpportunities() {
        this.marketOpportunities = [];
        // Generate opportunities based on Kelly calculations
        for (const [asset, calc] of this.kellyCalculations.entries()) {
            if (calc.kellyPercent > 0.1 && calc.confidence > 0.7) {
                let opportunityType = 'MOMENTUM';
                // Classify opportunity type based on parameters
                if (calc.winRate > 0.6 && calc.averageWin > calc.averageLoss * 2) {
                    opportunityType = 'MOMENTUM';
                }
                else if (calc.winRate < 0.5 && calc.averageWin > calc.averageLoss * 3) {
                    opportunityType = 'MEAN_REVERSION';
                }
                else if (calc.averageWin / calc.averageLoss > 3) {
                    opportunityType = 'VOLATILITY';
                }
                this.marketOpportunities.push({
                    asset,
                    opportunity: opportunityType,
                    expectedEdge: calc.expectedValue,
                    confidence: calc.confidence,
                    timeframe: '1D',
                    kellySize: calc.fractionalKelly,
                    reasoning: `Win rate: ${(calc.winRate * 100).toFixed(1)}%, Avg Win: ${(calc.averageWin * 100).toFixed(2)}%`,
                    riskFactors: [
                        `Max drawdown: ${(calc.maxDrawdown * 100).toFixed(1)}%`,
                        `Sample size: ${calc.sampleSize} observations`
                    ],
                    timestamp: Date.now()
                });
            }
        }
        // Limit to top opportunities
        this.marketOpportunities = this.marketOpportunities
            .sort((a, b) => (b.expectedEdge * b.confidence) - (a.expectedEdge * a.confidence))
            .slice(0, 10);
    }
    /**
     * Construct Kelly portfolio with diversification
     */
    constructKellyPortfolio() {
        const validPositions = this.currentPortfolio.positions.filter(p => p.kelly > 0);
        if (validPositions.length === 0) {
            this.currentPortfolio.totalKellyPercent = 0;
            this.currentPortfolio.adjustedKellyPercent = 0;
            return;
        }
        // Calculate total Kelly allocation
        const totalKelly = validPositions.reduce((sum, p) => sum + p.kellyWeight, 0);
        // Apply diversification factor (reduces total allocation due to correlation)
        const diversificationFactor = Math.max(0.5, 1 - (validPositions.length - 1) * 0.05);
        const adjustedKelly = Math.min(1.0, totalKelly * diversificationFactor);
        // Calculate portfolio risk metrics
        const avgReturn = validPositions.reduce((sum, p) => sum + p.expectedReturn * p.fractionalWeight, 0);
        const avgVolatility = validPositions.reduce((sum, p) => sum + p.risk * p.fractionalWeight, 0);
        const sharpeRatio = avgVolatility > 0 ? avgReturn / avgVolatility : 0;
        const maxDrawdown = validPositions.reduce((max, p) => Math.max(max, p.risk), 0);
        // Check if rebalancing is required
        const rebalanceRequired = validPositions.some(p => p.recommendedAction !== 'MAINTAIN' && p.urgency !== 'LOW');
        // Calculate confidence score
        const confidenceScore = validPositions.length > 0
            ? validPositions.reduce((sum, p) => {
                const calc = this.kellyCalculations.get(p.asset);
                return sum + (calc?.confidence || 0);
            }, 0) / validPositions.length
            : 0;
        this.currentPortfolio = {
            ...this.currentPortfolio,
            totalKellyPercent: totalKelly,
            diversificationFactor,
            adjustedKellyPercent: adjustedKelly,
            portfolioVolatility: avgVolatility,
            sharpeRatio,
            maxDrawdown,
            rebalanceRequired,
            confidenceScore,
            timestamp: Date.now()
        };
    }
    /**
     * Update Kelly metrics
     */
    updateKellyMetrics() {
        const validPositions = this.currentPortfolio.positions.filter(p => p.kelly > 0);
        if (validPositions.length === 0) {
            this.kellyMetrics = this.initializeMetrics();
            return;
        }
        const averageKelly = validPositions.reduce((sum, p) => sum + p.kelly, 0) / validPositions.length;
        const maxSinglePosition = Math.max(...validPositions.map(p => p.fractionalWeight));
        const diversificationBenefit = 1 - this.currentPortfolio.diversificationFactor;
        const riskBudgetUtilization = this.currentPortfolio.adjustedKellyPercent;
        const expectedReturn = validPositions.reduce((sum, p) => sum + p.expectedReturn * p.fractionalWeight, 0);
        const expectedVolatility = this.currentPortfolio.portfolioVolatility;
        const informationRatio = expectedVolatility > 0 ? expectedReturn / expectedVolatility : 0;
        const calmarRatio = this.currentPortfolio.maxDrawdown > 0
            ? expectedReturn / this.currentPortfolio.maxDrawdown : 0;
        this.kellyMetrics = {
            totalPositions: validPositions.length,
            averageKelly,
            maxSinglePosition,
            diversificationBenefit,
            riskBudgetUtilization,
            expectedReturn,
            expectedVolatility,
            informationRatio,
            calmarRatio,
            timestamp: Date.now()
        };
    }
    // Public getters
    getKellyCalculations() {
        return new Map(this.kellyCalculations);
    }
    getCurrentPortfolio() {
        return { ...this.currentPortfolio };
    }
    getKellyMetrics() {
        return { ...this.kellyMetrics };
    }
    getMarketOpportunities() {
        return [...this.marketOpportunities];
    }
    getTopPositions(count = 5) {
        return this.currentPortfolio.positions
            .filter(p => p.kelly > 0)
            .sort((a, b) => b.fractionalWeight - a.fractionalWeight)
            .slice(0, count);
    }
    getRebalanceRecommendations() {
        return this.currentPortfolio.positions.filter(p => p.recommendedAction !== 'MAINTAIN' && p.urgency !== 'LOW');
    }
    isRunning() {
        return this.isRunning;
    }
    /**
     * Calculate Kelly for a specific trade setup
     */
    calculateTradeKelly(winRate, averageWin, averageLoss) {
        const kelly = this.calculateKelly(winRate, averageWin, averageLoss);
        const fractionalKelly = kelly * this.config.fractionalKellyRatio;
        return {
            asset: 'Custom Trade',
            winRate,
            averageWin,
            averageLoss,
            kellyPercent: kelly,
            fractionalKelly,
            expectedValue: (winRate * averageWin) - ((1 - winRate) * averageLoss),
            maxDrawdown: this.calculateMaxDrawdown(averageWin, averageLoss, winRate),
            confidence: 0.5, // Unknown for custom trade
            sampleSize: 0,
            timestamp: Date.now()
        };
    }
    /**
     * Get Kelly insights and recommendations
     */
    getKellyInsights() {
        const insights = {
            totalEdge: this.kellyMetrics.expectedReturn,
            riskUtilization: this.kellyMetrics.riskBudgetUtilization,
            recommendations: [],
            warnings: []
        };
        if (this.currentPortfolio.adjustedKellyPercent > 0.8) {
            insights.warnings.push('High Kelly allocation - consider fractional Kelly for safety');
        }
        if (this.kellyMetrics.maxSinglePosition > this.config.maxSinglePosition) {
            insights.warnings.push(`Single position exceeds ${(this.config.maxSinglePosition * 100).toFixed(0)}% limit`);
        }
        if (this.currentPortfolio.confidenceScore < 0.6) {
            insights.warnings.push('Low confidence in Kelly calculations - need more data');
        }
        if (this.kellyMetrics.diversificationBenefit < 0.1) {
            insights.recommendations.push('Add more uncorrelated positions for diversification benefit');
        }
        if (this.currentPortfolio.sharpeRatio > 2.0) {
            insights.recommendations.push('Excellent risk-adjusted returns - consider scaling up');
        }
        if (this.kellyMetrics.calmarRatio > 1.0) {
            insights.recommendations.push('Strong Calmar ratio - good drawdown management');
        }
        return insights;
    }
}
exports.KellyCriterionEngine = KellyCriterionEngine;
exports.default = KellyCriterionEngine;
//# sourceMappingURL=KellyCriterionEngine.js.map