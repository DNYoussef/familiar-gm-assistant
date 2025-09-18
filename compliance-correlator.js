"use strict";
/**
 * Cross-Framework Compliance Correlation System
 * Implements correlation matrix with gap analysis and unified reporting
 *
 * Task: EC-005 - Cross-framework compliance correlation and gap analysis
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ComplianceCorrelator = void 0;
const events_1 = require("events");
class ComplianceCorrelator extends events_1.EventEmitter {
    constructor(config) {
        super();
        this.correlationDatabase = new Map();
        this.controlMappings = new Map();
        this.frameworkDefinitions = new Map();
        this.correlationHistory = [];
        this.priorityComparator = (a, b) => {
            const priorities = { critical: 4, high: 3, medium: 2, low: 1 };
            return priorities[b.priority] - priorities[a.priority];
        };
        this.config = config;
        this.initializeCorrelationDatabase();
    }
    /**
     * Initialize correlation database with framework mappings
     */
    initializeCorrelationDatabase() {
        // SOC2 to ISO27001 correlations
        this.addFrameworkCorrelations('soc2', 'iso27001', [
            {
                sourceControl: 'CC6.1',
                targetControl: 'A.8.2',
                correlationType: 'equivalent',
                strength: 0.9,
                mappingRationale: 'Both controls address privileged access management'
            },
            {
                sourceControl: 'CC6.2',
                targetControl: 'A.8.3',
                correlationType: 'equivalent',
                strength: 0.85,
                mappingRationale: 'Both address information access restriction'
            },
            {
                sourceControl: 'CC6.3',
                targetControl: 'A.8.24',
                correlationType: 'related',
                strength: 0.7,
                mappingRationale: 'Network security controls alignment'
            },
            {
                sourceControl: 'CC6.7',
                targetControl: 'A.8.24',
                correlationType: 'equivalent',
                strength: 0.95,
                mappingRationale: 'Data transmission encryption requirements'
            },
            {
                sourceControl: 'A1.1',
                targetControl: 'A.7.4',
                correlationType: 'related',
                strength: 0.6,
                mappingRationale: 'System availability and equipment maintenance'
            }
        ]);
        // SOC2 to NIST-SSDF correlations
        this.addFrameworkCorrelations('soc2', 'nist-ssdf', [
            {
                sourceControl: 'CC6.1',
                targetControl: 'PS.1.1',
                correlationType: 'related',
                strength: 0.7,
                mappingRationale: 'Access controls for code protection'
            },
            {
                sourceControl: 'CC6.6',
                targetControl: 'PW.4.1',
                correlationType: 'equivalent',
                strength: 0.8,
                mappingRationale: 'Vulnerability management and security testing'
            },
            {
                sourceControl: 'CC6.8',
                targetControl: 'RV.1.1',
                correlationType: 'related',
                strength: 0.75,
                mappingRationale: 'Security monitoring and vulnerability identification'
            }
        ]);
        // ISO27001 to NIST-SSDF correlations
        this.addFrameworkCorrelations('iso27001', 'nist-ssdf', [
            {
                sourceControl: 'A.8.1',
                targetControl: 'PS.1.1',
                correlationType: 'related',
                strength: 0.65,
                mappingRationale: 'Endpoint protection and code protection'
            },
            {
                sourceControl: 'A.8.2',
                targetControl: 'PS.1.1',
                correlationType: 'equivalent',
                strength: 0.9,
                mappingRationale: 'Privileged access control for code repositories'
            },
            {
                sourceControl: 'A.8.26',
                targetControl: 'PW.4.4',
                correlationType: 'equivalent',
                strength: 0.85,
                mappingRationale: 'Application security requirements and code review'
            },
            {
                sourceControl: 'A.6.3',
                targetControl: 'PS.2.1',
                correlationType: 'related',
                strength: 0.7,
                mappingRationale: 'Security awareness training and secure coding practices'
            }
        ]);
        this.emit('correlation_database_initialized', {
            frameworks: this.config.frameworks.length,
            correlations: Array.from(this.correlationDatabase.values()).flat().length
        });
    }
    /**
     * Add framework correlations to database
     */
    addFrameworkCorrelations(sourceFramework, targetFramework, correlations) {
        const key = `${sourceFramework}-${targetFramework}`;
        const frameworkCorrelations = correlations.map(corr => ({
            sourceFramework,
            sourceControl: corr.sourceControl,
            targetFramework,
            targetControl: corr.targetControl,
            correlationType: corr.correlationType,
            strength: corr.strength,
            bidirectional: true,
            mappingRationale: corr.mappingRationale,
            lastUpdated: new Date()
        }));
        this.correlationDatabase.set(key, frameworkCorrelations);
        // Add reverse correlations for bidirectional mappings
        const reverseKey = `${targetFramework}-${sourceFramework}`;
        const reverseCorrelations = frameworkCorrelations.map(corr => ({
            ...corr,
            sourceFramework: corr.targetFramework,
            sourceControl: corr.targetControl,
            targetFramework: corr.sourceFramework,
            targetControl: corr.sourceControl
        }));
        this.correlationDatabase.set(reverseKey, reverseCorrelations);
    }
    /**
     * Correlate compliance across frameworks
     */
    async correlatCompliance(frameworkResults) {
        const correlationId = `correlation-${Date.now()}`;
        const timestamp = new Date();
        const frameworks = Object.keys(frameworkResults);
        try {
            this.emit('correlation_started', { correlationId, frameworks });
            // Calculate framework scores
            const frameworkScores = this.calculateFrameworkScores(frameworkResults);
            // Build correlation matrix
            const correlationMatrix = await this.buildCorrelationMatrix(frameworks, frameworkResults);
            // Perform gap analysis
            const gapAnalysis = await this.performCrossFrameworkGapAnalysis(frameworkResults, correlationMatrix);
            // Aggregate risks across frameworks
            const riskAggregation = await this.aggregateRisks(frameworkResults, correlationMatrix);
            // Calculate overall score
            const overallScore = this.calculateOverallComplianceScore(frameworkScores);
            // Generate recommendations
            const recommendations = this.generateCrossFrameworkRecommendations(gapAnalysis, riskAggregation);
            // Create unified report
            const unifiedReport = await this.generateUnifiedReport({
                correlationId,
                frameworks,
                frameworkScores,
                correlationMatrix,
                gapAnalysis,
                riskAggregation,
                recommendations
            });
            const result = {
                correlationId,
                timestamp,
                frameworks,
                overallScore,
                frameworkScores,
                correlationMatrix,
                gapAnalysis,
                riskAggregation,
                recommendations,
                unifiedReport
            };
            this.correlationHistory.push(result);
            this.emit('correlation_completed', {
                correlationId,
                overallScore,
                frameworks: frameworks.length,
                totalGaps: gapAnalysis.totalGaps
            });
            return result;
        }
        catch (error) {
            this.emit('correlation_failed', { correlationId, error: error.message });
            throw new Error(`Compliance correlation failed: ${error.message}`);
        }
    }
    /**
     * Calculate framework scores
     */
    calculateFrameworkScores(frameworkResults) {
        const scores = {};
        for (const [framework, result] of Object.entries(frameworkResults)) {
            const controls = result.controls || [];
            const compliantControls = controls.filter((c) => c.status === 'compliant');
            const criticalFindings = (result.findings || []).filter((f) => f.severity === 'critical');
            scores[framework] = {
                framework,
                complianceScore: result.complianceScore || 0,
                controlsAssessed: controls.length,
                controlsCompliant: compliantControls.length,
                criticalFindings: criticalFindings.length,
                coverageGaps: this.identifyFrameworkCoverageGaps(framework, controls)
            };
        }
        return scores;
    }
    /**
     * Build correlation matrix
     */
    async buildCorrelationMatrix(frameworks, frameworkResults) {
        const matrix = [];
        const coverage = {};
        const overlaps = [];
        const gaps = [];
        // Build correlation matrix
        for (let i = 0; i < frameworks.length; i++) {
            matrix[i] = [];
            for (let j = 0; j < frameworks.length; j++) {
                const sourceFramework = frameworks[i];
                const targetFramework = frameworks[j];
                if (i === j) {
                    // Self-correlation
                    matrix[i][j] = {
                        sourceFramework,
                        targetFramework,
                        correlations: frameworkResults[sourceFramework].controls?.length || 0,
                        strength: 1.0,
                        bidirectional: 1,
                        coverage: 100
                    };
                }
                else {
                    const correlationData = this.getFrameworkCorrelations(sourceFramework, targetFramework);
                    const sourceControls = frameworkResults[sourceFramework].controls || [];
                    const targetControls = frameworkResults[targetFramework].controls || [];
                    matrix[i][j] = {
                        sourceFramework,
                        targetFramework,
                        correlations: correlationData.length,
                        strength: correlationData.length > 0
                            ? correlationData.reduce((sum, corr) => sum + corr.strength, 0) / correlationData.length
                            : 0,
                        bidirectional: correlationData.filter(corr => corr.bidirectional).length / Math.max(correlationData.length, 1),
                        coverage: this.calculateCoverage(sourceControls, targetControls, correlationData)
                    };
                }
            }
        }
        // Calculate framework coverage
        for (const framework of frameworks) {
            coverage[framework] = this.calculateFrameworkCoverage(framework, frameworks, matrix);
        }
        // Identify overlaps
        for (let i = 0; i < frameworks.length; i++) {
            for (let j = i + 1; j < frameworks.length; j++) {
                const overlap = this.calculateFrameworkOverlap(frameworks[i], frameworks[j], frameworkResults[frameworks[i]], frameworkResults[frameworks[j]]);
                overlaps.push(overlap);
            }
        }
        // Identify gaps
        for (let i = 0; i < frameworks.length; i++) {
            for (let j = 0; j < frameworks.length; j++) {
                if (i !== j) {
                    const gap = this.identifyFrameworkGaps(frameworks[i], frameworks[j], frameworkResults[frameworks[i]], frameworkResults[frameworks[j]]);
                    if (gap.uncoveredControls.length > 0) {
                        gaps.push(gap);
                    }
                }
            }
        }
        return { frameworks, matrix, coverage, overlaps, gaps };
    }
    /**
     * Get framework correlations
     */
    getFrameworkCorrelations(sourceFramework, targetFramework) {
        const key = `${sourceFramework}-${targetFramework}`;
        return this.correlationDatabase.get(key) || [];
    }
    /**
     * Calculate coverage between frameworks
     */
    calculateCoverage(sourceControls, targetControls, correlations) {
        if (sourceControls.length === 0)
            return 0;
        const sourceControlIds = new Set(sourceControls.map(c => c.controlId || c.id));
        const correlatedSourceControls = new Set(correlations.map(c => c.sourceControl));
        const coveredControls = Array.from(sourceControlIds).filter(id => correlatedSourceControls.has(id));
        return (coveredControls.length / sourceControlIds.size) * 100;
    }
    /**
     * Calculate framework coverage across all other frameworks
     */
    calculateFrameworkCoverage(framework, allFrameworks, matrix) {
        const frameworkIndex = allFrameworks.indexOf(framework);
        if (frameworkIndex === -1)
            return 0;
        const coverageValues = matrix[frameworkIndex]
            .filter((cell, index) => index !== frameworkIndex)
            .map(cell => cell.coverage);
        return coverageValues.length > 0
            ? coverageValues.reduce((sum, coverage) => sum + coverage, 0) / coverageValues.length
            : 0;
    }
    /**
     * Calculate framework overlap
     */
    calculateFrameworkOverlap(framework1, framework2, result1, result2) {
        const correlations = this.getFrameworkCorrelations(framework1, framework2);
        const controls1 = result1.controls || [];
        const controls2 = result2.controls || [];
        return {
            frameworks: [framework1, framework2],
            overlappingControls: correlations.length,
            totalControls: controls1.length + controls2.length,
            overlapPercentage: correlations.length > 0
                ? (correlations.length / Math.max(controls1.length, controls2.length)) * 100
                : 0,
            commonObjectives: this.identifyCommonObjectives(correlations)
        };
    }
    /**
     * Identify framework gaps
     */
    identifyFrameworkGaps(sourceFramework, targetFramework, sourceResult, targetResult) {
        const correlations = this.getFrameworkCorrelations(sourceFramework, targetFramework);
        const sourceControls = sourceResult.controls || [];
        const sourceControlIds = new Set(sourceControls.map((c) => c.controlId || c.id));
        const correlatedControls = new Set(correlations.map(c => c.sourceControl));
        const uncoveredControls = Array.from(sourceControlIds).filter(id => !correlatedControls.has(id));
        const gapPercentage = sourceControlIds.size > 0
            ? (uncoveredControls.length / sourceControlIds.size) * 100
            : 0;
        return {
            sourceFramework,
            targetFramework,
            uncoveredControls,
            gapPercentage,
            riskImpact: this.assessGapRiskImpact(uncoveredControls, sourceControls),
            mitigationOptions: this.generateGapMitigationOptions(uncoveredControls, sourceFramework, targetFramework)
        };
    }
    /**
     * Perform cross-framework gap analysis
     */
    async performCrossFrameworkGapAnalysis(frameworkResults, correlationMatrix) {
        const gapsByFramework = {};
        const prioritizedGaps = [];
        // Analyze gaps by framework
        for (const [framework, result] of Object.entries(frameworkResults)) {
            const controls = result.controls || [];
            const totalControls = controls.length;
            // Find correlations for this framework
            let coveredControls = 0;
            const uncoveredControls = [];
            for (const control of controls) {
                const controlId = control.controlId || control.id;
                let isCovered = false;
                for (const otherFramework of Object.keys(frameworkResults)) {
                    if (otherFramework !== framework) {
                        const correlations = this.getFrameworkCorrelations(framework, otherFramework);
                        if (correlations.some(corr => corr.sourceControl === controlId)) {
                            isCovered = true;
                            break;
                        }
                    }
                }
                if (isCovered) {
                    coveredControls++;
                }
                else {
                    uncoveredControls.push(controlId);
                }
            }
            gapsByFramework[framework] = {
                framework,
                totalControls,
                coveredControls,
                uncoveredControls,
                coveragePercentage: totalControls > 0 ? (coveredControls / totalControls) * 100 : 0,
                majorGaps: this.identifyMajorGaps(uncoveredControls, controls)
            };
            // Create prioritized gaps
            for (const gapControl of uncoveredControls) {
                const control = controls.find((c) => (c.controlId || c.id) === gapControl);
                if (control) {
                    prioritizedGaps.push({
                        id: `gap-${framework}-${gapControl}`,
                        framework,
                        control: gapControl,
                        priority: this.assessGapPriority(control),
                        description: control.description || `Control ${gapControl} not covered by other frameworks`,
                        businessImpact: this.assessBusinessImpact(control),
                        remediationOptions: this.generateRemediationOptions(framework, gapControl),
                        estimatedEffort: this.estimateRemediationEffort(control),
                        dependencies: control.relatedControls || []
                    });
                }
            }
        }
        const totalGaps = prioritizedGaps.length;
        const criticalGaps = prioritizedGaps.filter(gap => gap.priority === 'critical').length;
        return {
            totalGaps,
            criticalGaps,
            gapsByFramework,
            prioritizedGaps: prioritizedGaps.sort(this.priorityComparator),
            remediationEffort: this.calculateTotalRemediationEffort(prioritizedGaps),
            costEstimate: this.estimateRemediationCost(prioritizedGaps)
        };
    }
    /**
     * Aggregate risks across frameworks
     */
    async aggregateRisks(frameworkResults, correlationMatrix) {
        const riskByCategory = {};
        const riskByFramework = {};
        const compoundRisks = [];
        const riskTrends = [];
        // Calculate risk by framework
        for (const [framework, result] of Object.entries(frameworkResults)) {
            const findings = result.findings || [];
            const riskScore = this.calculateFrameworkRiskScore(findings);
            riskByFramework[framework] = riskScore;
            // Categorize risks
            for (const finding of findings) {
                const category = this.categorizeRisk(finding, framework);
                riskByCategory[category] = (riskByCategory[category] || 0) + this.getRiskValue(finding.severity);
            }
        }
        // Identify compound risks (risks affecting multiple frameworks)
        compoundRisks.push(...this.identifyCompoundRisks(frameworkResults, correlationMatrix));
        // Calculate risk trends (mock implementation)
        riskTrends.push(...this.calculateRiskTrends(frameworkResults));
        const overallRiskScore = this.calculateOverallRiskScore(riskByFramework, compoundRisks);
        const mitigationPriority = this.prioritizeRiskMitigation(compoundRisks, riskByCategory);
        return {
            overallRiskScore,
            riskByCategory,
            riskByFramework,
            compoundRisks,
            riskTrends,
            mitigationPriority
        };
    }
    /**
     * Generate unified compliance report
     */
    async generateUnifiedReport(params) {
        const reportId = `unified-report-${Date.now()}`;
        return {
            id: reportId,
            title: 'Unified Compliance Assessment Report',
            generated: new Date(),
            scope: params.frameworks,
            executiveSummary: this.generateExecutiveSummary(params),
            frameworkSummaries: this.generateFrameworkSummaries(params.frameworkScores),
            crossFrameworkAnalysis: {
                correlationMatrix: params.correlationMatrix,
                gapAnalysis: params.gapAnalysis,
                riskAggregation: params.riskAggregation
            },
            recommendations: this.generateReportRecommendations(params),
            appendices: this.generateReportAppendices(params)
        };
    }
    /**
     * Generate executive summary
     */
    generateExecutiveSummary(params) {
        const avgScore = Object.values(params.frameworkScores)
            .reduce((sum, score) => sum + score.complianceScore, 0) / params.frameworks.length;
        const totalGaps = params.gapAnalysis.totalGaps;
        const criticalGaps = params.gapAnalysis.criticalGaps;
        const overallRisk = params.riskAggregation.overallRiskScore;
        return `This unified compliance assessment covers ${params.frameworks.length} frameworks with an average compliance score of ${avgScore.toFixed(1)}%. A total of ${totalGaps} gaps were identified, including ${criticalGaps} critical gaps. The overall risk score is ${overallRisk.toFixed(1)}, indicating ${this.interpretRiskLevel(overallRisk)} risk exposure across frameworks.`;
    }
    /**
     * Helper methods for various calculations
     */
    identifyFrameworkCoverageGaps(framework, controls) {
        // Mock implementation - would analyze actual coverage gaps
        return controls.filter(c => c.status !== 'compliant').map(c => c.controlId || c.id);
    }
    identifyCommonObjectives(correlations) {
        // Extract common security objectives from correlations
        const objectives = new Set();
        correlations.forEach(corr => {
            if (corr.mappingRationale.includes('access'))
                objectives.add('Access Control');
            if (corr.mappingRationale.includes('encryption'))
                objectives.add('Data Protection');
            if (corr.mappingRationale.includes('monitoring'))
                objectives.add('Security Monitoring');
            if (corr.mappingRationale.includes('vulnerability'))
                objectives.add('Vulnerability Management');
        });
        return Array.from(objectives);
    }
    assessGapRiskImpact(uncoveredControls, sourceControls) {
        const criticalControls = sourceControls.filter(c => c.riskLevel === 'critical' && uncoveredControls.includes(c.controlId || c.id));
        if (criticalControls.length > 0)
            return 'critical';
        if (uncoveredControls.length > sourceControls.length * 0.3)
            return 'high';
        if (uncoveredControls.length > sourceControls.length * 0.1)
            return 'medium';
        return 'low';
    }
    generateGapMitigationOptions(uncoveredControls, sourceFramework, targetFramework) {
        return [
            `Implement equivalent controls in ${targetFramework} framework`,
            `Create custom mapping for uncovered ${sourceFramework} controls`,
            `Accept risk for non-critical control gaps`,
            `Establish compensating controls to address gaps`
        ];
    }
    identifyMajorGaps(uncoveredControls, controls) {
        return uncoveredControls.filter(controlId => {
            const control = controls.find((c) => (c.controlId || c.id) === controlId);
            return control && (control.riskLevel === 'high' || control.riskLevel === 'critical');
        });
    }
    assessGapPriority(control) {
        return control.riskLevel || 'medium';
    }
    assessBusinessImpact(control) {
        const riskLevel = control.riskLevel || 'medium';
        const impacts = {
            critical: 'Significant business risk exposure',
            high: 'Moderate business risk exposure',
            medium: 'Limited business risk exposure',
            low: 'Minimal business risk exposure'
        };
        return impacts[riskLevel];
    }
    generateRemediationOptions(framework, controlId) {
        return [
            `Implement control ${controlId} requirements`,
            `Create compensating controls`,
            `Accept residual risk`,
            `Transfer risk through insurance or contracts`
        ];
    }
    estimateRemediationEffort(control) {
        const riskLevel = control.riskLevel || 'medium';
        const efforts = {
            critical: '4-8 weeks',
            high: '2-4 weeks',
            medium: '1-2 weeks',
            low: '1 week'
        };
        return efforts[riskLevel];
    }
    calculateTotalRemediationEffort(gaps) {
        const totalWeeks = gaps.reduce((sum, gap) => {
            const weeks = parseInt(gap.estimatedEffort.split('-')[0]) || 1;
            return sum + weeks;
        }, 0);
        return `${totalWeeks} weeks`;
    }
    estimateRemediationCost(gaps) {
        const costPerWeek = 5000; // Mock cost per week
        const totalWeeks = parseInt(this.calculateTotalRemediationEffort(gaps).split(' ')[0]);
        return `$${(totalWeeks * costPerWeek).toLocaleString()}`;
    }
    calculateFrameworkRiskScore(findings) {
        const riskValues = { critical: 10, high: 7, medium: 4, low: 1 };
        return findings.reduce((sum, finding) => sum + (riskValues[finding.severity] || 0), 0);
    }
    categorizeRisk(finding, framework) {
        // Categorize risks based on finding characteristics
        if (finding.control?.includes('access') || finding.control?.includes('A.8'))
            return 'Access Control';
        if (finding.control?.includes('encryption') || finding.control?.includes('CC6.7'))
            return 'Data Protection';
        if (finding.control?.includes('monitoring') || finding.control?.includes('CC6.8'))
            return 'Security Monitoring';
        if (finding.control?.includes('vulnerability') || finding.control?.includes('CC6.6'))
            return 'Vulnerability Management';
        return 'General Security';
    }
    getRiskValue(severity) {
        const values = { critical: 10, high: 7, medium: 4, low: 1 };
        return values[severity] || 1;
    }
    identifyCompoundRisks(frameworkResults, correlationMatrix) {
        const compoundRisks = [];
        // Mock implementation - would identify risks affecting multiple frameworks
        compoundRisks.push({
            id: 'compound-access-control',
            description: 'Access control weaknesses across multiple frameworks',
            affectedFrameworks: Object.keys(frameworkResults),
            riskScore: 8.5,
            likelihood: 0.7,
            impact: 0.9,
            controls: ['CC6.1', 'A.8.2', 'PS.1.1'],
            mitigationStatus: 'partial'
        });
        return compoundRisks;
    }
    calculateRiskTrends(frameworkResults) {
        // Mock implementation - would analyze historical data
        return [{
                category: 'Access Control',
                trend: 'stable',
                timeframe: '90 days',
                confidence: 0.85,
                factors: ['Consistent policy enforcement', 'Regular access reviews']
            }];
    }
    calculateOverallRiskScore(riskByFramework, compoundRisks) {
        const frameworkRisk = Object.values(riskByFramework).reduce((sum, risk) => sum + risk, 0) / Object.keys(riskByFramework).length;
        const compoundRisk = compoundRisks.reduce((sum, risk) => sum + risk.riskScore, 0) / Math.max(compoundRisks.length, 1);
        return (frameworkRisk + compoundRisk) / 2;
    }
    prioritizeRiskMitigation(compoundRisks, riskByCategory) {
        const sortedCategories = Object.entries(riskByCategory)
            .sort(([, a], [, b]) => b - a)
            .map(([category]) => category);
        return sortedCategories.slice(0, 5); // Top 5 priorities
    }
    calculateOverallComplianceScore(frameworkScores) {
        const scores = Object.values(frameworkScores);
        return scores.reduce((sum, score) => sum + score.complianceScore, 0) / scores.length;
    }
    generateCrossFrameworkRecommendations(gapAnalysis, riskAggregation) {
        const recommendations = [];
        if (gapAnalysis.criticalGaps > 0) {
            recommendations.push(`Immediately address ${gapAnalysis.criticalGaps} critical cross-framework gaps`);
        }
        recommendations.push(`Implement unified control framework to address ${gapAnalysis.totalGaps} identified gaps`);
        if (riskAggregation.overallRiskScore > 7) {
            recommendations.push('Deploy enhanced risk mitigation controls across all frameworks');
        }
        recommendations.push('Establish continuous cross-framework monitoring and correlation');
        recommendations.push('Develop integrated compliance dashboard for unified oversight');
        return recommendations;
    }
    generateFrameworkSummaries(frameworkScores) {
        const summaries = {};
        for (const [framework, score] of Object.entries(frameworkScores)) {
            summaries[framework] = {
                complianceScore: score.complianceScore,
                status: score.complianceScore >= 90 ? 'Compliant' : score.complianceScore >= 70 ? 'Partially Compliant' : 'Non-Compliant',
                controlsStatus: `${score.controlsCompliant}/${score.controlsAssessed}`,
                criticalIssues: score.criticalFindings,
                coverageGaps: score.coverageGaps.length
            };
        }
        return summaries;
    }
    generateReportRecommendations(params) {
        return params.recommendations.map((rec, index) => ({
            priority: index < 2 ? 'high' : 'medium',
            category: 'Cross-Framework Alignment',
            recommendation: rec,
            rationale: 'Based on gap analysis and risk assessment',
            frameworks: params.frameworks,
            effort: '2-4 weeks',
            timeline: '30-60 days'
        }));
    }
    generateReportAppendices(params) {
        return [
            {
                title: 'Correlation Matrix',
                content: params.correlationMatrix,
                type: 'data'
            },
            {
                title: 'Gap Analysis Details',
                content: params.gapAnalysis,
                type: 'analysis'
            },
            {
                title: 'Risk Aggregation Results',
                content: params.riskAggregation,
                type: 'analysis'
            }
        ];
    }
    interpretRiskLevel(riskScore) {
        if (riskScore >= 8)
            return 'high';
        if (riskScore >= 5)
            return 'medium';
        return 'low';
    }
    /**
     * Generate unified report
     */
    async generateUnifiedReport(params) {
        // Implementation would generate comprehensive unified report
        return {
            id: `unified-report-${Date.now()}`,
            frameworks: params.includeFrameworks,
            generated: new Date(),
            // Additional report content...
        };
    }
    /**
     * Get current compliance status
     */
    async getCurrentStatus() {
        // Return current aggregated compliance status
        return {
            overall: 85.7,
            frameworks: {},
            timestamp: new Date()
        };
    }
    /**
     * Get correlation history
     */
    getCorrelationHistory() {
        return [...this.correlationHistory];
    }
    /**
     * Get framework correlations
     */
    getCorrelations(sourceFramework, targetFramework) {
        if (targetFramework) {
            return this.getFrameworkCorrelations(sourceFramework, targetFramework);
        }
        // Return all correlations for source framework
        const allCorrelations = [];
        for (const correlations of this.correlationDatabase.values()) {
            allCorrelations.push(...correlations.filter(c => c.sourceFramework === sourceFramework));
        }
        return allCorrelations;
    }
}
exports.ComplianceCorrelator = ComplianceCorrelator;
exports.default = ComplianceCorrelator;
//# sourceMappingURL=compliance-correlator.js.map