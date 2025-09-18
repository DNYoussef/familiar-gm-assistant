"use strict";
/**
 * Core Quality Gate Enforcement Engine
 *
 * Implements comprehensive quality gates with Six Sigma metrics,
 * automated decisions, and enterprise compliance validation.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.QualityGateEngine = void 0;
const events_1 = require("events");
const SixSigmaMetrics_1 = require("../metrics/SixSigmaMetrics");
const AutomatedDecisionEngine_1 = require("../decisions/AutomatedDecisionEngine");
const ComplianceGateManager_1 = require("../compliance/ComplianceGateManager");
const PerformanceMonitor_1 = require("../monitoring/PerformanceMonitor");
const SecurityGateValidator_1 = require("../compliance/SecurityGateValidator");
const QualityDashboard_1 = require("../dashboard/QualityDashboard");
class QualityGateEngine extends events_1.EventEmitter {
    constructor(config) {
        super();
        this.gateResults = new Map();
        this.config = config;
        this.initializeComponents();
    }
    initializeComponents() {
        this.sixSigmaMetrics = new SixSigmaMetrics_1.SixSigmaMetrics(this.config.thresholds.sixSigma);
        this.decisionEngine = new AutomatedDecisionEngine_1.AutomatedDecisionEngine(this.config);
        this.complianceManager = new ComplianceGateManager_1.ComplianceGateManager(this.config.thresholds.nasa);
        this.performanceMonitor = new PerformanceMonitor_1.PerformanceMonitor(this.config.thresholds.performance);
        this.securityValidator = new SecurityGateValidator_1.SecurityGateValidator(this.config.thresholds.security);
        this.dashboard = new QualityDashboard_1.QualityDashboard();
        // Setup event listeners
        this.setupEventListeners();
    }
    setupEventListeners() {
        this.performanceMonitor.on('regression-detected', this.handlePerformanceRegression.bind(this));
        this.securityValidator.on('critical-vulnerability', this.handleSecurityViolation.bind(this));
        this.complianceManager.on('compliance-violation', this.handleComplianceViolation.bind(this));
    }
    /**
     * Execute comprehensive quality gate validation
     */
    async executeQualityGate(gateId, artifacts, context) {
        const startTime = Date.now();
        const violations = [];
        const metrics = {};
        const recommendations = [];
        const automatedActions = [];
        try {
            // Six Sigma Metrics Validation (QG-001)
            if (this.config.enableSixSigma) {
                const sixSigmaResults = await this.sixSigmaMetrics.validateMetrics(artifacts, context);
                metrics.sixSigma = sixSigmaResults.metrics;
                violations.push(...sixSigmaResults.violations);
                recommendations.push(...sixSigmaResults.recommendations);
            }
            // NASA POT10 Compliance Gate (QG-003)
            if (this.config.nasaCompliance) {
                const complianceResults = await this.complianceManager.validateCompliance(artifacts, context);
                metrics.nasa = complianceResults.metrics;
                violations.push(...complianceResults.violations);
                recommendations.push(...complianceResults.recommendations);
            }
            // Performance Regression Detection (QG-004)
            if (this.config.performanceMonitoring) {
                const performanceResults = await this.performanceMonitor.detectRegressions(artifacts, context);
                metrics.performance = performanceResults.metrics;
                violations.push(...performanceResults.violations);
                recommendations.push(...performanceResults.recommendations);
            }
            // Security Vulnerability Gate (QG-005)
            if (this.config.securityValidation) {
                const securityResults = await this.securityValidator.validateSecurity(artifacts, context);
                metrics.security = securityResults.metrics;
                violations.push(...securityResults.violations);
                recommendations.push(...securityResults.recommendations);
            }
            // Automated Decision Processing (QG-002)
            if (this.config.automatedDecisions) {
                const decisionResults = await this.decisionEngine.processDecisions(violations, metrics, context);
                automatedActions.push(...decisionResults.actions);
                // Execute auto-remediation if enabled
                if (decisionResults.autoRemediate) {
                    await this.executeAutoRemediation(decisionResults.remediationPlan);
                }
            }
            // Determine gate pass/fail status
            const passed = this.determineGateStatus(violations, metrics);
            const result = {
                passed,
                gateId,
                timestamp: new Date(),
                metrics,
                violations,
                recommendations,
                automatedActions
            };
            // Store result and update dashboard
            this.gateResults.set(gateId, result);
            await this.dashboard.updateGateResult(result);
            // Emit events for downstream processing
            this.emit('gate-completed', result);
            if (!passed) {
                this.emit('gate-failed', result);
            }
            // Validate performance overhead budget
            const executionTime = Date.now() - startTime;
            await this.validatePerformanceOverhead(executionTime);
            return result;
        }
        catch (error) {
            const errorResult = {
                passed: false,
                gateId,
                timestamp: new Date(),
                metrics: { error: error.message },
                violations: [{
                        severity: 'critical',
                        category: 'six-sigma',
                        description: `Quality gate execution failed: ${error.message}`,
                        impact: 'Gate validation incomplete',
                        remediation: 'Review gate configuration and retry',
                        autoRemediable: false
                    }],
                recommendations: ['Review gate configuration', 'Check system resources'],
                automatedActions: []
            };
            this.emit('gate-error', errorResult);
            return errorResult;
        }
    }
    /**
     * Determine overall gate status based on violations and metrics
     */
    determineGateStatus(violations, metrics) {
        // Check for critical violations
        const criticalViolations = violations.filter(v => v.severity === 'critical');
        if (criticalViolations.length > 0) {
            return false;
        }
        // Check NASA compliance threshold (95%+)
        if (metrics.nasa?.complianceScore < this.config.thresholds.nasa.complianceThreshold) {
            return false;
        }
        // Check security violations (zero tolerance for critical/high)
        if (metrics.security?.criticalVulnerabilities > 0 || metrics.security?.highVulnerabilities > 0) {
            return false;
        }
        // Check Six Sigma thresholds
        if (metrics.sixSigma?.defectRate > this.config.thresholds.sixSigma.defectRate) {
            return false;
        }
        // Check performance regressions
        if (metrics.performance?.regressionPercentage > this.config.thresholds.performance.regressionThreshold) {
            return false;
        }
        return true;
    }
    /**
     * Execute automated remediation plan
     */
    async executeAutoRemediation(remediationPlan) {
        // Implementation for automated remediation
        // This would integrate with various systems to automatically fix issues
        this.emit('auto-remediation-started', remediationPlan);
        // Placeholder for actual remediation logic
        // Would include code fixes, configuration updates, etc.
        this.emit('auto-remediation-completed', remediationPlan);
    }
    /**
     * Handle performance regression detection
     */
    async handlePerformanceRegression(regression) {
        this.emit('performance-regression', regression);
        if (regression.severity === 'critical') {
            // Trigger automated rollback
            await this.triggerAutomatedRollback(regression);
        }
    }
    /**
     * Handle security violations
     */
    async handleSecurityViolation(violation) {
        this.emit('security-violation', violation);
        if (violation.severity === 'critical') {
            // Immediate blocking action
            await this.blockDeployment(violation);
        }
    }
    /**
     * Handle compliance violations
     */
    async handleComplianceViolation(violation) {
        this.emit('compliance-violation', violation);
        if (violation.nasaScore < this.config.thresholds.nasa.complianceThreshold) {
            // Block deployment until compliance is restored
            await this.blockDeployment(violation);
        }
    }
    /**
     * Trigger automated rollback for critical issues
     */
    async triggerAutomatedRollback(issue) {
        this.emit('automated-rollback-triggered', issue);
        // Implementation would integrate with deployment systems
    }
    /**
     * Block deployment for critical violations
     */
    async blockDeployment(violation) {
        this.emit('deployment-blocked', violation);
        // Implementation would integrate with CI/CD systems
    }
    /**
     * Validate performance overhead budget compliance
     */
    async validatePerformanceOverhead(executionTime) {
        const overheadPercentage = (executionTime / 1000) * 100; // Convert to percentage
        if (overheadPercentage > this.config.performanceBudget) {
            this.emit('performance-budget-exceeded', {
                actual: overheadPercentage,
                budget: this.config.performanceBudget,
                executionTime
            });
        }
    }
    /**
     * Get quality gate history and trends
     */
    getGateHistory(gateId) {
        if (gateId) {
            const result = this.gateResults.get(gateId);
            return result ? [result] : [];
        }
        return Array.from(this.gateResults.values());
    }
    /**
     * Get real-time quality metrics
     */
    async getRealTimeMetrics() {
        return {
            sixSigma: await this.sixSigmaMetrics.getCurrentMetrics(),
            nasa: await this.complianceManager.getCurrentCompliance(),
            performance: await this.performanceMonitor.getCurrentMetrics(),
            security: await this.securityValidator.getCurrentStatus(),
            gates: {
                total: this.gateResults.size,
                passed: Array.from(this.gateResults.values()).filter(r => r.passed).length,
                failed: Array.from(this.gateResults.values()).filter(r => !r.passed).length
            }
        };
    }
    /**
     * Update configuration at runtime
     */
    updateConfiguration(newConfig) {
        this.config = { ...this.config, ...newConfig };
        this.emit('configuration-updated', this.config);
    }
}
exports.QualityGateEngine = QualityGateEngine;
//# sourceMappingURL=QualityGateEngine.js.map