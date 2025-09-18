"use strict";
/**
 * Compliance Drift Detection System
 * Monitors compliance degradation and triggers automatic rollback
 * Supports NASA POT10, DFARS, and NIST compliance standards
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ComplianceDriftDetector = void 0;
class ComplianceDriftDetector {
    constructor(rollbackSystem) {
        this.baselines = new Map();
        this.detectedDrifts = new Map();
        this.alerts = new Map();
        this.monitoring = false;
        // Drift thresholds
        this.DRIFT_THRESHOLDS = {
            WARNING: 0.02, // 2% drift
            ERROR: 0.05, // 5% drift
            CRITICAL: 0.1, // 10% drift
            ROLLBACK: 0.15 // 15% drift triggers rollback
        };
        // Compliance targets
        this.COMPLIANCE_TARGETS = {
            NASA_POT10: 0.90, // 90% minimum
            DFARS: 0.85, // 85% minimum
            NIST: 0.88, // 88% minimum
            ISO27001: 0.85 // 85% minimum
        };
        this.rollbackSystem = rollbackSystem;
        this.alertSystem = new ComplianceAlertSystem();
        this.auditLogger = new ComplianceAuditLogger();
        this.complianceScanner = new ComplianceRuleScanner();
    }
    async startDriftDetection() {
        if (this.monitoring) {
            return;
        }
        this.monitoring = true;
        console.log('[ComplianceDriftDetector] Starting compliance drift monitoring');
        // Establish initial baselines
        await this.establishBaselines();
        // Start monitoring loops
        await Promise.all([
            this.startContinuousDriftDetection(),
            this.startTrendAnalysis(),
            this.startAutomaticRemediation(),
            this.startBaselineRefresh()
        ]);
    }
    async stopDriftDetection() {
        this.monitoring = false;
        console.log('[ComplianceDriftDetector] Stopping drift detection');
        await this.auditLogger.finalizeSession();
    }
    async establishBaselines() {
        const standards = ['NASA_POT10', 'DFARS', 'NIST', 'ISO27001'];
        for (const standard of standards) {
            try {
                console.log(`[ComplianceDriftDetector] Establishing baseline for ${standard}`);
                const complianceResult = await this.complianceScanner.scanStandard(standard);
                const baseline = {
                    standard,
                    timestamp: Date.now(),
                    overallScore: complianceResult.overallScore,
                    ruleScores: new Map(complianceResult.ruleScores),
                    checksum: await this.calculateBaselineChecksum(complianceResult),
                    validUntil: Date.now() + 86400000 // Valid for 24 hours
                };
                this.baselines.set(standard, baseline);
                await this.auditLogger.logBaselineEstablished(baseline);
                console.log(`[ComplianceDriftDetector] Baseline for ${standard}: ${(baseline.overallScore * 100).toFixed(1)}%`);
            }
            catch (error) {
                console.error(`[ComplianceDriftDetector] Failed to establish baseline for ${standard}:`, error);
                await this.auditLogger.logError('BASELINE_ESTABLISHMENT', standard, error);
            }
        }
    }
    async startContinuousDriftDetection() {
        while (this.monitoring) {
            try {
                // Check drift for each compliance standard
                for (const [standard, baseline] of this.baselines) {
                    await this.detectDriftForStandard(standard, baseline);
                }
                // Process any detected drifts
                await this.processDriftAlerts();
            }
            catch (error) {
                console.error('[ComplianceDriftDetector] Drift detection error:', error);
                await this.auditLogger.logError('DRIFT_DETECTION', 'ALL_STANDARDS', error);
            }
            await this.sleep(10000); // Check drift every 10 seconds
        }
    }
    async detectDriftForStandard(standard, baseline) {
        const currentResult = await this.complianceScanner.scanStandard(standard);
        const driftPercentage = Math.abs(currentResult.overallScore - baseline.overallScore) / baseline.overallScore;
        if (driftPercentage > this.DRIFT_THRESHOLDS.WARNING) {
            const driftId = `drift_${standard}_${Date.now()}`;
            // Analyze affected rules
            const affectedRules = await this.analyzeAffectedRules(baseline.ruleScores, new Map(currentResult.ruleScores));
            const drift = {
                id: driftId,
                timestamp: Date.now(),
                standard: standard,
                currentScore: currentResult.overallScore,
                baselineScore: baseline.overallScore,
                driftPercentage,
                affectedRules,
                severity: this.calculateDriftSeverity(driftPercentage),
                trendDirection: this.calculateTrendDirection(standard, currentResult.overallScore),
                timeToViolation: this.estimateTimeToViolation(standard, driftPercentage),
                automaticRollbackTriggered: false
            };
            this.detectedDrifts.set(driftId, drift);
            await this.createDriftAlert(drift);
            console.log(`[ComplianceDriftDetector] Drift detected for ${standard}: ${(driftPercentage * 100).toFixed(2)}%`);
            // Trigger automatic rollback if drift is critical
            if (driftPercentage > this.DRIFT_THRESHOLDS.ROLLBACK && this.rollbackSystem) {
                await this.triggerAutomaticRollback(drift);
            }
        }
    }
    async analyzeAffectedRules(baselineRules, currentRules) {
        const violations = [];
        for (const [ruleId, baselineScore] of baselineRules) {
            const currentScore = currentRules.get(ruleId) || 0;
            const scoreDiff = Math.abs(currentScore - baselineScore);
            if (scoreDiff > 0.05) { // 5% rule-level drift
                const ruleDetails = await this.complianceScanner.getRuleDetails(ruleId);
                violations.push({
                    ruleId,
                    ruleName: ruleDetails.name,
                    description: ruleDetails.description,
                    currentValue: currentScore,
                    requiredValue: baselineScore,
                    violationType: currentScore < baselineScore ? 'INSUFFICIENT' : 'INCORRECT',
                    impactScore: scoreDiff,
                    autoFixable: ruleDetails.autoFixable,
                    fixActions: ruleDetails.fixActions || []
                });
            }
        }
        return violations;
    }
    calculateDriftSeverity(driftPercentage) {
        if (driftPercentage >= this.DRIFT_THRESHOLDS.CRITICAL)
            return 'CRITICAL';
        if (driftPercentage >= this.DRIFT_THRESHOLDS.ERROR)
            return 'HIGH';
        if (driftPercentage >= this.DRIFT_THRESHOLDS.WARNING)
            return 'MEDIUM';
        return 'LOW';
    }
    calculateTrendDirection(standard, currentScore) {
        // Get recent scores for trend analysis
        const recentDrifts = Array.from(this.detectedDrifts.values())
            .filter(d => d.standard === standard)
            .sort((a, b) => b.timestamp - a.timestamp)
            .slice(0, 5);
        if (recentDrifts.length < 2)
            return 'STABLE';
        const oldScore = recentDrifts[recentDrifts.length - 1].currentScore;
        if (currentScore > oldScore + 0.01)
            return 'IMPROVING';
        if (currentScore < oldScore - 0.01)
            return 'DEGRADING';
        return 'STABLE';
    }
    estimateTimeToViolation(standard, driftPercentage) {
        const target = this.COMPLIANCE_TARGETS[standard];
        const baseline = this.baselines.get(standard);
        if (!baseline || baseline.overallScore - (baseline.overallScore * driftPercentage) > target) {
            return -1; // No violation expected
        }
        // Estimate based on drift rate (simplified calculation)
        const driftRate = driftPercentage / 3600; // Assume drift occurred over last hour
        const scoreToTarget = (baseline.overallScore - (baseline.overallScore * driftPercentage)) - target;
        return Math.abs(scoreToTarget / driftRate);
    }
    async createDriftAlert(drift) {
        const alertId = `alert_${drift.id}`;
        const alert = {
            id: alertId,
            timestamp: Date.now(),
            drift,
            alertLevel: this.mapSeverityToAlertLevel(drift.severity),
            escalationRequired: drift.severity === 'CRITICAL' || drift.severity === 'HIGH',
            rollbackRecommended: drift.driftPercentage > this.DRIFT_THRESHOLDS.ERROR,
            suppressUntil: undefined
        };
        this.alerts.set(alertId, alert);
        await this.auditLogger.logDriftAlert(alert);
        // Send alert through alert system
        await this.alertSystem.sendDriftAlert(alert);
    }
    async processDriftAlerts() {
        const activeAlerts = Array.from(this.alerts.values()).filter(alert => !alert.suppressUntil || alert.suppressUntil < Date.now());
        for (const alert of activeAlerts) {
            if (alert.escalationRequired && alert.alertLevel === 'CRITICAL') {
                await this.escalateCriticalAlert(alert);
            }
            if (alert.rollbackRecommended && alert.drift.driftPercentage > this.DRIFT_THRESHOLDS.ROLLBACK) {
                await this.recommendRollback(alert);
            }
        }
    }
    async triggerAutomaticRollback(drift) {
        if (!this.rollbackSystem || drift.automaticRollbackTriggered) {
            return;
        }
        try {
            console.log(`[ComplianceDriftDetector] Triggering automatic rollback for ${drift.standard} drift`);
            const latestSnapshot = await this.rollbackSystem.getLatestSnapshot();
            if (latestSnapshot) {
                await this.rollbackSystem.executeRollback(latestSnapshot.id, `Automatic rollback: ${drift.standard} compliance drift ${(drift.driftPercentage * 100).toFixed(2)}%`);
                drift.automaticRollbackTriggered = true;
                await this.auditLogger.logAutomaticRollback(drift);
            }
        }
        catch (error) {
            console.error('[ComplianceDriftDetector] Automatic rollback failed:', error);
            await this.auditLogger.logError('AUTOMATIC_ROLLBACK', drift.standard, error);
        }
    }
    async startTrendAnalysis() {
        while (this.monitoring) {
            try {
                // Analyze compliance trends for predictive alerting
                for (const standard of this.baselines.keys()) {
                    await this.analyzeTrends(standard);
                }
            }
            catch (error) {
                console.error('[ComplianceDriftDetector] Trend analysis error:', error);
            }
            await this.sleep(60000); // Trend analysis every minute
        }
    }
    async startAutomaticRemediation() {
        while (this.monitoring) {
            try {
                // Attempt automatic fixes for auto-fixable violations
                const fixableViolations = Array.from(this.detectedDrifts.values())
                    .flatMap(drift => drift.affectedRules)
                    .filter(rule => rule.autoFixable && rule.impactScore < 0.1); // Only minor violations
                for (const violation of fixableViolations) {
                    await this.applyAutomaticFix(violation);
                }
            }
            catch (error) {
                console.error('[ComplianceDriftDetector] Automatic remediation error:', error);
            }
            await this.sleep(30000); // Remediation every 30 seconds
        }
    }
    async startBaselineRefresh() {
        while (this.monitoring) {
            try {
                // Refresh baselines that have expired
                const now = Date.now();
                for (const [standard, baseline] of this.baselines) {
                    if (baseline.validUntil < now) {
                        await this.refreshBaseline(standard);
                    }
                }
            }
            catch (error) {
                console.error('[ComplianceDriftDetector] Baseline refresh error:', error);
            }
            await this.sleep(300000); // Check baselines every 5 minutes
        }
    }
    async refreshBaseline(standard) {
        console.log(`[ComplianceDriftDetector] Refreshing baseline for ${standard}`);
        const complianceResult = await this.complianceScanner.scanStandard(standard);
        const newBaseline = {
            standard,
            timestamp: Date.now(),
            overallScore: complianceResult.overallScore,
            ruleScores: new Map(complianceResult.ruleScores),
            checksum: await this.calculateBaselineChecksum(complianceResult),
            validUntil: Date.now() + 86400000 // Valid for 24 hours
        };
        this.baselines.set(standard, newBaseline);
        await this.auditLogger.logBaselineRefreshed(newBaseline);
    }
    async getDriftReport() {
        const activeDrifts = Array.from(this.detectedDrifts.values())
            .filter(drift => Date.now() - drift.timestamp < 3600000); // Active in last hour
        const criticalDrifts = activeDrifts.filter(drift => drift.severity === 'CRITICAL');
        const highDrifts = activeDrifts.filter(drift => drift.severity === 'HIGH');
        return {
            timestamp: Date.now(),
            totalDrifts: activeDrifts.length,
            criticalDrifts: criticalDrifts.length,
            highDrifts: highDrifts.length,
            activeDrifts,
            baselineStatus: this.getBaselineStatus(),
            complianceScores: await this.getCurrentComplianceScores(),
            recommendations: await this.generateDriftRecommendations()
        };
    }
    // Helper methods
    async calculateBaselineChecksum(result) {
        return `checksum_${Date.now()}_${JSON.stringify(result).length}`;
    }
    mapSeverityToAlertLevel(severity) {
        const mapping = {
            'LOW': 'INFO',
            'MEDIUM': 'WARNING',
            'HIGH': 'ERROR',
            'CRITICAL': 'CRITICAL'
        };
        return mapping[severity] || 'INFO';
    }
    async escalateCriticalAlert(alert) {
        console.log(`[ComplianceDriftDetector] Escalating critical alert: ${alert.id}`);
        await this.alertSystem.escalateAlert(alert);
    }
    async recommendRollback(alert) {
        console.log(`[ComplianceDriftDetector] Recommending rollback for alert: ${alert.id}`);
        await this.alertSystem.recommendRollback(alert);
    }
    async analyzeTrends(standard) {
        // Implementation would analyze compliance trends
        console.log(`[ComplianceDriftDetector] Analyzing trends for ${standard}`);
    }
    async applyAutomaticFix(violation) {
        console.log(`[ComplianceDriftDetector] Applying automatic fix for ${violation.ruleId}`);
        for (const action of violation.fixActions) {
            try {
                await this.executeFixAction(action);
                await this.auditLogger.logAutomaticFix(violation.ruleId, action);
            }
            catch (error) {
                await this.auditLogger.logFixError(violation.ruleId, action, error);
            }
        }
    }
    async executeFixAction(action) {
        // Implementation would execute the fix action
        console.log(`[ComplianceDriftDetector] Executing fix action: ${action}`);
    }
    getBaselineStatus() {
        return Array.from(this.baselines.entries()).map(([standard, baseline]) => ({
            standard,
            score: baseline.overallScore,
            age: Date.now() - baseline.timestamp,
            valid: baseline.validUntil > Date.now()
        }));
    }
    async getCurrentComplianceScores() {
        const scores = {};
        for (const standard of this.baselines.keys()) {
            const result = await this.complianceScanner.scanStandard(standard);
            scores[standard] = result.overallScore;
        }
        return scores;
    }
    async generateDriftRecommendations() {
        return [
            'Review recent system changes',
            'Validate configuration integrity',
            'Consider baseline refresh',
            'Implement additional monitoring'
        ];
    }
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
exports.ComplianceDriftDetector = ComplianceDriftDetector;
// Supporting classes and interfaces
class ComplianceAlertSystem {
    async sendDriftAlert(alert) {
        console.log(`[DRIFT ALERT] ${alert.alertLevel}: ${alert.drift.standard} drift ${(alert.drift.driftPercentage * 100).toFixed(2)}%`);
    }
    async escalateAlert(alert) {
        console.log(`[ESCALATION] Critical drift alert: ${alert.id}`);
    }
    async recommendRollback(alert) {
        console.log(`[ROLLBACK RECOMMENDATION] Alert: ${alert.id}`);
    }
}
class ComplianceAuditLogger {
    async logBaselineEstablished(baseline) {
        // Implementation would log baseline establishment
    }
    async logDriftAlert(alert) {
        // Implementation would log drift alert
    }
    async logAutomaticRollback(drift) {
        // Implementation would log automatic rollback
    }
    async logBaselineRefreshed(baseline) {
        // Implementation would log baseline refresh
    }
    async logAutomaticFix(ruleId, action) {
        // Implementation would log automatic fix
    }
    async logFixError(ruleId, action, error) {
        // Implementation would log fix error
    }
    async logError(component, context, error) {
        // Implementation would log errors
    }
    async finalizeSession() {
        // Implementation would finalize audit session
    }
}
class ComplianceRuleScanner {
    async scanStandard(standard) {
        // Mock implementation
        return {
            overallScore: 0.92 + (Math.random() * 0.06), // 92-98% score
            ruleScores: [
                ['rule_1', 0.95],
                ['rule_2', 0.88],
                ['rule_3', 0.97]
            ]
        };
    }
    async getRuleDetails(ruleId) {
        return {
            name: `Rule ${ruleId}`,
            description: `Description for ${ruleId}`,
            autoFixable: Math.random() > 0.5,
            fixActions: ['action_1', 'action_2']
        };
    }
}
//# sourceMappingURL=ComplianceDriftDetector.js.map