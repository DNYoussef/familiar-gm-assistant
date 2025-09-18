"use strict";
/**
 * Defense Monitoring Orchestrator
 * Coordinates all monitoring systems with unified alerting and response
 * Provides single point of control for defense-grade monitoring
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DefenseMonitoringOrchestrator = void 0;
const DefenseGradeMonitor_1 = require("./advanced/DefenseGradeMonitor");
const DefenseRollbackSystem_1 = require("../rollback/systems/DefenseRollbackSystem");
const DefenseSecurityMonitor_1 = require("../security/monitoring/DefenseSecurityMonitor");
const ComplianceDriftDetector_1 = require("../compliance/monitoring/ComplianceDriftDetector");
class DefenseMonitoringOrchestrator {
    constructor(config) {
        this.activeAlerts = new Map();
        this.monitoringActive = false;
        this.lastStatus = null;
        this.configuration = this.mergeConfiguration(config);
        // Initialize monitoring components
        this.performanceMonitor = new DefenseGradeMonitor_1.DefenseGradeMonitor();
        this.rollbackSystem = new DefenseRollbackSystem_1.DefenseRollbackSystem();
        this.securityMonitor = new DefenseSecurityMonitor_1.DefenseSecurityMonitor();
        this.complianceDetector = new ComplianceDriftDetector_1.ComplianceDriftDetector(this.rollbackSystem);
        // Initialize support systems
        this.unifiedLogger = new UnifiedAuditLogger();
        this.dashboardUpdater = new DashboardUpdater();
        this.escalationManager = new EscalationManager();
    }
    async startDefenseMonitoring() {
        if (this.monitoringActive) {
            console.log('[DefenseOrchestrator] Monitoring already active');
            return;
        }
        console.log('[DefenseOrchestrator] Starting unified defense monitoring');
        this.monitoringActive = true;
        try {
            // Start all monitoring systems
            await Promise.all([
                this.performanceMonitor.startMonitoring(),
                this.rollbackSystem.startRollbackSystem(),
                this.securityMonitor.startSecurityMonitoring(),
                this.complianceDetector.startDriftDetection()
            ]);
            // Start orchestration loops
            await Promise.all([
                this.startUnifiedStatusMonitoring(),
                this.startAlertCorrelation(),
                this.startAutomaticResponse(),
                this.startDashboardUpdates(),
                this.startHealthChecks()
            ]);
            console.log('[DefenseOrchestrator] Defense monitoring fully operational');
        }
        catch (error) {
            console.error('[DefenseOrchestrator] Failed to start monitoring:', error);
            this.monitoringActive = false;
            throw error;
        }
    }
    async stopDefenseMonitoring() {
        console.log('[DefenseOrchestrator] Stopping defense monitoring');
        this.monitoringActive = false;
        // Stop all monitoring systems
        await Promise.all([
            this.performanceMonitor.stopMonitoring(),
            this.rollbackSystem.stopRollbackSystem(),
            this.securityMonitor.stopSecurityMonitoring(),
            this.complianceDetector.stopDriftDetection()
        ]);
        await this.unifiedLogger.finalizeSession();
        console.log('[DefenseOrchestrator] Defense monitoring stopped');
    }
    async startUnifiedStatusMonitoring() {
        while (this.monitoringActive) {
            try {
                const status = await this.generateUnifiedStatus();
                // Update last status
                this.lastStatus = status;
                // Check for critical conditions
                if (status.overall.status === 'CRITICAL' || status.overall.status === 'EMERGENCY') {
                    await this.handleCriticalStatus(status);
                }
                // Log status for audit
                await this.unifiedLogger.logSystemStatus(status);
                // Update dashboard
                await this.dashboardUpdater.updateStatus(status);
            }
            catch (error) {
                console.error('[DefenseOrchestrator] Status monitoring error:', error);
                await this.unifiedLogger.logError('STATUS_MONITORING', error);
            }
            await this.sleep(5000); // Status update every 5 seconds
        }
    }
    async generateUnifiedStatus() {
        // Collect status from all monitoring systems
        const [performanceReport, securityDashboard, complianceReport, rollbackValidation] = await Promise.all([
            this.performanceMonitor.getPerformanceReport(),
            this.securityMonitor.getSecurityDashboardData(),
            this.complianceDetector.getDriftReport(),
            this.rollbackSystem.validateSystem()
        ]);
        // Calculate overall status
        const overallScore = this.calculateOverallScore(performanceReport, securityDashboard, complianceReport);
        const overallStatus = this.determineOverallStatus(overallScore, {
            performance: performanceReport,
            security: securityDashboard,
            compliance: complianceReport
        });
        return {
            timestamp: Date.now(),
            overall: {
                status: overallStatus,
                score: overallScore,
                lastUpdate: Date.now()
            },
            performance: {
                overhead: performanceReport.currentOverhead,
                target: performanceReport.targetOverhead,
                status: performanceReport.complianceWithTarget ? 'OK' : 'WARNING',
                trend: performanceReport.predictions.trendDirection
            },
            security: {
                threatLevel: securityDashboard.metrics.threatLevel,
                incidentCount: securityDashboard.activeIncidents.length,
                complianceScore: securityDashboard.metrics.complianceScore,
                lastThreat: securityDashboard.recentThreats[0]?.timestamp || 0
            },
            compliance: {
                overallScore: complianceReport.activeDrifts.reduce((sum, drift) => sum + drift.currentScore, 0) / complianceReport.activeDrifts.length || 0.95,
                driftCount: complianceReport.totalDrifts,
                criticalViolations: complianceReport.criticalDrifts,
                lastViolation: complianceReport.activeDrifts[0]?.timestamp || 0
            },
            rollback: {
                ready: rollbackValidation.rollbackCapability,
                lastSnapshot: rollbackValidation.lastSnapshot,
                estimatedTime: rollbackValidation.estimatedRollbackTime,
                historyCount: rollbackValidation.snapshotCount
            },
            alerts: {
                active: this.activeAlerts.size,
                critical: Array.from(this.activeAlerts.values()).filter(alert => alert.level === 'CRITICAL' || alert.level === 'EMERGENCY').length,
                suppressedUntil: Math.max(...Array.from(this.activeAlerts.values()).map(alert => alert.suppressUntil || 0))
            }
        };
    }
    async startAlertCorrelation() {
        while (this.monitoringActive) {
            try {
                // Correlate alerts from different monitoring systems
                await this.correlatePerformanceAlerts();
                await this.correlateSecurityAlerts();
                await this.correlateComplianceAlerts();
                // Process correlated alerts
                await this.processCorrelatedAlerts();
            }
            catch (error) {
                console.error('[DefenseOrchestrator] Alert correlation error:', error);
                await this.unifiedLogger.logError('ALERT_CORRELATION', error);
            }
            await this.sleep(3000); // Alert correlation every 3 seconds
        }
    }
    async startAutomaticResponse() {
        while (this.monitoringActive) {
            try {
                const criticalAlerts = Array.from(this.activeAlerts.values())
                    .filter(alert => alert.level === 'CRITICAL' || alert.level === 'EMERGENCY')
                    .filter(alert => !alert.acknowledged);
                for (const alert of criticalAlerts) {
                    await this.executeAutomaticResponse(alert);
                }
                // Check for rollback triggers
                await this.checkRollbackTriggers();
            }
            catch (error) {
                console.error('[DefenseOrchestrator] Automatic response error:', error);
                await this.unifiedLogger.logError('AUTOMATIC_RESPONSE', error);
            }
            await this.sleep(2000); // Response check every 2 seconds
        }
    }
    async executeAutomaticResponse(alert) {
        console.log(`[DefenseOrchestrator] Executing automatic response for alert: ${alert.id}`);
        for (const action of alert.actions.filter(a => a.automated && !a.executed)) {
            try {
                const result = await this.executeAlertAction(action, alert);
                action.executed = true;
                action.result = result;
                if (action.type === 'ROLLBACK') {
                    alert.rollbackTriggered = true;
                }
                await this.unifiedLogger.logAutomaticResponse(alert.id, action, result);
            }
            catch (error) {
                console.error(`[DefenseOrchestrator] Failed to execute action ${action.type}:`, error);
                await this.unifiedLogger.logResponseError(alert.id, action, error);
            }
        }
    }
    async checkRollbackTriggers() {
        if (!this.lastStatus)
            return;
        const triggers = this.configuration.rollbackTriggers;
        let rollbackTriggered = false;
        let rollbackReason = '';
        // Performance trigger
        if (this.lastStatus.performance.overhead > triggers.performanceOverhead) {
            rollbackTriggered = true;
            rollbackReason = `Performance overhead ${this.lastStatus.performance.overhead}% exceeds ${triggers.performanceOverhead}%`;
        }
        // Security trigger
        if (this.lastStatus.security.threatLevel === triggers.securityThreatLevel) {
            rollbackTriggered = true;
            rollbackReason = `Security threat level: ${this.lastStatus.security.threatLevel}`;
        }
        // Compliance trigger
        if (this.lastStatus.compliance.criticalViolations > 0) {
            rollbackTriggered = true;
            rollbackReason = `Critical compliance violations: ${this.lastStatus.compliance.criticalViolations}`;
        }
        if (rollbackTriggered) {
            await this.triggerEmergencyRollback(rollbackReason);
        }
    }
    async triggerEmergencyRollback(reason) {
        try {
            console.log(`[DefenseOrchestrator] EMERGENCY ROLLBACK TRIGGERED: ${reason}`);
            const rollbackResult = await this.rollbackSystem.executeRollback((await this.rollbackSystem.getSnapshotHistory())[0]?.id || 'latest', `EMERGENCY: ${reason}`);
            // Create emergency alert
            const emergencyAlert = {
                id: `emergency_${Date.now()}`,
                timestamp: Date.now(),
                source: 'SYSTEM',
                level: 'EMERGENCY',
                title: 'Emergency Rollback Executed',
                description: reason,
                details: rollbackResult,
                actions: [],
                escalated: true,
                acknowledged: false,
                rollbackTriggered: true
            };
            this.activeAlerts.set(emergencyAlert.id, emergencyAlert);
            await this.escalationManager.escalateEmergency(emergencyAlert);
            await this.unifiedLogger.logEmergencyRollback(emergencyAlert);
        }
        catch (error) {
            console.error('[DefenseOrchestrator] Emergency rollback failed:', error);
            await this.unifiedLogger.logError('EMERGENCY_ROLLBACK', error);
        }
    }
    async getDefenseStatus() {
        if (this.lastStatus) {
            return this.lastStatus;
        }
        return await this.generateUnifiedStatus();
    }
    async acknowledgeAlert(alertId, operator) {
        const alert = this.activeAlerts.get(alertId);
        if (!alert) {
            return false;
        }
        alert.acknowledged = true;
        await this.unifiedLogger.logAlertAcknowledged(alertId, operator);
        return true;
    }
    async suppressAlert(alertId, duration, operator) {
        const alert = this.activeAlerts.get(alertId);
        if (!alert) {
            return false;
        }
        alert.suppressUntil = Date.now() + duration;
        await this.unifiedLogger.logAlertSuppressed(alertId, duration, operator);
        return true;
    }
    getActiveAlerts() {
        return Array.from(this.activeAlerts.values())
            .filter(alert => !alert.suppressUntil || alert.suppressUntil < Date.now())
            .sort((a, b) => b.timestamp - a.timestamp);
    }
    // Helper methods
    mergeConfiguration(config) {
        const defaultConfig = {
            performanceThresholds: {
                overheadWarning: 0.8,
                overheadCritical: 1.2,
                responseTimeMax: 1000,
                memoryMax: 512
            },
            securityThresholds: {
                threatEscalation: 5,
                incidentEscalation: 3,
                complianceMin: 0.9
            },
            complianceThresholds: {
                driftWarning: 0.05,
                driftCritical: 0.1,
                violationMax: 5
            },
            rollbackTriggers: {
                performanceOverhead: 1.5,
                securityThreatLevel: 'CRITICAL',
                complianceDrift: 0.15,
                manualTrigger: true
            },
            alertingConfig: {
                suppressDuplicates: true,
                escalationDelay: 300000, // 5 minutes
                maxAlertsPerHour: 50
            }
        };
        return { ...defaultConfig, ...config };
    }
    calculateOverallScore(performance, security, compliance) {
        // Weight: 40% performance, 35% security, 25% compliance
        const perfScore = performance.complianceWithTarget ? 100 : Math.max(0, 100 - (performance.currentOverhead * 50));
        const secScore = security.metrics.overallScore;
        const compScore = (compliance.totalDrifts === 0 ? 100 : Math.max(0, 100 - (compliance.criticalDrifts * 20)));
        return Math.round((perfScore * 0.4) + (secScore * 0.35) + (compScore * 0.25));
    }
    determineOverallStatus(score, data) {
        if (score >= 90)
            return 'HEALTHY';
        if (score >= 75)
            return 'WARNING';
        if (score >= 60)
            return 'CRITICAL';
        return 'EMERGENCY';
    }
    async handleCriticalStatus(status) {
        console.log(`[DefenseOrchestrator] CRITICAL STATUS DETECTED: ${status.overall.status}`);
        const criticalAlert = {
            id: `critical_${Date.now()}`,
            timestamp: Date.now(),
            source: 'SYSTEM',
            level: 'CRITICAL',
            title: 'Critical System Status',
            description: `Overall system status: ${status.overall.status} (Score: ${status.overall.score})`,
            details: status,
            actions: [
                {
                    type: 'INVESTIGATE',
                    description: 'Investigate system status degradation',
                    automated: false,
                    executed: false
                },
                {
                    type: 'ESCALATE',
                    description: 'Escalate to operations team',
                    automated: true,
                    executed: false
                }
            ],
            escalated: false,
            acknowledged: false
        };
        this.activeAlerts.set(criticalAlert.id, criticalAlert);
        await this.escalationManager.escalateCritical(criticalAlert);
    }
    // Mock implementations for supporting methods
    async correlatePerformanceAlerts() {
        // Implementation would correlate performance alerts
    }
    async correlateSecurityAlerts() {
        // Implementation would correlate security alerts
    }
    async correlateComplianceAlerts() {
        // Implementation would correlate compliance alerts
    }
    async processCorrelatedAlerts() {
        // Implementation would process correlated alerts
    }
    async executeAlertAction(action, alert) {
        console.log(`[DefenseOrchestrator] Executing action: ${action.type}`);
        return { success: true };
    }
    async startDashboardUpdates() {
        while (this.monitoringActive) {
            try {
                if (this.lastStatus) {
                    await this.dashboardUpdater.updateDashboard(this.lastStatus, this.getActiveAlerts());
                }
            }
            catch (error) {
                console.error('[DefenseOrchestrator] Dashboard update error:', error);
            }
            await this.sleep(10000); // Dashboard update every 10 seconds
        }
    }
    async startHealthChecks() {
        while (this.monitoringActive) {
            try {
                const healthStatus = await this.performSystemHealthCheck();
                if (!healthStatus.healthy) {
                    await this.handleUnhealthySystem(healthStatus);
                }
            }
            catch (error) {
                console.error('[DefenseOrchestrator] Health check error:', error);
            }
            await this.sleep(30000); // Health check every 30 seconds
        }
    }
    async performSystemHealthCheck() {
        return { healthy: true, issues: [] };
    }
    async handleUnhealthySystem(healthStatus) {
        console.log('[DefenseOrchestrator] Unhealthy system detected:', healthStatus);
    }
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
exports.DefenseMonitoringOrchestrator = DefenseMonitoringOrchestrator;
// Supporting classes
class UnifiedAuditLogger {
    async logSystemStatus(status) {
        // Implementation would log system status
    }
    async logError(component, error) {
        // Implementation would log errors
    }
    async logAutomaticResponse(alertId, action, result) {
        // Implementation would log automatic responses
    }
    async logResponseError(alertId, action, error) {
        // Implementation would log response errors
    }
    async logEmergencyRollback(alert) {
        // Implementation would log emergency rollbacks
    }
    async logAlertAcknowledged(alertId, operator) {
        // Implementation would log alert acknowledgments
    }
    async logAlertSuppressed(alertId, duration, operator) {
        // Implementation would log alert suppressions
    }
    async finalizeSession() {
        // Implementation would finalize logging session
    }
}
class DashboardUpdater {
    async updateStatus(status) {
        // Implementation would update monitoring dashboard
    }
    async updateDashboard(status, alerts) {
        // Implementation would update full dashboard
    }
}
class EscalationManager {
    async escalateCritical(alert) {
        console.log(`[ESCALATION] Critical alert: ${alert.id}`);
    }
    async escalateEmergency(alert) {
        console.log(`[ESCALATION] Emergency alert: ${alert.id}`);
    }
}
//# sourceMappingURL=DefenseMonitoringOrchestrator.js.map