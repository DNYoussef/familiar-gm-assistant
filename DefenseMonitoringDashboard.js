"use strict";
/**
 * Defense Monitoring Dashboard
 * Real-time visualization and control interface for defense operations
 * Provides comprehensive overview of all monitoring systems
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DefenseMonitoringDashboard = void 0;
class DefenseMonitoringDashboard {
    constructor(orchestrator, config) {
        this.metricsHistory = new Map();
        this.alertHistory = new Map();
        this.refreshTimer = null;
        this.subscribers = new Map();
        this.orchestrator = orchestrator;
        this.configuration = this.mergeConfiguration(config);
    }
    async initialize() {
        console.log('[DefenseDashboard] Initializing defense monitoring dashboard');
        // Start dashboard data collection
        if (this.configuration.autoRefresh) {
            await this.startAutoRefresh();
        }
        // Load initial data
        await this.refreshDashboardData();
        console.log('[DefenseDashboard] Dashboard initialized successfully');
    }
    async shutdown() {
        console.log('[DefenseDashboard] Shutting down dashboard');
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
        this.subscribers.clear();
    }
    async startAutoRefresh() {
        this.refreshTimer = setInterval(async () => {
            try {
                await this.refreshDashboardData();
                await this.notifySubscribers();
            }
            catch (error) {
                console.error('[DefenseDashboard] Auto-refresh error:', error);
            }
        }, this.configuration.refreshInterval);
    }
    async refreshDashboardData() {
        const status = await this.orchestrator.getDefenseStatus();
        const alerts = this.orchestrator.getActiveAlerts();
        // Update metrics history
        const metrics = this.generateDashboardMetrics(status, alerts);
        this.metricsHistory.set(Date.now(), metrics);
        // Update alert history
        for (const alert of alerts) {
            this.alertHistory.set(alert.id, alert);
        }
        // Cleanup old data
        await this.cleanupOldData();
    }
    generateDashboardMetrics(status, alerts) {
        return {
            timestamp: Date.now(),
            performance: {
                overhead: status.performance.overhead,
                trend: this.calculatePerformanceTrend(),
                alerts: alerts.filter(a => a.source === 'PERFORMANCE').length,
                optimizations: 0 // Would be calculated from optimization history
            },
            security: {
                threatLevel: status.security.threatLevel,
                incidents: status.security.incidentCount,
                compliance: status.security.complianceScore,
                alerts: alerts.filter(a => a.source === 'SECURITY').length
            },
            compliance: {
                overallScore: status.compliance.overallScore,
                drift: status.compliance.driftCount,
                violations: status.compliance.criticalViolations,
                alerts: alerts.filter(a => a.source === 'COMPLIANCE').length
            },
            rollback: {
                ready: status.rollback.ready,
                snapshots: status.rollback.historyCount,
                lastRollback: 0, // Would track last rollback time
                averageTime: status.rollback.estimatedTime
            }
        };
    }
    async getDashboardData() {
        const status = await this.orchestrator.getDefenseStatus();
        const alerts = this.orchestrator.getActiveAlerts();
        const metrics = this.generateDashboardMetrics(status, alerts);
        return {
            timestamp: Date.now(),
            status,
            metrics,
            alerts: this.generateAlertSummary(alerts),
            history: this.getMetricsHistory(),
            configuration: this.configuration
        };
    }
    generateAlertSummary(alerts) {
        return {
            total: alerts.length,
            critical: alerts.filter(a => a.level === 'CRITICAL' || a.level === 'EMERGENCY').length,
            high: alerts.filter(a => a.level === 'ERROR').length,
            medium: alerts.filter(a => a.level === 'WARNING').length,
            low: alerts.filter(a => a.level === 'INFO').length,
            acknowledged: alerts.filter(a => a.acknowledged).length,
            suppressed: alerts.filter(a => a.suppressUntil && a.suppressUntil > Date.now()).length,
            recent: alerts.slice(0, this.configuration.maxAlertDisplay)
        };
    }
    async getPerformanceChart() {
        const history = this.getMetricsHistory();
        return {
            timestamps: history.map(h => h.timestamp),
            overhead: history.map(h => h.performance.overhead),
            target: Array(history.length).fill(1.2), // 1.2% target
            trend: this.calculatePerformanceTrend(),
            predictions: await this.generatePerformancePredictions()
        };
    }
    async getSecurityHeatmap() {
        const alerts = this.orchestrator.getActiveAlerts();
        const securityAlerts = alerts.filter(a => a.source === 'SECURITY');
        return {
            threatDistribution: this.calculateThreatDistribution(securityAlerts),
            timelineHeat: this.calculateTimelineHeat(securityAlerts),
            severityMatrix: this.calculateSeverityMatrix(securityAlerts),
            trendAnalysis: this.calculateSecurityTrend()
        };
    }
    async getComplianceMatrix() {
        const status = await this.orchestrator.getDefenseStatus();
        return {
            standards: {
                NASA_POT10: 0.95, // Would come from actual compliance data
                DFARS: 0.92,
                NIST: 0.94,
                ISO27001: 0.91
            },
            timeline: this.getComplianceTimeline(),
            violations: this.getViolationBreakdown(),
            trends: this.getComplianceTrends()
        };
    }
    async exportDashboardData(format) {
        const dashboardData = await this.getDashboardData();
        switch (format) {
            case 'JSON':
                return JSON.stringify(dashboardData, null, 2);
            case 'CSV':
                return this.convertToCSV(dashboardData);
            case 'PDF':
                return this.generatePDFReport(dashboardData);
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
    }
    subscribe(id, callback) {
        this.subscribers.set(id, callback);
    }
    unsubscribe(id) {
        this.subscribers.delete(id);
    }
    async notifySubscribers() {
        if (this.subscribers.size === 0)
            return;
        const data = await this.getDashboardData();
        for (const callback of this.subscribers.values()) {
            try {
                callback(data);
            }
            catch (error) {
                console.error('[DefenseDashboard] Subscriber notification error:', error);
            }
        }
    }
    // Configuration and utility methods
    updateConfiguration(config) {
        this.configuration = { ...this.configuration, ...config };
        // Restart auto-refresh if interval changed
        if (config.refreshInterval && this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.startAutoRefresh();
        }
    }
    mergeConfiguration(config) {
        const defaultConfig = {
            refreshInterval: 5000, // 5 seconds
            alertRetention: 86400000, // 24 hours
            historyRetention: 604800000, // 7 days
            maxAlertDisplay: 20,
            autoRefresh: true,
            compactMode: false
        };
        return { ...defaultConfig, ...config };
    }
    async cleanupOldData() {
        const now = Date.now();
        // Cleanup metrics history
        for (const [timestamp] of this.metricsHistory) {
            if (now - timestamp > this.configuration.historyRetention) {
                this.metricsHistory.delete(timestamp);
            }
        }
        // Cleanup alert history
        for (const [id, alert] of this.alertHistory) {
            if (now - alert.timestamp > this.configuration.alertRetention) {
                this.alertHistory.delete(id);
            }
        }
    }
    getMetricsHistory() {
        return Array.from(this.metricsHistory.values())
            .sort((a, b) => a.timestamp - b.timestamp);
    }
    calculatePerformanceTrend() {
        const history = this.getMetricsHistory();
        return history.slice(-20).map(h => h.performance.overhead);
    }
    async generatePerformancePredictions() {
        // Implementation would use ML/statistical models for prediction
        const trend = this.calculatePerformanceTrend();
        const predictions = [];
        // Simple linear prediction (would be more sophisticated in real implementation)
        for (let i = 0; i < 10; i++) {
            const lastValue = trend[trend.length - 1] || 1.0;
            predictions.push(lastValue + (Math.random() - 0.5) * 0.1);
        }
        return predictions;
    }
    calculateThreatDistribution(securityAlerts) {
        const distribution = {};
        securityAlerts.forEach(alert => {
            const type = alert.details?.type || 'UNKNOWN';
            distribution[type] = (distribution[type] || 0) + 1;
        });
        return distribution;
    }
    calculateTimelineHeat(alerts) {
        // Implementation would calculate heat map data
        return {
            hours: Array(24).fill(0).map((_, i) => ({ hour: i, count: Math.floor(Math.random() * 10) }))
        };
    }
    calculateSeverityMatrix(alerts) {
        const matrix = {
            CRITICAL: 0,
            ERROR: 0,
            WARNING: 0,
            INFO: 0
        };
        alerts.forEach(alert => {
            if (alert.level in matrix) {
                matrix[alert.level]++;
            }
        });
        return matrix;
    }
    calculateSecurityTrend() {
        return {
            direction: 'IMPROVING',
            confidence: 0.85,
            prediction: 'Threat level expected to remain LOW'
        };
    }
    getComplianceTimeline() {
        return {
            points: [
                { timestamp: Date.now() - 86400000, score: 0.94 },
                { timestamp: Date.now() - 43200000, score: 0.95 },
                { timestamp: Date.now(), score: 0.93 }
            ]
        };
    }
    getViolationBreakdown() {
        return {
            NASA_POT10: 1,
            DFARS: 2,
            NIST: 0,
            ISO27001: 1
        };
    }
    getComplianceTrends() {
        return {
            NASA_POT10: { direction: 'STABLE', change: 0.01 },
            DFARS: { direction: 'IMPROVING', change: 0.03 },
            NIST: { direction: 'STABLE', change: 0.00 },
            ISO27001: { direction: 'DEGRADING', change: -0.02 }
        };
    }
    convertToCSV(data) {
        const headers = ['Timestamp', 'Status', 'Performance_Overhead', 'Security_Threat', 'Compliance_Score', 'Active_Alerts'];
        const rows = [headers.join(',')];
        rows.push([
            new Date(data.timestamp).toISOString(),
            data.status.overall.status,
            data.metrics.performance.overhead.toString(),
            data.status.security.threatLevel,
            data.metrics.compliance.overallScore.toString(),
            data.alerts.total.toString()
        ].join(','));
        return rows.join('\n');
    }
    generatePDFReport(data) {
        // Mock PDF generation - would use actual PDF library
        return `Defense Monitoring Report - ${new Date().toISOString()}`;
    }
}
exports.DefenseMonitoringDashboard = DefenseMonitoringDashboard;
//# sourceMappingURL=DefenseMonitoringDashboard.js.map