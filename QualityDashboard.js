"use strict";
/**
 * Unified Quality Dashboard (QG-006)
 *
 * Implements cross-domain quality orchestration with unified quality dashboard
 * for comprehensive quality gate visualization and monitoring.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.QualityDashboard = void 0;
const events_1 = require("events");
class QualityDashboard extends events_1.EventEmitter {
    constructor() {
        super();
        this.alerts = new Map();
        this.layouts = new Map();
        this.currentLayout = 'default';
        this.refreshInterval = null;
        this.websocketClients = new Set();
        this.metrics = this.initializeDefaultMetrics();
        this.initializeDefaultLayouts();
        this.startPeriodicRefresh();
    }
    /**
     * Initialize default dashboard metrics
     */
    initializeDefaultMetrics() {
        return {
            overall: {
                qualityScore: 0,
                status: 'warning',
                lastUpdated: new Date(),
                environment: 'development',
                version: '1.0.0'
            },
            sixSigma: {
                currentLevel: 3,
                defectRate: 6210,
                processCapability: 1.0,
                qualityScore: 60,
                trend: 'stable'
            },
            nasa: {
                complianceScore: 0,
                pot10Compliance: 0,
                criticalViolations: 0,
                status: 'non-compliant',
                trend: 'stable'
            },
            performance: {
                responseTime: 0,
                throughput: 0,
                errorRate: 0,
                regressionStatus: 'none',
                trend: 'stable'
            },
            security: {
                overallScore: 0,
                criticalVulnerabilities: 0,
                highVulnerabilities: 0,
                owaspCompliance: 0,
                status: 'critical',
                trend: 'stable'
            },
            gates: {
                totalGates: 0,
                passedGates: 0,
                failedGates: 0,
                warningGates: 0,
                blockedGates: 0,
                successRate: 0,
                averageExecutionTime: 0
            },
            trends: {
                timeframe: '24h',
                qualityTrend: [],
                gateTrend: [],
                complianceTrend: [],
                performanceTrend: [],
                securityTrend: []
            }
        };
    }
    /**
     * Initialize default dashboard layouts
     */
    initializeDefaultLayouts() {
        const defaultLayout = {
            id: 'default',
            name: 'Default Quality Dashboard',
            created: new Date(),
            modified: new Date(),
            isDefault: true,
            widgets: [
                {
                    id: 'overall-score',
                    type: 'metric',
                    title: 'Overall Quality Score',
                    position: { x: 0, y: 0, width: 6, height: 4 },
                    config: { metric: 'overall.qualityScore', format: 'percentage' },
                    refreshInterval: 30,
                    visible: true
                },
                {
                    id: 'six-sigma-level',
                    type: 'metric',
                    title: 'Six Sigma Level',
                    position: { x: 6, y: 0, width: 6, height: 4 },
                    config: { metric: 'sixSigma.currentLevel', format: 'number' },
                    refreshInterval: 60,
                    visible: true
                },
                {
                    id: 'nasa-compliance',
                    type: 'metric',
                    title: 'NASA POT10 Compliance',
                    position: { x: 0, y: 4, width: 6, height: 4 },
                    config: { metric: 'nasa.complianceScore', format: 'percentage' },
                    refreshInterval: 300,
                    visible: true
                },
                {
                    id: 'security-score',
                    type: 'metric',
                    title: 'Security Score',
                    position: { x: 6, y: 4, width: 6, height: 4 },
                    config: { metric: 'security.overallScore', format: 'percentage' },
                    refreshInterval: 60,
                    visible: true
                },
                {
                    id: 'performance-chart',
                    type: 'chart',
                    title: 'Performance Trends',
                    position: { x: 0, y: 8, width: 12, height: 6 },
                    config: {
                        chartType: 'line',
                        metrics: ['performance.responseTime', 'performance.throughput'],
                        timeframe: '24h'
                    },
                    refreshInterval: 30,
                    visible: true
                },
                {
                    id: 'quality-gates',
                    type: 'table',
                    title: 'Quality Gate Status',
                    position: { x: 0, y: 14, width: 12, height: 6 },
                    config: {
                        columns: ['gate', 'status', 'score', 'lastRun'],
                        sortBy: 'lastRun',
                        sortOrder: 'desc'
                    },
                    refreshInterval: 10,
                    visible: true
                },
                {
                    id: 'active-alerts',
                    type: 'alert',
                    title: 'Active Alerts',
                    position: { x: 0, y: 20, width: 12, height: 4 },
                    config: {
                        maxAlerts: 10,
                        severityFilter: ['critical', 'warning']
                    },
                    refreshInterval: 5,
                    visible: true
                }
            ]
        };
        this.layouts.set('default', defaultLayout);
        // Executive summary layout
        const executiveLayout = {
            id: 'executive',
            name: 'Executive Summary',
            created: new Date(),
            modified: new Date(),
            isDefault: false,
            widgets: [
                {
                    id: 'executive-overview',
                    type: 'metric',
                    title: 'Quality Overview',
                    position: { x: 0, y: 0, width: 12, height: 8 },
                    config: {
                        layout: 'executive',
                        metrics: [
                            'overall.qualityScore',
                            'nasa.complianceScore',
                            'security.overallScore',
                            'gates.successRate'
                        ]
                    },
                    refreshInterval: 60,
                    visible: true
                },
                {
                    id: 'trend-summary',
                    type: 'trend',
                    title: 'Quality Trends',
                    position: { x: 0, y: 8, width: 12, height: 8 },
                    config: {
                        timeframe: '30d',
                        showAll: true
                    },
                    refreshInterval: 300,
                    visible: true
                }
            ]
        };
        this.layouts.set('executive', executiveLayout);
    }
    /**
     * Update dashboard with quality gate result
     */
    async updateGateResult(gateResult) {
        try {
            // Update gate metrics
            this.updateGateMetrics(gateResult);
            // Update domain-specific metrics
            await this.updateDomainMetrics(gateResult);
            // Update overall quality score
            this.updateOverallScore();
            // Check for new alerts
            await this.checkAndCreateAlerts(gateResult);
            // Update trends
            this.updateTrends();
            // Notify clients of updates
            this.notifyClients('metrics-updated', this.metrics);
            this.emit('dashboard-updated', this.metrics);
        }
        catch (error) {
            this.emit('dashboard-error', error);
        }
    }
    /**
     * Update gate-specific metrics
     */
    updateGateMetrics(gateResult) {
        this.metrics.gates.totalGates++;
        if (gateResult.passed) {
            this.metrics.gates.passedGates++;
        }
        else {
            this.metrics.gates.failedGates++;
        }
        // Calculate success rate
        this.metrics.gates.successRate =
            (this.metrics.gates.passedGates / this.metrics.gates.totalGates) * 100;
        // Update average execution time (would track actual execution time)
        this.metrics.gates.averageExecutionTime =
            (this.metrics.gates.averageExecutionTime + (gateResult.executionTime || 5000)) / 2;
    }
    /**
     * Update domain-specific metrics
     */
    async updateDomainMetrics(gateResult) {
        const metrics = gateResult.metrics || {};
        // Update Six Sigma metrics
        if (metrics.sixSigma) {
            this.metrics.sixSigma.currentLevel = metrics.sixSigma.sigma?.level || this.metrics.sixSigma.currentLevel;
            this.metrics.sixSigma.defectRate = metrics.sixSigma.defectRate || this.metrics.sixSigma.defectRate;
            this.metrics.sixSigma.processCapability = metrics.sixSigma.processCapability?.cpk || this.metrics.sixSigma.processCapability;
            this.metrics.sixSigma.qualityScore = metrics.sixSigma.qualityScore || this.metrics.sixSigma.qualityScore;
        }
        // Update NASA compliance metrics
        if (metrics.nasa) {
            this.metrics.nasa.complianceScore = metrics.nasa.overallScore || this.metrics.nasa.complianceScore;
            this.metrics.nasa.pot10Compliance = metrics.nasa.pot10Compliance || this.metrics.nasa.pot10Compliance;
            this.metrics.nasa.criticalViolations = this.countCriticalViolations(gateResult.violations || []);
            this.updateNASAStatus();
        }
        // Update performance metrics
        if (metrics.performance) {
            this.metrics.performance.responseTime = metrics.performance.responseTime?.mean || this.metrics.performance.responseTime;
            this.metrics.performance.throughput = metrics.performance.throughput?.requestsPerSecond || this.metrics.performance.throughput;
            this.metrics.performance.errorRate = metrics.performance.errors?.errorRate || this.metrics.performance.errorRate;
            this.updatePerformanceRegressionStatus(metrics.performance);
        }
        // Update security metrics
        if (metrics.security) {
            this.metrics.security.overallScore = metrics.security.overallScore || this.metrics.security.overallScore;
            this.metrics.security.criticalVulnerabilities = metrics.security.vulnerabilities?.critical || this.metrics.security.criticalVulnerabilities;
            this.metrics.security.highVulnerabilities = metrics.security.vulnerabilities?.high || this.metrics.security.highVulnerabilities;
            this.metrics.security.owaspCompliance = metrics.security.compliance?.owasp?.score || this.metrics.security.owaspCompliance;
            this.updateSecurityStatus();
        }
    }
    /**
     * Update overall quality score
     */
    updateOverallScore() {
        const weights = {
            sixSigma: 0.25,
            nasa: 0.25,
            performance: 0.20,
            security: 0.25,
            gates: 0.05
        };
        // Calculate performance score (inverse of response time and error rate)
        const performanceScore = Math.max(0, 100 - ((this.metrics.performance.responseTime / 10) +
            (this.metrics.performance.errorRate * 10)));
        this.metrics.overall.qualityScore = (this.metrics.sixSigma.qualityScore * weights.sixSigma +
            this.metrics.nasa.complianceScore * weights.nasa +
            performanceScore * weights.performance +
            this.metrics.security.overallScore * weights.security +
            this.metrics.gates.successRate * weights.gates);
        this.updateOverallStatus();
        this.metrics.overall.lastUpdated = new Date();
    }
    /**
     * Update overall status based on quality score
     */
    updateOverallStatus() {
        const score = this.metrics.overall.qualityScore;
        if (score >= 90) {
            this.metrics.overall.status = 'excellent';
        }
        else if (score >= 75) {
            this.metrics.overall.status = 'good';
        }
        else if (score >= 60) {
            this.metrics.overall.status = 'warning';
        }
        else {
            this.metrics.overall.status = 'critical';
        }
    }
    /**
     * Count critical violations from gate result
     */
    countCriticalViolations(violations) {
        return violations.filter(v => v.severity === 'critical').length;
    }
    /**
     * Update NASA compliance status
     */
    updateNASAStatus() {
        const score = this.metrics.nasa.complianceScore;
        const criticalViolations = this.metrics.nasa.criticalViolations;
        if (criticalViolations > 0) {
            this.metrics.nasa.status = 'non-compliant';
        }
        else if (score >= 95) {
            this.metrics.nasa.status = 'compliant';
        }
        else if (score >= 80) {
            this.metrics.nasa.status = 'minor-issues';
        }
        else {
            this.metrics.nasa.status = 'major-issues';
        }
    }
    /**
     * Update performance regression status
     */
    updatePerformanceRegressionStatus(performanceMetrics) {
        const regressionPercentage = performanceMetrics.regressionPercentage || 0;
        if (regressionPercentage >= 20) {
            this.metrics.performance.regressionStatus = 'critical';
        }
        else if (regressionPercentage >= 10) {
            this.metrics.performance.regressionStatus = 'major';
        }
        else if (regressionPercentage >= 5) {
            this.metrics.performance.regressionStatus = 'minor';
        }
        else {
            this.metrics.performance.regressionStatus = 'none';
        }
    }
    /**
     * Update security status
     */
    updateSecurityStatus() {
        const criticalVulns = this.metrics.security.criticalVulnerabilities;
        const highVulns = this.metrics.security.highVulnerabilities;
        const score = this.metrics.security.overallScore;
        if (criticalVulns > 0) {
            this.metrics.security.status = 'critical';
        }
        else if (highVulns > 0 || score < 70) {
            this.metrics.security.status = 'major-issues';
        }
        else if (score < 85) {
            this.metrics.security.status = 'minor-issues';
        }
        else {
            this.metrics.security.status = 'secure';
        }
    }
    /**
     * Check for new alerts based on gate results
     */
    async checkAndCreateAlerts(gateResult) {
        const alerts = [];
        // Overall quality score alert
        if (this.metrics.overall.qualityScore < 60) {
            alerts.push({
                id: `quality-score-${Date.now()}`,
                timestamp: new Date(),
                severity: 'critical',
                domain: 'gate',
                title: 'Low Quality Score',
                description: `Overall quality score ${this.metrics.overall.qualityScore.toFixed(1)}% below acceptable threshold`,
                acknowledged: false,
                autoResolved: false
            });
        }
        // NASA compliance alert
        if (this.metrics.nasa.criticalViolations > 0) {
            alerts.push({
                id: `nasa-critical-${Date.now()}`,
                timestamp: new Date(),
                severity: 'critical',
                domain: 'nasa',
                title: 'NASA Critical Violations',
                description: `${this.metrics.nasa.criticalViolations} critical NASA POT10 violations detected`,
                acknowledged: false,
                autoResolved: false
            });
        }
        // Security alerts
        if (this.metrics.security.criticalVulnerabilities > 0) {
            alerts.push({
                id: `security-critical-${Date.now()}`,
                timestamp: new Date(),
                severity: 'critical',
                domain: 'security',
                title: 'Critical Security Vulnerabilities',
                description: `${this.metrics.security.criticalVulnerabilities} critical security vulnerabilities found`,
                acknowledged: false,
                autoResolved: false
            });
        }
        // Performance regression alert
        if (this.metrics.performance.regressionStatus === 'critical') {
            alerts.push({
                id: `performance-regression-${Date.now()}`,
                timestamp: new Date(),
                severity: 'critical',
                domain: 'performance',
                title: 'Critical Performance Regression',
                description: 'Critical performance regression detected - consider rollback',
                acknowledged: false,
                autoResolved: false
            });
        }
        // Six Sigma alert
        if (this.metrics.sixSigma.currentLevel < 3) {
            alerts.push({
                id: `six-sigma-low-${Date.now()}`,
                timestamp: new Date(),
                severity: 'warning',
                domain: 'six-sigma',
                title: 'Low Six Sigma Level',
                description: `Six Sigma level ${this.metrics.sixSigma.currentLevel} below target threshold`,
                acknowledged: false,
                autoResolved: false
            });
        }
        // Store new alerts
        alerts.forEach(alert => {
            this.alerts.set(alert.id, alert);
        });
        if (alerts.length > 0) {
            this.notifyClients('new-alerts', alerts);
            this.emit('new-alerts', alerts);
        }
    }
    /**
     * Update trend data
     */
    updateTrends() {
        const now = new Date();
        // Update quality trend
        this.metrics.trends.qualityTrend.push({
            timestamp: now,
            score: this.metrics.overall.qualityScore
        });
        // Update gate trend
        this.metrics.trends.gateTrend.push({
            timestamp: now,
            passed: this.metrics.gates.passedGates,
            failed: this.metrics.gates.failedGates
        });
        // Update compliance trend
        this.metrics.trends.complianceTrend.push({
            timestamp: now,
            score: this.metrics.nasa.complianceScore
        });
        // Update performance trend
        this.metrics.trends.performanceTrend.push({
            timestamp: now,
            responseTime: this.metrics.performance.responseTime,
            throughput: this.metrics.performance.throughput
        });
        // Update security trend
        this.metrics.trends.securityTrend.push({
            timestamp: now,
            score: this.metrics.security.overallScore,
            vulnerabilities: this.metrics.security.criticalVulnerabilities + this.metrics.security.highVulnerabilities
        });
        // Keep only data for the selected timeframe
        this.trimTrendData();
    }
    /**
     * Trim trend data to keep only relevant timeframe
     */
    trimTrendData() {
        const timeframes = {
            '1h': 60 * 60 * 1000,
            '24h': 24 * 60 * 60 * 1000,
            '7d': 7 * 24 * 60 * 60 * 1000,
            '30d': 30 * 24 * 60 * 60 * 1000
        };
        const cutoff = new Date(Date.now() - timeframes[this.metrics.trends.timeframe]);
        this.metrics.trends.qualityTrend = this.metrics.trends.qualityTrend.filter(point => point.timestamp >= cutoff);
        this.metrics.trends.gateTrend = this.metrics.trends.gateTrend.filter(point => point.timestamp >= cutoff);
        this.metrics.trends.complianceTrend = this.metrics.trends.complianceTrend.filter(point => point.timestamp >= cutoff);
        this.metrics.trends.performanceTrend = this.metrics.trends.performanceTrend.filter(point => point.timestamp >= cutoff);
        this.metrics.trends.securityTrend = this.metrics.trends.securityTrend.filter(point => point.timestamp >= cutoff);
    }
    /**
     * Start periodic dashboard refresh
     */
    startPeriodicRefresh() {
        this.refreshInterval = setInterval(() => {
            this.refreshDashboard();
        }, 30000); // Refresh every 30 seconds
    }
    /**
     * Refresh dashboard data
     */
    async refreshDashboard() {
        try {
            // Update trends if needed
            this.updateTrends();
            // Check for alert auto-resolution
            this.checkAlertAutoResolution();
            // Notify clients of refresh
            this.notifyClients('dashboard-refreshed', {
                timestamp: new Date(),
                metrics: this.metrics
            });
        }
        catch (error) {
            this.emit('refresh-error', error);
        }
    }
    /**
     * Check for alerts that can be auto-resolved
     */
    checkAlertAutoResolution() {
        const resolvedAlerts = [];
        this.alerts.forEach((alert, id) => {
            if (alert.autoResolved || alert.acknowledged)
                return;
            let shouldResolve = false;
            // Auto-resolve based on current metrics
            switch (alert.domain) {
                case 'gate':
                    if (alert.title.includes('Quality Score') && this.metrics.overall.qualityScore >= 60) {
                        shouldResolve = true;
                    }
                    break;
                case 'nasa':
                    if (alert.title.includes('Critical Violations') && this.metrics.nasa.criticalViolations === 0) {
                        shouldResolve = true;
                    }
                    break;
                case 'security':
                    if (alert.title.includes('Critical Security') && this.metrics.security.criticalVulnerabilities === 0) {
                        shouldResolve = true;
                    }
                    break;
                case 'performance':
                    if (alert.title.includes('Performance Regression') && this.metrics.performance.regressionStatus === 'none') {
                        shouldResolve = true;
                    }
                    break;
            }
            if (shouldResolve) {
                alert.autoResolved = true;
                resolvedAlerts.push(id);
            }
        });
        if (resolvedAlerts.length > 0) {
            this.notifyClients('alerts-resolved', resolvedAlerts);
            this.emit('alerts-resolved', resolvedAlerts);
        }
    }
    /**
     * Notify WebSocket clients of updates
     */
    notifyClients(event, data) {
        const message = JSON.stringify({ event, data, timestamp: new Date() });
        this.websocketClients.forEach(client => {
            try {
                if (client.readyState === 1) { // WebSocket.OPEN
                    client.send(message);
                }
            }
            catch (error) {
                // Remove failed client
                this.websocketClients.delete(client);
            }
        });
    }
    /**
     * Get current dashboard metrics
     */
    getCurrentMetrics() {
        return { ...this.metrics };
    }
    /**
     * Get active alerts
     */
    getActiveAlerts() {
        return Array.from(this.alerts.values()).filter(alert => !alert.acknowledged && !alert.autoResolved);
    }
    /**
     * Get dashboard layout
     */
    getLayout(layoutId) {
        const id = layoutId || this.currentLayout;
        return this.layouts.get(id);
    }
    /**
     * Set active layout
     */
    setLayout(layoutId) {
        if (this.layouts.has(layoutId)) {
            this.currentLayout = layoutId;
            this.emit('layout-changed', layoutId);
        }
    }
    /**
     * Acknowledge alert
     */
    acknowledgeAlert(alertId) {
        const alert = this.alerts.get(alertId);
        if (alert) {
            alert.acknowledged = true;
            this.notifyClients('alert-acknowledged', alertId);
            this.emit('alert-acknowledged', alertId);
        }
    }
    /**
     * Add WebSocket client
     */
    addWebSocketClient(client) {
        this.websocketClients.add(client);
        // Send initial data
        client.send(JSON.stringify({
            event: 'initial-data',
            data: {
                metrics: this.metrics,
                alerts: this.getActiveAlerts(),
                layout: this.getLayout()
            },
            timestamp: new Date()
        }));
    }
    /**
     * Remove WebSocket client
     */
    removeWebSocketClient(client) {
        this.websocketClients.delete(client);
    }
    /**
     * Export dashboard data
     */
    exportData(format = 'json') {
        if (format === 'json') {
            return {
                metrics: this.metrics,
                alerts: Array.from(this.alerts.values()),
                layouts: Array.from(this.layouts.values()),
                exportedAt: new Date()
            };
        }
        else {
            // CSV export would be implemented here
            return 'CSV export not implemented';
        }
    }
    /**
     * Clean up resources
     */
    destroy() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
        this.websocketClients.clear();
        this.removeAllListeners();
    }
}
exports.QualityDashboard = QualityDashboard;
//# sourceMappingURL=QualityDashboard.js.map