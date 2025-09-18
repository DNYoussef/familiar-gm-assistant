"use strict";
/**
 * CI/CD Performance Monitor
 * Phase 4 Step 8: Production Performance Monitoring Framework
 *
 * Real-time monitoring and alerting for CI/CD system performance
 * with <2% overhead constraint enforcement.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.CICDPerformanceMonitor = void 0;
const events_1 = require("events");
const fs = __importStar(require("fs/promises"));
const path = __importStar(require("path"));
class CICDPerformanceMonitor extends events_1.EventEmitter {
    constructor(config) {
        super();
        this.isMonitoring = false;
        this.metricsHistory = new Map();
        this.activeAlerts = new Map();
        this.monitoringInterval = null;
        this.baselineMetrics = new Map();
        this.config = config;
        this.trendAnalyzer = new TrendAnalyzer();
        this.initializeMonitoring();
    }
    /**
     * Start performance monitoring
     */
    async startMonitoring() {
        if (this.isMonitoring) {
            console.log('Monitoring already active');
            return;
        }
        console.log('[SEARCH] Starting CI/CD performance monitoring...');
        console.log(`   Overhead threshold: <${this.config.overheadThreshold}%`);
        console.log(`   Sampling interval: ${this.config.samplingInterval}ms`);
        console.log(`   Monitoring domains: ${this.config.domains.join(', ')}`);
        // Establish baseline metrics
        await this.establishBaseline();
        // Start monitoring loop
        this.isMonitoring = true;
        this.monitoringInterval = setInterval(() => this.collectMetrics(), this.config.samplingInterval);
        // Start trend analysis if enabled
        if (this.config.enableTrendAnalysis) {
            this.startTrendAnalysis();
        }
        this.emit('monitoring-started', {
            timestamp: new Date(),
            domains: this.config.domains,
            baseline: Object.fromEntries(this.baselineMetrics)
        });
        console.log('[OK] Performance monitoring active');
    }
    /**
     * Stop performance monitoring
     */
    async stopMonitoring() {
        if (!this.isMonitoring) {
            console.log('Monitoring not active');
            return this.generateSummary();
        }
        console.log(' Stopping CI/CD performance monitoring...');
        this.isMonitoring = false;
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }
        // Generate final summary
        const summary = this.generateSummary();
        // Save monitoring data
        await this.saveMonitoringData();
        this.emit('monitoring-stopped', {
            timestamp: new Date(),
            summary
        });
        console.log('[OK] Performance monitoring stopped');
        return summary;
    }
    /**
     * Get current performance status
     */
    getCurrentStatus() {
        const currentMetrics = new Map();
        // Get latest metrics for each domain
        for (const domain of this.config.domains) {
            const history = this.metricsHistory.get(domain) || [];
            if (history.length > 0) {
                currentMetrics.set(domain, history[history.length - 1]);
            }
        }
        const overallOverhead = this.calculateOverallOverhead(Array.from(currentMetrics.values()));
        const activeAlertsCount = this.activeAlerts.size;
        const complianceRate = this.calculateComplianceRate(Array.from(currentMetrics.values()));
        return {
            timestamp: new Date(),
            overallOverhead,
            overheadCompliant: overallOverhead <= this.config.overheadThreshold,
            activeAlerts: activeAlertsCount,
            complianceRate,
            domains: Object.fromEntries(Array.from(currentMetrics.entries()).map(([domain, sample]) => [
                domain,
                {
                    overhead: sample.overheadPercentage,
                    compliant: sample.compliance.overallScore >= 80,
                    lastUpdated: sample.timestamp
                }
            ]))
        };
    }
    /**
     * Initialize monitoring system
     */
    initializeMonitoring() {
        // Initialize metrics history for each domain
        for (const domain of this.config.domains) {
            this.metricsHistory.set(domain, []);
        }
        // Setup alert handlers
        this.on('alert-triggered', this.handleAlert.bind(this));
        this.on('alert-resolved', this.handleAlertResolution.bind(this));
    }
    /**
     * Establish baseline performance metrics
     */
    async establishBaseline() {
        console.log('[CHART] Establishing baseline metrics...');
        for (const domain of this.config.domains) {
            // Simulate baseline establishment
            const baselineMetrics = await this.captureBaselineMetrics(domain);
            this.baselineMetrics.set(domain, baselineMetrics);
            console.log(`   ${domain}: ${baselineMetrics.throughput.toFixed(1)} ops/sec, ${baselineMetrics.resourceUsage.memory.toFixed(1)} MB`);
        }
    }
    /**
     * Capture baseline metrics for domain
     */
    async captureBaselineMetrics(domain) {
        // Simulate baseline capture with realistic values
        return {
            throughput: 20 + Math.random() * 30, // 20-50 ops/sec
            latency: {
                mean: 50 + Math.random() * 50,
                p95: 100 + Math.random() * 100,
                p99: 200 + Math.random() * 200,
                max: 500 + Math.random() * 500
            },
            errorRate: Math.random() * 2, // 0-2%
            resourceUsage: {
                memory: 50 + Math.random() * 50, // 50-100 MB
                cpu: 10 + Math.random() * 20, // 10-30%
                network: Math.random() * 5, // 0-5 MB/s
                disk: Math.random() * 2 // 0-2 MB/s
            },
            activeOperations: Math.floor(Math.random() * 10)
        };
    }
    /**
     * Collect performance metrics
     */
    async collectMetrics() {
        for (const domain of this.config.domains) {
            try {
                const metrics = await this.captureDomainMetrics(domain);
                const baseline = this.baselineMetrics.get(domain);
                if (!baseline) {
                    console.warn(`No baseline found for domain: ${domain}`);
                    continue;
                }
                // Calculate overhead
                const overheadPercentage = this.calculateOverhead(metrics, baseline);
                // Validate compliance
                const compliance = this.validateCompliance(metrics, overheadPercentage);
                // Create sample
                const sample = {
                    timestamp: new Date(),
                    domain,
                    metrics,
                    overheadPercentage,
                    compliance
                };
                // Store sample
                this.storeSample(sample);
                // Check for alerts
                await this.checkAlerts(sample);
                // Emit monitoring data
                this.emit('metrics-collected', sample);
            }
            catch (error) {
                console.error(`Failed to collect metrics for ${domain}:`, error);
            }
        }
    }
    /**
     * Capture current metrics for domain
     */
    async captureDomainMetrics(domain) {
        // Simulate metric capture with realistic fluctuations
        const baseline = this.baselineMetrics.get(domain);
        const variance = 0.2; // 20% variance
        const fluctuation = () => 1 + (Math.random() - 0.5) * variance * 2;
        return {
            throughput: baseline.throughput * fluctuation(),
            latency: {
                mean: baseline.latency.mean * fluctuation(),
                p95: baseline.latency.p95 * fluctuation(),
                p99: baseline.latency.p99 * fluctuation(),
                max: baseline.latency.max * fluctuation()
            },
            errorRate: Math.max(0, baseline.errorRate * fluctuation()),
            resourceUsage: {
                memory: baseline.resourceUsage.memory * fluctuation(),
                cpu: Math.min(100, baseline.resourceUsage.cpu * fluctuation()),
                network: baseline.resourceUsage.network * fluctuation(),
                disk: baseline.resourceUsage.disk * fluctuation()
            },
            activeOperations: Math.floor(baseline.activeOperations * fluctuation())
        };
    }
    /**
     * Calculate performance overhead
     */
    calculateOverhead(current, baseline) {
        const memoryOverhead = ((current.resourceUsage.memory - baseline.resourceUsage.memory) / baseline.resourceUsage.memory) * 100;
        const cpuOverhead = ((current.resourceUsage.cpu - baseline.resourceUsage.cpu) / baseline.resourceUsage.cpu) * 100;
        const latencyOverhead = ((current.latency.mean - baseline.latency.mean) / baseline.latency.mean) * 100;
        // Calculate weighted average overhead
        const totalOverhead = (memoryOverhead * 0.4 + cpuOverhead * 0.4 + latencyOverhead * 0.2);
        return Math.max(0, totalOverhead);
    }
    /**
     * Validate performance compliance
     */
    validateCompliance(metrics, overhead) {
        const overheadCompliant = overhead <= this.config.overheadThreshold;
        const latencyCompliant = metrics.latency.p95 <= 1000; // 1 second
        const throughputCompliant = metrics.throughput >= 10; // Min 10 ops/sec
        const errorRateCompliant = metrics.errorRate <= 5; // Max 5%
        const compliances = [overheadCompliant, latencyCompliant, throughputCompliant, errorRateCompliant];
        const overallScore = (compliances.filter(c => c).length / compliances.length) * 100;
        return {
            overheadCompliant,
            latencyCompliant,
            throughputCompliant,
            errorRateCompliant,
            overallScore
        };
    }
    /**
     * Store metrics sample
     */
    storeSample(sample) {
        const history = this.metricsHistory.get(sample.domain) || [];
        history.push(sample);
        // Maintain retention period
        const cutoff = new Date(Date.now() - (this.config.retentionPeriod * 60 * 60 * 1000));
        const filtered = history.filter(s => s.timestamp > cutoff);
        this.metricsHistory.set(sample.domain, filtered);
    }
    /**
     * Check for performance alerts
     */
    async checkAlerts(sample) {
        const alerts = [];
        // Check overhead threshold
        if (sample.overheadPercentage >= this.config.alertThresholds.critical.overhead) {
            alerts.push(this.createAlert('critical', 'overhead', sample, `Critical overhead violation: ${sample.overheadPercentage.toFixed(2)}%`, 'Immediate investigation required. Check for resource leaks or inefficient operations.'));
        }
        else if (sample.overheadPercentage >= this.config.alertThresholds.warning.overhead) {
            alerts.push(this.createAlert('warning', 'overhead', sample, `Warning: High overhead detected: ${sample.overheadPercentage.toFixed(2)}%`, 'Monitor closely and consider optimization if trend continues.'));
        }
        // Check latency threshold
        if (sample.metrics.latency.p95 >= this.config.alertThresholds.critical.latency) {
            alerts.push(this.createAlert('critical', 'latency', sample, `Critical latency violation: ${sample.metrics.latency.p95.toFixed(1)}ms P95`, 'Check for bottlenecks in processing pipeline.'));
        }
        // Check error rate
        if (sample.metrics.errorRate >= this.config.alertThresholds.critical.errorRate) {
            alerts.push(this.createAlert('critical', 'error_rate', sample, `Critical error rate: ${sample.metrics.errorRate.toFixed(1)}%`, 'Investigate error sources and implement error handling improvements.'));
        }
        // Process alerts
        for (const alert of alerts) {
            await this.triggerAlert(alert);
        }
    }
    /**
     * Create performance alert
     */
    createAlert(severity, type, sample, message, recommendation) {
        const id = `${sample.domain}-${type}-${Date.now()}`;
        let currentValue;
        let threshold;
        switch (type) {
            case 'overhead':
                currentValue = sample.overheadPercentage;
                threshold = severity === 'critical' ?
                    this.config.alertThresholds.critical.overhead :
                    this.config.alertThresholds.warning.overhead;
                break;
            case 'latency':
                currentValue = sample.metrics.latency.p95;
                threshold = severity === 'critical' ?
                    this.config.alertThresholds.critical.latency :
                    this.config.alertThresholds.warning.latency;
                break;
            case 'error_rate':
                currentValue = sample.metrics.errorRate;
                threshold = severity === 'critical' ?
                    this.config.alertThresholds.critical.errorRate :
                    this.config.alertThresholds.warning.errorRate;
                break;
            default:
                currentValue = 0;
                threshold = 0;
        }
        return {
            id,
            timestamp: new Date(),
            domain: sample.domain,
            severity,
            type,
            message,
            currentValue,
            threshold,
            recommendation
        };
    }
    /**
     * Trigger performance alert
     */
    async triggerAlert(alert) {
        // Check if alert already exists
        const existingKey = `${alert.domain}-${alert.type}`;
        if (this.activeAlerts.has(existingKey)) {
            return; // Don't duplicate alerts
        }
        // Store alert
        this.activeAlerts.set(existingKey, alert);
        // Emit alert event
        this.emit('alert-triggered', alert);
        // Send notifications if enabled
        if (this.config.enableRealTimeAlerts) {
            await this.sendAlert(alert);
        }
        console.log(`[ALERT] ${alert.severity.toUpperCase()} ALERT: ${alert.message}`);
    }
    /**
     * Send alert notification
     */
    async sendAlert(alert) {
        // Simulate alert notification
        const notification = {
            timestamp: alert.timestamp,
            subject: `CI/CD Performance Alert - ${alert.domain}`,
            body: `${alert.message}\n\nRecommendation: ${alert.recommendation}\n\nCurrent: ${alert.currentValue.toFixed(2)}\nThreshold: ${alert.threshold}`,
            severity: alert.severity
        };
        // Would integrate with actual notification system
        this.emit('notification-sent', notification);
    }
    /**
     * Handle alert events
     */
    handleAlert(alert) {
        console.log(` Alert triggered: ${alert.id} - ${alert.message}`);
    }
    /**
     * Handle alert resolution
     */
    handleAlertResolution(alertId) {
        console.log(`[OK] Alert resolved: ${alertId}`);
    }
    /**
     * Start trend analysis
     */
    startTrendAnalysis() {
        setInterval(() => {
            for (const domain of this.config.domains) {
                const trend = this.trendAnalyzer.analyzeTrend(domain, this.metricsHistory.get(domain) || []);
                if (trend) {
                    this.emit('trend-analysis', trend);
                }
            }
        }, 300000); // Every 5 minutes
    }
    /**
     * Calculate overall system overhead
     */
    calculateOverallOverhead(samples) {
        if (samples.length === 0)
            return 0;
        return samples.reduce((sum, s) => sum + s.overheadPercentage, 0) / samples.length;
    }
    /**
     * Calculate overall compliance rate
     */
    calculateComplianceRate(samples) {
        if (samples.length === 0)
            return 100;
        return samples.reduce((sum, s) => sum + s.compliance.overallScore, 0) / samples.length;
    }
    /**
     * Generate monitoring summary
     */
    generateSummary() {
        const totalSamples = Array.from(this.metricsHistory.values()).reduce((sum, history) => sum + history.length, 0);
        const allSamples = Array.from(this.metricsHistory.values()).flat();
        const avgOverhead = allSamples.length > 0 ?
            allSamples.reduce((sum, s) => sum + s.overheadPercentage, 0) / allSamples.length : 0;
        const maxOverhead = allSamples.length > 0 ?
            Math.max(...allSamples.map(s => s.overheadPercentage)) : 0;
        const complianceRate = allSamples.length > 0 ?
            allSamples.reduce((sum, s) => sum + s.compliance.overallScore, 0) / allSamples.length : 100;
        return {
            monitoringPeriod: {
                start: allSamples.length > 0 ? allSamples[0].timestamp : new Date(),
                end: new Date(),
                duration: 0 // Would calculate actual duration
            },
            totalSamples,
            overheadStats: {
                average: avgOverhead,
                maximum: maxOverhead,
                violations: allSamples.filter(s => s.overheadPercentage > this.config.overheadThreshold).length
            },
            complianceRate,
            alertsSummary: {
                total: this.activeAlerts.size,
                critical: Array.from(this.activeAlerts.values()).filter(a => a.severity === 'critical').length,
                warning: Array.from(this.activeAlerts.values()).filter(a => a.severity === 'warning').length
            },
            domainSummaries: Object.fromEntries(this.config.domains.map(domain => [
                domain,
                this.generateDomainSummary(domain)
            ]))
        };
    }
    /**
     * Generate domain-specific summary
     */
    generateDomainSummary(domain) {
        const history = this.metricsHistory.get(domain) || [];
        if (history.length === 0) {
            return {
                samples: 0,
                averageOverhead: 0,
                maxOverhead: 0,
                complianceRate: 100,
                alerts: 0
            };
        }
        const avgOverhead = history.reduce((sum, s) => sum + s.overheadPercentage, 0) / history.length;
        const maxOverhead = Math.max(...history.map(s => s.overheadPercentage));
        const complianceRate = history.reduce((sum, s) => sum + s.compliance.overallScore, 0) / history.length;
        const alerts = Array.from(this.activeAlerts.values()).filter(a => a.domain === domain).length;
        return {
            samples: history.length,
            averageOverhead: avgOverhead,
            maxOverhead: maxOverhead,
            complianceRate: complianceRate,
            alerts: alerts
        };
    }
    /**
     * Save monitoring data
     */
    async saveMonitoringData() {
        try {
            const dataDir = path.join(process.cwd(), '.claude', '.artifacts', 'monitoring');
            await fs.mkdir(dataDir, { recursive: true });
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const dataFile = path.join(dataDir, `monitoring-data-${timestamp}.json`);
            const data = {
                config: this.config,
                baseline: Object.fromEntries(this.baselineMetrics),
                history: Object.fromEntries(this.metricsHistory),
                alerts: Array.from(this.activeAlerts.values()),
                summary: this.generateSummary()
            };
            await fs.writeFile(dataFile, JSON.stringify(data, null, 2), 'utf8');
            console.log(`[DISK] Monitoring data saved: ${dataFile}`);
        }
        catch (error) {
            console.error('Failed to save monitoring data:', error);
        }
    }
}
exports.CICDPerformanceMonitor = CICDPerformanceMonitor;
// Supporting classes
class TrendAnalyzer {
    analyzeTrend(domain, samples) {
        if (samples.length < 10)
            return null; // Need minimum samples for trend analysis
        // Simple trend analysis implementation
        const recentSamples = samples.slice(-10);
        const overheadTrend = this.calculateTrend(recentSamples.map(s => s.overheadPercentage));
        const latencyTrend = this.calculateTrend(recentSamples.map(s => s.metrics.latency.p95));
        return {
            domain,
            timespan: '10-sample window',
            trends: {
                overhead: overheadTrend,
                latency: latencyTrend,
                throughput: this.calculateTrend(recentSamples.map(s => s.metrics.throughput)),
                errorRate: this.calculateTrend(recentSamples.map(s => s.metrics.errorRate))
            },
            predictions: {
                overheadViolationProbability: overheadTrend.direction === 'increasing' ? 30 : 10,
                estimatedTimeToViolation: -1,
                recommendedActions: ['Monitor trend', 'Consider optimization if increasing']
            }
        };
    }
    calculateTrend(values) {
        if (values.length < 2) {
            return { direction: 'stable', rate: 0, confidence: 0 };
        }
        // Simple linear regression slope
        const n = values.length;
        const sumX = (n * (n - 1)) / 2;
        const sumY = values.reduce((sum, val) => sum + val, 0);
        const sumXY = values.reduce((sum, val, i) => sum + (i * val), 0);
        const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const direction = slope > 0.1 ? 'increasing' : slope < -0.1 ? 'decreasing' : 'stable';
        const rate = Math.abs(slope);
        const confidence = Math.min(100, rate * 100);
        return { direction, rate, confidence };
    }
}
//# sourceMappingURL=CICDPerformanceMonitor.js.map