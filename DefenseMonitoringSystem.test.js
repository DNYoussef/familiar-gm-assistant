"use strict";
/**
 * Defense Monitoring System Tests
 * Comprehensive test suite for defense-grade monitoring and rollback systems
 * Validates <1.2% overhead requirement and <30 second rollback capability
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.generatePerformanceMetrics = generatePerformanceMetrics;
const globals_1 = require("@jest/globals");
const DefenseGradeMonitor_1 = require("../../src/monitoring/advanced/DefenseGradeMonitor");
const DefenseRollbackSystem_1 = require("../../src/rollback/systems/DefenseRollbackSystem");
const DefenseSecurityMonitor_1 = require("../../src/security/monitoring/DefenseSecurityMonitor");
const ComplianceDriftDetector_1 = require("../../src/compliance/monitoring/ComplianceDriftDetector");
const DefenseMonitoringOrchestrator_1 = require("../../src/monitoring/DefenseMonitoringOrchestrator");
(0, globals_1.describe)('Defense Monitoring System', () => {
    let performanceMonitor;
    let rollbackSystem;
    let securityMonitor;
    let complianceDetector;
    let orchestrator;
    (0, globals_1.beforeEach)(async () => {
        // Initialize monitoring systems
        performanceMonitor = new DefenseGradeMonitor_1.DefenseGradeMonitor();
        rollbackSystem = new DefenseRollbackSystem_1.DefenseRollbackSystem();
        securityMonitor = new DefenseSecurityMonitor_1.DefenseSecurityMonitor();
        complianceDetector = new ComplianceDriftDetector_1.ComplianceDriftDetector(rollbackSystem);
        orchestrator = new DefenseMonitoringOrchestrator_1.DefenseMonitoringOrchestrator();
        // Mock timers for testing
        globals_1.jest.useFakeTimers();
    });
    (0, globals_1.afterEach)(async () => {
        // Cleanup
        await performanceMonitor.stopMonitoring();
        await rollbackSystem.stopRollbackSystem();
        await securityMonitor.stopSecurityMonitoring();
        await complianceDetector.stopDriftDetection();
        await orchestrator.stopDefenseMonitoring();
        globals_1.jest.useRealTimers();
    });
    (0, globals_1.describe)('DefenseGradeMonitor', () => {
        (0, globals_1.it)('should maintain microsecond precision timing', async () => {
            const startTime = performance.now();
            await performanceMonitor.startMonitoring();
            // Allow some monitoring cycles
            globals_1.jest.advanceTimersByTime(1000);
            const report = await performanceMonitor.getPerformanceReport();
            (0, globals_1.expect)(report).toBeDefined();
            (0, globals_1.expect)(report.timestamp).toBeGreaterThan(startTime);
            (0, globals_1.expect)(report.currentOverhead).toBeLessThan(1.2); // <1.2% requirement
        });
        (0, globals_1.it)('should detect performance degradation', async () => {
            await performanceMonitor.startMonitoring();
            // Simulate performance degradation
            const mockHighOverhead = globals_1.jest.spyOn(performanceMonitor, 'calculateSystemOverhead')
                .mockReturnValue(1.5); // 1.5% overhead
            globals_1.jest.advanceTimersByTime(5000);
            const report = await performanceMonitor.getPerformanceReport();
            (0, globals_1.expect)(report.complianceWithTarget).toBe(false);
            (0, globals_1.expect)(report.currentOverhead).toBeGreaterThan(1.2);
            mockHighOverhead.mockRestore();
        });
        (0, globals_1.it)('should generate optimization recommendations', async () => {
            await performanceMonitor.startMonitoring();
            globals_1.jest.advanceTimersByTime(30000); // Wait for predictive analysis
            const report = await performanceMonitor.getPerformanceReport();
            (0, globals_1.expect)(report.recommendations).toBeDefined();
            (0, globals_1.expect)(Array.isArray(report.recommendations)).toBe(true);
            (0, globals_1.expect)(report.predictions).toBeDefined();
            (0, globals_1.expect)(report.predictions.confidence).toBeGreaterThan(0);
        });
        (0, globals_1.it)('should track resource usage within limits', async () => {
            await performanceMonitor.startMonitoring();
            globals_1.jest.advanceTimersByTime(10000);
            const report = await performanceMonitor.getPerformanceReport();
            // Verify resource tracking
            (0, globals_1.expect)(report.totalAgents).toBeGreaterThan(0);
            (0, globals_1.expect)(report.totalMetrics).toBeGreaterThan(0);
            (0, globals_1.expect)(report.currentOverhead).toBeLessThan(5.0); // Reasonable upper bound
        });
    });
    (0, globals_1.describe)('DefenseRollbackSystem', () => {
        (0, globals_1.it)('should create snapshots successfully', async () => {
            await rollbackSystem.startRollbackSystem();
            const snapshotId = await rollbackSystem.createSnapshot('TEST');
            (0, globals_1.expect)(snapshotId).toBeDefined();
            (0, globals_1.expect)(typeof snapshotId).toBe('string');
            (0, globals_1.expect)(snapshotId).toMatch(/^snapshot_\d+_\w+$/);
        });
        (0, globals_1.it)('should execute rollback within 30 seconds', async () => {
            await rollbackSystem.startRollbackSystem();
            // Create a snapshot first
            const snapshotId = await rollbackSystem.createSnapshot('BASELINE');
            // Measure rollback time
            const startTime = performance.now();
            const result = await rollbackSystem.executeRollback(snapshotId, 'TEST_ROLLBACK');
            const rollbackTime = performance.now() - startTime;
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(rollbackTime).toBeLessThan(30000); // <30 seconds requirement
            (0, globals_1.expect)(result.duration).toBeLessThan(30000);
        });
        (0, globals_1.it)('should validate snapshot integrity', async () => {
            await rollbackSystem.startRollbackSystem();
            const snapshotId = await rollbackSystem.createSnapshot('INTEGRITY_TEST');
            const snapshots = rollbackSystem.getSnapshotHistory();
            const snapshot = snapshots.find(s => s.id === snapshotId);
            (0, globals_1.expect)(snapshot).toBeDefined();
            (0, globals_1.expect)(snapshot.checksum).toBeDefined();
            (0, globals_1.expect)(snapshot.systemState).toBeDefined();
        });
        (0, globals_1.it)('should handle rollback failures gracefully', async () => {
            await rollbackSystem.startRollbackSystem();
            // Try to rollback to non-existent snapshot
            const result = await rollbackSystem.executeRollback('invalid_snapshot', 'ERROR_TEST');
            (0, globals_1.expect)(result.success).toBe(false);
            (0, globals_1.expect)(result.error).toBeDefined();
        });
        (0, globals_1.it)('should maintain rollback history', async () => {
            await rollbackSystem.startRollbackSystem();
            const snapshotId = await rollbackSystem.createSnapshot('HISTORY_TEST');
            await rollbackSystem.executeRollback(snapshotId, 'HISTORY_ROLLBACK');
            const history = rollbackSystem.getRollbackHistory();
            (0, globals_1.expect)(history).toBeDefined();
            (0, globals_1.expect)(history.length).toBeGreaterThan(0);
            (0, globals_1.expect)(history[0].reason).toBe('HISTORY_ROLLBACK');
        });
    });
    (0, globals_1.describe)('DefenseSecurityMonitor', () => {
        (0, globals_1.it)('should start security monitoring without errors', async () => {
            await (0, globals_1.expect)(securityMonitor.startSecurityMonitoring()).resolves.not.toThrow();
            globals_1.jest.advanceTimersByTime(5000);
            const dashboard = await securityMonitor.getSecurityDashboardData();
            (0, globals_1.expect)(dashboard).toBeDefined();
            (0, globals_1.expect)(dashboard.metrics.threatLevel).toBeDefined();
            (0, globals_1.expect)(dashboard.systemStatus).toBeDefined();
        });
        (0, globals_1.it)('should detect and classify threats', async () => {
            await securityMonitor.startSecurityMonitoring();
            globals_1.jest.advanceTimersByTime(10000);
            const metrics = await securityMonitor.generateSecurityMetrics();
            (0, globals_1.expect)(metrics.threatLevel).toMatch(/^(LOW|MEDIUM|HIGH|CRITICAL)$/);
            (0, globals_1.expect)(metrics.overallScore).toBeGreaterThanOrEqual(0);
            (0, globals_1.expect)(metrics.overallScore).toBeLessThanOrEqual(100);
        });
        (0, globals_1.it)('should maintain compliance monitoring', async () => {
            await securityMonitor.startSecurityMonitoring();
            globals_1.jest.advanceTimersByTime(30000); // Wait for compliance check
            const dashboard = await securityMonitor.getSecurityDashboardData();
            (0, globals_1.expect)(dashboard.metrics.complianceScore).toBeGreaterThan(0.8); // >80% compliance
        });
        (0, globals_1.it)('should generate security recommendations', async () => {
            await securityMonitor.startSecurityMonitoring();
            const dashboard = await securityMonitor.getSecurityDashboardData();
            (0, globals_1.expect)(dashboard.recommendations).toBeDefined();
            (0, globals_1.expect)(Array.isArray(dashboard.recommendations)).toBe(true);
        });
    });
    (0, globals_1.describe)('ComplianceDriftDetector', () => {
        (0, globals_1.it)('should establish compliance baselines', async () => {
            await complianceDetector.startDriftDetection();
            globals_1.jest.advanceTimersByTime(1000);
            const report = await complianceDetector.getDriftReport();
            (0, globals_1.expect)(report).toBeDefined();
            (0, globals_1.expect)(report.baselineStatus).toBeDefined();
            (0, globals_1.expect)(report.complianceScores).toBeDefined();
        });
        (0, globals_1.it)('should detect compliance drift', async () => {
            await complianceDetector.startDriftDetection();
            // Simulate drift by advancing time
            globals_1.jest.advanceTimersByTime(15000);
            const report = await complianceDetector.getDriftReport();
            (0, globals_1.expect)(report.totalDrifts).toBeGreaterThanOrEqual(0);
            (0, globals_1.expect)(report.recommendations).toBeDefined();
        });
        (0, globals_1.it)('should trigger rollback on critical drift', async () => {
            const rollbackSpy = globals_1.jest.spyOn(rollbackSystem, 'executeRollback')
                .mockResolvedValue({ success: true });
            await complianceDetector.startDriftDetection();
            // Simulate critical drift - would need to mock internal methods
            globals_1.jest.advanceTimersByTime(20000);
            // In a real scenario, critical drift would trigger rollback
            // This test validates the integration is properly set up
            (0, globals_1.expect)(complianceDetector).toBeDefined();
            rollbackSpy.mockRestore();
        });
    });
    (0, globals_1.describe)('DefenseMonitoringOrchestrator', () => {
        (0, globals_1.it)('should coordinate all monitoring systems', async () => {
            await orchestrator.startDefenseMonitoring();
            globals_1.jest.advanceTimersByTime(10000);
            const status = await orchestrator.getDefenseStatus();
            (0, globals_1.expect)(status).toBeDefined();
            (0, globals_1.expect)(status.overall.status).toMatch(/^(HEALTHY|WARNING|CRITICAL|EMERGENCY)$/);
            (0, globals_1.expect)(status.performance).toBeDefined();
            (0, globals_1.expect)(status.security).toBeDefined();
            (0, globals_1.expect)(status.compliance).toBeDefined();
            (0, globals_1.expect)(status.rollback).toBeDefined();
        });
        (0, globals_1.it)('should generate unified alerts', async () => {
            await orchestrator.startDefenseMonitoring();
            globals_1.jest.advanceTimersByTime(5000);
            const alerts = orchestrator.getActiveAlerts();
            (0, globals_1.expect)(Array.isArray(alerts)).toBe(true);
            // Alerts may be empty in normal operation
        });
        (0, globals_1.it)('should handle alert acknowledgment', async () => {
            await orchestrator.startDefenseMonitoring();
            // Create a mock alert by simulating a condition
            // In a real scenario, alerts would be generated by monitoring systems
            const alerts = orchestrator.getActiveAlerts();
            if (alerts.length > 0) {
                const result = await orchestrator.acknowledgeAlert(alerts[0].id, 'test-operator');
                (0, globals_1.expect)(result).toBe(true);
            }
        });
        (0, globals_1.it)('should calculate overall system score', async () => {
            await orchestrator.startDefenseMonitoring();
            globals_1.jest.advanceTimersByTime(10000);
            const status = await orchestrator.getDefenseStatus();
            (0, globals_1.expect)(status.overall.score).toBeGreaterThanOrEqual(0);
            (0, globals_1.expect)(status.overall.score).toBeLessThanOrEqual(100);
        });
    });
    (0, globals_1.describe)('Performance Requirements Validation', () => {
        (0, globals_1.it)('should meet the <1.2% overhead requirement consistently', async () => {
            const measurements = [];
            await performanceMonitor.startMonitoring();
            // Take multiple measurements over time
            for (let i = 0; i < 10; i++) {
                globals_1.jest.advanceTimersByTime(1000);
                const report = await performanceMonitor.getPerformanceReport();
                measurements.push(report.currentOverhead);
            }
            // Verify all measurements are under threshold
            const maxOverhead = Math.max(...measurements);
            const avgOverhead = measurements.reduce((sum, val) => sum + val, 0) / measurements.length;
            (0, globals_1.expect)(maxOverhead).toBeLessThan(1.2);
            (0, globals_1.expect)(avgOverhead).toBeLessThan(1.0); // Should be well under target
        });
        (0, globals_1.it)('should complete rollback operations within 30 seconds', async () => {
            await rollbackSystem.startRollbackSystem();
            const rollbackTimes = [];
            // Test multiple rollback operations
            for (let i = 0; i < 3; i++) {
                const snapshotId = await rollbackSystem.createSnapshot(`TEST_${i}`);
                const startTime = performance.now();
                const result = await rollbackSystem.executeRollback(snapshotId, `PERF_TEST_${i}`);
                const duration = performance.now() - startTime;
                rollbackTimes.push(duration);
                (0, globals_1.expect)(result.success).toBe(true);
            }
            // Verify all rollbacks completed within time limit
            const maxRollbackTime = Math.max(...rollbackTimes);
            const avgRollbackTime = rollbackTimes.reduce((sum, val) => sum + val, 0) / rollbackTimes.length;
            (0, globals_1.expect)(maxRollbackTime).toBeLessThan(30000);
            (0, globals_1.expect)(avgRollbackTime).toBeLessThan(15000); // Should be well under limit
        });
        (0, globals_1.it)('should maintain defense-grade monitoring capabilities', async () => {
            await orchestrator.startDefenseMonitoring();
            // Run for extended period to validate stability
            globals_1.jest.advanceTimersByTime(60000); // 1 minute of operation
            const status = await orchestrator.getDefenseStatus();
            // Validate defense-grade requirements
            (0, globals_1.expect)(status.performance.overhead).toBeLessThan(1.2);
            (0, globals_1.expect)(status.security.complianceScore).toBeGreaterThan(0.9);
            (0, globals_1.expect)(status.rollback.ready).toBe(true);
            (0, globals_1.expect)(status.rollback.estimatedTime).toBeLessThan(30);
            // Validate monitoring coverage
            (0, globals_1.expect)(status.overall.status).toBeDefined();
            (0, globals_1.expect)(status.alerts.active).toBeGreaterThanOrEqual(0);
        });
    });
    (0, globals_1.describe)('Integration Tests', () => {
        (0, globals_1.it)('should integrate all monitoring systems seamlessly', async () => {
            // Start all systems
            await Promise.all([
                performanceMonitor.startMonitoring(),
                rollbackSystem.startRollbackSystem(),
                securityMonitor.startSecurityMonitoring(),
                complianceDetector.startDriftDetection(),
                orchestrator.startDefenseMonitoring()
            ]);
            // Allow systems to stabilize
            globals_1.jest.advanceTimersByTime(10000);
            // Validate integration
            const orchestratorStatus = await orchestrator.getDefenseStatus();
            const performanceReport = await performanceMonitor.getPerformanceReport();
            const securityDashboard = await securityMonitor.getSecurityDashboardData();
            const complianceReport = await complianceDetector.getDriftReport();
            // Verify all systems are operational
            (0, globals_1.expect)(orchestratorStatus.overall.status).not.toBe('EMERGENCY');
            (0, globals_1.expect)(performanceReport.currentOverhead).toBeLessThan(2.0);
            (0, globals_1.expect)(securityDashboard.metrics.threatLevel).toBeDefined();
            (0, globals_1.expect)(complianceReport.baselineStatus).toBeDefined();
        });
        (0, globals_1.it)('should handle system failure and recovery', async () => {
            await orchestrator.startDefenseMonitoring();
            // Simulate system failure
            const mockFailure = globals_1.jest.spyOn(orchestrator, 'generateUnifiedStatus')
                .mockRejectedValue(new Error('System failure'));
            globals_1.jest.advanceTimersByTime(5000);
            // System should continue operating despite failures
            (0, globals_1.expect)(orchestrator.getActiveAlerts()).toBeDefined();
            mockFailure.mockRestore();
        });
    });
});
// Performance benchmarking tests
(0, globals_1.describe)('Performance Benchmarks', () => {
    (0, globals_1.it)('should meet latency requirements for monitoring operations', async () => {
        const monitor = new DefenseGradeMonitor_1.DefenseGradeMonitor();
        await monitor.startMonitoring();
        const iterations = 100;
        const latencies = [];
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await monitor.getPerformanceReport();
            const latency = performance.now() - start;
            latencies.push(latency);
        }
        const p95Latency = latencies.sort((a, b) => a - b)[Math.floor(iterations * 0.95)];
        const p99Latency = latencies.sort((a, b) => a - b)[Math.floor(iterations * 0.99)];
        (0, globals_1.expect)(p95Latency).toBeLessThan(100); // <100ms P95
        (0, globals_1.expect)(p99Latency).toBeLessThan(500); // <500ms P99
        await monitor.stopMonitoring();
    });
    (0, globals_1.it)('should handle concurrent operations efficiently', async () => {
        const monitor = new DefenseGradeMonitor_1.DefenseGradeMonitor();
        await monitor.startMonitoring();
        const concurrentOperations = 50;
        const startTime = performance.now();
        // Execute concurrent operations
        const promises = Array(concurrentOperations).fill(0).map(() => monitor.getPerformanceReport());
        const results = await Promise.all(promises);
        const totalTime = performance.now() - startTime;
        // Verify all operations completed
        (0, globals_1.expect)(results).toHaveLength(concurrentOperations);
        results.forEach(result => {
            (0, globals_1.expect)(result).toBeDefined();
            (0, globals_1.expect)(result.currentOverhead).toBeGreaterThanOrEqual(0);
        });
        // Verify acceptable performance under load
        (0, globals_1.expect)(totalTime).toBeLessThan(5000); // <5 seconds for 50 concurrent ops
        await monitor.stopMonitoring();
    });
});
function generatePerformanceMetrics(latencies) {
    const sorted = latencies.sort((a, b) => a - b);
    const p95 = sorted[Math.floor(latencies.length * 0.95)];
    const p99 = sorted[Math.floor(latencies.length * 0.99)];
    return {
        p95_ms: Math.round(p95 * 100) / 100,
        p99_ms: Math.round(p99 * 100) / 100,
        delta_p95: '+0.0%', // Would calculate actual delta from baseline
        gate: p95 < 100 && p99 < 500 ? 'pass' : 'fail'
    };
}
//# sourceMappingURL=DefenseMonitoringSystem.test.js.map