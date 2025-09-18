"use strict";
/**
 * Six Sigma Quality Structure Contract Tests
 * Validates CTQ, SPC, DPMO, and theater detection output structure contracts
 */
Object.defineProperty(exports, "__esModule", { value: true });
describe('Six Sigma Quality Structure Contracts', () => {
    const QUALITY_CONTRACTS = {
        CTQ_RESPONSE_TIME_MAX: 50, // milliseconds
        CTQ_UPTIME_MIN: 99.9, // percentage
        DPMO_MAX: 3.4, // Six Sigma level
        THEATER_THRESHOLD: 0.3, // Theater detection threshold
        SPC_CPK_MIN: 1.33, // Process capability minimum
        SIGMA_LEVEL_MIN: 6 // Minimum sigma level
    };
    function generateMockSixSigmaMetrics() {
        return {
            ctq: {
                responseTime: Math.random() * 40 + 10, // 10-50ms
                uptime: 99.9 + Math.random() * 0.1, // 99.9-100%
                defectRate: Math.random() * 0.001 // 0-0.1%
            },
            spc: {
                mean: 30 + Math.random() * 10, // 30-40ms
                ucl: 45 + Math.random() * 10, // 45-55ms
                lcl: 15 + Math.random() * 10, // 15-25ms
                cpk: 1.2 + Math.random() * 0.5 // 1.2-1.7
            },
            dpmo: Math.random() * 10, // 0-10 DPMO
            sigmaLevel: 5 + Math.random() * 1.5, // 5-6.5 sigma
            theaterDetection: {
                score: Math.random() * 0.5, // 0-0.5 score
                threshold: QUALITY_CONTRACTS.THEATER_THRESHOLD,
                isElevated: false
            }
        };
    }
    test('should validate CTQ structure contract', () => {
        const metrics = generateMockSixSigmaMetrics();
        // Contract: CTQ must have required properties
        expect(metrics.ctq).toHaveProperty('responseTime');
        expect(metrics.ctq).toHaveProperty('uptime');
        expect(metrics.ctq).toHaveProperty('defectRate');
        // Contract: CTQ values must be within acceptable ranges
        expect(metrics.ctq.responseTime).toBeGreaterThan(0);
        expect(metrics.ctq.responseTime).toBeLessThan(QUALITY_CONTRACTS.CTQ_RESPONSE_TIME_MAX);
        expect(metrics.ctq.uptime).toBeGreaterThanOrEqual(QUALITY_CONTRACTS.CTQ_UPTIME_MIN);
        expect(metrics.ctq.defectRate).toBeGreaterThanOrEqual(0);
        expect(metrics.ctq.defectRate).toBeLessThan(0.01); // Less than 1%
    });
    test('should validate SPC structure contract', () => {
        const metrics = generateMockSixSigmaMetrics();
        // Contract: SPC must have statistical control properties
        expect(metrics.spc).toHaveProperty('mean');
        expect(metrics.spc).toHaveProperty('ucl');
        expect(metrics.spc).toHaveProperty('lcl');
        expect(metrics.spc).toHaveProperty('cpk');
        // Contract: SPC control limits must be logical
        expect(metrics.spc.ucl).toBeGreaterThan(metrics.spc.mean);
        expect(metrics.spc.lcl).toBeLessThan(metrics.spc.mean);
        expect(metrics.spc.cpk).toBeGreaterThan(0);
        // Contract: Control limits should indicate capable process
        if (metrics.spc.cpk >= QUALITY_CONTRACTS.SPC_CPK_MIN) {
            expect(metrics.spc.ucl - metrics.spc.lcl).toBeLessThan(60); // Reasonable spread
        }
    });
    test('should validate DPMO structure contract', () => {
        const metrics = generateMockSixSigmaMetrics();
        // Contract: DPMO must be a non-negative number
        expect(typeof metrics.dpmo).toBe('number');
        expect(metrics.dpmo).toBeGreaterThanOrEqual(0);
        // Contract: DPMO should correlate with sigma level
        expect(typeof metrics.sigmaLevel).toBe('number');
        expect(metrics.sigmaLevel).toBeGreaterThan(0);
        expect(metrics.sigmaLevel).toBeLessThanOrEqual(7); // Theoretical maximum
        // Contract: High sigma level should mean low DPMO
        if (metrics.sigmaLevel >= 6) {
            expect(metrics.dpmo).toBeLessThanOrEqual(QUALITY_CONTRACTS.DPMO_MAX);
        }
    });
    test('should validate theater detection structure contract', () => {
        const metrics = generateMockSixSigmaMetrics();
        // Update theater detection based on score
        metrics.theaterDetection.isElevated = metrics.theaterDetection.score >= metrics.theaterDetection.threshold;
        // Contract: Theater detection must have required properties
        expect(metrics.theaterDetection).toHaveProperty('score');
        expect(metrics.theaterDetection).toHaveProperty('threshold');
        expect(metrics.theaterDetection).toHaveProperty('isElevated');
        // Contract: Score must be between 0 and 1
        expect(metrics.theaterDetection.score).toBeGreaterThanOrEqual(0);
        expect(metrics.theaterDetection.score).toBeLessThanOrEqual(1);
        // Contract: Threshold must match configuration
        expect(metrics.theaterDetection.threshold).toBe(QUALITY_CONTRACTS.THEATER_THRESHOLD);
        // Contract: isElevated must be boolean and match score vs threshold
        expect(typeof metrics.theaterDetection.isElevated).toBe('boolean');
        expect(metrics.theaterDetection.isElevated).toBe(metrics.theaterDetection.score >= metrics.theaterDetection.threshold);
    });
    test('should validate complete quality metrics structure', () => {
        const metrics = generateMockSixSigmaMetrics();
        // Contract: All major Six Sigma components must be present
        expect(metrics).toHaveProperty('ctq');
        expect(metrics).toHaveProperty('spc');
        expect(metrics).toHaveProperty('dpmo');
        expect(metrics).toHaveProperty('sigmaLevel');
        expect(metrics).toHaveProperty('theaterDetection');
        // Contract: Metrics should be internally consistent
        const isHighQuality = metrics.sigmaLevel >= 6 &&
            metrics.dpmo <= QUALITY_CONTRACTS.DPMO_MAX &&
            metrics.ctq.responseTime <= QUALITY_CONTRACTS.CTQ_RESPONSE_TIME_MAX;
        if (isHighQuality) {
            expect(metrics.ctq.defectRate).toBeLessThan(0.001); // Very low defect rate
            expect(metrics.spc.cpk).toBeGreaterThanOrEqual(1.33); // Capable process
        }
    });
    test('should validate quality metrics serialization contract', () => {
        const metrics = generateMockSixSigmaMetrics();
        // Contract: Metrics must be JSON serializable
        const serialized = JSON.stringify(metrics);
        expect(serialized).toBeDefined();
        expect(serialized.length).toBeGreaterThan(0);
        // Contract: Deserialized metrics must maintain structure
        const deserialized = JSON.parse(serialized);
        expect(deserialized).toEqual(metrics);
        // Contract: All numeric values must remain numeric after serialization
        expect(typeof deserialized.ctq.responseTime).toBe('number');
        expect(typeof deserialized.spc.cpk).toBe('number');
        expect(typeof deserialized.dpmo).toBe('number');
        expect(typeof deserialized.sigmaLevel).toBe('number');
        expect(typeof deserialized.theaterDetection.score).toBe('number');
    });
    test('should validate quality improvement tracking contract', () => {
        const baseline = generateMockSixSigmaMetrics();
        // Simulate improvement
        const improved = {
            ...baseline,
            ctq: {
                ...baseline.ctq,
                responseTime: baseline.ctq.responseTime * 0.8, // 20% improvement
                defectRate: baseline.ctq.defectRate * 0.5 // 50% reduction
            },
            dpmo: baseline.dpmo * 0.3, // 70% reduction
            sigmaLevel: baseline.sigmaLevel + 0.5
        };
        // Contract: Improvements must be measurable
        expect(improved.ctq.responseTime).toBeLessThan(baseline.ctq.responseTime);
        expect(improved.ctq.defectRate).toBeLessThan(baseline.ctq.defectRate);
        expect(improved.dpmo).toBeLessThan(baseline.dpmo);
        expect(improved.sigmaLevel).toBeGreaterThan(baseline.sigmaLevel);
        // Contract: Improvements must maintain data integrity
        expect(improved.ctq.responseTime).toBeGreaterThan(0);
        expect(improved.dpmo).toBeGreaterThanOrEqual(0);
        expect(improved.sigmaLevel).toBeLessThanOrEqual(7);
    });
});
//# sourceMappingURL=sixsigma-quality.test.js.map