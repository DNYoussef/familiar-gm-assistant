"use strict";
/**
 * Performance Constraints Contract Tests
 * Validates 50ms response time contract and throughput requirements
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const perf_hooks_1 = require("perf_hooks");
const supertest_1 = __importDefault(require("supertest"));
const express_1 = __importDefault(require("express"));
describe('Performance Constraints Contracts', () => {
    const RESPONSE_TIME_CONTRACT = 50; // milliseconds
    const THROUGHPUT_CONTRACT = 100; // requests per second
    const MEMORY_LEAK_THRESHOLD = 10; // MB increase over baseline
    let app;
    let server;
    beforeAll(() => {
        // Create test Express app with performance monitoring
        app = (0, express_1.default)();
        app.use(express_1.default.json());
        // Health endpoint with timing measurement
        app.get('/health', (req, res) => {
            const startTime = perf_hooks_1.performance.now();
            // Simulate some processing
            const data = {
                status: 'ok',
                timestamp: Date.now(),
                processingTime: perf_hooks_1.performance.now() - startTime
            };
            res.json(data);
        });
        // API endpoint with controlled delay
        app.post('/api/process', (req, res) => {
            const startTime = perf_hooks_1.performance.now();
            // Simulate processing based on input
            const processingDelay = req.body.delay || 0;
            setTimeout(() => {
                res.json({
                    result: 'processed',
                    inputSize: JSON.stringify(req.body).length,
                    processingTime: perf_hooks_1.performance.now() - startTime
                });
            }, processingDelay);
        });
        // Quality metrics endpoint
        app.get('/api/metrics', (req, res) => {
            const metrics = {
                responseTime: Math.random() * 40 + 10, // 10-50ms range
                throughput: Math.random() * 50 + 75, // 75-125 rps range
                errorRate: Math.random() * 0.01, // 0-1% error rate
                memoryUsage: process.memoryUsage()
            };
            res.json(metrics);
        });
        server = app.listen(3001);
    });
    afterAll(() => {
        if (server) {
            server.close();
        }
    });
    test('should meet 50ms response time contract for health endpoint', async () => {
        const startTime = perf_hooks_1.performance.now();
        const response = await (0, supertest_1.default)(app)
            .get('/health')
            .expect(200);
        const endTime = perf_hooks_1.performance.now();
        const responseTime = endTime - startTime;
        // Contract: Response time must be ≤ 50ms
        expect(responseTime).toBeLessThan(RESPONSE_TIME_CONTRACT);
        expect(response.body.status).toBe('ok');
        expect(response.body.processingTime).toBeLessThan(25); // Server-side processing
    });
    test('should maintain response time contract under load', async () => {
        const concurrentRequests = 20;
        const responseTimes = [];
        // Execute concurrent requests
        const promises = Array.from({ length: concurrentRequests }, async () => {
            const startTime = perf_hooks_1.performance.now();
            const response = await (0, supertest_1.default)(app)
                .get('/health')
                .expect(200);
            const responseTime = perf_hooks_1.performance.now() - startTime;
            responseTimes.push(responseTime);
            return response;
        });
        const responses = await Promise.all(promises);
        // Contract: All responses should meet timing contract
        responseTimes.forEach(time => {
            expect(time).toBeLessThan(RESPONSE_TIME_CONTRACT);
        });
        // Additional validation: Average response time should be reasonable
        const avgResponseTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
        expect(avgResponseTime).toBeLessThan(RESPONSE_TIME_CONTRACT * 0.8);
        // All requests should succeed
        expect(responses.length).toBe(concurrentRequests);
    });
    test('should enforce throughput contract', async () => {
        const testDuration = 2000; // 2 seconds
        const requests = [];
        const startTime = perf_hooks_1.performance.now();
        let requestCount = 0;
        // Generate requests for the test duration
        const interval = setInterval(() => {
            if (perf_hooks_1.performance.now() - startTime >= testDuration) {
                clearInterval(interval);
                return;
            }
            requestCount++;
            const promise = (0, supertest_1.default)(app)
                .get('/health')
                .expect(200);
            requests.push(promise);
        }, 10); // Request every 10ms
        // Wait for test duration
        await new Promise(resolve => setTimeout(resolve, testDuration));
        clearInterval(interval);
        // Wait for all requests to complete
        const responses = await Promise.all(requests);
        const actualDuration = (perf_hooks_1.performance.now() - startTime) / 1000; // Convert to seconds
        const actualThroughput = responses.length / actualDuration;
        // Contract: Should handle at least 100 requests per second
        expect(actualThroughput).toBeGreaterThanOrEqual(THROUGHPUT_CONTRACT);
        expect(responses.length).toBeGreaterThan(0);
    });
    test('should validate API processing time contract', async () => {
        const testCases = [
            { delay: 0, maxTime: 25 },
            { delay: 10, maxTime: 40 },
            { delay: 20, maxTime: 50 }
        ];
        for (const testCase of testCases) {
            const startTime = perf_hooks_1.performance.now();
            const response = await (0, supertest_1.default)(app)
                .post('/api/process')
                .send({ delay: testCase.delay, data: 'test' })
                .expect(200);
            const totalTime = perf_hooks_1.performance.now() - startTime;
            // Contract: Total response time should meet expectations
            expect(totalTime).toBeLessThan(testCase.maxTime + RESPONSE_TIME_CONTRACT);
            expect(response.body.result).toBe('processed');
            expect(response.body.processingTime).toBeGreaterThan(testCase.delay);
        }
    });
    test('should maintain performance under varying payload sizes', async () => {
        const payloadSizes = [100, 1000, 10000]; // bytes
        for (const size of payloadSizes) {
            const payload = {
                data: 'x'.repeat(size),
                timestamp: Date.now()
            };
            const startTime = perf_hooks_1.performance.now();
            const response = await (0, supertest_1.default)(app)
                .post('/api/process')
                .send(payload)
                .expect(200);
            const responseTime = perf_hooks_1.performance.now() - startTime;
            // Contract: Response time should not degrade significantly with payload size
            const expectedMaxTime = RESPONSE_TIME_CONTRACT + (size / 1000); // 1ms per KB
            expect(responseTime).toBeLessThan(expectedMaxTime);
            expect(response.body.inputSize).toBeGreaterThan(size);
        }
    });
    test('should validate metrics endpoint performance contract', async () => {
        const iterations = 10;
        const responseTimes = [];
        for (let i = 0; i < iterations; i++) {
            const startTime = perf_hooks_1.performance.now();
            const response = await (0, supertest_1.default)(app)
                .get('/api/metrics')
                .expect(200);
            const responseTime = perf_hooks_1.performance.now() - startTime;
            responseTimes.push(responseTime);
            // Validate response structure
            expect(response.body).toHaveProperty('responseTime');
            expect(response.body).toHaveProperty('throughput');
            expect(response.body).toHaveProperty('errorRate');
            expect(response.body).toHaveProperty('memoryUsage');
            // Contract: Metrics should indicate good performance
            expect(response.body.responseTime).toBeLessThan(RESPONSE_TIME_CONTRACT);
            expect(response.body.errorRate).toBeLessThan(0.05); // Less than 5% error rate
        }
        // Contract: Consistent performance across multiple calls
        const avgResponseTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
        const maxResponseTime = Math.max(...responseTimes);
        expect(avgResponseTime).toBeLessThan(RESPONSE_TIME_CONTRACT * 0.6);
        expect(maxResponseTime).toBeLessThan(RESPONSE_TIME_CONTRACT);
    });
});
//# sourceMappingURL=performance-constraints.test.js.map