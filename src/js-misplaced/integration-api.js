"use strict";
/**
 * Integration API Specification
 * RESTful, WebSocket, and GraphQL endpoints for linter integration system
 * MESH NODE AGENT: Integration Specialist for Linter Integration Architecture Swarm
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.IntegrationApiServer = void 0;
const events_1 = require("events");
const http_1 = require("http");
const ws_1 = require("ws");
const perf_hooks_1 = require("perf_hooks");
const crypto_1 = require("crypto");
/**
 * Integration API Server
 * Comprehensive API server supporting REST, WebSocket, and GraphQL
 */
class IntegrationApiServer extends events_1.EventEmitter {
    constructor(ingestionEngine, toolManager, correlationFramework, port = 3000) {
        super();
        this.ingestionEngine = ingestionEngine;
        this.toolManager = toolManager;
        this.correlationFramework = correlationFramework;
        this.apiKeys = new Map();
        this.rateLimits = new Map();
        this.activeConnections = new Map();
        this.subscriptions = new Map(); // channel -> connection IDs
        this.endpoints = new Map();
        this.middleware = [];
        this.isRunning = false;
        this.version = '1.0.0';
        this.port = port;
        this.httpServer = (0, http_1.createServer)();
        this.wsServer = new ws_1.WebSocketServer({ server: this.httpServer });
        this.initializeEndpoints();
        this.setupHttpHandlers();
        this.setupWebSocketHandlers();
        this.setupAuthentication();
        this.setupRateLimiting();
    }
    /**
     * Initialize all API endpoints
     */
    initializeEndpoints() {
        const endpoints = [
            // Health and Status Endpoints
            ['/health', {
                    path: '/health',
                    method: 'GET',
                    authentication: 'none',
                    rateLimit: 60,
                    timeout: 5000,
                    documentation: 'Check API health status',
                    examples: [{ response: { status: 'healthy', timestamp: Date.now() } }]
                }],
            ['/status', {
                    path: '/status',
                    method: 'GET',
                    authentication: 'required',
                    rateLimit: 30,
                    timeout: 10000,
                    documentation: 'Get comprehensive system status',
                    examples: []
                }],
            // Linter Execution Endpoints
            ['/api/v1/lint/execute', {
                    path: '/api/v1/lint/execute',
                    method: 'POST',
                    authentication: 'required',
                    rateLimit: 10,
                    timeout: 300000, // 5 minutes for linting
                    documentation: 'Execute linting on specified files',
                    examples: [{
                            request: { filePaths: ['src/file.ts'], tools: ['eslint', 'tsc'] },
                            response: { correlationId: 'abc123', status: 'started' }
                        }]
                }],
            ['/api/v1/lint/results/{correlationId}', {
                    path: '/api/v1/lint/results/{correlationId}',
                    method: 'GET',
                    authentication: 'required',
                    rateLimit: 30,
                    timeout: 30000,
                    documentation: 'Get linting results by correlation ID',
                    examples: []
                }],
            // Tool Management Endpoints
            ['/api/v1/tools', {
                    path: '/api/v1/tools',
                    method: 'GET',
                    authentication: 'required',
                    rateLimit: 60,
                    timeout: 10000,
                    documentation: 'List all registered linter tools',
                    examples: []
                }],
            ['/api/v1/tools/{toolId}/status', {
                    path: '/api/v1/tools/{toolId}/status',
                    method: 'GET',
                    authentication: 'required',
                    rateLimit: 60,
                    timeout: 10000,
                    documentation: 'Get status of specific tool',
                    examples: []
                }],
            ['/api/v1/tools/{toolId}/execute', {
                    path: '/api/v1/tools/{toolId}/execute',
                    method: 'POST',
                    authentication: 'required',
                    rateLimit: 15,
                    timeout: 300000,
                    documentation: 'Execute specific tool',
                    examples: []
                }],
            // Correlation Analysis Endpoints
            ['/api/v1/correlations/analyze', {
                    path: '/api/v1/correlations/analyze',
                    method: 'POST',
                    authentication: 'required',
                    rateLimit: 5,
                    timeout: 120000,
                    documentation: 'Perform correlation analysis on results',
                    examples: []
                }],
            ['/api/v1/correlations/clusters', {
                    path: '/api/v1/correlations/clusters',
                    method: 'GET',
                    authentication: 'required',
                    rateLimit: 30,
                    timeout: 30000,
                    documentation: 'Get violation clusters',
                    examples: []
                }],
            // Metrics and Monitoring Endpoints
            ['/api/v1/metrics/tools', {
                    path: '/api/v1/metrics/tools',
                    method: 'GET',
                    authentication: 'required',
                    rateLimit: 30,
                    timeout: 15000,
                    documentation: 'Get tool performance metrics',
                    examples: []
                }],
            ['/api/v1/metrics/correlations', {
                    path: '/api/v1/metrics/correlations',
                    method: 'GET',
                    authentication: 'required',
                    rateLimit: 30,
                    timeout: 15000,
                    documentation: 'Get correlation metrics',
                    examples: []
                }],
            // Configuration Endpoints
            ['/api/v1/config/tools/{toolId}', {
                    path: '/api/v1/config/tools/{toolId}',
                    method: 'PUT',
                    authentication: 'required',
                    rateLimit: 10,
                    timeout: 30000,
                    documentation: 'Update tool configuration',
                    examples: []
                }],
            // GraphQL Endpoint
            ['/graphql', {
                    path: '/graphql',
                    method: 'POST',
                    authentication: 'required',
                    rateLimit: 20,
                    timeout: 60000,
                    documentation: 'GraphQL endpoint for complex queries',
                    examples: []
                }]
        ];
        endpoints.forEach(([path, config]) => {
            this.endpoints.set(path, config);
        });
    }
    /**
     * Setup HTTP request handlers
     */
    setupHttpHandlers() {
        this.httpServer.on('request', async (req, res) => {
            const startTime = perf_hooks_1.performance.now();
            const requestId = this.generateRequestId();
            try {
                // Parse request
                const apiRequest = await this.parseHttpRequest(req, requestId);
                // Apply middleware
                const middlewareResult = await this.applyMiddleware(apiRequest);
                if (!middlewareResult.allowed) {
                    this.sendHttpResponse(res, {
                        id: requestId,
                        status: middlewareResult.status || 403,
                        error: middlewareResult.message || 'Access denied',
                        metadata: this.createResponseMetadata(startTime, apiRequest)
                    });
                    return;
                }
                // Route request
                const response = await this.routeRequest(apiRequest);
                response.metadata = this.createResponseMetadata(startTime, apiRequest);
                this.sendHttpResponse(res, response);
                // Log request
                this.emit('api_request', {
                    request: apiRequest,
                    response,
                    duration: perf_hooks_1.performance.now() - startTime
                });
            }
            catch (error) {
                const errorResponse = {
                    id: requestId,
                    status: 500,
                    error: error.message,
                    metadata: this.createResponseMetadata(startTime)
                };
                this.sendHttpResponse(res, errorResponse);
                this.emit('api_error', { requestId, error: error.message });
            }
        });
    }
    /**
     * Setup WebSocket handlers
     */
    setupWebSocketHandlers() {
        this.wsServer.on('connection', (ws, req) => {
            const connectionId = this.generateConnectionId();
            this.activeConnections.set(connectionId, ws);
            ws.on('message', async (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    await this.handleWebSocketMessage(connectionId, message);
                }
                catch (error) {
                    this.sendWebSocketError(ws, 'Invalid message format', error.message);
                }
            });
            ws.on('close', () => {
                this.cleanupConnection(connectionId);
            });
            ws.on('error', (error) => {
                this.emit('websocket_error', { connectionId, error: error.message });
                this.cleanupConnection(connectionId);
            });
            // Send welcome message
            this.sendWebSocketMessage(ws, {
                type: 'data',
                data: {
                    message: 'Connected to Linter Integration API',
                    connectionId,
                    version: this.version
                },
                timestamp: Date.now(),
                id: this.generateMessageId()
            });
        });
    }
    /**
     * Setup authentication system
     */
    setupAuthentication() {
        // Default API key for development
        this.apiKeys.set('dev-key-12345', {
            apiKey: 'dev-key-12345',
            userId: 'developer',
            permissions: ['read', 'write', 'admin'],
            rateLimit: 100,
            quotaUsed: 0,
            expiresAt: Date.now() + (365 * 24 * 60 * 60 * 1000) // 1 year
        });
    }
    /**
     * Setup rate limiting
     */
    setupRateLimiting() {
        // Clean up rate limit data every minute
        setInterval(() => {
            const now = Date.now();
            const oneMinute = 60 * 1000;
            for (const [key, limit] of this.rateLimits.entries()) {
                if (now - limit.windowStart > oneMinute) {
                    this.rateLimits.delete(key);
                }
            }
        }, 60000);
    }
    /**
     * Route HTTP requests to appropriate handlers
     */
    async routeRequest(request) {
        const { path, method } = request;
        // Health endpoint
        if (path === '/health' && method === 'GET') {
            return this.handleHealthCheck(request);
        }
        // Status endpoint
        if (path === '/status' && method === 'GET') {
            return this.handleStatusCheck(request);
        }
        // Lint execution
        if (path === '/api/v1/lint/execute' && method === 'POST') {
            return this.handleLintExecution(request);
        }
        // Lint results
        if (path.startsWith('/api/v1/lint/results/') && method === 'GET') {
            const correlationId = path.split('/').pop();
            return this.handleLintResults(request, correlationId);
        }
        // Tool management
        if (path === '/api/v1/tools' && method === 'GET') {
            return this.handleToolsList(request);
        }
        if (path.startsWith('/api/v1/tools/') && path.endsWith('/status') && method === 'GET') {
            const toolId = path.split('/')[4];
            return this.handleToolStatus(request, toolId);
        }
        if (path.startsWith('/api/v1/tools/') && path.endsWith('/execute') && method === 'POST') {
            const toolId = path.split('/')[4];
            return this.handleToolExecution(request, toolId);
        }
        // Correlation analysis
        if (path === '/api/v1/correlations/analyze' && method === 'POST') {
            return this.handleCorrelationAnalysis(request);
        }
        if (path === '/api/v1/correlations/clusters' && method === 'GET') {
            return this.handleClustersList(request);
        }
        // Metrics
        if (path === '/api/v1/metrics/tools' && method === 'GET') {
            return this.handleToolMetrics(request);
        }
        if (path === '/api/v1/metrics/correlations' && method === 'GET') {
            return this.handleCorrelationMetrics(request);
        }
        // GraphQL
        if (path === '/graphql' && method === 'POST') {
            return this.handleGraphQL(request);
        }
        // Not found
        return {
            id: request.id,
            status: 404,
            error: 'Endpoint not found',
            metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
        };
    }
    /**
     * Handle individual endpoint requests
     */
    async handleHealthCheck(request) {
        return {
            id: request.id,
            status: 200,
            data: {
                status: 'healthy',
                timestamp: Date.now(),
                version: this.version,
                uptime: process.uptime(),
                services: {
                    ingestionEngine: 'healthy',
                    toolManager: 'healthy',
                    correlationFramework: 'healthy'
                }
            },
            metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
        };
    }
    async handleStatusCheck(request) {
        try {
            const toolStatus = this.toolManager.getAllToolStatus();
            const connectionCount = this.activeConnections.size;
            return {
                id: request.id,
                status: 200,
                data: {
                    tools: toolStatus,
                    activeConnections: connectionCount,
                    subscriptions: Object.fromEntries(this.subscriptions),
                    performance: {
                        memoryUsage: process.memoryUsage(),
                        cpuUsage: process.cpuUsage()
                    }
                },
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
        catch (error) {
            return {
                id: request.id,
                status: 500,
                error: error.message,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
    }
    async handleLintExecution(request) {
        try {
            const { filePaths, tools, options } = request.body;
            if (!Array.isArray(filePaths) || filePaths.length === 0) {
                return {
                    id: request.id,
                    status: 400,
                    error: 'filePaths is required and must be a non-empty array',
                    metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
                };
            }
            // Start execution (non-blocking)
            const executionPromise = this.ingestionEngine.executeRealtimeLinting(filePaths, {
                ...options,
                allowConcurrent: true
            });
            // Generate correlation ID for tracking
            const correlationId = this.generateCorrelationId();
            // Handle execution in background
            executionPromise
                .then(result => {
                this.broadcastToChannel('lint-results', {
                    type: 'execution-complete',
                    correlationId,
                    result
                });
            })
                .catch(error => {
                this.broadcastToChannel('lint-results', {
                    type: 'execution-error',
                    correlationId,
                    error: error.message
                });
            });
            return {
                id: request.id,
                status: 202, // Accepted
                data: {
                    correlationId,
                    status: 'started',
                    filePaths,
                    tools: tools || 'all',
                    estimatedDuration: filePaths.length * 5000 // rough estimate
                },
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
        catch (error) {
            return {
                id: request.id,
                status: 500,
                error: error.message,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
    }
    async handleLintResults(request, correlationId) {
        // This would retrieve stored results by correlation ID
        // For now, return a placeholder
        return {
            id: request.id,
            status: 200,
            data: {
                correlationId,
                status: 'completed',
                results: [],
                message: 'Results retrieval not yet implemented - use WebSocket for real-time results'
            },
            metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
        };
    }
    async handleToolsList(request) {
        try {
            const status = this.toolManager.getAllToolStatus();
            return {
                id: request.id,
                status: 200,
                data: {
                    tools: Object.keys(status),
                    detailed: status
                },
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
        catch (error) {
            return {
                id: request.id,
                status: 500,
                error: error.message,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
    }
    async handleToolStatus(request, toolId) {
        try {
            const status = this.toolManager.getToolStatus(toolId);
            return {
                id: request.id,
                status: 200,
                data: status,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
        catch (error) {
            return {
                id: request.id,
                status: 404,
                error: error.message,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
    }
    async handleToolExecution(request, toolId) {
        try {
            const { filePaths, options } = request.body;
            if (!Array.isArray(filePaths) || filePaths.length === 0) {
                return {
                    id: request.id,
                    status: 400,
                    error: 'filePaths is required and must be a non-empty array',
                    metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
                };
            }
            const result = await this.toolManager.executeTool(toolId, filePaths, options);
            return {
                id: request.id,
                status: 200,
                data: result,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
        catch (error) {
            return {
                id: request.id,
                status: 500,
                error: error.message,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
    }
    async handleCorrelationAnalysis(request) {
        try {
            const { results } = request.body;
            if (!Array.isArray(results)) {
                return {
                    id: request.id,
                    status: 400,
                    error: 'results is required and must be an array',
                    metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
                };
            }
            const analysis = await this.correlationFramework.correlateResults(results);
            return {
                id: request.id,
                status: 200,
                data: analysis,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
        catch (error) {
            return {
                id: request.id,
                status: 500,
                error: error.message,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
    }
    async handleClustersList(request) {
        // This would retrieve stored clusters
        // For now, return a placeholder
        return {
            id: request.id,
            status: 200,
            data: {
                clusters: [],
                message: 'Cluster retrieval not yet implemented'
            },
            metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
        };
    }
    async handleToolMetrics(request) {
        try {
            const status = this.toolManager.getAllToolStatus();
            const metrics = Object.fromEntries(Object.entries(status).map(([toolId, toolStatus]) => [
                toolId,
                toolStatus.metrics
            ]));
            return {
                id: request.id,
                status: 200,
                data: metrics,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
        catch (error) {
            return {
                id: request.id,
                status: 500,
                error: error.message,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
    }
    async handleCorrelationMetrics(request) {
        // This would retrieve correlation metrics
        // For now, return a placeholder
        return {
            id: request.id,
            status: 200,
            data: {
                totalCorrelations: 0,
                averageConfidence: 0,
                message: 'Correlation metrics retrieval not yet implemented'
            },
            metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
        };
    }
    async handleGraphQL(request) {
        try {
            const query = request.body;
            if (!query.query) {
                return {
                    id: request.id,
                    status: 400,
                    error: 'GraphQL query is required',
                    metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
                };
            }
            const result = await this.executeGraphQLQuery(query);
            return {
                id: request.id,
                status: 200,
                data: result,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
        catch (error) {
            return {
                id: request.id,
                status: 500,
                error: error.message,
                metadata: this.createResponseMetadata(perf_hooks_1.performance.now())
            };
        }
    }
    /**
     * Handle WebSocket messages
     */
    async handleWebSocketMessage(connectionId, message) {
        const ws = this.activeConnections.get(connectionId);
        if (!ws)
            return;
        switch (message.type) {
            case 'subscribe':
                if (message.channel) {
                    this.subscribeToChannel(connectionId, message.channel);
                    this.sendWebSocketMessage(ws, {
                        type: 'data',
                        data: { subscribed: message.channel },
                        timestamp: Date.now(),
                        id: this.generateMessageId()
                    });
                }
                break;
            case 'unsubscribe':
                if (message.channel) {
                    this.unsubscribeFromChannel(connectionId, message.channel);
                    this.sendWebSocketMessage(ws, {
                        type: 'data',
                        data: { unsubscribed: message.channel },
                        timestamp: Date.now(),
                        id: this.generateMessageId()
                    });
                }
                break;
            case 'ping':
                this.sendWebSocketMessage(ws, {
                    type: 'pong',
                    timestamp: Date.now(),
                    id: this.generateMessageId()
                });
                break;
            default:
                this.sendWebSocketError(ws, 'Unknown message type', `Type '${message.type}' not supported`);
        }
    }
    /**
     * Execute GraphQL query (simplified implementation)
     */
    async executeGraphQLQuery(query) {
        // This is a simplified GraphQL implementation
        // In production, you would use a proper GraphQL library
        if (query.query.includes('tools')) {
            const tools = this.toolManager.getAllToolStatus();
            return {
                data: {
                    tools: Object.entries(tools).map(([id, status]) => ({
                        id,
                        name: status.tool.name,
                        isHealthy: status.health.isHealthy,
                        executionCount: status.metrics.totalExecutions
                    }))
                }
            };
        }
        if (query.query.includes('correlations')) {
            return {
                data: {
                    correlations: {
                        total: 0,
                        recent: []
                    }
                }
            };
        }
        return {
            errors: [{
                    message: 'Query not supported in simplified GraphQL implementation'
                }]
        };
    }
    /**
     * Start the API server
     */
    async start() {
        if (this.isRunning) {
            throw new Error('Server is already running');
        }
        return new Promise((resolve, reject) => {
            this.httpServer.listen(this.port, (error) => {
                if (error) {
                    reject(error);
                }
                else {
                    this.isRunning = true;
                    this.emit('server_started', { port: this.port });
                    console.log(`Integration API Server running on port ${this.port}`);
                    resolve();
                }
            });
        });
    }
    /**
     * Stop the API server
     */
    async stop() {
        if (!this.isRunning) {
            return;
        }
        return new Promise((resolve) => {
            // Close all WebSocket connections
            this.activeConnections.forEach(ws => ws.close());
            this.activeConnections.clear();
            // Close HTTP server
            this.httpServer.close(() => {
                this.isRunning = false;
                this.emit('server_stopped');
                resolve();
            });
        });
    }
    // Helper methods
    async parseHttpRequest(req, requestId) {
        const url = new URL(req.url, `http://${req.headers.host}`);
        const query = Object.fromEntries(url.searchParams.entries());
        let body;
        if (req.method === 'POST' || req.method === 'PUT' || req.method === 'PATCH') {
            body = await this.parseRequestBody(req);
        }
        return {
            id: requestId,
            method: req.method,
            path: url.pathname,
            query,
            body,
            headers: req.headers,
            timestamp: Date.now()
        };
    }
    async parseRequestBody(req) {
        return new Promise((resolve, reject) => {
            let body = '';
            req.on('data', (chunk) => {
                body += chunk.toString();
            });
            req.on('end', () => {
                try {
                    resolve(body ? JSON.parse(body) : {});
                }
                catch (error) {
                    reject(new Error('Invalid JSON in request body'));
                }
            });
            req.on('error', reject);
        });
    }
    async applyMiddleware(request) {
        // Authentication check
        const endpoint = this.endpoints.get(request.path);
        if (endpoint?.authentication === 'required') {
            const apiKey = request.headers['x-api-key'] || request.headers['authorization']?.replace('Bearer ', '');
            if (!apiKey || !this.apiKeys.has(apiKey)) {
                return { allowed: false, status: 401, message: 'Invalid or missing API key' };
            }
            request.authentication = this.apiKeys.get(apiKey);
        }
        // Rate limiting
        if (request.authentication) {
            const rateLimitResult = this.checkRateLimit(request.authentication.apiKey, endpoint?.rateLimit || 60);
            if (!rateLimitResult.allowed) {
                return { allowed: false, status: 429, message: 'Rate limit exceeded' };
            }
        }
        return { allowed: true };
    }
    checkRateLimit(apiKey, limit) {
        const now = Date.now();
        const windowStart = Math.floor(now / 60000) * 60000; // Start of current minute
        const key = `${apiKey}_${windowStart}`;
        let rateLimitInfo = this.rateLimits.get(key);
        if (!rateLimitInfo) {
            rateLimitInfo = {
                limit,
                remaining: limit - 1,
                resetTime: windowStart + 60000,
                windowStart
            };
        }
        else {
            rateLimitInfo.remaining = Math.max(0, rateLimitInfo.remaining - 1);
        }
        this.rateLimits.set(key, rateLimitInfo);
        return {
            allowed: rateLimitInfo.remaining >= 0,
            info: rateLimitInfo
        };
    }
    sendHttpResponse(res, response) {
        res.writeHead(response.status, {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key'
        });
        res.end(JSON.stringify(response, null, 2));
    }
    sendWebSocketMessage(ws, message) {
        if (ws.readyState === ws_1.WebSocket.OPEN) {
            ws.send(JSON.stringify(message));
        }
    }
    sendWebSocketError(ws, error, details) {
        this.sendWebSocketMessage(ws, {
            type: 'error',
            data: { error, details },
            timestamp: Date.now(),
            id: this.generateMessageId()
        });
    }
    subscribeToChannel(connectionId, channel) {
        if (!this.subscriptions.has(channel)) {
            this.subscriptions.set(channel, new Set());
        }
        this.subscriptions.get(channel).add(connectionId);
    }
    unsubscribeFromChannel(connectionId, channel) {
        const subscribers = this.subscriptions.get(channel);
        if (subscribers) {
            subscribers.delete(connectionId);
            if (subscribers.size === 0) {
                this.subscriptions.delete(channel);
            }
        }
    }
    broadcastToChannel(channel, data) {
        const subscribers = this.subscriptions.get(channel);
        if (subscribers) {
            const message = {
                type: 'data',
                channel,
                data,
                timestamp: Date.now(),
                id: this.generateMessageId()
            };
            subscribers.forEach(connectionId => {
                const ws = this.activeConnections.get(connectionId);
                if (ws) {
                    this.sendWebSocketMessage(ws, message);
                }
            });
        }
    }
    cleanupConnection(connectionId) {
        this.activeConnections.delete(connectionId);
        // Remove from all subscriptions
        this.subscriptions.forEach((subscribers, channel) => {
            subscribers.delete(connectionId);
            if (subscribers.size === 0) {
                this.subscriptions.delete(channel);
            }
        });
    }
    createResponseMetadata(startTime, request) {
        const metadata = {
            executionTime: perf_hooks_1.performance.now() - startTime,
            timestamp: Date.now(),
            version: this.version
        };
        if (request?.authentication) {
            const rateLimitKey = `${request.authentication.apiKey}_${Math.floor(Date.now() / 60000) * 60000}`;
            const rateLimitInfo = this.rateLimits.get(rateLimitKey);
            if (rateLimitInfo) {
                metadata.rateLimit = rateLimitInfo;
            }
        }
        return metadata;
    }
    generateRequestId() {
        return `req_${Date.now()}_${(0, crypto_1.randomBytes)(4).toString('hex')}`;
    }
    generateConnectionId() {
        return `conn_${Date.now()}_${(0, crypto_1.randomBytes)(4).toString('hex')}`;
    }
    generateMessageId() {
        return `msg_${Date.now()}_${(0, crypto_1.randomBytes)(4).toString('hex')}`;
    }
    generateCorrelationId() {
        return `corr_${Date.now()}_${(0, crypto_1.randomBytes)(8).toString('hex')}`;
    }
}
exports.IntegrationApiServer = IntegrationApiServer;
//# sourceMappingURL=integration-api.js.map