"use strict";
/**
 * Tool Management System
 * Manages linter tool lifecycle, configuration, and resource allocation
 * MESH NODE AGENT: Integration Specialist for Linter Integration Architecture Swarm
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ToolManagementSystem = void 0;
const events_1 = require("events");
const child_process_1 = require("child_process");
const perf_hooks_1 = require("perf_hooks");
/**
 * Tool Management System
 * Comprehensive lifecycle management for linter tools
 */
class ToolManagementSystem extends events_1.EventEmitter {
    constructor(workspaceRoot) {
        super();
        this.workspaceRoot = workspaceRoot;
        this.tools = new Map();
        this.configurations = new Map();
        this.environments = new Map();
        this.resourceAllocations = new Map();
        this.healthStatus = new Map();
        this.metrics = new Map();
        this.circuitBreakers = new Map();
        this.recoveryProcedures = new Map();
        this.runningProcesses = new Map();
        this.executionQueue = new Map();
        this.maxGlobalConcurrency = 10;
        this.healthCheckInterval = 30000;
        this.metricsRetentionPeriod = 86400000; // 24 hours
        this.initializeDefaultEnvironments();
        this.initializeDefaultResourceAllocations();
        this.setupPeriodicTasks();
    }
    /**
     * Initialize default tool environments
     */
    initializeDefaultEnvironments() {
        // Node.js environment for TypeScript/JavaScript tools
        this.environments.set('nodejs', {
            nodeVersion: '18.0.0',
            environmentVariables: {
                NODE_ENV: 'production',
                NODE_OPTIONS: '--max-old-space-size=4096'
            },
            workingDirectory: this.workspaceRoot,
            pathExtensions: ['node_modules/.bin']
        });
        // Python environment for Python tools
        this.environments.set('python', {
            pythonVersion: '3.8.0',
            environmentVariables: {
                PYTHONPATH: this.workspaceRoot,
                PYTHONUNBUFFERED: '1',
                PYTHONDONTWRITEBYTECODE: '1'
            },
            workingDirectory: this.workspaceRoot,
            pathExtensions: ['.venv/bin', 'venv/Scripts']
        });
        // Generic system environment
        this.environments.set('system', {
            environmentVariables: {},
            workingDirectory: this.workspaceRoot,
            pathExtensions: []
        });
    }
    /**
     * Initialize default resource allocations
     */
    initializeDefaultResourceAllocations() {
        const defaultAllocations = {
            eslint: {
                concurrencyLimit: 3,
                priorityWeight: 0.8,
                executionQuota: 100,
                throttleInterval: 1000
            },
            tsc: {
                concurrencyLimit: 1, // TypeScript compiler is resource-intensive
                priorityWeight: 0.9,
                executionQuota: 50,
                throttleInterval: 2000
            },
            flake8: {
                concurrencyLimit: 2,
                priorityWeight: 0.7,
                executionQuota: 80,
                throttleInterval: 1500
            },
            pylint: {
                concurrencyLimit: 1, // Pylint is slow
                priorityWeight: 0.6,
                executionQuota: 30,
                throttleInterval: 3000
            },
            ruff: {
                concurrencyLimit: 4, // Ruff is very fast
                priorityWeight: 0.8,
                executionQuota: 150,
                throttleInterval: 500
            },
            mypy: {
                concurrencyLimit: 2,
                priorityWeight: 0.7,
                executionQuota: 60,
                throttleInterval: 2000
            },
            bandit: {
                concurrencyLimit: 2,
                priorityWeight: 0.9, // Security is high priority
                executionQuota: 70,
                throttleInterval: 1500
            }
        };
        Object.entries(defaultAllocations).forEach(([toolId, allocation]) => {
            this.resourceAllocations.set(toolId, allocation);
        });
    }
    /**
     * Setup periodic health checks and maintenance tasks
     */
    setupPeriodicTasks() {
        // Health check interval
        setInterval(() => {
            this.performHealthChecks();
        }, this.healthCheckInterval);
        // Metrics cleanup interval
        setInterval(() => {
            this.cleanupOldMetrics();
        }, this.metricsRetentionPeriod / 24); // Check every hour
        // Resource usage monitoring
        setInterval(() => {
            this.monitorResourceUsage();
        }, 5000); // Every 5 seconds
    }
    /**
     * Register a new linter tool
     */
    async registerTool(tool, configuration) {
        try {
            // Validate tool installation
            await this.validateToolInstallation(tool);
            // Store tool and configuration
            this.tools.set(tool.id, tool);
            if (configuration) {
                this.configurations.set(tool.id, configuration);
            }
            // Initialize health status and metrics
            this.initializeToolHealth(tool.id);
            this.initializeToolMetrics(tool.id);
            // Setup circuit breaker
            this.circuitBreakers.set(tool.id, {
                isOpen: false,
                failureCount: 0,
                lastFailureTime: 0,
                successCount: 0,
                nextAttemptTime: 0
            });
            // Setup recovery procedures
            this.setupRecoveryProcedures(tool);
            this.emit('tool_registered', { toolId: tool.id, name: tool.name });
        }
        catch (error) {
            this.emit('tool_registration_failed', {
                toolId: tool.id,
                error: error.message
            });
            throw error;
        }
    }
    /**
     * Validate that a tool is properly installed and accessible
     */
    async validateToolInstallation(tool) {
        return new Promise((resolve, reject) => {
            const testCommand = tool.healthCheckCommand || `${tool.command} --version`;
            const [command, ...args] = testCommand.split(' ');
            const process = (0, child_process_1.spawn)(command, args, {
                stdio: ['pipe', 'pipe', 'pipe'],
                timeout: 10000
            });
            let stdout = '';
            let stderr = '';
            process.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            process.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            process.on('close', (code) => {
                if (code === 0) {
                    resolve();
                }
                else {
                    reject(new Error(`Tool validation failed for ${tool.name}: ${stderr || stdout}`));
                }
            });
            process.on('error', (error) => {
                reject(new Error(`Failed to execute ${tool.name}: ${error.message}`));
            });
        });
    }
    /**
     * Initialize health status for a tool
     */
    initializeToolHealth(toolId) {
        this.healthStatus.set(toolId, {
            isHealthy: true,
            lastHealthCheck: Date.now(),
            healthScore: 100,
            failureRate: 0,
            averageExecutionTime: 0,
            successfulExecutions: 0,
            failedExecutions: 0
        });
    }
    /**
     * Initialize metrics for a tool
     */
    initializeToolMetrics(toolId) {
        this.metrics.set(toolId, {
            totalExecutions: 0,
            successfulExecutions: 0,
            failedExecutions: 0,
            averageExecutionTime: 0,
            minExecutionTime: Infinity,
            maxExecutionTime: 0,
            totalViolationsFound: 0,
            uniqueRulesTriggered: new Set(),
            resourceUsage: {
                peakMemory: 0,
                totalCpuTime: 0,
                diskUsage: 0
            }
        });
    }
    /**
     * Setup recovery procedures for a tool
     */
    setupRecoveryProcedures(tool) {
        const procedures = {
            resetConfiguration: true,
            reinstallTool: false,
            clearCache: true,
            restartEnvironment: false,
            escalateToAdmin: false,
            customRecoverySteps: []
        };
        // Tool-specific recovery procedures
        switch (tool.id) {
            case 'eslint':
                procedures.customRecoverySteps = [
                    'npm install eslint --save-dev',
                    'rm -rf node_modules/.cache/eslint'
                ];
                break;
            case 'tsc':
                procedures.customRecoverySteps = [
                    'npm install typescript --save-dev',
                    'rm -rf node_modules/.cache/tsc'
                ];
                break;
            case 'flake8':
            case 'pylint':
            case 'mypy':
            case 'bandit':
                procedures.customRecoverySteps = [
                    `pip install --upgrade ${tool.id}`,
                    'rm -rf __pycache__',
                    'rm -rf .mypy_cache'
                ];
                break;
        }
        this.recoveryProcedures.set(tool.id, procedures);
    }
    /**
     * Execute a tool with full lifecycle management
     */
    async executeTool(toolId, filePaths, options = {}) {
        const tool = this.tools.get(toolId);
        if (!tool) {
            throw new Error(`Tool not found: ${toolId}`);
        }
        const allocation = this.resourceAllocations.get(toolId);
        const health = this.healthStatus.get(toolId);
        const circuitBreaker = this.circuitBreakers.get(toolId);
        // Check circuit breaker
        if (circuitBreaker.isOpen) {
            if (Date.now() < circuitBreaker.nextAttemptTime) {
                throw new Error(`Circuit breaker open for tool ${toolId}`);
            }
        }
        // Check health status
        if (!health.isHealthy && !options.forceExecution) {
            throw new Error(`Tool ${toolId} is unhealthy. Use forceExecution to override.`);
        }
        // Wait for resource availability
        await this.waitForResourceAvailability(toolId);
        const startTime = perf_hooks_1.performance.now();
        let executionResult;
        try {
            // Execute with resource monitoring
            executionResult = await this.executeWithMonitoring(tool, filePaths, options);
            // Update success metrics
            this.updateSuccessMetrics(toolId, perf_hooks_1.performance.now() - startTime, executionResult);
            // Reset circuit breaker on success
            circuitBreaker.failureCount = 0;
            circuitBreaker.successCount++;
            circuitBreaker.isOpen = false;
            return executionResult;
        }
        catch (error) {
            // Update failure metrics
            this.updateFailureMetrics(toolId, perf_hooks_1.performance.now() - startTime, error);
            // Update circuit breaker
            circuitBreaker.failureCount++;
            circuitBreaker.lastFailureTime = Date.now();
            if (circuitBreaker.failureCount >= 5) {
                circuitBreaker.isOpen = true;
                circuitBreaker.nextAttemptTime = Date.now() + 60000; // 1 minute
                // Attempt recovery
                await this.attemptToolRecovery(toolId);
            }
            throw error;
        }
    }
    /**
     * Wait for resource availability based on allocation limits
     */
    async waitForResourceAvailability(toolId) {
        const allocation = this.resourceAllocations.get(toolId);
        const runningCount = this.getRunningProcessCount(toolId);
        if (runningCount >= allocation.concurrencyLimit) {
            return new Promise((resolve) => {
                const queue = this.executionQueue.get(toolId) || [];
                queue.push(resolve);
                this.executionQueue.set(toolId, queue);
            });
        }
    }
    /**
     * Execute tool with comprehensive monitoring
     */
    async executeWithMonitoring(tool, filePaths, options) {
        const environment = this.getToolEnvironment(tool);
        const configuration = this.configurations.get(tool.id);
        // Prepare execution arguments
        const args = this.prepareExecutionArgs(tool, filePaths, configuration, options);
        return new Promise((resolve, reject) => {
            const startTime = perf_hooks_1.performance.now();
            const startMemory = process.memoryUsage();
            const childProcess = (0, child_process_1.spawn)(tool.command, args, {
                env: { ...process.env, ...environment.environmentVariables },
                cwd: environment.workingDirectory,
                stdio: ['pipe', 'pipe', 'pipe']
            });
            this.runningProcesses.set(`${tool.id}_${Date.now()}`, childProcess);
            let stdout = '';
            let stderr = '';
            childProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            childProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            childProcess.on('close', (code) => {
                const executionTime = perf_hooks_1.performance.now() - startTime;
                const endMemory = process.memoryUsage();
                const memoryUsed = endMemory.heapUsed - startMemory.heapUsed;
                if (code === 0 || (code === 1 && stdout)) {
                    resolve({
                        success: true,
                        output: stdout,
                        stderr,
                        executionTime,
                        memoryUsed,
                        exitCode: code,
                        violationsFound: this.countViolationsInOutput(tool, stdout)
                    });
                }
                else {
                    reject(new Error(`Tool execution failed with code ${code}: ${stderr}`));
                }
                // Clean up process reference
                this.runningProcesses.delete(`${tool.id}_${Date.now()}`);
                // Process next in queue
                this.processExecutionQueue(tool.id);
            });
            childProcess.on('error', (error) => {
                reject(error);
                this.runningProcesses.delete(`${tool.id}_${Date.now()}`);
                this.processExecutionQueue(tool.id);
            });
            // Set timeout
            setTimeout(() => {
                if (!childProcess.killed) {
                    childProcess.kill('SIGTERM');
                    reject(new Error(`Tool execution timed out after ${tool.timeout}ms`));
                }
            }, tool.timeout);
        });
    }
    /**
     * Perform health checks on all registered tools
     */
    async performHealthChecks() {
        const healthPromises = Array.from(this.tools.keys()).map(toolId => this.performToolHealthCheck(toolId));
        await Promise.allSettled(healthPromises);
    }
    /**
     * Perform health check on a specific tool
     */
    async performToolHealthCheck(toolId) {
        const tool = this.tools.get(toolId);
        const health = this.healthStatus.get(toolId);
        try {
            await this.validateToolInstallation(tool);
            health.isHealthy = true;
            health.lastHealthCheck = Date.now();
            health.healthScore = Math.min(100, health.healthScore + 10);
            this.emit('tool_health_ok', { toolId, healthScore: health.healthScore });
        }
        catch (error) {
            health.isHealthy = false;
            health.lastHealthCheck = Date.now();
            health.healthScore = Math.max(0, health.healthScore - 20);
            health.lastError = error.message;
            this.emit('tool_health_degraded', {
                toolId,
                error: error.message,
                healthScore: health.healthScore
            });
            // Attempt recovery if health is critically low
            if (health.healthScore <= 20) {
                await this.attemptToolRecovery(toolId);
            }
        }
    }
    /**
     * Attempt to recover a failed tool
     */
    async attemptToolRecovery(toolId) {
        const procedures = this.recoveryProcedures.get(toolId);
        if (!procedures)
            return;
        this.emit('tool_recovery_started', { toolId });
        try {
            if (procedures.resetConfiguration) {
                await this.resetToolConfiguration(toolId);
            }
            if (procedures.clearCache) {
                await this.clearToolCache(toolId);
            }
            // Execute custom recovery steps
            for (const step of procedures.customRecoverySteps) {
                await this.executeRecoveryStep(step);
            }
            // Reinitialize tool health
            this.initializeToolHealth(toolId);
            this.emit('tool_recovery_completed', { toolId });
        }
        catch (error) {
            this.emit('tool_recovery_failed', { toolId, error: error.message });
            if (procedures.escalateToAdmin) {
                this.emit('tool_recovery_escalation', { toolId, error: error.message });
            }
        }
    }
    /**
     * Get comprehensive tool status
     */
    getToolStatus(toolId) {
        const tool = this.tools.get(toolId);
        const health = this.healthStatus.get(toolId);
        const metrics = this.metrics.get(toolId);
        const circuitBreaker = this.circuitBreakers.get(toolId);
        const allocation = this.resourceAllocations.get(toolId);
        if (!tool || !health || !metrics || !circuitBreaker || !allocation) {
            throw new Error(`Tool not found: ${toolId}`);
        }
        return {
            tool,
            health,
            metrics,
            circuitBreaker,
            allocation,
            isRunning: this.getRunningProcessCount(toolId) > 0,
            queueLength: (this.executionQueue.get(toolId) || []).length
        };
    }
    /**
     * Get status of all tools
     */
    getAllToolStatus() {
        const status = {};
        this.tools.forEach((_, toolId) => {
            status[toolId] = this.getToolStatus(toolId);
        });
        return status;
    }
    // Helper methods
    getToolEnvironment(tool) {
        // Determine environment based on tool type
        if (['eslint', 'tsc'].includes(tool.id)) {
            return this.environments.get('nodejs');
        }
        else if (['flake8', 'pylint', 'mypy', 'bandit', 'ruff'].includes(tool.id)) {
            return this.environments.get('python');
        }
        else {
            return this.environments.get('system');
        }
    }
    prepareExecutionArgs(tool, filePaths, configuration, options) {
        let args = [...tool.args];
        if (configuration?.customArgs) {
            args = [...args, ...configuration.customArgs];
        }
        if (options?.additionalArgs) {
            args = [...args, ...options.additionalArgs];
        }
        args = [...args, ...filePaths];
        return args;
    }
    getRunningProcessCount(toolId) {
        return Array.from(this.runningProcesses.keys())
            .filter(key => key.startsWith(`${toolId}_`))
            .length;
    }
    processExecutionQueue(toolId) {
        const queue = this.executionQueue.get(toolId) || [];
        if (queue.length > 0) {
            const nextExecution = queue.shift();
            nextExecution();
            this.executionQueue.set(toolId, queue);
        }
    }
    countViolationsInOutput(tool, output) {
        // Basic violation counting logic - would be enhanced per tool
        try {
            if (tool.outputFormat === 'json') {
                const data = JSON.parse(output);
                if (Array.isArray(data)) {
                    return data.reduce((count, file) => count + (file.messages?.length || 0), 0);
                }
            }
        }
        catch (error) {
            // Fallback to line counting for text output
            return output.split('\n').filter(line => line.trim()).length;
        }
        return 0;
    }
    updateSuccessMetrics(toolId, executionTime, result) {
        const metrics = this.metrics.get(toolId);
        const health = this.healthStatus.get(toolId);
        metrics.totalExecutions++;
        metrics.successfulExecutions++;
        metrics.totalViolationsFound += result.violationsFound;
        // Update execution time statistics
        const currentAvg = metrics.averageExecutionTime;
        metrics.averageExecutionTime = (currentAvg * (metrics.totalExecutions - 1) + executionTime) / metrics.totalExecutions;
        metrics.minExecutionTime = Math.min(metrics.minExecutionTime, executionTime);
        metrics.maxExecutionTime = Math.max(metrics.maxExecutionTime, executionTime);
        // Update health
        health.successfulExecutions++;
        health.averageExecutionTime = metrics.averageExecutionTime;
        health.failureRate = metrics.failedExecutions / metrics.totalExecutions;
    }
    updateFailureMetrics(toolId, executionTime, error) {
        const metrics = this.metrics.get(toolId);
        const health = this.healthStatus.get(toolId);
        metrics.totalExecutions++;
        metrics.failedExecutions++;
        health.failedExecutions++;
        health.failureRate = metrics.failedExecutions / metrics.totalExecutions;
        health.lastError = error.message;
    }
    async resetToolConfiguration(toolId) {
        // Implementation for resetting tool configuration
    }
    async clearToolCache(toolId) {
        // Implementation for clearing tool cache
    }
    async executeRecoveryStep(step) {
        // Implementation for executing recovery commands
    }
    cleanupOldMetrics() {
        // Implementation for cleaning up old metrics data
    }
    monitorResourceUsage() {
        // Implementation for monitoring resource usage
    }
}
exports.ToolManagementSystem = ToolManagementSystem;
//# sourceMappingURL=tool-management-system.js.map