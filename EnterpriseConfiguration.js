"use strict";
/**
 * Enterprise Configuration & CTQ Specifications Support
 *
 * Provides enterprise-grade configuration management with CTQ (Critical-to-Quality)
 * specifications and comprehensive quality gate configuration support.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.EnterpriseConfiguration = void 0;
const events_1 = require("events");
class EnterpriseConfiguration extends events_1.EventEmitter {
    constructor(initialConfig) {
        super();
        this.configHistory = new Map();
        this.validators = new Map();
        this.config = this.initializeDefaultConfig(initialConfig);
        this.initializeValidators();
        this.validateConfiguration();
    }
    /**
     * Initialize default enterprise configuration
     */
    initializeDefaultConfig(partial) {
        const defaultConfig = {
            organization: {
                name: 'Default Organization',
                industry: 'general',
                complianceFrameworks: ['ISO-9001'],
                qualityMaturityLevel: 3,
                riskTolerance: 'medium',
                geographicalRegions: ['US']
            },
            qualityStandards: {
                primaryStandard: 'ISO-9001',
                secondaryStandards: [],
                customStandards: [],
                certificationRequirements: []
            },
            ctqSpecifications: this.getDefaultCTQSpecifications(),
            gateConfigurations: this.getDefaultGateConfigurations(),
            thresholds: this.getDefaultThresholds(),
            integrations: this.getDefaultIntegrations(),
            governance: this.getDefaultGovernance()
        };
        return { ...defaultConfig, ...partial };
    }
    /**
     * Get default CTQ specifications
     */
    getDefaultCTQSpecifications() {
        return [
            {
                id: 'ctq-response-time',
                name: 'System Response Time',
                description: 'Maximum acceptable response time for user requests',
                category: 'performance',
                importance: 'critical',
                target: {
                    value: 200,
                    unit: 'milliseconds',
                    direction: 'minimize',
                    baseline: 500,
                    stretchGoal: 100
                },
                limits: {
                    upperSpecLimit: 500,
                    lowerSpecLimit: 0,
                    upperControlLimit: 400,
                    lowerControlLimit: 50,
                    actionLimits: {
                        warning: 300,
                        critical: 450
                    }
                },
                measurementMethod: {
                    type: 'automated',
                    frequency: 'continuous',
                    tools: ['APM', 'Load Testing'],
                    dataSource: 'application-metrics',
                    calculationFormula: 'p95(response_time)'
                },
                validationCriteria: {
                    statistical: {
                        sampleSize: 1000,
                        confidenceLevel: 95,
                        significanceLevel: 0.05
                    },
                    acceptance: {
                        minimumCapability: 1.33,
                        maximumDefectRate: 3400,
                        requiredYield: 99.66
                    }
                },
                stakeholders: ['Product Team', 'Engineering', 'QA'],
                reviewFrequency: 30
            },
            {
                id: 'ctq-security-score',
                name: 'Security Compliance Score',
                description: 'Overall security posture and compliance score',
                category: 'security',
                importance: 'critical',
                target: {
                    value: 95,
                    unit: 'percentage',
                    direction: 'maximize',
                    baseline: 80,
                    stretchGoal: 98
                },
                limits: {
                    upperSpecLimit: 100,
                    lowerSpecLimit: 90,
                    upperControlLimit: 98,
                    lowerControlLimit: 92,
                    actionLimits: {
                        warning: 90,
                        critical: 85
                    }
                },
                measurementMethod: {
                    type: 'automated',
                    frequency: 'periodic',
                    tools: ['SAST', 'DAST', 'SCA'],
                    dataSource: 'security-scans',
                    calculationFormula: 'weighted_average(owasp_score, nist_score, custom_score)'
                },
                validationCriteria: {
                    statistical: {
                        sampleSize: 100,
                        confidenceLevel: 99,
                        significanceLevel: 0.01
                    },
                    acceptance: {
                        minimumCapability: 1.67,
                        maximumDefectRate: 233,
                        requiredYield: 99.977
                    }
                },
                stakeholders: ['Security Team', 'Compliance', 'Engineering'],
                reviewFrequency: 7
            },
            {
                id: 'ctq-defect-rate',
                name: 'Defect Rate',
                description: 'Production defect rate per million opportunities',
                category: 'reliability',
                importance: 'critical',
                target: {
                    value: 3400,
                    unit: 'PPM',
                    direction: 'minimize',
                    baseline: 10000,
                    stretchGoal: 233
                },
                limits: {
                    upperSpecLimit: 6000,
                    lowerSpecLimit: 0,
                    upperControlLimit: 5000,
                    lowerControlLimit: 100,
                    actionLimits: {
                        warning: 4000,
                        critical: 5500
                    }
                },
                measurementMethod: {
                    type: 'automated',
                    frequency: 'continuous',
                    tools: ['Bug Tracking', 'Monitoring'],
                    dataSource: 'defect-tracking',
                    calculationFormula: '(defects / opportunities) * 1000000'
                },
                validationCriteria: {
                    statistical: {
                        sampleSize: 500,
                        confidenceLevel: 95,
                        significanceLevel: 0.05
                    },
                    acceptance: {
                        minimumCapability: 1.33,
                        maximumDefectRate: 3400,
                        requiredYield: 99.66
                    }
                },
                stakeholders: ['QA Team', 'Engineering', 'Product'],
                reviewFrequency: 14
            }
        ];
    }
    /**
     * Get default gate configurations
     */
    getDefaultGateConfigurations() {
        return [
            {
                id: 'gate-development',
                name: 'Development Quality Gate',
                description: 'Quality gate for development phase',
                stage: 'development',
                mandatory: true,
                enabledValidators: [
                    {
                        type: 'six-sigma',
                        enabled: true,
                        weight: 0.3,
                        config: { enableCTQValidation: true },
                        thresholds: { defectRate: 10000, qualityScore: 70 },
                        ctqMappings: ['ctq-defect-rate']
                    },
                    {
                        type: 'security',
                        enabled: true,
                        weight: 0.4,
                        config: { enableOWASPValidation: true },
                        thresholds: { criticalVulnerabilities: 0, minimumSecurityScore: 80 },
                        ctqMappings: ['ctq-security-score']
                    }
                ],
                dependencies: [],
                executionConditions: [
                    {
                        type: 'code-change',
                        condition: 'pull-request',
                        parameters: { minChanges: 1 }
                    }
                ],
                escalationPolicies: [
                    {
                        level: 'team',
                        trigger: 'failure',
                        recipients: ['dev-team@company.com'],
                        notificationMethod: 'email',
                        template: 'dev-gate-failure'
                    }
                ]
            },
            {
                id: 'gate-production',
                name: 'Production Readiness Gate',
                description: 'Comprehensive quality gate for production deployment',
                stage: 'production',
                mandatory: true,
                enabledValidators: [
                    {
                        type: 'six-sigma',
                        enabled: true,
                        weight: 0.25,
                        config: { enableCTQValidation: true, requireFullCompliance: true },
                        thresholds: { defectRate: 3400, qualityScore: 85 },
                        ctqMappings: ['ctq-defect-rate']
                    },
                    {
                        type: 'nasa',
                        enabled: true,
                        weight: 0.25,
                        config: { enablePOT10Rules: true },
                        thresholds: { complianceScore: 95, criticalViolations: 0 },
                        ctqMappings: []
                    },
                    {
                        type: 'performance',
                        enabled: true,
                        weight: 0.2,
                        config: { enableRegressionDetection: true },
                        thresholds: { regressionThreshold: 5, responseTimeLimit: 200 },
                        ctqMappings: ['ctq-response-time']
                    },
                    {
                        type: 'security',
                        enabled: true,
                        weight: 0.3,
                        config: { enableOWASPValidation: true, vulnerabilityScanning: true },
                        thresholds: { criticalVulnerabilities: 0, minimumSecurityScore: 95 },
                        ctqMappings: ['ctq-security-score']
                    }
                ],
                dependencies: ['gate-development'],
                executionConditions: [
                    {
                        type: 'trigger-event',
                        condition: 'deployment-request',
                        parameters: { environment: 'production' }
                    }
                ],
                escalationPolicies: [
                    {
                        level: 'management',
                        trigger: 'failure',
                        recipients: ['engineering-leads@company.com'],
                        notificationMethod: 'slack',
                        template: 'prod-gate-failure'
                    }
                ]
            }
        ];
    }
    /**
     * Get default thresholds
     */
    getDefaultThresholds() {
        return {
            global: {
                minimumQualityScore: 80,
                maximumDefectRate: 3400,
                minimumProcessCapability: 1.33,
                maximumRegressionTolerance: 5,
                securityBaseline: 90,
                complianceMinimum: 95
            },
            contextual: [
                {
                    context: 'development',
                    environment: 'dev',
                    thresholds: {
                        minimumQualityScore: 70,
                        maximumDefectRate: 10000,
                        minimumSecurityScore: 80
                    },
                    overrides: []
                },
                {
                    context: 'production',
                    environment: 'prod',
                    thresholds: {
                        minimumQualityScore: 95,
                        maximumDefectRate: 233,
                        minimumSecurityScore: 95
                    },
                    overrides: []
                }
            ],
            adaptive: {
                enabled: false,
                learningPeriod: 30,
                adjustmentFactor: 0.1,
                minimumDataPoints: 100,
                maxAdjustmentPercentage: 10
            }
        };
    }
    /**
     * Get default integrations
     */
    getDefaultIntegrations() {
        return {
            cicd: {
                platform: 'github',
                webhookUrl: '',
                authentication: {
                    type: 'token',
                    credentials: {}
                },
                qualityGateIntegration: true,
                blockDeploymentOnFailure: true
            },
            monitoring: {
                platform: 'prometheus',
                endpoint: '',
                authentication: {},
                metricPrefix: 'quality_gates',
                dashboardTemplate: 'default'
            },
            ticketing: {
                platform: 'jira',
                endpoint: '',
                authentication: {},
                autoCreateTickets: false,
                ticketTemplate: 'quality-gate-failure'
            },
            communication: {
                platform: 'slack',
                webhookUrl: '',
                channels: {
                    alerts: '#quality-alerts',
                    reports: '#quality-reports',
                    notifications: '#quality-notifications'
                },
                messageTemplates: {}
            }
        };
    }
    /**
     * Get default governance settings
     */
    getDefaultGovernance() {
        return {
            approvalWorkflows: [
                {
                    trigger: 'threshold-change',
                    approvers: [
                        {
                            level: 1,
                            roles: ['Quality Manager'],
                            requireAll: true,
                            delegationAllowed: false
                        }
                    ],
                    timeout: 24,
                    autoApprovalConditions: []
                }
            ],
            auditSettings: {
                enabled: true,
                retentionPeriod: 24,
                auditEvents: ['config-change', 'threshold-change', 'gate-bypass'],
                encryptionRequired: true,
                externalAuditorAccess: false
            },
            reportingSettings: {
                enabledReports: [
                    {
                        type: 'quality-dashboard',
                        format: 'html',
                        template: 'standard',
                        dataScope: ['overall', 'gates', 'trends']
                    }
                ],
                distributionLists: [
                    {
                        name: 'Quality Team',
                        recipients: ['quality@company.com'],
                        reportTypes: ['quality-dashboard'],
                        frequency: 'weekly'
                    }
                ],
                scheduledReports: []
            },
            dataRetention: {
                qualityMetrics: 12,
                auditLogs: 84,
                reports: 24,
                violations: 36,
                archiveLocation: 's3://quality-archives',
                encryptionRequired: true
            }
        };
    }
    /**
     * Initialize configuration validators
     */
    initializeValidators() {
        this.validators.set('ctq-specification', this.validateCTQSpecification.bind(this));
        this.validators.set('gate-configuration', this.validateGateConfiguration.bind(this));
        this.validators.set('thresholds', this.validateThresholds.bind(this));
        this.validators.set('integrations', this.validateIntegrations.bind(this));
    }
    /**
     * Validate CTQ specification
     */
    validateCTQSpecification(ctq) {
        const errors = [];
        if (!ctq.id || !ctq.name) {
            errors.push('CTQ must have valid ID and name');
        }
        if (ctq.target.value <= 0) {
            errors.push('CTQ target value must be positive');
        }
        if (ctq.limits.upperSpecLimit <= ctq.limits.lowerSpecLimit) {
            errors.push('Upper spec limit must be greater than lower spec limit');
        }
        if (ctq.validationCriteria.statistical.confidenceLevel <= 0 ||
            ctq.validationCriteria.statistical.confidenceLevel >= 100) {
            errors.push('Confidence level must be between 0 and 100');
        }
        return errors;
    }
    /**
     * Validate gate configuration
     */
    validateGateConfiguration(gate) {
        const errors = [];
        if (!gate.id || !gate.name) {
            errors.push('Gate must have valid ID and name');
        }
        const totalWeight = gate.enabledValidators.reduce((sum, v) => sum + v.weight, 0);
        if (Math.abs(totalWeight - 1.0) > 0.01) {
            errors.push('Validator weights must sum to 1.0');
        }
        if (gate.enabledValidators.length === 0) {
            errors.push('Gate must have at least one enabled validator');
        }
        return errors;
    }
    /**
     * Validate thresholds
     */
    validateThresholds(thresholds) {
        const errors = [];
        if (thresholds.global.minimumQualityScore < 0 || thresholds.global.minimumQualityScore > 100) {
            errors.push('Quality score must be between 0 and 100');
        }
        if (thresholds.global.maximumDefectRate < 0) {
            errors.push('Defect rate must be non-negative');
        }
        if (thresholds.global.minimumProcessCapability < 0) {
            errors.push('Process capability must be non-negative');
        }
        return errors;
    }
    /**
     * Validate integrations
     */
    validateIntegrations(integrations) {
        const errors = [];
        if (integrations.cicd.webhookUrl && !this.isValidUrl(integrations.cicd.webhookUrl)) {
            errors.push('CI/CD webhook URL must be valid');
        }
        if (integrations.monitoring.endpoint && !this.isValidUrl(integrations.monitoring.endpoint)) {
            errors.push('Monitoring endpoint must be valid URL');
        }
        return errors;
    }
    /**
     * Validate URL format
     */
    isValidUrl(url) {
        try {
            new URL(url);
            return true;
        }
        catch {
            return false;
        }
    }
    /**
     * Validate entire configuration
     */
    validateConfiguration() {
        const errors = [];
        // Validate CTQ specifications
        this.config.ctqSpecifications.forEach(ctq => {
            const ctqErrors = this.validators.get('ctq-specification')(ctq);
            errors.push(...ctqErrors);
        });
        // Validate gate configurations
        this.config.gateConfigurations.forEach(gate => {
            const gateErrors = this.validators.get('gate-configuration')(gate);
            errors.push(...gateErrors);
        });
        // Validate thresholds
        const thresholdErrors = this.validators.get('thresholds')(this.config.thresholds);
        errors.push(...thresholdErrors);
        // Validate integrations
        const integrationErrors = this.validators.get('integrations')(this.config.integrations);
        errors.push(...integrationErrors);
        if (errors.length > 0) {
            this.emit('validation-errors', errors);
            throw new Error(`Configuration validation failed: ${errors.join(', ')}`);
        }
        this.emit('configuration-validated', this.config);
    }
    /**
     * Get current configuration
     */
    getConfiguration() {
        return JSON.parse(JSON.stringify(this.config));
    }
    /**
     * Update configuration
     */
    updateConfiguration(updates) {
        // Store current config in history
        const timestamp = new Date().toISOString();
        this.configHistory.set(timestamp, JSON.parse(JSON.stringify(this.config)));
        // Apply updates
        this.config = { ...this.config, ...updates };
        // Validate updated configuration
        this.validateConfiguration();
        this.emit('configuration-updated', {
            timestamp,
            changes: updates,
            config: this.config
        });
    }
    /**
     * Get CTQ specification by ID
     */
    getCTQSpecification(id) {
        return this.config.ctqSpecifications.find(ctq => ctq.id === id);
    }
    /**
     * Add or update CTQ specification
     */
    updateCTQSpecification(ctq) {
        const errors = this.validators.get('ctq-specification')(ctq);
        if (errors.length > 0) {
            throw new Error(`CTQ validation failed: ${errors.join(', ')}`);
        }
        const index = this.config.ctqSpecifications.findIndex(c => c.id === ctq.id);
        if (index >= 0) {
            this.config.ctqSpecifications[index] = ctq;
        }
        else {
            this.config.ctqSpecifications.push(ctq);
        }
        this.emit('ctq-updated', ctq);
    }
    /**
     * Get gate configuration by ID
     */
    getGateConfiguration(id) {
        return this.config.gateConfigurations.find(gate => gate.id === id);
    }
    /**
     * Add or update gate configuration
     */
    updateGateConfiguration(gate) {
        const errors = this.validators.get('gate-configuration')(gate);
        if (errors.length > 0) {
            throw new Error(`Gate validation failed: ${errors.join(', ')}`);
        }
        const index = this.config.gateConfigurations.findIndex(g => g.id === gate.id);
        if (index >= 0) {
            this.config.gateConfigurations[index] = gate;
        }
        else {
            this.config.gateConfigurations.push(gate);
        }
        this.emit('gate-updated', gate);
    }
    /**
     * Get thresholds for context
     */
    getThresholdsForContext(context, environment) {
        const contextualThreshold = this.config.thresholds.contextual.find(ct => ct.context === context && ct.environment === environment);
        if (contextualThreshold) {
            return { ...this.config.thresholds.global, ...contextualThreshold.thresholds };
        }
        return this.config.thresholds.global;
    }
    /**
     * Export configuration
     */
    exportConfiguration(format = 'json') {
        if (format === 'json') {
            return JSON.stringify(this.config, null, 2);
        }
        else {
            // YAML export would be implemented here
            return 'YAML export not implemented';
        }
    }
    /**
     * Import configuration
     */
    importConfiguration(configData, format = 'json') {
        let importedConfig;
        try {
            if (format === 'json') {
                importedConfig = JSON.parse(configData);
            }
            else {
                throw new Error('YAML import not implemented');
            }
            // Store current config in history
            const timestamp = new Date().toISOString();
            this.configHistory.set(timestamp, JSON.parse(JSON.stringify(this.config)));
            // Apply imported configuration
            this.config = importedConfig;
            // Validate imported configuration
            this.validateConfiguration();
            this.emit('configuration-imported', {
                timestamp,
                source: 'import',
                config: this.config
            });
        }
        catch (error) {
            this.emit('import-error', error);
            throw new Error(`Configuration import failed: ${error.message}`);
        }
    }
    /**
     * Get configuration history
     */
    getConfigurationHistory() {
        return Array.from(this.configHistory.entries()).map(([timestamp, config]) => ({
            timestamp,
            config
        }));
    }
    /**
     * Rollback to previous configuration
     */
    rollbackConfiguration(timestamp) {
        const historicalConfig = this.configHistory.get(timestamp);
        if (!historicalConfig) {
            throw new Error(`Configuration not found for timestamp: ${timestamp}`);
        }
        // Store current config before rollback
        const currentTimestamp = new Date().toISOString();
        this.configHistory.set(currentTimestamp, JSON.parse(JSON.stringify(this.config)));
        // Apply historical configuration
        this.config = JSON.parse(JSON.stringify(historicalConfig));
        // Validate rolled back configuration
        this.validateConfiguration();
        this.emit('configuration-rollback', {
            timestamp: currentTimestamp,
            rolledBackTo: timestamp,
            config: this.config
        });
    }
}
exports.EnterpriseConfiguration = EnterpriseConfiguration;
//# sourceMappingURL=EnterpriseConfiguration.js.map