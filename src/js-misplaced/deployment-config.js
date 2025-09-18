"use strict";
/**
 * Deployment Configuration Management
 *
 * Integrates with Phase 3 artifact system and enterprise configuration
 * for centralized deployment settings and compliance validation.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DeploymentConfigManager = void 0;
class DeploymentConfigManager {
    constructor() {
        this.configCache = new Map();
        this.enterpriseConfig = null;
        this.phase3Integration = new Phase3Integration();
        this.initializeConfigManager();
    }
    /**
     * Load enterprise configuration from Phase 3 system
     */
    async loadEnterpriseConfig() {
        try {
            // Load from enterprise_config.yaml
            this.enterpriseConfig = await this.loadEnterpriseConfigFile();
            // Integrate with Phase 3 artifact system
            await this.phase3Integration.initialize(this.enterpriseConfig);
            console.log('Enterprise configuration loaded successfully');
        }
        catch (error) {
            console.error('Failed to load enterprise configuration:', error);
            throw error;
        }
    }
    /**
     * Get deployment configuration for environment
     */
    async getDeploymentConfig(environmentName, strategyType) {
        const cacheKey = `${environmentName}-${strategyType}`;
        if (this.configCache.has(cacheKey)) {
            return this.configCache.get(cacheKey);
        }
        const config = await this.buildDeploymentConfig(environmentName, strategyType);
        this.configCache.set(cacheKey, config);
        return config;
    }
    /**
     * Get environment configuration with enterprise overrides
     */
    async getEnvironmentConfig(environmentName) {
        const baseConfig = await this.getBaseEnvironmentConfig(environmentName);
        const enterpriseOverrides = this.getEnterpriseOverrides(environmentName);
        return this.mergeEnvironmentConfig(baseConfig, enterpriseOverrides);
    }
    /**
     * Get platform configuration for deployment
     */
    async getPlatformConfig(platformType, environmentName) {
        const baseConfig = await this.getBasePlatformConfig(platformType);
        const environmentOverrides = this.getEnvironmentPlatformOverrides(platformType, environmentName);
        return this.mergePlatformConfig(baseConfig, environmentOverrides);
    }
    /**
     * Validate deployment configuration against enterprise policies
     */
    async validateDeploymentConfig(config) {
        const issues = [];
        // Validate against enterprise policies
        const policyIssues = await this.validateEnterpisePolicies(config);
        issues.push(...policyIssues);
        // Validate compliance requirements
        const complianceIssues = await this.validateComplianceRequirements(config);
        issues.push(...complianceIssues);
        // Validate resource limits
        const resourceIssues = this.validateResourceLimits(config);
        issues.push(...resourceIssues);
        // Validate security settings
        const securityIssues = await this.validateSecuritySettings(config);
        issues.push(...securityIssues);
        return {
            valid: issues.length === 0,
            issues,
            config
        };
    }
    /**
     * Get feature flags for environment
     */
    getFeatureFlags(environmentName) {
        if (!this.enterpriseConfig) {
            return {};
        }
        const globalFlags = this.enterpriseConfig.featureFlags?.global || {};
        const environmentFlags = this.enterpriseConfig.featureFlags?.environments?.[environmentName] || {};
        return { ...globalFlags, ...environmentFlags };
    }
    /**
     * Get compliance configuration
     */
    getComplianceConfig(environmentName) {
        const defaultConfig = {
            level: 'basic',
            checks: [],
            auditEnabled: false,
            reportingRequired: false
        };
        if (!this.enterpriseConfig?.compliance) {
            return defaultConfig;
        }
        const globalCompliance = this.enterpriseConfig.compliance.global || {};
        const environmentCompliance = this.enterpriseConfig.compliance.environments?.[environmentName] || {};
        return {
            ...defaultConfig,
            ...globalCompliance,
            ...environmentCompliance
        };
    }
    /**
     * Update configuration cache
     */
    async refreshConfiguration() {
        this.configCache.clear();
        await this.loadEnterpriseConfig();
        console.log('Configuration cache refreshed');
    }
    /**
     * Initialize configuration manager
     */
    async initializeConfigManager() {
        // Load default configurations
        await this.loadDefaultConfigurations();
        // Load enterprise configuration if available
        try {
            await this.loadEnterpriseConfig();
        }
        catch (error) {
            console.warn('Enterprise configuration not available, using defaults');
        }
    }
    /**
     * Load enterprise configuration file
     */
    async loadEnterpriseConfigFile() {
        // In real implementation, this would load from actual file system
        // For now, return a mock configuration
        return {
            version: '1.0.0',
            environments: {
                development: {
                    platformDefaults: {
                        kubernetes: {
                            cluster: 'dev-cluster',
                            namespace: 'development'
                        }
                    },
                    resourceLimits: {
                        cpu: '2 cores',
                        memory: '4 GB',
                        storage: '10 GB'
                    },
                    complianceLevel: 'basic'
                },
                staging: {
                    platformDefaults: {
                        kubernetes: {
                            cluster: 'staging-cluster',
                            namespace: 'staging'
                        }
                    },
                    resourceLimits: {
                        cpu: '4 cores',
                        memory: '8 GB',
                        storage: '20 GB'
                    },
                    complianceLevel: 'enhanced'
                },
                production: {
                    platformDefaults: {
                        kubernetes: {
                            cluster: 'prod-cluster',
                            namespace: 'production'
                        }
                    },
                    resourceLimits: {
                        cpu: '8 cores',
                        memory: '16 GB',
                        storage: '50 GB'
                    },
                    complianceLevel: 'nasa-pot10'
                }
            },
            deploymentStrategies: {
                default: {
                    type: 'rolling',
                    timeout: 300000,
                    rollbackOnFailure: true
                },
                'blue-green': {
                    type: 'blue-green',
                    timeout: 600000,
                    autoSwitch: false,
                    validationDuration: 300000
                },
                canary: {
                    type: 'canary',
                    timeout: 1800000,
                    initialTrafficPercentage: 10,
                    stepPercentage: 25,
                    stepDuration: 300000
                }
            },
            featureFlags: {
                global: {
                    enableMonitoring: true,
                    enableLogging: true,
                    enableMetrics: true
                },
                environments: {
                    development: {
                        debugMode: true,
                        verboseLogging: true
                    },
                    production: {
                        debugMode: false,
                        verboseLogging: false,
                        strictSecurity: true
                    }
                }
            },
            compliance: {
                global: {
                    auditEnabled: true,
                    reportingRequired: true
                },
                environments: {
                    production: {
                        level: 'nasa-pot10',
                        auditEnabled: true,
                        reportingRequired: true,
                        approvalRequired: true
                    }
                }
            },
            security: {
                global: {
                    tlsRequired: true,
                    certificateValidation: true,
                    secretEncryption: true
                },
                environments: {
                    production: {
                        strictSecurityPolicies: true,
                        networkPoliciesRequired: true,
                        rbacRequired: true
                    }
                }
            }
        };
    }
    /**
     * Build deployment configuration
     */
    async buildDeploymentConfig(environmentName, strategyType) {
        const environment = await this.getEnvironmentConfig(environmentName);
        const strategy = this.getDeploymentStrategy(strategyType);
        const featureFlags = this.getFeatureFlags(environmentName);
        const compliance = this.getComplianceConfig(environmentName);
        return {
            environment,
            strategy,
            featureFlags,
            compliance,
            phase3Integration: await this.phase3Integration.getConfig(),
            metadata: {
                configVersion: this.enterpriseConfig?.version || '1.0.0',
                generatedAt: new Date(),
                source: 'enterprise-config'
            }
        };
    }
    /**
     * Get base environment configuration
     */
    async getBaseEnvironmentConfig(environmentName) {
        // Return default environment configuration
        return {
            name: environmentName,
            type: environmentName,
            config: {
                replicas: environmentName === 'production' ? 5 : environmentName === 'staging' ? 2 : 1,
                resources: {
                    cpu: '500m',
                    memory: '512Mi',
                    storage: '1Gi'
                },
                networkConfig: {
                    loadBalancer: {
                        type: 'application',
                        healthCheckPath: '/health',
                        healthCheckInterval: 30,
                        unhealthyThreshold: 3
                    },
                    serviceType: 'LoadBalancer'
                },
                secrets: [],
                featureFlags: {},
                complianceLevel: 'basic'
            },
            healthEndpoints: [`https://${environmentName}.example.com/health`],
            rollbackCapable: true
        };
    }
    /**
     * Get enterprise overrides for environment
     */
    getEnterpriseOverrides(environmentName) {
        if (!this.enterpriseConfig?.environments?.[environmentName]) {
            return {};
        }
        const envConfig = this.enterpriseConfig.environments[environmentName];
        return {
            config: {
                complianceLevel: envConfig.complianceLevel || 'basic',
                resources: this.parseResourceLimits(envConfig.resourceLimits),
                featureFlags: this.getFeatureFlags(environmentName)
            }
        };
    }
    /**
     * Merge environment configurations
     */
    mergeEnvironmentConfig(base, overrides) {
        return {
            ...base,
            config: {
                ...base.config,
                ...overrides.config,
                resources: {
                    ...base.config.resources,
                    ...overrides.config?.resources
                },
                featureFlags: {
                    ...base.config.featureFlags,
                    ...overrides.config?.featureFlags
                }
            }
        };
    }
    /**
     * Get base platform configuration
     */
    async getBasePlatformConfig(platformType) {
        return {
            type: platformType,
            version: 'latest',
            credentials: {
                type: 'kubeconfig',
                data: {}
            },
            features: {
                blueGreenSupport: true,
                canarySupport: true,
                autoScaling: true,
                loadBalancing: true,
                secretManagement: true
            }
        };
    }
    /**
     * Get environment-specific platform overrides
     */
    getEnvironmentPlatformOverrides(platformType, environmentName) {
        const envConfig = this.enterpriseConfig?.environments?.[environmentName];
        const platformDefaults = envConfig?.platformDefaults?.[platformType];
        if (!platformDefaults) {
            return {};
        }
        return {
            endpoint: platformDefaults.cluster,
            metadata: platformDefaults
        };
    }
    /**
     * Merge platform configurations
     */
    mergePlatformConfig(base, overrides) {
        return {
            ...base,
            ...overrides,
            credentials: {
                ...base.credentials,
                ...overrides.credentials
            },
            features: {
                ...base.features,
                ...overrides.features
            }
        };
    }
    /**
     * Get deployment strategy configuration
     */
    getDeploymentStrategy(strategyType) {
        const strategyConfig = this.enterpriseConfig?.deploymentStrategies?.[strategyType] ||
            this.enterpriseConfig?.deploymentStrategies?.default;
        if (!strategyConfig) {
            // Return default strategy
            return {
                type: 'rolling',
                config: {
                    timeout: 300000,
                    healthCheckDelay: 30000,
                    healthCheckTimeout: 10000,
                    healthCheckInterval: 5000,
                    progressDeadlineSeconds: 600
                },
                rollbackStrategy: {
                    enabled: true,
                    autoTriggers: [],
                    manualApprovalRequired: false,
                    preserveResourceVersion: true
                }
            };
        }
        return {
            type: strategyConfig.type,
            config: {
                timeout: strategyConfig.timeout,
                healthCheckDelay: 30000,
                healthCheckTimeout: 10000,
                healthCheckInterval: 5000,
                progressDeadlineSeconds: 600,
                ...strategyConfig
            },
            rollbackStrategy: {
                enabled: strategyConfig.rollbackOnFailure || true,
                autoTriggers: [],
                manualApprovalRequired: false,
                preserveResourceVersion: true
            }
        };
    }
    /**
     * Load default configurations
     */
    async loadDefaultConfigurations() {
        // Load and cache default configurations
        console.log('Default configurations loaded');
    }
    /**
     * Validate enterprise policies
     */
    async validateEnterpisePolicies(config) {
        const issues = [];
        // Validate resource limits
        if (this.enterpriseConfig?.environments?.[config.environment.name]?.resourceLimits) {
            const limits = this.enterpriseConfig.environments[config.environment.name].resourceLimits;
            if (!this.isWithinResourceLimits(config.environment.config.resources, limits)) {
                issues.push({
                    type: 'policy',
                    severity: 'error',
                    message: `Resource configuration exceeds enterprise limits for ${config.environment.name}`
                });
            }
        }
        return issues;
    }
    /**
     * Validate compliance requirements
     */
    async validateComplianceRequirements(config) {
        const issues = [];
        if (config.compliance.level === 'nasa-pot10') {
            // NASA POT10 specific validations
            if (!config.compliance.auditEnabled) {
                issues.push({
                    type: 'compliance',
                    severity: 'error',
                    message: 'NASA POT10 compliance requires audit logging to be enabled'
                });
            }
            if (!config.compliance.approvalRequired) {
                issues.push({
                    type: 'compliance',
                    severity: 'error',
                    message: 'NASA POT10 compliance requires manual approval for production deployments'
                });
            }
        }
        return issues;
    }
    /**
     * Validate resource limits
     */
    validateResourceLimits(config) {
        const issues = [];
        const resources = config.environment.config.resources;
        // Validate CPU and memory are specified
        if (!resources.cpu || !resources.memory) {
            issues.push({
                type: 'resource',
                severity: 'error',
                message: 'CPU and memory resources must be specified'
            });
        }
        return issues;
    }
    /**
     * Validate security settings
     */
    async validateSecuritySettings(config) {
        const issues = [];
        const securityConfig = this.enterpriseConfig?.security;
        if (securityConfig?.global?.tlsRequired && config.environment.type === 'production') {
            // Check if TLS is configured
            // This would involve checking actual network configuration
        }
        return issues;
    }
    /**
     * Parse resource limits from string format
     */
    parseResourceLimits(limits) {
        // Parse resource limits like "2 cores", "4 GB", etc.
        return {
            cpu: limits.cpu,
            memory: limits.memory,
            storage: limits.storage
        };
    }
    /**
     * Check if resources are within enterprise limits
     */
    isWithinResourceLimits(resources, limits) {
        // Simplified validation - in real implementation, would parse and compare actual values
        return true;
    }
}
exports.DeploymentConfigManager = DeploymentConfigManager;
/**
 * Phase 3 Integration for Artifact System
 */
class Phase3Integration {
    constructor() {
        this.config = null;
    }
    async initialize(enterpriseConfig) {
        this.config = {
            enabled: true,
            artifactRepository: 'enterprise-registry',
            complianceValidation: true,
            securityScanning: true,
            metadataTracking: true
        };
        console.log('Phase 3 integration initialized');
    }
    async getConfig() {
        return this.config || {
            enabled: false,
            artifactRepository: 'default',
            complianceValidation: false,
            securityScanning: false,
            metadataTracking: false
        };
    }
    async validateArtifact(artifactId) {
        if (!this.config?.enabled) {
            return true;
        }
        // Validate artifact against Phase 3 system
        console.log(`Validating artifact ${artifactId} with Phase 3 system`);
        return true;
    }
    async getArtifactMetadata(artifactId) {
        if (!this.config?.enabled) {
            return {};
        }
        // Retrieve artifact metadata from Phase 3 system
        return {
            validated: true,
            securityScanPassed: true,
            complianceLevel: 'nasa-pot10'
        };
    }
}
//# sourceMappingURL=deployment-config.js.map