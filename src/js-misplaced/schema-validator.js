"use strict";
/**
 * Enterprise Configuration Schema Validator
 * Comprehensive validation system for enterprise configuration integrity
 * Supports JSON Schema validation, runtime validation, and configuration drift detection
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.EnterpriseConfigValidator = void 0;
const zod_1 = require("zod");
const ajv_1 = __importDefault(require("ajv"));
const ajv_formats_1 = __importDefault(require("ajv-formats"));
const js_yaml_1 = __importDefault(require("js-yaml"));
const promises_1 = __importDefault(require("fs/promises"));
const crypto_1 = require("crypto");
// Security configuration schema
const SecurityConfigSchema = zod_1.z.object({
    authentication: zod_1.z.object({
        enabled: zod_1.z.boolean(),
        method: zod_1.z.enum(['basic', 'oauth2', 'saml', 'ldap', 'multi_factor']),
        session_timeout: zod_1.z.number().min(300).max(86400),
        max_concurrent_sessions: zod_1.z.number().min(1).max(100),
        password_policy: zod_1.z.object({
            min_length: zod_1.z.number().min(8).max(128),
            require_uppercase: zod_1.z.boolean(),
            require_lowercase: zod_1.z.boolean(),
            require_numbers: zod_1.z.boolean(),
            require_special_chars: zod_1.z.boolean(),
            expiry_days: zod_1.z.number().min(30).max(365)
        })
    }),
    authorization: zod_1.z.object({
        rbac_enabled: zod_1.z.boolean(),
        default_role: zod_1.z.string(),
        roles: zod_1.z.record(zod_1.z.object({
            permissions: zod_1.z.array(zod_1.z.string())
        }))
    }),
    audit: zod_1.z.object({
        enabled: zod_1.z.boolean(),
        log_level: zod_1.z.enum(['basic', 'detailed', 'comprehensive']),
        retention_days: zod_1.z.number().min(30).max(2555), // 7 years max
        export_format: zod_1.z.enum(['json', 'csv', 'xml']),
        real_time_monitoring: zod_1.z.boolean(),
        anomaly_detection: zod_1.z.boolean()
    }),
    encryption: zod_1.z.object({
        at_rest: zod_1.z.boolean(),
        in_transit: zod_1.z.boolean(),
        algorithm: zod_1.z.string(),
        key_rotation_days: zod_1.z.number().min(30).max(365)
    })
});
// Performance configuration schema
const PerformanceConfigSchema = zod_1.z.object({
    scaling: zod_1.z.object({
        auto_scaling_enabled: zod_1.z.boolean(),
        min_workers: zod_1.z.number().min(1).max(100),
        max_workers: zod_1.z.number().min(1).max(1000),
        scale_up_threshold: zod_1.z.number().min(0.1).max(1.0),
        scale_down_threshold: zod_1.z.number().min(0.1).max(1.0),
        cooldown_period: zod_1.z.number().min(60).max(3600)
    }),
    resource_limits: zod_1.z.object({
        max_memory_mb: zod_1.z.number().min(512).max(32768),
        max_cpu_cores: zod_1.z.number().min(1).max(64),
        max_file_size_mb: zod_1.z.number().min(1).max(1024),
        max_analysis_time_seconds: zod_1.z.number().min(60).max(7200),
        max_concurrent_analyses: zod_1.z.number().min(1).max(100)
    }),
    caching: zod_1.z.object({
        enabled: zod_1.z.boolean(),
        provider: zod_1.z.enum(['memory', 'redis', 'memcached']),
        ttl_seconds: zod_1.z.number().min(60).max(86400),
        max_cache_size_mb: zod_1.z.number().min(64).max(8192),
        cache_compression: zod_1.z.boolean()
    }),
    database: zod_1.z.object({
        connection_pool_size: zod_1.z.number().min(5).max(100),
        query_timeout_seconds: zod_1.z.number().min(5).max(300),
        read_replica_enabled: zod_1.z.boolean(),
        indexing_strategy: zod_1.z.string()
    })
});
// Main enterprise configuration schema
const EnterpriseConfigSchema = zod_1.z.object({
    schema: zod_1.z.object({
        version: zod_1.z.string(),
        format_version: zod_1.z.string(),
        compatibility_level: zod_1.z.enum(['forward', 'backward', 'strict']),
        migration_required: zod_1.z.boolean()
    }),
    enterprise: zod_1.z.object({
        enabled: zod_1.z.boolean(),
        license_mode: zod_1.z.enum(['community', 'professional', 'enterprise']),
        compliance_level: zod_1.z.enum(['standard', 'strict', 'nasa-pot10', 'defense']),
        features: zod_1.z.record(zod_1.z.boolean())
    }),
    security: SecurityConfigSchema,
    multi_tenancy: zod_1.z.object({
        enabled: zod_1.z.boolean(),
        isolation_level: zod_1.z.enum(['basic', 'enhanced', 'complete']),
        tenant_specific_config: zod_1.z.boolean(),
        resource_quotas: zod_1.z.object({
            max_users_per_tenant: zod_1.z.number().min(1).max(10000),
            max_projects_per_tenant: zod_1.z.number().min(1).max(1000),
            max_analysis_jobs_per_day: zod_1.z.number().min(100).max(100000),
            storage_limit_gb: zod_1.z.number().min(10).max(10000)
        }),
        default_tenant: zod_1.z.object({
            name: zod_1.z.string().min(1).max(100),
            admin_email: zod_1.z.string().email(),
            compliance_profile: zod_1.z.string()
        })
    }),
    performance: PerformanceConfigSchema,
    integrations: zod_1.z.object({
        api: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            version: zod_1.z.string(),
            rate_limiting: zod_1.z.object({
                enabled: zod_1.z.boolean(),
                requests_per_minute: zod_1.z.number().min(10).max(10000),
                burst_limit: zod_1.z.number().min(10).max(1000)
            }),
            authentication_required: zod_1.z.boolean(),
            cors_enabled: zod_1.z.boolean(),
            swagger_ui_enabled: zod_1.z.boolean()
        }),
        webhooks: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            max_endpoints: zod_1.z.number().min(1).max(100),
            timeout_seconds: zod_1.z.number().min(5).max(300),
            retry_attempts: zod_1.z.number().min(0).max(10),
            signature_verification: zod_1.z.boolean()
        }),
        external_systems: zod_1.z.record(zod_1.z.object({
            enabled: zod_1.z.boolean(),
            url: zod_1.z.string().optional(),
            api_version: zod_1.z.string().optional()
        }).passthrough()),
        ci_cd: zod_1.z.record(zod_1.z.object({
            enabled: zod_1.z.boolean(),
            url: zod_1.z.string().optional()
        }).passthrough())
    }),
    monitoring: zod_1.z.object({
        metrics: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            provider: zod_1.z.enum(['prometheus', 'datadog', 'new_relic']),
            collection_interval: zod_1.z.number().min(10).max(300),
            retention_days: zod_1.z.number().min(7).max(365),
            custom_metrics: zod_1.z.boolean()
        }),
        logging: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            level: zod_1.z.enum(['debug', 'info', 'warn', 'error']),
            format: zod_1.z.enum(['text', 'json', 'structured']),
            output: zod_1.z.array(zod_1.z.enum(['console', 'file', 'syslog', 'elasticsearch'])),
            file_rotation: zod_1.z.boolean(),
            max_file_size_mb: zod_1.z.number().min(10).max(1000),
            max_files: zod_1.z.number().min(1).max(100)
        }),
        tracing: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            sampling_rate: zod_1.z.number().min(0.0).max(1.0),
            provider: zod_1.z.enum(['jaeger', 'zipkin', 'datadog'])
        }),
        alerts: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            channels: zod_1.z.array(zod_1.z.enum(['email', 'slack', 'teams', 'pagerduty'])),
            thresholds: zod_1.z.object({
                error_rate: zod_1.z.number().min(0.0).max(1.0),
                response_time_p95: zod_1.z.number().min(100).max(60000),
                memory_usage: zod_1.z.number().min(0.0).max(1.0),
                cpu_usage: zod_1.z.number().min(0.0).max(1.0)
            })
        })
    }),
    analytics: zod_1.z.object({
        enabled: zod_1.z.boolean(),
        data_retention_days: zod_1.z.number().min(30).max(2555),
        trend_analysis: zod_1.z.boolean(),
        predictive_insights: zod_1.z.boolean(),
        custom_dashboards: zod_1.z.boolean(),
        scheduled_reports: zod_1.z.boolean(),
        machine_learning: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            model_training: zod_1.z.boolean(),
            anomaly_detection: zod_1.z.boolean(),
            pattern_recognition: zod_1.z.boolean(),
            automated_insights: zod_1.z.boolean()
        }),
        export_formats: zod_1.z.array(zod_1.z.enum(['pdf', 'excel', 'csv', 'json'])),
        real_time_streaming: zod_1.z.boolean()
    }),
    governance: zod_1.z.object({
        quality_gates: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            enforce_blocking: zod_1.z.boolean(),
            custom_rules: zod_1.z.boolean(),
            nasa_compliance: zod_1.z.object({
                enabled: zod_1.z.boolean(),
                minimum_score: zod_1.z.number().min(0.0).max(1.0),
                critical_violations_allowed: zod_1.z.number().min(0).max(100),
                high_violations_allowed: zod_1.z.number().min(0).max(100),
                automated_remediation_suggestions: zod_1.z.boolean()
            }),
            custom_gates: zod_1.z.record(zod_1.z.union([zod_1.z.number(), zod_1.z.boolean(), zod_1.z.string()]))
        }),
        policies: zod_1.z.object({
            code_standards: zod_1.z.string(),
            security_requirements: zod_1.z.string(),
            documentation_mandatory: zod_1.z.boolean(),
            review_requirements: zod_1.z.object({
                min_approvers: zod_1.z.number().min(1).max(10),
                security_review_required: zod_1.z.boolean(),
                architecture_review_threshold: zod_1.z.number().min(1).max(1000)
            })
        })
    }),
    notifications: zod_1.z.object({
        enabled: zod_1.z.boolean(),
        channels: zod_1.z.record(zod_1.z.object({
            enabled: zod_1.z.boolean()
        }).passthrough()),
        templates: zod_1.z.record(zod_1.z.string()),
        escalation: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            levels: zod_1.z.array(zod_1.z.object({
                delay: zod_1.z.number().min(60).max(86400),
                recipients: zod_1.z.array(zod_1.z.string())
            }))
        })
    }),
    environments: zod_1.z.record(zod_1.z.record(zod_1.z.any())).optional(),
    legacy_integration: zod_1.z.object({
        preserve_existing_configs: zod_1.z.boolean(),
        migration_warnings: zod_1.z.boolean(),
        detector_config_path: zod_1.z.string(),
        analysis_config_path: zod_1.z.string(),
        conflict_resolution: zod_1.z.enum(['legacy_wins', 'enterprise_wins', 'merge'])
    }),
    extensions: zod_1.z.object({
        custom_detectors: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            directory: zod_1.z.string(),
            auto_discovery: zod_1.z.boolean()
        }),
        custom_reporters: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            directory: zod_1.z.string(),
            formats: zod_1.z.array(zod_1.z.string())
        }),
        plugins: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            directory: zod_1.z.string(),
            sandboxing: zod_1.z.boolean(),
            security_scanning: zod_1.z.boolean()
        })
    }),
    backup: zod_1.z.object({
        enabled: zod_1.z.boolean(),
        schedule: zod_1.z.string(),
        retention_days: zod_1.z.number().min(7).max(2555),
        encryption: zod_1.z.boolean(),
        offsite_storage: zod_1.z.boolean(),
        disaster_recovery: zod_1.z.object({
            enabled: zod_1.z.boolean(),
            rpo_minutes: zod_1.z.number().min(5).max(1440),
            rto_minutes: zod_1.z.number().min(30).max(28800),
            failover_testing: zod_1.z.boolean(),
            automated_failover: zod_1.z.boolean()
        })
    }),
    validation: zod_1.z.object({
        schema_validation: zod_1.z.boolean(),
        runtime_validation: zod_1.z.boolean(),
        configuration_drift_detection: zod_1.z.boolean(),
        rules: zod_1.z.array(zod_1.z.object({
            name: zod_1.z.string(),
            condition: zod_1.z.string(),
            environment: zod_1.z.string().optional(),
            severity: zod_1.z.enum(['info', 'warning', 'error'])
        }))
    })
});
/**
 * Enterprise Configuration Schema Validator
 * Provides comprehensive validation for enterprise configuration files
 */
class EnterpriseConfigValidator {
    constructor() {
        this.configCache = new Map();
        this.validationRules = new Map();
        this.ajv = new ajv_1.default({ allErrors: true, verbose: true });
        (0, ajv_formats_1.default)(this.ajv);
        this.initializeCustomRules();
    }
    /**
     * Initialize custom validation rules
     */
    initializeCustomRules() {
        // Production security requirements
        this.validationRules.set('production-security', (config) => {
            const errors = [];
            if (!config.security.encryption.at_rest) {
                errors.push({
                    path: 'security.encryption.at_rest',
                    message: 'Encryption at rest must be enabled in production environments',
                    severity: 'critical',
                    rule: 'production-security',
                    suggestion: 'Set security.encryption.at_rest to true'
                });
            }
            if (!config.security.audit.enabled) {
                errors.push({
                    path: 'security.audit.enabled',
                    message: 'Audit logging must be enabled in production environments',
                    severity: 'critical',
                    rule: 'production-security',
                    suggestion: 'Set security.audit.enabled to true'
                });
            }
            if (config.security.audit.retention_days < 365) {
                errors.push({
                    path: 'security.audit.retention_days',
                    message: 'Audit logs must be retained for at least 365 days in production',
                    severity: 'error',
                    rule: 'production-security',
                    suggestion: 'Set security.audit.retention_days to 365 or higher'
                });
            }
            return errors;
        });
        // Performance limits validation
        this.validationRules.set('performance-limits', (config) => {
            const errors = [];
            if (config.performance.resource_limits.max_memory_mb > 16384) {
                errors.push({
                    path: 'performance.resource_limits.max_memory_mb',
                    message: 'Memory limit exceeds recommended maximum of 16GB',
                    severity: 'error',
                    rule: 'performance-limits',
                    suggestion: 'Consider reducing max_memory_mb to 16384 or lower'
                });
            }
            if (config.performance.scaling.max_workers > 100) {
                errors.push({
                    path: 'performance.scaling.max_workers',
                    message: 'Max workers exceeds recommended limit of 100',
                    severity: 'error',
                    rule: 'performance-limits',
                    suggestion: 'Consider reducing max_workers to 100 or implementing worker pools'
                });
            }
            return errors;
        });
        // NASA compliance validation
        this.validationRules.set('nasa-compliance', (config) => {
            const errors = [];
            if (config.enterprise.compliance_level === 'nasa-pot10') {
                if (!config.governance.quality_gates.nasa_compliance.enabled) {
                    errors.push({
                        path: 'governance.quality_gates.nasa_compliance.enabled',
                        message: 'NASA POT10 compliance gates must be enabled for nasa-pot10 compliance level',
                        severity: 'critical',
                        rule: 'nasa-compliance'
                    });
                }
                if (config.governance.quality_gates.nasa_compliance.minimum_score < 0.95) {
                    errors.push({
                        path: 'governance.quality_gates.nasa_compliance.minimum_score',
                        message: 'NASA POT10 compliance requires minimum score of 0.95',
                        severity: 'critical',
                        rule: 'nasa-compliance'
                    });
                }
                if (config.governance.quality_gates.nasa_compliance.critical_violations_allowed > 0) {
                    errors.push({
                        path: 'governance.quality_gates.nasa_compliance.critical_violations_allowed',
                        message: 'NASA POT10 compliance requires zero critical violations',
                        severity: 'critical',
                        rule: 'nasa-compliance'
                    });
                }
            }
            return errors;
        });
    }
    /**
     * Validate enterprise configuration file
     */
    async validateConfig(configPath, environment) {
        try {
            const configContent = await promises_1.default.readFile(configPath, 'utf-8');
            const config = js_yaml_1.default.load(configContent);
            return this.validateConfigObject(config, environment);
        }
        catch (error) {
            return {
                isValid: false,
                errors: [{
                        path: 'root',
                        message: `Failed to load configuration: ${error.message}`,
                        severity: 'critical'
                    }],
                warnings: [],
                metadata: {
                    validatedAt: new Date(),
                    validator: 'EnterpriseConfigValidator',
                    schemaVersion: '1.0',
                    configHash: '',
                    environment
                }
            };
        }
    }
    /**
     * Validate configuration object
     */
    validateConfigObject(config, environment) {
        const errors = [];
        const warnings = [];
        const configHash = this.calculateConfigHash(config);
        // Schema validation using Zod
        try {
            const validatedConfig = EnterpriseConfigSchema.parse(config);
            // Apply environment-specific overrides
            if (environment && config.environments?.[environment]) {
                this.applyEnvironmentOverrides(validatedConfig, config.environments[environment]);
            }
            // Custom validation rules
            for (const [ruleName, rule] of this.validationRules.entries()) {
                try {
                    const ruleErrors = rule(validatedConfig);
                    // Filter errors by environment if specified
                    if (environment) {
                        errors.push(...ruleErrors.filter(error => !error.rule || this.isRuleApplicableToEnvironment(error.rule, environment)));
                    }
                    else {
                        errors.push(...ruleErrors);
                    }
                }
                catch (ruleError) {
                    warnings.push({
                        path: 'validation',
                        message: `Custom rule '${ruleName}' failed: ${ruleError.message}`,
                        rule: ruleName
                    });
                }
            }
            // Environment-specific validation
            if (environment) {
                const envErrors = this.validateEnvironmentSpecificRules(validatedConfig, environment);
                errors.push(...envErrors);
            }
            // Cache validated configuration
            this.configCache.set(configHash, {
                config: validatedConfig,
                hash: configHash,
                timestamp: new Date()
            });
        }
        catch (zodError) {
            if (zodError instanceof zod_1.z.ZodError) {
                errors.push(...zodError.errors.map(err => ({
                    path: err.path.join('.'),
                    message: err.message,
                    severity: 'error',
                    suggestion: this.generateSuggestion(err.path.join('.'), err.code)
                })));
            }
            else {
                errors.push({
                    path: 'schema',
                    message: `Schema validation failed: ${zodError.message}`,
                    severity: 'critical'
                });
            }
        }
        return {
            isValid: errors.length === 0,
            errors,
            warnings,
            metadata: {
                validatedAt: new Date(),
                validator: 'EnterpriseConfigValidator',
                schemaVersion: '1.0',
                configHash,
                environment
            }
        };
    }
    /**
     * Detect configuration drift
     */
    async detectConfigurationDrift(currentConfigPath, baselineConfigPath) {
        try {
            const [currentConfig, baselineConfig] = await Promise.all([
                this.loadConfigFile(currentConfigPath),
                this.loadConfigFile(baselineConfigPath)
            ]);
            const changes = this.detectChanges(baselineConfig, currentConfig);
            const riskLevel = this.calculateRiskLevel(changes);
            return {
                hasDrift: changes.length > 0,
                changes,
                riskLevel,
                lastValidConfig: baselineConfigPath
            };
        }
        catch (error) {
            throw new Error(`Configuration drift detection failed: ${error.message}`);
        }
    }
    /**
     * Apply environment-specific configuration overrides
     */
    applyEnvironmentOverrides(config, overrides) {
        for (const [path, value] of Object.entries(overrides)) {
            this.setNestedProperty(config, path, value);
        }
    }
    /**
     * Validate environment-specific rules
     */
    validateEnvironmentSpecificRules(config, environment) {
        const errors = [];
        // Production-specific rules
        if (environment === 'production') {
            const productionRules = this.validationRules.get('production-security');
            if (productionRules) {
                errors.push(...productionRules(config));
            }
        }
        // Validate environment-specific configuration rules
        if (config.validation.rules) {
            for (const rule of config.validation.rules) {
                if (!rule.environment || rule.environment === environment) {
                    const conditionResult = this.evaluateCondition(rule.condition, config);
                    if (!conditionResult && rule.severity === 'error') {
                        errors.push({
                            path: rule.name,
                            message: `Validation rule '${rule.name}' failed: ${rule.condition}`,
                            severity: 'error',
                            rule: rule.name
                        });
                    }
                }
            }
        }
        return errors;
    }
    /**
     * Load and parse configuration file
     */
    async loadConfigFile(filePath) {
        const content = await promises_1.default.readFile(filePath, 'utf-8');
        return js_yaml_1.default.load(content);
    }
    /**
     * Detect changes between two configuration objects
     */
    detectChanges(baseline, current, path = '') {
        const changes = [];
        const baselineKeys = new Set(Object.keys(baseline || {}));
        const currentKeys = new Set(Object.keys(current || {}));
        // Detect removed keys
        for (const key of baselineKeys) {
            if (!currentKeys.has(key)) {
                changes.push({
                    path: path ? `${path}.${key}` : key,
                    type: 'removed',
                    oldValue: baseline[key],
                    impact: this.calculateChangeImpact(`${path}.${key}`, baseline[key], undefined)
                });
            }
        }
        // Detect added keys
        for (const key of currentKeys) {
            if (!baselineKeys.has(key)) {
                changes.push({
                    path: path ? `${path}.${key}` : key,
                    type: 'added',
                    newValue: current[key],
                    impact: this.calculateChangeImpact(`${path}.${key}`, undefined, current[key])
                });
            }
        }
        // Detect modified values
        for (const key of currentKeys) {
            if (baselineKeys.has(key)) {
                const currentPath = path ? `${path}.${key}` : key;
                if (typeof baseline[key] === 'object' && typeof current[key] === 'object') {
                    changes.push(...this.detectChanges(baseline[key], current[key], currentPath));
                }
                else if (baseline[key] !== current[key]) {
                    changes.push({
                        path: currentPath,
                        type: 'modified',
                        oldValue: baseline[key],
                        newValue: current[key],
                        impact: this.calculateChangeImpact(currentPath, baseline[key], current[key])
                    });
                }
            }
        }
        return changes;
    }
    /**
     * Calculate risk level based on changes
     */
    calculateRiskLevel(changes) {
        if (changes.length === 0)
            return 'low';
        const criticalChanges = changes.filter(c => c.impact === 'critical').length;
        const highChanges = changes.filter(c => c.impact === 'high').length;
        const mediumChanges = changes.filter(c => c.impact === 'medium').length;
        if (criticalChanges > 0)
            return 'critical';
        if (highChanges > 5)
            return 'critical';
        if (highChanges > 0 || mediumChanges > 10)
            return 'high';
        if (mediumChanges > 0 || changes.length > 20)
            return 'medium';
        return 'low';
    }
    /**
     * Calculate the impact of a configuration change
     */
    calculateChangeImpact(path, oldValue, newValue) {
        // Security-related changes are high impact
        if (path.includes('security.') || path.includes('encryption') || path.includes('authentication')) {
            return 'critical';
        }
        // Performance and resource changes
        if (path.includes('performance.') || path.includes('resource_limits')) {
            return 'high';
        }
        // Quality gates and governance
        if (path.includes('governance.') || path.includes('quality_gates')) {
            return 'high';
        }
        // Monitoring and alerting
        if (path.includes('monitoring.') || path.includes('alerts')) {
            return 'medium';
        }
        // Default to low impact
        return 'low';
    }
    /**
     * Calculate configuration hash for drift detection
     */
    calculateConfigHash(config) {
        const configString = JSON.stringify(config, Object.keys(config).sort());
        return (0, crypto_1.createHash)('sha256').update(configString).digest('hex');
    }
    /**
     * Set nested property using dot notation
     */
    setNestedProperty(obj, path, value) {
        const keys = path.split('.');
        let current = obj;
        for (let i = 0; i < keys.length - 1; i++) {
            if (!(keys[i] in current)) {
                current[keys[i]] = {};
            }
            current = current[keys[i]];
        }
        current[keys[keys.length - 1]] = value;
    }
    /**
     * Check if validation rule applies to specific environment
     */
    isRuleApplicableToEnvironment(ruleName, environment) {
        const environmentSpecificRules = {
            'production-security': ['production'],
            'nasa-compliance': ['production', 'staging'],
            'performance-limits': ['production', 'staging', 'development']
        };
        return environmentSpecificRules[ruleName]?.includes(environment) ?? true;
    }
    /**
     * Generate suggestion for validation error
     */
    generateSuggestion(path, errorCode) {
        const suggestions = {
            'invalid_type': `Check the data type for ${path}`,
            'required': `${path} is required and must be provided`,
            'too_small': `${path} value is too small, increase the value`,
            'too_big': `${path} value is too large, decrease the value`,
            'invalid_enum_value': `${path} must be one of the allowed values`
        };
        return suggestions[errorCode] || `Please check the value for ${path}`;
    }
    /**
     * Evaluate a condition string against configuration
     */
    evaluateCondition(condition, config) {
        try {
            // Simple condition evaluation - in production, use a proper expression parser
            const cleanCondition = condition.replace(/(\w+(?:\.\w+)*)/g, (match) => {
                const value = this.getNestedProperty(config, match);
                return JSON.stringify(value);
            });
            return new Function('return ' + cleanCondition)();
        }
        catch (error) {
            console.warn(`Failed to evaluate condition: ${condition}`, error);
            return false;
        }
    }
    /**
     * Get nested property using dot notation
     */
    getNestedProperty(obj, path) {
        return path.split('.').reduce((current, key) => current?.[key], obj);
    }
    /**
     * Clear validation cache
     */
    clearCache() {
        this.configCache.clear();
    }
    /**
     * Get cached configuration
     */
    getCachedConfig(hash) {
        const cached = this.configCache.get(hash);
        return cached ? cached.config : null;
    }
}
exports.EnterpriseConfigValidator = EnterpriseConfigValidator;
exports.default = EnterpriseConfigValidator;
//# sourceMappingURL=schema-validator.js.map