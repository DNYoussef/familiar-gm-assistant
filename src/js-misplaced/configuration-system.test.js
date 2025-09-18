"use strict";
/**
 * Comprehensive Unit Tests for Enterprise Configuration System
 * Tests all major components including validation, compatibility, and environment overrides
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const promises_1 = __importDefault(require("fs/promises"));
const js_yaml_1 = __importDefault(require("js-yaml"));
const schema_validator_1 = require("../../src/config/schema-validator");
const backward_compatibility_1 = require("../../src/config/backward-compatibility");
const configuration_manager_1 = require("../../src/config/configuration-manager");
const environment_overrides_1 = require("../../src/config/environment-overrides");
const migration_versioning_1 = require("../../src/config/migration-versioning");
// Mock file system operations
globals_1.jest.mock('fs/promises');
const mockFs = promises_1.default;
(0, globals_1.describe)('Enterprise Configuration System', () => {
    (0, globals_1.beforeEach)(() => {
        globals_1.jest.clearAllMocks();
    });
    (0, globals_1.afterEach)(() => {
        globals_1.jest.resetAllMocks();
    });
    (0, globals_1.describe)('EnterpriseConfigValidator', () => {
        let validator;
        (0, globals_1.beforeEach)(() => {
            validator = new schema_validator_1.EnterpriseConfigValidator();
        });
        (0, globals_1.describe)('Schema Validation', () => {
            (0, globals_1.it)('should validate a valid enterprise configuration', async () => {
                const validConfig = {
                    schema: {
                        version: '1.0',
                        format_version: '2024.1',
                        compatibility_level: 'backward',
                        migration_required: false
                    },
                    enterprise: {
                        enabled: true,
                        license_mode: 'enterprise',
                        compliance_level: 'nasa-pot10',
                        features: {
                            advanced_analytics: true,
                            multi_tenant_support: true
                        }
                    },
                    security: {
                        authentication: {
                            enabled: true,
                            method: 'oauth2',
                            session_timeout: 3600,
                            max_concurrent_sessions: 5,
                            password_policy: {
                                min_length: 12,
                                require_uppercase: true,
                                require_lowercase: true,
                                require_numbers: true,
                                require_special_chars: true,
                                expiry_days: 90
                            }
                        },
                        authorization: {
                            rbac_enabled: true,
                            default_role: 'viewer',
                            roles: {
                                admin: { permissions: ['read', 'write', 'admin'] }
                            }
                        },
                        audit: {
                            enabled: true,
                            log_level: 'detailed',
                            retention_days: 365,
                            export_format: 'json',
                            real_time_monitoring: true,
                            anomaly_detection: true
                        },
                        encryption: {
                            at_rest: true,
                            in_transit: true,
                            algorithm: 'AES-256-GCM',
                            key_rotation_days: 90
                        }
                    },
                    multi_tenancy: {
                        enabled: true,
                        isolation_level: 'complete',
                        tenant_specific_config: true,
                        resource_quotas: {
                            max_users_per_tenant: 1000,
                            max_projects_per_tenant: 100,
                            max_analysis_jobs_per_day: 10000,
                            storage_limit_gb: 1000
                        },
                        default_tenant: {
                            name: 'default',
                            admin_email: 'admin@company.com',
                            compliance_profile: 'standard'
                        }
                    },
                    performance: {
                        scaling: {
                            auto_scaling_enabled: true,
                            min_workers: 2,
                            max_workers: 20,
                            scale_up_threshold: 0.8,
                            scale_down_threshold: 0.3,
                            cooldown_period: 300
                        },
                        resource_limits: {
                            max_memory_mb: 8192,
                            max_cpu_cores: 8,
                            max_file_size_mb: 100,
                            max_analysis_time_seconds: 3600,
                            max_concurrent_analyses: 10
                        },
                        caching: {
                            enabled: true,
                            provider: 'redis',
                            ttl_seconds: 3600,
                            max_cache_size_mb: 1024,
                            cache_compression: true
                        },
                        database: {
                            connection_pool_size: 20,
                            query_timeout_seconds: 30,
                            read_replica_enabled: true,
                            indexing_strategy: 'optimized'
                        }
                    },
                    integrations: {
                        api: {
                            enabled: true,
                            version: 'v1',
                            rate_limiting: {
                                enabled: true,
                                requests_per_minute: 1000,
                                burst_limit: 100
                            },
                            authentication_required: true,
                            cors_enabled: true,
                            swagger_ui_enabled: true
                        },
                        webhooks: {
                            enabled: true,
                            max_endpoints: 50,
                            timeout_seconds: 30,
                            retry_attempts: 3,
                            signature_verification: true
                        },
                        external_systems: {
                            github: {
                                enabled: true,
                                url: 'https://api.github.com'
                            }
                        },
                        ci_cd: {
                            github_actions: {
                                enabled: true
                            }
                        }
                    },
                    monitoring: {
                        metrics: {
                            enabled: true,
                            provider: 'prometheus',
                            collection_interval: 30,
                            retention_days: 30,
                            custom_metrics: true
                        },
                        logging: {
                            enabled: true,
                            level: 'info',
                            format: 'json',
                            output: ['console', 'file'],
                            file_rotation: true,
                            max_file_size_mb: 100,
                            max_files: 10
                        },
                        tracing: {
                            enabled: true,
                            sampling_rate: 0.1,
                            provider: 'jaeger'
                        },
                        alerts: {
                            enabled: true,
                            channels: ['email', 'slack'],
                            thresholds: {
                                error_rate: 0.05,
                                response_time_p95: 2000,
                                memory_usage: 0.85,
                                cpu_usage: 0.90
                            }
                        }
                    },
                    analytics: {
                        enabled: true,
                        data_retention_days: 730,
                        trend_analysis: true,
                        predictive_insights: true,
                        custom_dashboards: true,
                        scheduled_reports: true,
                        machine_learning: {
                            enabled: true,
                            model_training: true,
                            anomaly_detection: true,
                            pattern_recognition: true,
                            automated_insights: true
                        },
                        export_formats: ['pdf', 'excel', 'csv', 'json'],
                        real_time_streaming: true
                    },
                    governance: {
                        quality_gates: {
                            enabled: true,
                            enforce_blocking: true,
                            custom_rules: true,
                            nasa_compliance: {
                                enabled: true,
                                minimum_score: 0.95,
                                critical_violations_allowed: 0,
                                high_violations_allowed: 0,
                                automated_remediation_suggestions: true
                            },
                            custom_gates: {
                                code_coverage: 0.80
                            }
                        },
                        policies: {
                            code_standards: 'enterprise',
                            security_requirements: 'strict',
                            documentation_mandatory: true,
                            review_requirements: {
                                min_approvers: 2,
                                security_review_required: true,
                                architecture_review_threshold: 100
                            }
                        }
                    },
                    notifications: {
                        enabled: true,
                        channels: {
                            email: { enabled: true }
                        },
                        templates: {
                            alert: 'templates/alert.html'
                        },
                        escalation: {
                            enabled: true,
                            levels: [
                                { delay: 300, recipients: ['team'] }
                            ]
                        }
                    },
                    legacy_integration: {
                        preserve_existing_configs: true,
                        migration_warnings: true,
                        detector_config_path: 'analyzer/config/detector_config.yaml',
                        analysis_config_path: 'analyzer/config/analysis_config.yaml',
                        conflict_resolution: 'merge'
                    },
                    extensions: {
                        custom_detectors: {
                            enabled: true,
                            directory: 'extensions/detectors',
                            auto_discovery: true
                        },
                        custom_reporters: {
                            enabled: true,
                            directory: 'extensions/reporters',
                            formats: ['json']
                        },
                        plugins: {
                            enabled: true,
                            directory: 'extensions/plugins',
                            sandboxing: true,
                            security_scanning: true
                        }
                    },
                    backup: {
                        enabled: true,
                        schedule: '0 2 * * *',
                        retention_days: 90,
                        encryption: true,
                        offsite_storage: true,
                        disaster_recovery: {
                            enabled: true,
                            rpo_minutes: 60,
                            rto_minutes: 240,
                            failover_testing: true,
                            automated_failover: false
                        }
                    },
                    validation: {
                        schema_validation: true,
                        runtime_validation: true,
                        configuration_drift_detection: true,
                        rules: []
                    }
                };
                const result = validator.validateConfigObject(validConfig);
                (0, globals_1.expect)(result.isValid).toBe(true);
                (0, globals_1.expect)(result.errors).toHaveLength(0);
                (0, globals_1.expect)(result.metadata.validator).toBe('EnterpriseConfigValidator');
            });
            (0, globals_1.it)('should detect invalid configuration values', async () => {
                const invalidConfig = {
                    schema: {
                        version: '1.0',
                        format_version: '2024.1',
                        compatibility_level: 'invalid', // Invalid enum value
                        migration_required: false
                    },
                    enterprise: {
                        enabled: 'yes', // Should be boolean
                        license_mode: 'enterprise',
                        compliance_level: 'nasa-pot10',
                        features: {}
                    }
                };
                const result = validator.validateConfigObject(invalidConfig);
                (0, globals_1.expect)(result.isValid).toBe(false);
                (0, globals_1.expect)(result.errors.length).toBeGreaterThan(0);
                (0, globals_1.expect)(result.errors.some(e => e.path.includes('compatibility_level'))).toBe(true);
                (0, globals_1.expect)(result.errors.some(e => e.path.includes('enabled'))).toBe(true);
            });
            (0, globals_1.it)('should validate NASA POT10 compliance rules', async () => {
                const nasaConfig = {
                    schema: { version: '1.0', format_version: '2024.1', compatibility_level: 'backward', migration_required: false },
                    enterprise: { enabled: true, license_mode: 'enterprise', compliance_level: 'nasa-pot10', features: {} },
                    governance: {
                        quality_gates: {
                            enabled: true,
                            enforce_blocking: true,
                            custom_rules: true,
                            nasa_compliance: {
                                enabled: true,
                                minimum_score: 0.90, // Below required 0.95
                                critical_violations_allowed: 1, // Should be 0
                                high_violations_allowed: 0,
                                automated_remediation_suggestions: true
                            },
                            custom_gates: {}
                        },
                        policies: {
                            code_standards: 'standard',
                            security_requirements: 'standard',
                            documentation_mandatory: false,
                            review_requirements: { min_approvers: 1, security_review_required: false, architecture_review_threshold: 100 }
                        }
                    }
                };
                const result = validator.validateConfigObject(nasaConfig, 'production');
                (0, globals_1.expect)(result.isValid).toBe(false);
                (0, globals_1.expect)(result.errors.some(e => e.rule === 'nasa-compliance')).toBe(true);
            });
        });
        (0, globals_1.describe)('Configuration Drift Detection', () => {
            (0, globals_1.it)('should detect configuration drift', async () => {
                const baselineConfig = { version: '1.0', feature_a: true, setting_x: 100 };
                const currentConfig = { version: '1.0', feature_a: false, setting_y: 200 }; // Changed and added
                mockFs.readFile
                    .mockResolvedValueOnce(js_yaml_1.default.dump(baselineConfig))
                    .mockResolvedValueOnce(js_yaml_1.default.dump(currentConfig));
                const result = await validator.detectConfigurationDrift('current.yaml', 'baseline.yaml');
                (0, globals_1.expect)(result.hasDrift).toBe(true);
                (0, globals_1.expect)(result.changes).toHaveLength(2); // One modified, one added
                (0, globals_1.expect)(result.changes.some(c => c.type === 'modified' && c.path === 'feature_a')).toBe(true);
                (0, globals_1.expect)(result.changes.some(c => c.type === 'added' && c.path === 'setting_y')).toBe(true);
            });
            (0, globals_1.it)('should calculate risk level correctly', async () => {
                const baselineConfig = { security: { enabled: true } };
                const currentConfig = { security: { enabled: false } }; // High impact change
                mockFs.readFile
                    .mockResolvedValueOnce(js_yaml_1.default.dump(baselineConfig))
                    .mockResolvedValueOnce(js_yaml_1.default.dump(currentConfig));
                const result = await validator.detectConfigurationDrift('current.yaml', 'baseline.yaml');
                (0, globals_1.expect)(result.riskLevel).toBe('critical');
            });
        });
    });
    (0, globals_1.describe)('BackwardCompatibilityManager', () => {
        let compatibilityManager;
        (0, globals_1.beforeEach)(() => {
            compatibilityManager = new backward_compatibility_1.BackwardCompatibilityManager();
        });
        (0, globals_1.describe)('Legacy Configuration Loading', () => {
            (0, globals_1.it)('should load legacy detector configuration', async () => {
                const legacyDetectorConfig = {
                    god_object_detector: {
                        method_threshold: 20,
                        loc_threshold: 500,
                        parameter_threshold: 10
                    },
                    magic_literal_detector: {
                        thresholds: {
                            number_repetition: 3,
                            string_repetition: 2
                        }
                    }
                };
                mockFs.readFile.mockResolvedValueOnce(js_yaml_1.default.dump(legacyDetectorConfig));
                const result = await compatibilityManager.loadLegacyConfigs('detector_config.yaml');
                (0, globals_1.expect)(result.detector).toBeDefined();
                (0, globals_1.expect)(result.detector?.god_object_detector.method_threshold).toBe(20);
                (0, globals_1.expect)(result.detector?.magic_literal_detector.thresholds.number_repetition).toBe(3);
            });
            (0, globals_1.it)('should load legacy analysis configuration', async () => {
                const legacyAnalysisConfig = {
                    analysis: {
                        default_policy: 'standard',
                        max_file_size_mb: 10,
                        max_analysis_time_seconds: 300,
                        parallel_workers: 4,
                        cache_enabled: true
                    },
                    quality_gates: {
                        overall_quality_threshold: 0.75,
                        critical_violation_limit: 0,
                        high_violation_limit: 5
                    }
                };
                mockFs.readFile.mockResolvedValueOnce(js_yaml_1.default.dump(legacyAnalysisConfig));
                const result = await compatibilityManager.loadLegacyConfigs(undefined, 'analysis_config.yaml');
                (0, globals_1.expect)(result.analysis).toBeDefined();
                (0, globals_1.expect)(result.analysis?.analysis.max_file_size_mb).toBe(10);
                (0, globals_1.expect)(result.analysis?.quality_gates.overall_quality_threshold).toBe(0.75);
            });
        });
        (0, globals_1.describe)('Legacy Configuration Migration', () => {
            (0, globals_1.it)('should migrate legacy configuration to enterprise format', async () => {
                const legacyConfigs = {
                    detector: {
                        god_object_detector: {
                            method_threshold: 20,
                            loc_threshold: 500,
                            parameter_threshold: 10
                        }
                    },
                    analysis: {
                        analysis: {
                            max_file_size_mb: 10,
                            parallel_workers: 4
                        },
                        quality_gates: {
                            overall_quality_threshold: 0.85
                        }
                    }
                };
                const result = await compatibilityManager.migrateLegacyConfig(legacyConfigs);
                (0, globals_1.expect)(result.success).toBe(true);
                (0, globals_1.expect)(result.migratedConfig.performance?.resource_limits?.max_file_size_mb).toBe(10);
                (0, globals_1.expect)(result.migratedConfig.governance?.quality_gates?.custom_gates?.overall_threshold).toBe(0.85);
            });
            (0, globals_1.it)('should handle migration conflicts appropriately', async () => {
                const legacyConfigs = {
                    detector: null,
                    analysis: {
                        analysis: {
                            parallel_workers: 25 // Exceeds recommended limit
                        }
                    }
                };
                const result = await compatibilityManager.migrateLegacyConfig(legacyConfigs, 'merge');
                (0, globals_1.expect)(result.success).toBe(true);
                (0, globals_1.expect)(result.warnings.length).toBeGreaterThan(0);
                (0, globals_1.expect)(result.warnings.some(w => w.message.includes('exceeds recommended limit'))).toBe(true);
            });
        });
    });
    (0, globals_1.describe)('EnvironmentOverrideSystem', () => {
        let overrideSystem;
        let originalEnv;
        (0, globals_1.beforeEach)(() => {
            originalEnv = { ...process.env };
            overrideSystem = new environment_overrides_1.EnvironmentOverrideSystem();
        });
        (0, globals_1.afterEach)(() => {
            process.env = originalEnv;
        });
        (0, globals_1.describe)('Environment Variable Processing', () => {
            (0, globals_1.it)('should process boolean environment variables', async () => {
                process.env.ENTERPRISE_CONFIG_ENTERPRISE_ENABLED = 'true';
                process.env.ENTERPRISE_CONFIG_SECURITY_AUTH_ENABLED = 'false';
                const result = await overrideSystem.processEnvironmentOverrides();
                (0, globals_1.expect)(result.overrides['enterprise.enabled']).toBe(true);
                (0, globals_1.expect)(result.overrides['security.auth.enabled']).toBe(false);
                (0, globals_1.expect)(result.metadata.totalOverrides).toBeGreaterThan(0);
            });
            (0, globals_1.it)('should process numeric environment variables', async () => {
                process.env.ENTERPRISE_CONFIG_PERFORMANCE_MAX_WORKERS = '20';
                process.env.ENTERPRISE_CONFIG_SECURITY_SESSION_TIMEOUT = '7200';
                const result = await overrideSystem.processEnvironmentOverrides();
                (0, globals_1.expect)(result.overrides['performance.max.workers']).toBe(20);
                (0, globals_1.expect)(result.overrides['security.session.timeout']).toBe(7200);
            });
            (0, globals_1.it)('should detect and handle secrets', async () => {
                process.env.ENTERPRISE_CONFIG_OAUTH_CLIENT_SECRET = 'very-secret-key';
                process.env.DATABASE_PASSWORD = 'db-password';
                const result = await overrideSystem.processEnvironmentOverrides();
                (0, globals_1.expect)(Object.keys(result.secrets)).toContain('oauth.client.secret');
                (0, globals_1.expect)(result.metadata.secretsCount).toBeGreaterThan(0);
            });
            (0, globals_1.it)('should validate environment variable values', async () => {
                process.env.ENTERPRISE_CONFIG_PERFORMANCE_MAX_WORKERS = 'invalid-number';
                process.env.ENTERPRISE_CONFIG_LICENSE_MODE = 'invalid-mode';
                const result = await overrideSystem.processEnvironmentOverrides();
                (0, globals_1.expect)(result.errors.length).toBeGreaterThan(0);
                (0, globals_1.expect)(result.errors.some(e => e.error.includes('Invalid number'))).toBe(true);
            });
            (0, globals_1.it)('should parse array values correctly', async () => {
                process.env.ENTERPRISE_CONFIG_MONITORING_OUTPUT = 'console,file,elasticsearch';
                const result = await overrideSystem.processEnvironmentOverrides();
                (0, globals_1.expect)(Array.isArray(result.overrides['monitoring.output'])).toBe(true);
                (0, globals_1.expect)(result.overrides['monitoring.output']).toEqual(['console', 'file', 'elasticsearch']);
            });
            (0, globals_1.it)('should parse JSON object values', async () => {
                process.env.ENTERPRISE_CONFIG_CUSTOM_SETTINGS = '{"key1":"value1","key2":42}';
                const result = await overrideSystem.processEnvironmentOverrides();
                (0, globals_1.expect)(typeof result.overrides['custom.settings']).toBe('object');
                (0, globals_1.expect)(result.overrides['custom.settings'].key1).toBe('value1');
                (0, globals_1.expect)(result.overrides['custom.settings'].key2).toBe(42);
            });
        });
        (0, globals_1.describe)('Secret Handling', () => {
            (0, globals_1.it)('should assess secret strength', async () => {
                process.env.WEAK_SECRET = '123';
                process.env.STRONG_SECRET = 'Str0ng!P@ssw0rd#2024$';
                const result = await overrideSystem.processEnvironmentOverrides();
                const weakSecret = Object.values(result.secrets).find(s => s.strength === 'weak');
                const strongSecret = Object.values(result.secrets).find(s => s.strength === 'strong');
                (0, globals_1.expect)(weakSecret).toBeDefined();
                (0, globals_1.expect)(strongSecret).toBeDefined();
            });
        });
    });
    (0, globals_1.describe)('ConfigurationManager Integration', () => {
        let configManager;
        (0, globals_1.beforeEach)(() => {
            configManager = new configuration_manager_1.ConfigurationManager({
                configPath: 'test-config.yaml',
                validateOnLoad: true,
                enableHotReload: false
            });
        });
        (0, globals_1.afterEach)(async () => {
            await configManager.cleanup();
        });
        (0, globals_1.describe)('Configuration Loading', () => {
            (0, globals_1.it)('should load and validate enterprise configuration', async () => {
                const testConfig = {
                    schema: { version: '1.0', format_version: '2024.1', compatibility_level: 'backward', migration_required: false },
                    enterprise: { enabled: true, license_mode: 'enterprise', compliance_level: 'standard', features: {} },
                    performance: {
                        resource_limits: { max_memory_mb: 4096, max_cpu_cores: 4, max_file_size_mb: 10, max_analysis_time_seconds: 300, max_concurrent_analyses: 5 }
                    }
                };
                mockFs.readFile.mockResolvedValueOnce(js_yaml_1.default.dump(testConfig));
                const result = await configManager.initialize();
                (0, globals_1.expect)(result.success).toBe(true);
                (0, globals_1.expect)(result.config).toBeDefined();
                (0, globals_1.expect)(result.config?.enterprise?.enabled).toBe(true);
            });
            (0, globals_1.it)('should handle configuration loading errors gracefully', async () => {
                mockFs.readFile.mockRejectedValueOnce(new Error('File not found'));
                const result = await configManager.initialize();
                (0, globals_1.expect)(result.success).toBe(false);
                (0, globals_1.expect)(result.errors.length).toBeGreaterThan(0);
            });
        });
        (0, globals_1.describe)('Configuration Updates', () => {
            (0, globals_1.it)('should update configuration values', async () => {
                const testConfig = {
                    schema: { version: '1.0', format_version: '2024.1', compatibility_level: 'backward', migration_required: false },
                    enterprise: { enabled: true, license_mode: 'community', compliance_level: 'standard', features: {} }
                };
                mockFs.readFile.mockResolvedValueOnce(js_yaml_1.default.dump(testConfig));
                mockFs.writeFile.mockResolvedValueOnce();
                await configManager.initialize();
                const success = await configManager.updateConfigValue('enterprise.license_mode', 'professional');
                (0, globals_1.expect)(success).toBe(true);
                (0, globals_1.expect)(configManager.getConfigValue('enterprise.license_mode')).toBe('professional');
            });
        });
        (0, globals_1.describe)('Environment-Specific Configuration', () => {
            (0, globals_1.it)('should apply environment-specific overrides', async () => {
                const testConfig = {
                    schema: { version: '1.0', format_version: '2024.1', compatibility_level: 'backward', migration_required: false },
                    enterprise: { enabled: true, license_mode: 'enterprise', compliance_level: 'standard', features: {} },
                    security: { authentication: { enabled: true } },
                    environments: {
                        development: {
                            'security.authentication.enabled': false
                        }
                    }
                };
                mockFs.readFile.mockResolvedValueOnce(js_yaml_1.default.dump(testConfig));
                const devConfigManager = new configuration_manager_1.ConfigurationManager({
                    configPath: 'test-config.yaml',
                    environment: 'development'
                });
                const result = await devConfigManager.initialize();
                (0, globals_1.expect)(result.success).toBe(true);
                (0, globals_1.expect)(devConfigManager.getConfigValue('security.authentication.enabled')).toBe(false);
                await devConfigManager.cleanup();
            });
        });
    });
    (0, globals_1.describe)('ConfigurationMigrationManager', () => {
        let migrationManager;
        (0, globals_1.beforeEach)(async () => {
            migrationManager = new migration_versioning_1.ConfigurationMigrationManager({
                versionsDirectory: 'test-versions',
                migrationsDirectory: 'test-migrations',
                backupDirectory: 'test-backups',
                autoMigration: false
            });
            mockFs.mkdir.mockResolvedValue(undefined);
            mockFs.readFile.mockResolvedValue('[]'); // Empty history
            mockFs.writeFile.mockResolvedValue(undefined);
            await migrationManager.initialize();
        });
        (0, globals_1.describe)('Migration Execution', () => {
            (0, globals_1.it)('should execute migration successfully', async () => {
                const testConfig = {
                    version: '1.0.0'
                };
                mockFs.readFile.mockResolvedValueOnce(js_yaml_1.default.dump(testConfig));
                mockFs.writeFile.mockResolvedValue(undefined);
                const result = await migrationManager.migrate('1.1.0');
                (0, globals_1.expect)(result.success).toBe(true);
                (0, globals_1.expect)(result.fromVersion).toBe('1.0.0');
                (0, globals_1.expect)(result.toVersion).toBe('1.1.0');
                (0, globals_1.expect)(result.executedMigrations).toContain('1.1.0');
            });
            (0, globals_1.it)('should handle migration failures and provide rollback info', async () => {
                // Mock a configuration that will cause migration to fail
                const testConfig = {
                    version: '1.0.0',
                    invalid_structure: true
                };
                mockFs.readFile.mockResolvedValueOnce(js_yaml_1.default.dump(testConfig));
                // Add a custom migration that will fail
                migrationManager.addMigration({
                    version: '1.0.1',
                    description: 'Test failing migration',
                    breaking: false,
                    dependencies: [],
                    up: async () => {
                        throw new Error('Migration intentionally failed');
                    },
                    down: async (config) => config,
                    validate: async () => true
                });
                const result = await migrationManager.migrate('1.0.1');
                (0, globals_1.expect)(result.success).toBe(false);
                (0, globals_1.expect)(result.errors.length).toBeGreaterThan(0);
                (0, globals_1.expect)(result.rollbackInfo).toBeDefined();
            });
        });
        (0, globals_1.describe)('Version Management', () => {
            (0, globals_1.it)('should detect current version correctly', async () => {
                const status = migrationManager.getStatus();
                (0, globals_1.expect)(status.currentVersion).toBe('1.0.0');
            });
            (0, globals_1.it)('should check if migration is needed', async () => {
                const migrationNeeded = migrationManager.isMigrationNeeded('2.0.0');
                (0, globals_1.expect)(migrationNeeded).toBe(true);
            });
            (0, globals_1.it)('should list available migrations', async () => {
                const migrations = migrationManager.getAvailableMigrationsList();
                (0, globals_1.expect)(Array.isArray(migrations)).toBe(true);
                (0, globals_1.expect)(migrations.length).toBeGreaterThan(0);
                (0, globals_1.expect)(migrations[0]).toHaveProperty('version');
                (0, globals_1.expect)(migrations[0]).toHaveProperty('description');
                (0, globals_1.expect)(migrations[0]).toHaveProperty('breaking');
            });
        });
        (0, globals_1.describe)('Rollback Functionality', () => {
            (0, globals_1.it)('should perform rollback successfully', async () => {
                const testConfig = {
                    version: '2.0.0',
                    new_feature: true
                };
                mockFs.readFile.mockResolvedValueOnce(js_yaml_1.default.dump(testConfig));
                mockFs.writeFile.mockResolvedValue(undefined);
                const result = await migrationManager.rollback('1.3.0');
                (0, globals_1.expect)(result.success).toBe(true);
                (0, globals_1.expect)(result.toVersion).toBe('1.3.0');
            });
        });
    });
    (0, globals_1.describe)('Integration Tests', () => {
        (0, globals_1.describe)('Full Configuration Lifecycle', () => {
            (0, globals_1.it)('should handle complete configuration lifecycle', async () => {
                // Setup environment variables
                process.env.ENTERPRISE_CONFIG_ENTERPRISE_ENABLED = 'true';
                process.env.ENTERPRISE_CONFIG_LICENSE_MODE = 'enterprise';
                // Mock file system for configuration files
                const baseConfig = {
                    schema: { version: '1.0', format_version: '2024.1', compatibility_level: 'backward', migration_required: false },
                    enterprise: { enabled: false, license_mode: 'community', compliance_level: 'standard', features: {} }
                };
                mockFs.readFile.mockResolvedValue(js_yaml_1.default.dump(baseConfig));
                mockFs.writeFile.mockResolvedValue(undefined);
                mockFs.mkdir.mockResolvedValue(undefined);
                // Initialize configuration manager
                const configManager = new configuration_manager_1.ConfigurationManager({
                    validateOnLoad: true,
                    preserveLegacyConfigs: true
                });
                const result = await configManager.initialize();
                // Verify configuration loading
                (0, globals_1.expect)(result.success).toBe(true);
                // Verify environment overrides were applied
                (0, globals_1.expect)(result.appliedOverrides).toContain('enterprise.enabled');
                // Verify final configuration values
                const config = configManager.getConfig();
                (0, globals_1.expect)(config?.enterprise?.enabled).toBe(true); // Overridden by environment
                (0, globals_1.expect)(config?.enterprise?.license_mode).toBe('enterprise'); // Overridden by environment
                await configManager.cleanup();
                delete process.env.ENTERPRISE_CONFIG_ENTERPRISE_ENABLED;
                delete process.env.ENTERPRISE_CONFIG_LICENSE_MODE;
            });
        });
        (0, globals_1.describe)('Error Handling and Recovery', () => {
            (0, globals_1.it)('should gracefully handle validation errors', async () => {
                const invalidConfig = {
                    enterprise: {
                        enabled: 'invalid-boolean', // Invalid type
                        license_mode: 'invalid-license' // Invalid enum value
                    }
                };
                mockFs.readFile.mockResolvedValueOnce(js_yaml_1.default.dump(invalidConfig));
                const configManager = new configuration_manager_1.ConfigurationManager({
                    validateOnLoad: true
                });
                const result = await configManager.initialize();
                (0, globals_1.expect)(result.success).toBe(false);
                (0, globals_1.expect)(result.errors.length).toBeGreaterThan(0);
                (0, globals_1.expect)(result.config).toBeUndefined();
                await configManager.cleanup();
            });
            (0, globals_1.it)('should handle file system errors', async () => {
                mockFs.readFile.mockRejectedValueOnce(new Error('Permission denied'));
                const configManager = new configuration_manager_1.ConfigurationManager();
                const result = await configManager.initialize();
                (0, globals_1.expect)(result.success).toBe(false);
                (0, globals_1.expect)(result.errors.some(e => e.includes('Permission denied'))).toBe(true);
                await configManager.cleanup();
            });
        });
        (0, globals_1.describe)('Performance Tests', () => {
            (0, globals_1.it)('should process large configuration efficiently', async () => {
                const largeConfig = {
                    schema: { version: '1.0', format_version: '2024.1', compatibility_level: 'backward', migration_required: false },
                    enterprise: { enabled: true, license_mode: 'enterprise', compliance_level: 'standard', features: {} },
                    // Add many configuration entries to test performance
                    large_section: {}
                };
                // Generate large configuration section
                for (let i = 0; i < 1000; i++) {
                    largeConfig.large_section[`setting_${i}`] = {
                        value: i,
                        enabled: i % 2 === 0,
                        description: `Setting number ${i} for performance testing`
                    };
                }
                mockFs.readFile.mockResolvedValueOnce(js_yaml_1.default.dump(largeConfig));
                const startTime = Date.now();
                const configManager = new configuration_manager_1.ConfigurationManager({
                    validateOnLoad: true
                });
                const result = await configManager.initialize();
                const endTime = Date.now();
                const duration = endTime - startTime;
                (0, globals_1.expect)(result.success).toBe(true);
                (0, globals_1.expect)(duration).toBeLessThan(5000); // Should complete within 5 seconds
                await configManager.cleanup();
            });
        });
    });
});
//# sourceMappingURL=configuration-system.test.js.map