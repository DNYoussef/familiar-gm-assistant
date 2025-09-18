"use strict";
/**
 * Unified Configuration Manager
 * Integration patterns with existing configuration manager
 * Provides seamless integration between enterprise and analyzer configurations
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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ConfigurationManager = void 0;
const events_1 = require("events");
const promises_1 = __importDefault(require("fs/promises"));
const path_1 = __importDefault(require("path"));
const js_yaml_1 = __importDefault(require("js-yaml"));
const schema_validator_1 = require("./schema-validator");
const backward_compatibility_1 = require("./backward-compatibility");
/**
 * Unified Configuration Manager
 * Handles loading, validation, and integration of all configuration sources
 */
class ConfigurationManager extends events_1.EventEmitter {
    constructor(options = {}) {
        super();
        this.currentConfig = null;
        this.watchHandlers = new Map();
        this.lastLoadTime = null;
        this.configHash = '';
        this.options = {
            configPath: options.configPath || 'config/enterprise_config.yaml',
            legacyDetectorPath: options.legacyDetectorPath || 'analyzer/config/detector_config.yaml',
            legacyAnalysisPath: options.legacyAnalysisPath || 'analyzer/config/analysis_config.yaml',
            environment: options.environment || process.env.NODE_ENV || 'development',
            enableHotReload: options.enableHotReload ?? true,
            validateOnLoad: options.validateOnLoad ?? true,
            preserveLegacyConfigs: options.preserveLegacyConfigs ?? true,
            conflictResolution: options.conflictResolution || 'merge',
            backupEnabled: options.backupEnabled ?? true,
            auditLogging: options.auditLogging ?? true
        };
        this.validator = new schema_validator_1.EnterpriseConfigValidator();
        this.compatibilityManager = new backward_compatibility_1.BackwardCompatibilityManager();
        this.sources = {
            enterprise: null,
            legacyDetector: null,
            legacyAnalysis: null,
            environmentOverrides: {},
            merged: null
        };
    }
    /**
     * Initialize the configuration manager
     */
    async initialize() {
        try {
            this.emit('config_change', {
                type: 'loaded',
                timestamp: new Date(),
                source: 'file',
                metadata: { environment: this.options.environment }
            });
            // Load all configuration sources
            const loadResult = await this.loadAllSources();
            // Set up hot reload if enabled
            if (this.options.enableHotReload) {
                await this.setupHotReload();
            }
            // Log audit event
            if (this.options.auditLogging) {
                this.logAuditEvent('configuration_initialized', {
                    environment: this.options.environment,
                    success: loadResult.success,
                    sourcesLoaded: Object.entries(this.sources)
                        .filter(([, value]) => value !== null)
                        .map(([key]) => key)
                });
            }
            this.lastLoadTime = new Date();
            return loadResult;
        }
        catch (error) {
            const errorResult = {
                success: false,
                errors: [`Initialization failed: ${error.message}`],
                warnings: [],
                appliedOverrides: []
            };
            this.emit('config_change', {
                type: 'error',
                timestamp: new Date(),
                source: 'file',
                metadata: { error: error.message }
            });
            return errorResult;
        }
    }
    /**
     * Load all configuration sources and merge them
     */
    async loadAllSources() {
        const errors = [];
        const warnings = [];
        const appliedOverrides = [];
        let validation;
        let migration;
        try {
            // 1. Load enterprise configuration
            try {
                const enterpriseContent = await promises_1.default.readFile(this.options.configPath, 'utf-8');
                this.sources.enterprise = js_yaml_1.default.load(enterpriseContent);
            }
            catch (error) {
                if (error.code !== 'ENOENT') {
                    errors.push(`Failed to load enterprise config: ${error.message}`);
                }
                else {
                    warnings.push('Enterprise configuration file not found, using defaults');
                    this.sources.enterprise = this.createDefaultEnterpriseConfig();
                }
            }
            // 2. Load legacy configurations if preservation is enabled
            if (this.options.preserveLegacyConfigs) {
                try {
                    const legacyConfigs = await this.compatibilityManager.loadLegacyConfigs(this.options.legacyDetectorPath, this.options.legacyAnalysisPath);
                    this.sources.legacyDetector = legacyConfigs.detector;
                    this.sources.legacyAnalysis = legacyConfigs.analysis;
                    // Migrate legacy configuration if enterprise config doesn't exist
                    if (!this.sources.enterprise) {
                        const migrationResult = await this.compatibilityManager.migrateLegacyConfig(legacyConfigs, this.options.conflictResolution);
                        migration = migrationResult;
                        if (migrationResult.success) {
                            this.sources.enterprise = migrationResult.migratedConfig;
                            warnings.push('Generated enterprise config from legacy configuration');
                        }
                        else {
                            errors.push('Failed to migrate legacy configuration');
                        }
                    }
                }
                catch (error) {
                    warnings.push(`Could not load legacy configurations: ${error.message}`);
                }
            }
            // 3. Apply environment variable overrides
            this.sources.environmentOverrides = this.loadEnvironmentOverrides();
            if (Object.keys(this.sources.environmentOverrides).length > 0) {
                appliedOverrides.push(...Object.keys(this.sources.environmentOverrides));
            }
            // 4. Merge all sources
            if (this.sources.enterprise) {
                this.sources.merged = await this.mergeAllSources();
                // Apply environment-specific overrides from config
                if (this.sources.merged.environments?.[this.options.environment]) {
                    this.applyEnvironmentSpecificOverrides(this.sources.merged, this.sources.merged.environments[this.options.environment]);
                    appliedOverrides.push(`environment.${this.options.environment}`);
                }
            }
            // 5. Validate the final merged configuration
            if (this.options.validateOnLoad && this.sources.merged) {
                validation = this.validator.validateConfigObject(this.sources.merged, this.options.environment);
                if (!validation.isValid) {
                    errors.push(...validation.errors.map(e => `${e.path}: ${e.message}`));
                }
                warnings.push(...validation.warnings.map(w => `${w.path}: ${w.message}`));
            }
            // 6. Set as current configuration if valid
            if (this.sources.merged && (!validation || validation.isValid)) {
                this.currentConfig = this.sources.merged;
                this.configHash = this.calculateConfigHash(this.currentConfig);
            }
            return {
                success: errors.length === 0,
                config: this.currentConfig || undefined,
                validation,
                migration,
                errors,
                warnings,
                appliedOverrides
            };
        }
        catch (error) {
            return {
                success: false,
                errors: [`Configuration loading failed: ${error.message}`],
                warnings,
                appliedOverrides
            };
        }
    }
    /**
     * Merge all configuration sources according to priority
     */
    async mergeAllSources() {
        if (!this.sources.enterprise) {
            throw new Error('Enterprise configuration is required for merging');
        }
        let mergedConfig = JSON.parse(JSON.stringify(this.sources.enterprise));
        // 1. Apply legacy configuration compatibility if enabled
        if (this.options.preserveLegacyConfigs &&
            (this.sources.legacyDetector || this.sources.legacyAnalysis)) {
            const { mergedConfig: legacyMerged } = await this.compatibilityManager.mergeWithLegacyConfig(mergedConfig, {
                detector: this.sources.legacyDetector,
                analysis: this.sources.legacyAnalysis
            }, this.options.conflictResolution);
            mergedConfig = legacyMerged;
        }
        // 2. Apply environment variable overrides (highest priority)
        this.applyEnvironmentOverrides(mergedConfig, this.sources.environmentOverrides);
        return mergedConfig;
    }
    /**
     * Load environment variable overrides
     */
    loadEnvironmentOverrides() {
        const overrides = {};
        const envPrefix = 'ENTERPRISE_CONFIG_';
        for (const [key, value] of Object.entries(process.env)) {
            if (key.startsWith(envPrefix)) {
                const configPath = key
                    .substring(envPrefix.length)
                    .toLowerCase()
                    .replace(/_/g, '.');
                overrides[configPath] = this.parseEnvironmentValue(value);
            }
        }
        return overrides;
    }
    /**
     * Parse environment variable value to appropriate type
     */
    parseEnvironmentValue(value) {
        // Boolean values
        if (value.toLowerCase() === 'true')
            return true;
        if (value.toLowerCase() === 'false')
            return false;
        // Numeric values
        if (/^\d+$/.test(value))
            return parseInt(value, 10);
        if (/^\d+\.\d+$/.test(value))
            return parseFloat(value);
        // JSON values (for complex objects)
        if (value.startsWith('{') || value.startsWith('[')) {
            try {
                return JSON.parse(value);
            }
            catch {
                // Fall back to string if JSON parsing fails
            }
        }
        return value;
    }
    /**
     * Apply environment variable overrides to configuration
     */
    applyEnvironmentOverrides(config, overrides) {
        for (const [path, value] of Object.entries(overrides)) {
            this.setNestedProperty(config, path, value);
        }
    }
    /**
     * Apply environment-specific configuration overrides
     */
    applyEnvironmentSpecificOverrides(config, overrides) {
        for (const [path, value] of Object.entries(overrides)) {
            this.setNestedProperty(config, path, value);
        }
    }
    /**
     * Set up hot reload for configuration files
     */
    async setupHotReload() {
        const watchPaths = [
            this.options.configPath,
            this.options.legacyDetectorPath,
            this.options.legacyAnalysisPath
        ];
        for (const watchPath of watchPaths) {
            try {
                const { watch } = await Promise.resolve().then(() => __importStar(require('chokidar')));
                const watcher = watch(watchPath, {
                    ignored: /(^|[\/\\])\../,
                    persistent: true
                });
                watcher.on('change', async (changedPath) => {
                    try {
                        await this.reloadConfiguration();
                        this.emit('config_change', {
                            type: 'updated',
                            path: changedPath,
                            timestamp: new Date(),
                            source: 'file',
                            metadata: { trigger: 'file_change' }
                        });
                    }
                    catch (error) {
                        this.emit('config_change', {
                            type: 'error',
                            path: changedPath,
                            timestamp: new Date(),
                            source: 'file',
                            metadata: { error: error.message, trigger: 'hot_reload' }
                        });
                    }
                });
                this.watchHandlers.set(watchPath, watcher);
            }
            catch (error) {
                console.warn(`Could not set up hot reload for ${watchPath}:`, error.message);
            }
        }
    }
    /**
     * Reload configuration from all sources
     */
    async reloadConfiguration() {
        const oldConfig = this.currentConfig;
        const oldHash = this.configHash;
        const result = await this.loadAllSources();
        if (result.success && this.configHash !== oldHash) {
            this.emit('config_change', {
                type: 'updated',
                oldValue: oldConfig,
                newValue: this.currentConfig,
                timestamp: new Date(),
                source: 'file',
                metadata: {
                    oldHash,
                    newHash: this.configHash,
                    trigger: 'reload'
                }
            });
            if (this.options.auditLogging) {
                this.logAuditEvent('configuration_reloaded', {
                    environment: this.options.environment,
                    oldHash,
                    newHash: this.configHash,
                    success: result.success
                });
            }
        }
        return result;
    }
    /**
     * Get current configuration
     */
    getConfig() {
        return this.currentConfig;
    }
    /**
     * Get configuration sources
     */
    getConfigurationSources() {
        return { ...this.sources };
    }
    /**
     * Get specific configuration value by path
     */
    getConfigValue(path, defaultValue) {
        if (!this.currentConfig) {
            return defaultValue;
        }
        const value = this.getNestedProperty(this.currentConfig, path);
        return value !== undefined ? value : defaultValue;
    }
    /**
     * Update configuration value
     */
    async updateConfigValue(path, value, persist = true) {
        if (!this.currentConfig) {
            throw new Error('No configuration loaded');
        }
        const oldValue = this.getNestedProperty(this.currentConfig, path);
        this.setNestedProperty(this.currentConfig, path, value);
        // Validate updated configuration
        if (this.options.validateOnLoad) {
            const validation = this.validator.validateConfigObject(this.currentConfig, this.options.environment);
            if (!validation.isValid) {
                // Rollback on validation failure
                this.setNestedProperty(this.currentConfig, path, oldValue);
                throw new Error(`Configuration update failed validation: ${validation.errors.map(e => e.message).join(', ')}`);
            }
        }
        // Persist to file if requested
        if (persist) {
            try {
                await this.persistConfiguration();
            }
            catch (error) {
                // Rollback on persistence failure
                this.setNestedProperty(this.currentConfig, path, oldValue);
                throw new Error(`Failed to persist configuration: ${error.message}`);
            }
        }
        // Update hash and emit change event
        this.configHash = this.calculateConfigHash(this.currentConfig);
        this.emit('config_change', {
            type: 'updated',
            path,
            oldValue,
            newValue: value,
            timestamp: new Date(),
            source: 'api',
            metadata: { persisted: persist }
        });
        if (this.options.auditLogging) {
            this.logAuditEvent('configuration_updated', {
                path,
                oldValue,
                newValue: value,
                persisted: persist,
                environment: this.options.environment
            });
        }
        return true;
    }
    /**
     * Persist current configuration to file
     */
    async persistConfiguration() {
        if (!this.currentConfig) {
            throw new Error('No configuration to persist');
        }
        // Create backup if enabled
        if (this.options.backupEnabled) {
            try {
                const backupPath = await this.createConfigBackup();
                console.log(`Configuration backup created: ${backupPath}`);
            }
            catch (error) {
                console.warn(`Failed to create backup: ${error.message}`);
            }
        }
        // Write configuration to file
        const configContent = js_yaml_1.default.dump(this.currentConfig, {
            indent: 2,
            lineWidth: 120,
            noRefs: true
        });
        await promises_1.default.writeFile(this.options.configPath, configContent, 'utf-8');
    }
    /**
     * Create a backup of the current configuration
     */
    async createConfigBackup() {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const backupDir = 'config/backups';
        const backupPath = path_1.default.join(backupDir, `enterprise_config_${timestamp}.yaml`);
        await promises_1.default.mkdir(backupDir, { recursive: true });
        const currentContent = await promises_1.default.readFile(this.options.configPath, 'utf-8');
        await promises_1.default.writeFile(backupPath, currentContent);
        return backupPath;
    }
    /**
     * Create default enterprise configuration
     */
    createDefaultEnterpriseConfig() {
        return {
            schema: {
                version: "1.0",
                format_version: "2024.1",
                compatibility_level: "backward",
                migration_required: false
            },
            enterprise: {
                enabled: true,
                license_mode: "community",
                compliance_level: "standard",
                features: {
                    advanced_analytics: false,
                    multi_tenant_support: false,
                    enterprise_security: false,
                    audit_logging: false,
                    performance_monitoring: true,
                    custom_detectors: true,
                    integration_platform: false,
                    governance_framework: true,
                    compliance_reporting: false,
                    advanced_visualization: false,
                    ml_insights: false,
                    risk_assessment: false,
                    automated_remediation: false,
                    multi_environment_sync: false,
                    enterprise_apis: false
                }
            },
            security: {
                authentication: {
                    enabled: false,
                    method: "basic",
                    session_timeout: 3600,
                    max_concurrent_sessions: 5,
                    password_policy: {
                        min_length: 8,
                        require_uppercase: true,
                        require_lowercase: true,
                        require_numbers: true,
                        require_special_chars: false,
                        expiry_days: 90
                    }
                },
                authorization: {
                    rbac_enabled: false,
                    default_role: "viewer",
                    roles: {
                        viewer: { permissions: ["read"] },
                        developer: { permissions: ["read", "execute"] },
                        admin: { permissions: ["read", "write", "execute", "admin"] }
                    }
                },
                audit: {
                    enabled: false,
                    log_level: "basic",
                    retention_days: 90,
                    export_format: "json",
                    real_time_monitoring: false,
                    anomaly_detection: false
                },
                encryption: {
                    at_rest: false,
                    in_transit: false,
                    algorithm: "AES-256-GCM",
                    key_rotation_days: 90
                }
            },
            multi_tenancy: {
                enabled: false,
                isolation_level: "basic",
                tenant_specific_config: false,
                resource_quotas: {
                    max_users_per_tenant: 100,
                    max_projects_per_tenant: 10,
                    max_analysis_jobs_per_day: 1000,
                    storage_limit_gb: 10
                },
                default_tenant: {
                    name: "default",
                    admin_email: "admin@example.com",
                    compliance_profile: "standard"
                }
            },
            performance: {
                scaling: {
                    auto_scaling_enabled: false,
                    min_workers: 1,
                    max_workers: 4,
                    scale_up_threshold: 0.8,
                    scale_down_threshold: 0.3,
                    cooldown_period: 300
                },
                resource_limits: {
                    max_memory_mb: 4096,
                    max_cpu_cores: 4,
                    max_file_size_mb: 10,
                    max_analysis_time_seconds: 300,
                    max_concurrent_analyses: 5
                },
                caching: {
                    enabled: true,
                    provider: "memory",
                    ttl_seconds: 3600,
                    max_cache_size_mb: 512,
                    cache_compression: false
                },
                database: {
                    connection_pool_size: 10,
                    query_timeout_seconds: 30,
                    read_replica_enabled: false,
                    indexing_strategy: "basic"
                }
            },
            integrations: {
                api: {
                    enabled: true,
                    version: "v1",
                    rate_limiting: {
                        enabled: false,
                        requests_per_minute: 100,
                        burst_limit: 10
                    },
                    authentication_required: false,
                    cors_enabled: true,
                    swagger_ui_enabled: true
                },
                webhooks: {
                    enabled: false,
                    max_endpoints: 10,
                    timeout_seconds: 30,
                    retry_attempts: 3,
                    signature_verification: false
                },
                external_systems: {
                    github: {
                        enabled: false
                    }
                },
                ci_cd: {
                    github_actions: {
                        enabled: false
                    }
                }
            },
            monitoring: {
                metrics: {
                    enabled: true,
                    provider: "prometheus",
                    collection_interval: 30,
                    retention_days: 7,
                    custom_metrics: false
                },
                logging: {
                    enabled: true,
                    level: "info",
                    format: "text",
                    output: ["console"],
                    file_rotation: false,
                    max_file_size_mb: 100,
                    max_files: 5
                },
                tracing: {
                    enabled: false,
                    sampling_rate: 0.1,
                    provider: "jaeger"
                },
                alerts: {
                    enabled: false,
                    channels: [],
                    thresholds: {
                        error_rate: 0.05,
                        response_time_p95: 5000,
                        memory_usage: 0.85,
                        cpu_usage: 0.90
                    }
                }
            },
            analytics: {
                enabled: false,
                data_retention_days: 30,
                trend_analysis: false,
                predictive_insights: false,
                custom_dashboards: false,
                scheduled_reports: false,
                machine_learning: {
                    enabled: false,
                    model_training: false,
                    anomaly_detection: false,
                    pattern_recognition: false,
                    automated_insights: false
                },
                export_formats: ["json"],
                real_time_streaming: false
            },
            governance: {
                quality_gates: {
                    enabled: true,
                    enforce_blocking: false,
                    custom_rules: false,
                    nasa_compliance: {
                        enabled: false,
                        minimum_score: 0.75,
                        critical_violations_allowed: 0,
                        high_violations_allowed: 5,
                        automated_remediation_suggestions: false
                    },
                    custom_gates: {}
                },
                policies: {
                    code_standards: "standard",
                    security_requirements: "basic",
                    documentation_mandatory: false,
                    review_requirements: {
                        min_approvers: 1,
                        security_review_required: false,
                        architecture_review_threshold: 100
                    }
                }
            },
            notifications: {
                enabled: false,
                channels: {},
                templates: {},
                escalation: {
                    enabled: false,
                    levels: []
                }
            },
            legacy_integration: {
                preserve_existing_configs: true,
                migration_warnings: true,
                detector_config_path: "analyzer/config/detector_config.yaml",
                analysis_config_path: "analyzer/config/analysis_config.yaml",
                conflict_resolution: "merge"
            },
            extensions: {
                custom_detectors: {
                    enabled: true,
                    directory: "extensions/detectors",
                    auto_discovery: true
                },
                custom_reporters: {
                    enabled: true,
                    directory: "extensions/reporters",
                    formats: ["json"]
                },
                plugins: {
                    enabled: false,
                    directory: "extensions/plugins",
                    sandboxing: true,
                    security_scanning: true
                }
            },
            backup: {
                enabled: false,
                schedule: "0 2 * * *",
                retention_days: 30,
                encryption: false,
                offsite_storage: false,
                disaster_recovery: {
                    enabled: false,
                    rpo_minutes: 60,
                    rto_minutes: 240,
                    failover_testing: false,
                    automated_failover: false
                }
            },
            validation: {
                schema_validation: true,
                runtime_validation: false,
                configuration_drift_detection: false,
                rules: []
            }
        };
    }
    /**
     * Calculate configuration hash for change detection
     */
    calculateConfigHash(config) {
        const crypto = require('crypto');
        const configString = JSON.stringify(config, Object.keys(config).sort());
        return crypto.createHash('sha256').update(configString).digest('hex');
    }
    /**
     * Get nested property using dot notation
     */
    getNestedProperty(obj, path) {
        return path.split('.').reduce((current, key) => current?.[key], obj);
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
     * Log audit event
     */
    logAuditEvent(event, metadata) {
        const auditEntry = {
            timestamp: new Date().toISOString(),
            event,
            metadata,
            environment: this.options.environment,
            configHash: this.configHash
        };
        // In production, this would go to a proper audit logging system
        console.log('[AUDIT]', JSON.stringify(auditEntry));
    }
    /**
     * Cleanup resources
     */
    async cleanup() {
        // Close file watchers
        for (const [path, watcher] of this.watchHandlers.entries()) {
            try {
                await watcher.close();
            }
            catch (error) {
                console.warn(`Failed to close watcher for ${path}:`, error.message);
            }
        }
        this.watchHandlers.clear();
        this.removeAllListeners();
    }
    /**
     * Get configuration manager status
     */
    getStatus() {
        return {
            initialized: this.currentConfig !== null,
            configLoaded: this.currentConfig !== null,
            lastLoadTime: this.lastLoadTime,
            hotReloadEnabled: this.options.enableHotReload,
            watchedFiles: Array.from(this.watchHandlers.keys()),
            environment: this.options.environment,
            configHash: this.configHash
        };
    }
}
exports.ConfigurationManager = ConfigurationManager;
exports.default = ConfigurationManager;
//# sourceMappingURL=configuration-manager.js.map