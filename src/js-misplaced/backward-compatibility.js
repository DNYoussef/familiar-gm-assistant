"use strict";
/**
 * Backward Compatibility Layer
 * Preserves existing analyzer configuration while integrating with enterprise features
 * Provides seamless migration path and conflict resolution strategies
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.BackwardCompatibilityManager = void 0;
const js_yaml_1 = __importDefault(require("js-yaml"));
const promises_1 = __importDefault(require("fs/promises"));
const path_1 = __importDefault(require("path"));
const zod_1 = require("zod");
// Legacy configuration schemas (based on existing analyzer configs)
const LegacyDetectorConfigSchema = zod_1.z.object({
    values_detector: zod_1.z.object({
        config_keywords: zod_1.z.array(zod_1.z.string()),
        thresholds: zod_1.z.object({
            duplicate_literal_minimum: zod_1.z.number(),
            configuration_coupling_limit: zod_1.z.number(),
            configuration_line_spread: zod_1.z.number()
        }),
        exclusions: zod_1.z.object({
            common_strings: zod_1.z.array(zod_1.z.string()),
            common_numbers: zod_1.z.array(zod_1.z.union([zod_1.z.number(), zod_1.z.string()]))
        })
    }),
    position_detector: zod_1.z.object({
        max_positional_params: zod_1.z.number(),
        severity_mapping: zod_1.z.record(zod_1.z.string())
    }),
    algorithm_detector: zod_1.z.object({
        minimum_function_lines: zod_1.z.number(),
        duplicate_threshold: zod_1.z.number(),
        normalization: zod_1.z.object({
            ignore_variable_names: zod_1.z.boolean(),
            ignore_comments: zod_1.z.boolean(),
            focus_on_structure: zod_1.z.boolean()
        })
    }),
    magic_literal_detector: zod_1.z.object({
        severity_rules: zod_1.z.record(zod_1.z.string()),
        thresholds: zod_1.z.object({
            number_repetition: zod_1.z.number(),
            string_repetition: zod_1.z.number()
        })
    }),
    god_object_detector: zod_1.z.object({
        method_threshold: zod_1.z.number(),
        loc_threshold: zod_1.z.number(),
        parameter_threshold: zod_1.z.number()
    }),
    timing_detector: zod_1.z.object({
        sleep_detection: zod_1.z.boolean(),
        timeout_patterns: zod_1.z.array(zod_1.z.string()),
        severity: zod_1.z.string()
    }),
    convention_detector: zod_1.z.object({
        naming_patterns: zod_1.z.record(zod_1.z.string())
    }),
    execution_detector: zod_1.z.object({
        dangerous_functions: zod_1.z.array(zod_1.z.string()),
        subprocess_patterns: zod_1.z.array(zod_1.z.string())
    })
});
const LegacyAnalysisConfigSchema = zod_1.z.object({
    analysis: zod_1.z.object({
        default_policy: zod_1.z.string(),
        max_file_size_mb: zod_1.z.number(),
        max_analysis_time_seconds: zod_1.z.number(),
        parallel_workers: zod_1.z.number(),
        cache_enabled: zod_1.z.boolean()
    }),
    quality_gates: zod_1.z.object({
        overall_quality_threshold: zod_1.z.number(),
        critical_violation_limit: zod_1.z.number(),
        high_violation_limit: zod_1.z.number(),
        policies: zod_1.z.record(zod_1.z.object({
            quality_threshold: zod_1.z.number(),
            violation_limits: zod_1.z.record(zod_1.z.number())
        }))
    }),
    connascence: zod_1.z.object({
        type_weights: zod_1.z.record(zod_1.z.number()),
        severity_multipliers: zod_1.z.record(zod_1.z.number())
    }),
    file_processing: zod_1.z.object({
        supported_extensions: zod_1.z.array(zod_1.z.string()),
        exclusion_patterns: zod_1.z.array(zod_1.z.string()),
        max_recursion_depth: zod_1.z.number(),
        follow_symlinks: zod_1.z.boolean()
    }),
    error_handling: zod_1.z.object({
        continue_on_syntax_error: zod_1.z.boolean(),
        log_all_errors: zod_1.z.boolean(),
        graceful_degradation: zod_1.z.boolean(),
        max_retry_attempts: zod_1.z.number()
    }),
    reporting: zod_1.z.object({
        default_format: zod_1.z.string(),
        include_recommendations: zod_1.z.boolean(),
        include_context: zod_1.z.boolean(),
        max_code_snippet_lines: zod_1.z.number(),
        formats: zod_1.z.record(zod_1.z.record(zod_1.z.union([zod_1.z.boolean(), zod_1.z.string()])))
    }),
    integrations: zod_1.z.object({
        mcp: zod_1.z.object({
            timeout_seconds: zod_1.z.number(),
            max_request_size_mb: zod_1.z.number(),
            rate_limit_requests_per_minute: zod_1.z.number()
        }),
        vscode: zod_1.z.object({
            live_analysis: zod_1.z.boolean(),
            max_diagnostics: zod_1.z.number(),
            debounce_ms: zod_1.z.number()
        }),
        cli: zod_1.z.object({
            colored_output: zod_1.z.boolean(),
            progress_bar: zod_1.z.boolean(),
            verbose_default: zod_1.z.boolean()
        })
    })
});
/**
 * Backward Compatibility Manager
 * Handles migration and preservation of existing analyzer configuration
 */
class BackwardCompatibilityManager {
    constructor() {
        this.legacyDetectorConfig = null;
        this.legacyAnalysisConfig = null;
        this.migrationMappings = [];
        this.initializeMigrationMappings();
    }
    /**
     * Initialize mapping between legacy and enterprise configuration paths
     */
    initializeMigrationMappings() {
        this.migrationMappings = [
            // Analysis configuration mappings
            {
                legacyPath: 'analysis.max_file_size_mb',
                enterprisePath: 'performance.resource_limits.max_file_size_mb'
            },
            {
                legacyPath: 'analysis.max_analysis_time_seconds',
                enterprisePath: 'performance.resource_limits.max_analysis_time_seconds'
            },
            {
                legacyPath: 'analysis.parallel_workers',
                enterprisePath: 'performance.scaling.max_workers'
            },
            {
                legacyPath: 'analysis.cache_enabled',
                enterprisePath: 'performance.caching.enabled'
            },
            // Quality gates mappings
            {
                legacyPath: 'quality_gates.overall_quality_threshold',
                enterprisePath: 'governance.quality_gates.custom_gates.overall_threshold',
                transformer: (value) => value
            },
            {
                legacyPath: 'quality_gates.critical_violation_limit',
                enterprisePath: 'governance.quality_gates.nasa_compliance.critical_violations_allowed'
            },
            {
                legacyPath: 'quality_gates.high_violation_limit',
                enterprisePath: 'governance.quality_gates.nasa_compliance.high_violations_allowed'
            },
            // NASA compliance mapping
            {
                legacyPath: 'quality_gates.policies.nasa-compliance',
                enterprisePath: 'governance.quality_gates.nasa_compliance',
                transformer: (policy) => ({
                    enabled: true,
                    minimum_score: policy.quality_threshold,
                    critical_violations_allowed: policy.violation_limits.critical || 0,
                    high_violations_allowed: policy.violation_limits.high || 0,
                    automated_remediation_suggestions: true
                })
            },
            // File processing mappings
            {
                legacyPath: 'file_processing.supported_extensions',
                enterprisePath: 'governance.policies.supported_file_types',
                transformer: (extensions) => extensions
            },
            {
                legacyPath: 'file_processing.exclusion_patterns',
                enterprisePath: 'governance.policies.exclusion_patterns'
            },
            // Reporting mappings
            {
                legacyPath: 'reporting.default_format',
                enterprisePath: 'analytics.export_formats',
                transformer: (format) => [format]
            },
            {
                legacyPath: 'reporting.include_recommendations',
                enterprisePath: 'analytics.predictive_insights'
            },
            // Integration mappings
            {
                legacyPath: 'integrations.mcp.timeout_seconds',
                enterprisePath: 'integrations.api.rate_limiting.timeout_seconds',
                transformer: (timeout) => timeout
            },
            {
                legacyPath: 'integrations.mcp.rate_limit_requests_per_minute',
                enterprisePath: 'integrations.api.rate_limiting.requests_per_minute'
            },
            {
                legacyPath: 'integrations.vscode.live_analysis',
                enterprisePath: 'monitoring.metrics.enabled'
            },
            // Detector configuration mappings
            {
                legacyPath: 'god_object_detector.method_threshold',
                enterprisePath: 'governance.quality_gates.custom_gates.god_object_method_threshold'
            },
            {
                legacyPath: 'god_object_detector.loc_threshold',
                enterprisePath: 'governance.quality_gates.custom_gates.god_object_loc_threshold'
            },
            {
                legacyPath: 'magic_literal_detector.thresholds',
                enterprisePath: 'governance.quality_gates.custom_gates.magic_literal_thresholds',
                transformer: (thresholds) => thresholds
            }
        ];
    }
    /**
     * Load existing legacy configuration files
     */
    async loadLegacyConfigs(detectorConfigPath, analysisConfigPath) {
        const detectorPath = detectorConfigPath || 'analyzer/config/detector_config.yaml';
        const analysisPath = analysisConfigPath || 'analyzer/config/analysis_config.yaml';
        try {
            const [detectorContent, analysisContent] = await Promise.allSettled([
                this.loadYamlFile(detectorPath),
                this.loadYamlFile(analysisPath)
            ]);
            let detectorConfig = null;
            let analysisConfig = null;
            if (detectorContent.status === 'fulfilled') {
                try {
                    detectorConfig = LegacyDetectorConfigSchema.parse(detectorContent.value);
                    this.legacyDetectorConfig = detectorConfig;
                }
                catch (error) {
                    console.warn('Failed to parse detector config, using as-is:', error.message);
                    detectorConfig = detectorContent.value;
                }
            }
            if (analysisContent.status === 'fulfilled') {
                try {
                    analysisConfig = LegacyAnalysisConfigSchema.parse(analysisContent.value);
                    this.legacyAnalysisConfig = analysisConfig;
                }
                catch (error) {
                    console.warn('Failed to parse analysis config, using as-is:', error.message);
                    analysisConfig = analysisContent.value;
                }
            }
            return { detector: detectorConfig, analysis: analysisConfig };
        }
        catch (error) {
            throw new Error(`Failed to load legacy configurations: ${error.message}`);
        }
    }
    /**
     * Migrate legacy configuration to enterprise format
     */
    async migrateLegacyConfig(legacyConfigs, conflictResolution = 'merge') {
        const conflicts = [];
        const warnings = [];
        const migratedConfig = {};
        try {
            // Create backup of legacy configs
            const backupPath = await this.createLegacyBackup(legacyConfigs);
            // Migrate analysis configuration
            if (legacyConfigs.analysis) {
                const analysisResult = this.migrateAnalysisConfig(legacyConfigs.analysis, conflictResolution);
                Object.assign(migratedConfig, analysisResult.config);
                conflicts.push(...analysisResult.conflicts);
                warnings.push(...analysisResult.warnings);
            }
            // Migrate detector configuration
            if (legacyConfigs.detector) {
                const detectorResult = this.migrateDetectorConfig(legacyConfigs.detector, conflictResolution);
                Object.assign(migratedConfig, detectorResult.config);
                conflicts.push(...detectorResult.conflicts);
                warnings.push(...detectorResult.warnings);
            }
            // Add compatibility layer configuration
            this.addCompatibilityLayerConfig(migratedConfig);
            return {
                success: true,
                migratedConfig,
                conflicts,
                warnings,
                preservedLegacyConfig: conflictResolution !== 'enterprise_wins',
                backupPath
            };
        }
        catch (error) {
            return {
                success: false,
                migratedConfig: {},
                conflicts,
                warnings: [...warnings, {
                        path: 'migration',
                        message: `Migration failed: ${error.message}`,
                        severity: 'error'
                    }],
                preservedLegacyConfig: false
            };
        }
    }
    /**
     * Migrate analysis configuration
     */
    migrateAnalysisConfig(analysisConfig, conflictResolution) {
        const config = {
            performance: {
                resource_limits: {
                    max_file_size_mb: analysisConfig.analysis.max_file_size_mb,
                    max_analysis_time_seconds: analysisConfig.analysis.max_analysis_time_seconds,
                    max_memory_mb: 8192, // default
                    max_cpu_cores: 8, // default
                    max_concurrent_analyses: analysisConfig.analysis.parallel_workers
                },
                scaling: {
                    auto_scaling_enabled: false,
                    min_workers: 1,
                    max_workers: analysisConfig.analysis.parallel_workers,
                    scale_up_threshold: 0.8,
                    scale_down_threshold: 0.3,
                    cooldown_period: 300
                },
                caching: {
                    enabled: analysisConfig.analysis.cache_enabled,
                    provider: 'memory',
                    ttl_seconds: 3600,
                    max_cache_size_mb: 1024,
                    cache_compression: false
                },
                database: {
                    connection_pool_size: 20,
                    query_timeout_seconds: 30,
                    read_replica_enabled: false,
                    indexing_strategy: 'optimized'
                }
            },
            governance: {
                quality_gates: {
                    enabled: true,
                    enforce_blocking: true,
                    custom_rules: true,
                    nasa_compliance: {
                        enabled: analysisConfig.quality_gates.policies?.['nasa-compliance']?.quality_threshold >= 0.95 || false,
                        minimum_score: analysisConfig.quality_gates.policies?.['nasa-compliance']?.quality_threshold || 0.75,
                        critical_violations_allowed: analysisConfig.quality_gates.policies?.['nasa-compliance']?.violation_limits?.critical || 0,
                        high_violations_allowed: analysisConfig.quality_gates.policies?.['nasa-compliance']?.violation_limits?.high || 5,
                        automated_remediation_suggestions: true
                    },
                    custom_gates: {
                        overall_threshold: analysisConfig.quality_gates.overall_quality_threshold,
                        critical_violation_limit: analysisConfig.quality_gates.critical_violation_limit,
                        high_violation_limit: analysisConfig.quality_gates.high_violation_limit
                    }
                },
                policies: {
                    code_standards: analysisConfig.analysis.default_policy,
                    security_requirements: 'standard',
                    documentation_mandatory: analysisConfig.reporting.include_context,
                    review_requirements: {
                        min_approvers: 2,
                        security_review_required: true,
                        architecture_review_threshold: 100
                    }
                }
            },
            analytics: {
                enabled: true,
                data_retention_days: 365,
                trend_analysis: analysisConfig.reporting.include_recommendations,
                predictive_insights: analysisConfig.reporting.include_recommendations,
                custom_dashboards: true,
                scheduled_reports: false,
                machine_learning: {
                    enabled: false,
                    model_training: false,
                    anomaly_detection: false,
                    pattern_recognition: false,
                    automated_insights: false
                },
                export_formats: [analysisConfig.reporting.default_format],
                real_time_streaming: false
            }
        };
        const conflicts = [];
        const warnings = [];
        // Check for potential conflicts and add warnings
        if (analysisConfig.analysis.parallel_workers > 20) {
            warnings.push({
                path: 'performance.scaling.max_workers',
                message: `Legacy parallel_workers (${analysisConfig.analysis.parallel_workers}) exceeds recommended limit`,
                severity: 'warning',
                recommendation: 'Consider reducing to 20 or implementing worker pools'
            });
        }
        if (analysisConfig.quality_gates.overall_quality_threshold < 0.75) {
            warnings.push({
                path: 'governance.quality_gates.custom_gates.overall_threshold',
                message: 'Legacy quality threshold is below enterprise standard',
                severity: 'warning',
                recommendation: 'Consider increasing to 0.75 or higher'
            });
        }
        return { config, conflicts, warnings };
    }
    /**
     * Migrate detector configuration
     */
    migrateDetectorConfig(detectorConfig, conflictResolution) {
        const config = {};
        const conflicts = [];
        const warnings = [];
        // Migrate god object detector settings to quality gates
        if (!config.governance) {
            config.governance = {
                quality_gates: {
                    enabled: true,
                    enforce_blocking: true,
                    custom_rules: true,
                    nasa_compliance: {
                        enabled: false,
                        minimum_score: 0.75,
                        critical_violations_allowed: 0,
                        high_violations_allowed: 5,
                        automated_remediation_suggestions: true
                    },
                    custom_gates: {}
                },
                policies: {
                    code_standards: 'standard',
                    security_requirements: 'standard',
                    documentation_mandatory: false,
                    review_requirements: {
                        min_approvers: 2,
                        security_review_required: true,
                        architecture_review_threshold: 100
                    }
                }
            };
        }
        // Migrate god object thresholds
        if (config.governance?.quality_gates?.custom_gates) {
            config.governance.quality_gates.custom_gates.god_object_method_threshold =
                detectorConfig.god_object_detector.method_threshold;
            config.governance.quality_gates.custom_gates.god_object_loc_threshold =
                detectorConfig.god_object_detector.loc_threshold;
            config.governance.quality_gates.custom_gates.god_object_parameter_threshold =
                detectorConfig.god_object_detector.parameter_threshold;
        }
        // Migrate magic literal thresholds
        if (config.governance?.quality_gates?.custom_gates) {
            config.governance.quality_gates.custom_gates.magic_literal_number_repetition =
                detectorConfig.magic_literal_detector.thresholds.number_repetition;
            config.governance.quality_gates.custom_gates.magic_literal_string_repetition =
                detectorConfig.magic_literal_detector.thresholds.string_repetition;
        }
        // Store detector-specific configuration for extensions
        if (!config.extensions) {
            config.extensions = {
                custom_detectors: {
                    enabled: true,
                    directory: 'extensions/detectors',
                    auto_discovery: true
                },
                custom_reporters: {
                    enabled: true,
                    directory: 'extensions/reporters',
                    formats: ['custom_json']
                },
                plugins: {
                    enabled: true,
                    directory: 'extensions/plugins',
                    sandboxing: true,
                    security_scanning: true
                }
            };
        }
        warnings.push({
            path: 'extensions.custom_detectors',
            message: 'Detector-specific configuration preserved in extensions section',
            severity: 'info',
            recommendation: 'Review custom detector settings in extensions configuration'
        });
        return { config, conflicts, warnings };
    }
    /**
     * Add compatibility layer configuration
     */
    addCompatibilityLayerConfig(config) {
        if (!config.legacy_integration) {
            config.legacy_integration = {
                preserve_existing_configs: true,
                migration_warnings: true,
                detector_config_path: 'analyzer/config/detector_config.yaml',
                analysis_config_path: 'analyzer/config/analysis_config.yaml',
                conflict_resolution: 'merge'
            };
        }
    }
    /**
     * Merge enterprise config with legacy settings
     */
    async mergeWithLegacyConfig(enterpriseConfig, legacyConfigs, conflictResolution = 'merge') {
        const mergedConfig = JSON.parse(JSON.stringify(enterpriseConfig));
        const conflicts = [];
        // Apply legacy mappings based on conflict resolution strategy
        for (const mapping of this.migrationMappings) {
            const legacyValue = this.getNestedProperty(legacyConfigs.analysis || {}, mapping.legacyPath);
            const enterpriseValue = this.getNestedProperty(mergedConfig, mapping.enterprisePath);
            if (legacyValue !== undefined && enterpriseValue !== undefined && legacyValue !== enterpriseValue) {
                const conflict = {
                    path: mapping.enterprisePath,
                    legacyValue,
                    enterpriseValue,
                    resolution: mapping.conflicts || conflictResolution,
                    rationale: `Legacy value differs from enterprise default`
                };
                conflicts.push(conflict);
                // Apply resolution strategy
                switch (conflict.resolution) {
                    case 'legacy_wins':
                        this.setNestedProperty(mergedConfig, mapping.enterprisePath, mapping.transformer ? mapping.transformer(legacyValue) : legacyValue);
                        break;
                    case 'enterprise_wins':
                        // Keep enterprise value (no change needed)
                        break;
                    case 'merge':
                        if (typeof legacyValue === 'object' && typeof enterpriseValue === 'object') {
                            const merged = { ...enterpriseValue, ...legacyValue };
                            this.setNestedProperty(mergedConfig, mapping.enterprisePath, merged);
                        }
                        else {
                            // For non-objects, prefer legacy value in merge mode
                            this.setNestedProperty(mergedConfig, mapping.enterprisePath, mapping.transformer ? mapping.transformer(legacyValue) : legacyValue);
                        }
                        break;
                }
            }
        }
        return { mergedConfig, conflicts };
    }
    /**
     * Create backup of legacy configuration files
     */
    async createLegacyBackup(legacyConfigs) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const backupDir = `config/backups/legacy-${timestamp}`;
        try {
            await promises_1.default.mkdir(backupDir, { recursive: true });
            if (legacyConfigs.detector) {
                await promises_1.default.writeFile(path_1.default.join(backupDir, 'detector_config.yaml'), js_yaml_1.default.dump(legacyConfigs.detector));
            }
            if (legacyConfigs.analysis) {
                await promises_1.default.writeFile(path_1.default.join(backupDir, 'analysis_config.yaml'), js_yaml_1.default.dump(legacyConfigs.analysis));
            }
            return backupDir;
        }
        catch (error) {
            throw new Error(`Failed to create legacy backup: ${error.message}`);
        }
    }
    /**
     * Load YAML file
     */
    async loadYamlFile(filePath) {
        const content = await promises_1.default.readFile(filePath, 'utf-8');
        return js_yaml_1.default.load(content);
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
     * Validate backward compatibility
     */
    async validateBackwardCompatibility(enterpriseConfig, legacyConfigPaths) {
        const issues = [];
        let isCompatible = true;
        try {
            const legacyConfigs = await this.loadLegacyConfigs(legacyConfigPaths.detector, legacyConfigPaths.analysis);
            // Check if enterprise settings conflict with critical legacy settings
            if (legacyConfigs.analysis) {
                // Check file size limits
                const legacyMaxSize = legacyConfigs.analysis.analysis.max_file_size_mb;
                const enterpriseMaxSize = enterpriseConfig.performance.resource_limits.max_file_size_mb;
                if (enterpriseMaxSize < legacyMaxSize) {
                    issues.push(`Enterprise max file size (${enterpriseMaxSize}MB) is smaller than legacy setting (${legacyMaxSize}MB)`);
                    isCompatible = false;
                }
                // Check analysis timeout
                const legacyTimeout = legacyConfigs.analysis.analysis.max_analysis_time_seconds;
                const enterpriseTimeout = enterpriseConfig.performance.resource_limits.max_analysis_time_seconds;
                if (enterpriseTimeout < legacyTimeout) {
                    issues.push(`Enterprise analysis timeout (${enterpriseTimeout}s) is shorter than legacy setting (${legacyTimeout}s)`);
                    isCompatible = false;
                }
            }
            // Check if legacy integration settings are preserved
            if (!enterpriseConfig.legacy_integration?.preserve_existing_configs) {
                issues.push('Legacy configuration preservation is disabled');
                isCompatible = false;
            }
        }
        catch (error) {
            issues.push(`Failed to validate backward compatibility: ${error.message}`);
            isCompatible = false;
        }
        return { isCompatible, issues };
    }
    /**
     * Get migration status summary
     */
    getMigrationStatus() {
        return {
            legacyConfigsLoaded: this.legacyDetectorConfig !== null || this.legacyAnalysisConfig !== null,
            detectorConfigValid: this.legacyDetectorConfig !== null,
            analysisConfigValid: this.legacyAnalysisConfig !== null,
            migrationMappingsCount: this.migrationMappings.length
        };
    }
}
exports.BackwardCompatibilityManager = BackwardCompatibilityManager;
exports.default = BackwardCompatibilityManager;
//# sourceMappingURL=backward-compatibility.js.map