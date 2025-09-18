"use strict";
/**
 * Configuration Migration and Versioning Strategy
 * Comprehensive migration system with version management and rollback capabilities
 * Supports semantic versioning, automated migrations, and compatibility tracking
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ConfigurationMigrationManager = void 0;
const promises_1 = __importDefault(require("fs/promises"));
const path_1 = __importDefault(require("path"));
const js_yaml_1 = __importDefault(require("js-yaml"));
const zod_1 = require("zod");
const crypto_1 = require("crypto");
const backward_compatibility_1 = require("./backward-compatibility");
// Version metadata schema
const VersionMetadataSchema = zod_1.z.object({
    version: zod_1.z.string(),
    timestamp: zod_1.z.string(),
    description: zod_1.z.string(),
    breaking_changes: zod_1.z.array(zod_1.z.string()),
    migration_required: zod_1.z.boolean(),
    compatibility_level: zod_1.z.enum(['major', 'minor', 'patch']),
    rollback_supported: zod_1.z.boolean(),
    checksum: zod_1.z.string(),
    size_bytes: zod_1.z.number(),
    author: zod_1.z.string().optional(),
    change_log: zod_1.z.array(zod_1.z.object({
        type: zod_1.z.enum(['added', 'changed', 'deprecated', 'removed', 'fixed', 'security']),
        description: zod_1.z.string(),
        impact: zod_1.z.enum(['low', 'medium', 'high', 'critical']).optional()
    }))
});
/**
 * Configuration Migration and Versioning Manager
 * Handles configuration evolution, migrations, and version management
 */
class ConfigurationMigrationManager {
    constructor(config = {}) {
        this.migrations = new Map();
        this.versionHistory = [];
        this.currentVersion = '1.0.0';
        this.config = {
            versionsDirectory: config.versionsDirectory || 'config/versions',
            migrationsDirectory: config.migrationsDirectory || 'config/migrations',
            backupDirectory: config.backupDirectory || 'config/backups',
            autoMigration: config.autoMigration ?? true,
            validationRequired: config.validationRequired ?? true,
            rollbackEnabled: config.rollbackEnabled ?? true,
            maxVersionHistory: config.maxVersionHistory || 50,
            compressionEnabled: config.compressionEnabled ?? true,
            encryptionEnabled: config.encryptionEnabled ?? false,
            notifications: {
                enabled: config.notifications?.enabled ?? false,
                channels: config.notifications?.channels || [],
                onSuccess: config.notifications?.onSuccess ?? true,
                onFailure: config.notifications?.onFailure ?? true,
                onRollback: config.notifications?.onRollback ?? true
            }
        };
        this.compatibilityManager = new backward_compatibility_1.BackwardCompatibilityManager();
        this.initializeMigrations();
    }
    /**
     * Initialize built-in migrations
     */
    initializeMigrations() {
        const migrations = [
            {
                version: '1.1.0',
                description: 'Add enterprise features support',
                breaking: false,
                dependencies: [],
                up: async (config) => {
                    if (!config.enterprise) {
                        config.enterprise = {
                            enabled: false,
                            license_mode: 'community',
                            compliance_level: 'standard',
                            features: {}
                        };
                    }
                    return config;
                },
                down: async (config) => {
                    delete config.enterprise;
                    return config;
                },
                validate: async (config) => {
                    return config.enterprise !== undefined;
                }
            },
            {
                version: '1.2.0',
                description: 'Add multi-tenancy support',
                breaking: false,
                dependencies: ['1.1.0'],
                up: async (config) => {
                    if (!config.multi_tenancy) {
                        config.multi_tenancy = {
                            enabled: false,
                            isolation_level: 'basic',
                            tenant_specific_config: false,
                            resource_quotas: {
                                max_users_per_tenant: 100,
                                max_projects_per_tenant: 10,
                                max_analysis_jobs_per_day: 1000,
                                storage_limit_gb: 10
                            },
                            default_tenant: {
                                name: 'default',
                                admin_email: 'admin@example.com',
                                compliance_profile: 'standard'
                            }
                        };
                    }
                    return config;
                },
                down: async (config) => {
                    delete config.multi_tenancy;
                    return config;
                },
                validate: async (config) => {
                    return config.multi_tenancy !== undefined;
                }
            },
            {
                version: '1.3.0',
                description: 'Enhanced security configuration',
                breaking: true,
                dependencies: ['1.2.0'],
                up: async (config) => {
                    if (config.security) {
                        // Migrate old security structure to new format
                        if (config.security.auth && !config.security.authentication) {
                            config.security.authentication = config.security.auth;
                            delete config.security.auth;
                        }
                        // Add encryption section if missing
                        if (!config.security.encryption) {
                            config.security.encryption = {
                                at_rest: false,
                                in_transit: false,
                                algorithm: 'AES-256-GCM',
                                key_rotation_days: 90
                            };
                        }
                        // Add audit section if missing
                        if (!config.security.audit) {
                            config.security.audit = {
                                enabled: false,
                                log_level: 'basic',
                                retention_days: 90,
                                export_format: 'json',
                                real_time_monitoring: false,
                                anomaly_detection: false
                            };
                        }
                    }
                    return config;
                },
                down: async (config) => {
                    if (config.security?.authentication) {
                        config.security.auth = config.security.authentication;
                        delete config.security.authentication;
                        delete config.security.encryption;
                        delete config.security.audit;
                    }
                    return config;
                },
                validate: async (config) => {
                    return config.security?.authentication !== undefined &&
                        config.security?.encryption !== undefined &&
                        config.security?.audit !== undefined;
                }
            },
            {
                version: '2.0.0',
                description: 'Major restructure for enterprise deployment',
                breaking: true,
                dependencies: ['1.3.0'],
                up: async (config) => {
                    // Major restructure - move legacy settings to new structure
                    const newConfig = {
                        schema: {
                            version: '2.0',
                            format_version: '2024.1',
                            compatibility_level: 'backward',
                            migration_required: false
                        },
                        ...config
                    };
                    // Restructure quality gates
                    if (config.quality_gates) {
                        newConfig.governance = {
                            quality_gates: {
                                enabled: true,
                                enforce_blocking: config.quality_gates.enforce_blocking || false,
                                custom_rules: true,
                                nasa_compliance: {
                                    enabled: config.quality_gates.nasa_compliance?.enabled || false,
                                    minimum_score: config.quality_gates.overall_quality_threshold || 0.75,
                                    critical_violations_allowed: config.quality_gates.critical_violation_limit || 0,
                                    high_violations_allowed: config.quality_gates.high_violation_limit || 5,
                                    automated_remediation_suggestions: true
                                },
                                custom_gates: config.quality_gates.custom_gates || {}
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
                        delete newConfig.quality_gates;
                    }
                    return newConfig;
                },
                down: async (config) => {
                    // Restore old structure
                    if (config.governance?.quality_gates) {
                        config.quality_gates = {
                            overall_quality_threshold: config.governance.quality_gates.nasa_compliance.minimum_score,
                            critical_violation_limit: config.governance.quality_gates.nasa_compliance.critical_violations_allowed,
                            high_violation_limit: config.governance.quality_gates.nasa_compliance.high_violations_allowed,
                            enforce_blocking: config.governance.quality_gates.enforce_blocking,
                            custom_gates: config.governance.quality_gates.custom_gates
                        };
                        delete config.governance;
                    }
                    delete config.schema;
                    return config;
                },
                validate: async (config) => {
                    return config.schema?.version === '2.0' && config.governance !== undefined;
                }
            },
            {
                version: '2.1.0',
                description: 'Add advanced analytics and ML capabilities',
                breaking: false,
                dependencies: ['2.0.0'],
                up: async (config) => {
                    if (!config.analytics) {
                        config.analytics = {
                            enabled: false,
                            data_retention_days: 365,
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
                            export_formats: ['json'],
                            real_time_streaming: false
                        };
                    }
                    return config;
                },
                down: async (config) => {
                    delete config.analytics;
                    return config;
                },
                validate: async (config) => {
                    return config.analytics !== undefined;
                }
            }
        ];
        migrations.forEach(migration => {
            this.migrations.set(migration.version, migration);
        });
    }
    /**
     * Initialize the migration system
     */
    async initialize() {
        try {
            // Create necessary directories
            await this.createDirectories();
            // Load version history
            await this.loadVersionHistory();
            // Detect current version
            await this.detectCurrentVersion();
        }
        catch (error) {
            throw new Error(`Migration system initialization failed: ${error.message}`);
        }
    }
    /**
     * Create necessary directories
     */
    async createDirectories() {
        const directories = [
            this.config.versionsDirectory,
            this.config.migrationsDirectory,
            this.config.backupDirectory
        ];
        for (const dir of directories) {
            try {
                await promises_1.default.mkdir(dir, { recursive: true });
            }
            catch (error) {
                console.warn(`Failed to create directory ${dir}:`, error);
            }
        }
    }
    /**
     * Load version history from disk
     */
    async loadVersionHistory() {
        try {
            const historyPath = path_1.default.join(this.config.versionsDirectory, 'history.json');
            const historyContent = await promises_1.default.readFile(historyPath, 'utf-8');
            this.versionHistory = JSON.parse(historyContent);
        }
        catch (error) {
            // History file doesn't exist - start fresh
            this.versionHistory = [];
        }
    }
    /**
     * Save version history to disk
     */
    async saveVersionHistory() {
        const historyPath = path_1.default.join(this.config.versionsDirectory, 'history.json');
        await promises_1.default.writeFile(historyPath, JSON.stringify(this.versionHistory, null, 2));
    }
    /**
     * Detect current configuration version
     */
    async detectCurrentVersion() {
        try {
            const configPath = 'config/enterprise_config.yaml';
            const configContent = await promises_1.default.readFile(configPath, 'utf-8');
            const config = js_yaml_1.default.load(configContent);
            this.currentVersion = config.schema?.version || '1.0.0';
        }
        catch (error) {
            // Configuration file doesn't exist or is invalid
            this.currentVersion = '1.0.0';
        }
    }
    /**
     * Get available migrations for a version range
     */
    getAvailableMigrations(fromVersion, toVersion) {
        const availableMigrations = [];
        for (const [version, migration] of this.migrations.entries()) {
            if (this.isVersionInRange(version, fromVersion, toVersion)) {
                availableMigrations.push(migration);
            }
        }
        // Sort by semantic version
        return availableMigrations.sort((a, b) => this.compareVersions(a.version, b.version));
    }
    /**
     * Execute migration from current version to target version
     */
    async migrate(targetVersion) {
        const startTime = new Date();
        const fromVersion = this.currentVersion;
        const toVersion = targetVersion || this.getLatestVersion();
        const result = {
            success: false,
            fromVersion,
            toVersion,
            executedMigrations: [],
            duration: 0,
            errors: [],
            warnings: [],
            metadata: {
                startTime,
                endTime: new Date(),
                totalMigrations: 0,
                skippedMigrations: 0,
                configSizeBefore: 0,
                configSizeAfter: 0,
                performanceMetrics: {
                    migrationTime: 0,
                    validationTime: 0,
                    backupTime: 0,
                    ioOperations: 0
                }
            }
        };
        try {
            // Load current configuration
            const configPath = 'config/enterprise_config.yaml';
            const configContent = await promises_1.default.readFile(configPath, 'utf-8');
            let currentConfig = js_yaml_1.default.load(configContent);
            result.metadata.configSizeBefore = Buffer.byteLength(configContent, 'utf-8');
            // Create backup
            const backupStart = performance.now();
            result.backupPath = await this.createBackup(currentConfig, fromVersion);
            result.metadata.performanceMetrics.backupTime = performance.now() - backupStart;
            // Get required migrations
            const migrations = this.getAvailableMigrations(fromVersion, toVersion);
            result.metadata.totalMigrations = migrations.length;
            // Execute migrations in sequence
            const migrationStart = performance.now();
            for (const migration of migrations) {
                try {
                    // Check if migration should be skipped
                    if (migration.skipIf && migration.skipIf(currentConfig)) {
                        result.metadata.skippedMigrations++;
                        result.warnings.push({
                            migration: migration.version,
                            message: 'Migration skipped due to skip condition',
                            recommendation: 'Review skip conditions if this is unexpected'
                        });
                        continue;
                    }
                    // Validate dependencies
                    const dependenciesValid = await this.validateDependencies(migration, result.executedMigrations);
                    if (!dependenciesValid) {
                        result.errors.push({
                            migration: migration.version,
                            error: 'Migration dependencies not satisfied',
                            severity: 'error',
                            rollbackRequired: true
                        });
                        break;
                    }
                    // Execute migration
                    currentConfig = await migration.up(currentConfig);
                    // Validate result if required
                    if (this.config.validationRequired) {
                        const validationStart = performance.now();
                        const isValid = await migration.validate(currentConfig);
                        result.metadata.performanceMetrics.validationTime += performance.now() - validationStart;
                        if (!isValid) {
                            result.errors.push({
                                migration: migration.version,
                                error: 'Post-migration validation failed',
                                severity: 'error',
                                rollbackRequired: true
                            });
                            break;
                        }
                    }
                    result.executedMigrations.push(migration.version);
                    // Notify about breaking changes
                    if (migration.breaking) {
                        result.warnings.push({
                            migration: migration.version,
                            message: 'This migration contains breaking changes',
                            recommendation: 'Review compatibility with existing configurations'
                        });
                    }
                }
                catch (error) {
                    result.errors.push({
                        migration: migration.version,
                        error: error.message,
                        severity: 'critical',
                        rollbackRequired: true
                    });
                    break;
                }
            }
            result.metadata.performanceMetrics.migrationTime = performance.now() - migrationStart;
            // Handle migration failure
            if (result.errors.length > 0) {
                if (this.config.rollbackEnabled) {
                    result.rollbackInfo = {
                        availableVersions: this.getAvailableVersions(),
                        recommendedVersion: fromVersion,
                        rollbackPath: result.backupPath,
                        estimatedDuration: result.metadata.performanceMetrics.migrationTime * 0.5
                    };
                }
                result.success = false;
            }
            else {
                // Save migrated configuration
                const newConfigContent = js_yaml_1.default.dump(currentConfig, {
                    indent: 2,
                    lineWidth: 120,
                    noRefs: true
                });
                await promises_1.default.writeFile(configPath, newConfigContent);
                result.metadata.performanceMetrics.ioOperations++;
                result.metadata.configSizeAfter = Buffer.byteLength(newConfigContent, 'utf-8');
                // Update current version
                this.currentVersion = toVersion;
                // Save version metadata
                await this.saveVersionMetadata(currentConfig, toVersion, result);
                result.success = true;
            }
        }
        catch (error) {
            result.errors.push({
                migration: 'system',
                error: `Migration system error: ${error.message}`,
                severity: 'critical',
                rollbackRequired: true
            });
        }
        // Calculate final metrics
        const endTime = new Date();
        result.metadata.endTime = endTime;
        result.duration = endTime.getTime() - startTime.getTime();
        // Send notifications
        if (this.config.notifications.enabled) {
            await this.sendNotification(result);
        }
        return result;
    }
    /**
     * Rollback to a specific version
     */
    async rollback(targetVersion) {
        if (!this.config.rollbackEnabled) {
            throw new Error('Rollback is disabled in configuration');
        }
        const startTime = new Date();
        const fromVersion = this.currentVersion;
        const result = {
            success: false,
            fromVersion,
            toVersion: targetVersion,
            executedMigrations: [],
            duration: 0,
            errors: [],
            warnings: [],
            metadata: {
                startTime,
                endTime: new Date(),
                totalMigrations: 0,
                skippedMigrations: 0,
                configSizeBefore: 0,
                configSizeAfter: 0,
                performanceMetrics: {
                    migrationTime: 0,
                    validationTime: 0,
                    backupTime: 0,
                    ioOperations: 0
                }
            }
        };
        try {
            // Load current configuration
            const configPath = 'config/enterprise_config.yaml';
            const configContent = await promises_1.default.readFile(configPath, 'utf-8');
            let currentConfig = js_yaml_1.default.load(configContent);
            result.metadata.configSizeBefore = Buffer.byteLength(configContent, 'utf-8');
            // Create backup before rollback
            result.backupPath = await this.createBackup(currentConfig, fromVersion);
            // Get migrations to rollback (in reverse order)
            const migrations = this.getAvailableMigrations(targetVersion, fromVersion).reverse();
            result.metadata.totalMigrations = migrations.length;
            // Execute rollback migrations
            for (const migration of migrations) {
                try {
                    currentConfig = await migration.down(currentConfig);
                    result.executedMigrations.push(migration.version);
                }
                catch (error) {
                    result.errors.push({
                        migration: migration.version,
                        error: `Rollback failed: ${error.message}`,
                        severity: 'critical',
                        rollbackRequired: false
                    });
                    break;
                }
            }
            if (result.errors.length === 0) {
                // Save rolled back configuration
                const newConfigContent = js_yaml_1.default.dump(currentConfig);
                await promises_1.default.writeFile(configPath, newConfigContent);
                this.currentVersion = targetVersion;
                result.success = true;
            }
        }
        catch (error) {
            result.errors.push({
                migration: 'system',
                error: `Rollback system error: ${error.message}`,
                severity: 'critical',
                rollbackRequired: false
            });
        }
        result.metadata.endTime = new Date();
        result.duration = result.metadata.endTime.getTime() - startTime.getTime();
        return result;
    }
    /**
     * Create configuration backup
     */
    async createBackup(config, version) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const backupFileName = `config-backup-${version}-${timestamp}.yaml`;
        const backupPath = path_1.default.join(this.config.backupDirectory, backupFileName);
        const configContent = js_yaml_1.default.dump(config);
        if (this.config.compressionEnabled) {
            const zlib = require('zlib');
            const compressed = zlib.gzipSync(configContent);
            await promises_1.default.writeFile(`${backupPath}.gz`, compressed);
            return `${backupPath}.gz`;
        }
        else {
            await promises_1.default.writeFile(backupPath, configContent);
            return backupPath;
        }
    }
    /**
     * Save version metadata
     */
    async saveVersionMetadata(config, version, migrationResult) {
        const configContent = js_yaml_1.default.dump(config);
        const checksum = (0, crypto_1.createHash)('sha256').update(configContent).digest('hex');
        const metadata = {
            version,
            timestamp: new Date().toISOString(),
            description: `Migration to version ${version}`,
            breaking_changes: migrationResult.errors.length > 0 ? ['Migration contained errors'] : [],
            migration_required: false,
            compatibility_level: this.getCompatibilityLevel(version),
            rollback_supported: this.config.rollbackEnabled,
            checksum,
            size_bytes: Buffer.byteLength(configContent, 'utf-8'),
            author: process.env.USER || 'system',
            change_log: migrationResult.executedMigrations.map(v => ({
                type: 'changed',
                description: `Applied migration ${v}`,
                impact: 'medium'
            }))
        };
        // Add to history
        this.versionHistory.push(metadata);
        // Trim history if needed
        if (this.versionHistory.length > this.config.maxVersionHistory) {
            this.versionHistory = this.versionHistory.slice(-this.config.maxVersionHistory);
        }
        // Save to disk
        await this.saveVersionHistory();
        // Save individual version file
        const versionPath = path_1.default.join(this.config.versionsDirectory, `${version}.json`);
        await promises_1.default.writeFile(versionPath, JSON.stringify(metadata, null, 2));
    }
    /**
     * Validate migration dependencies
     */
    async validateDependencies(migration, executedMigrations) {
        for (const dependency of migration.dependencies) {
            if (!executedMigrations.includes(dependency) && this.compareVersions(dependency, this.currentVersion) > 0) {
                return false;
            }
        }
        return true;
    }
    /**
     * Get compatibility level for version
     */
    getCompatibilityLevel(version) {
        const [major, minor] = version.split('.').map(Number);
        const [currentMajor, currentMinor] = this.currentVersion.split('.').map(Number);
        if (major !== currentMajor)
            return 'major';
        if (minor !== currentMinor)
            return 'minor';
        return 'patch';
    }
    /**
     * Check if version is in range
     */
    isVersionInRange(version, fromVersion, toVersion) {
        return this.compareVersions(version, fromVersion) > 0 &&
            this.compareVersions(version, toVersion) <= 0;
    }
    /**
     * Compare semantic versions
     */
    compareVersions(a, b) {
        const [aMajor, aMinor, aPatch] = a.split('.').map(Number);
        const [bMajor, bMinor, bPatch] = b.split('.').map(Number);
        if (aMajor !== bMajor)
            return aMajor - bMajor;
        if (aMinor !== bMinor)
            return aMinor - bMinor;
        return aPatch - bPatch;
    }
    /**
     * Get latest available version
     */
    getLatestVersion() {
        const versions = Array.from(this.migrations.keys()).sort(this.compareVersions.bind(this));
        return versions[versions.length - 1] || this.currentVersion;
    }
    /**
     * Get available versions for rollback
     */
    getAvailableVersions() {
        return this.versionHistory
            .map(v => v.version)
            .filter(v => v !== this.currentVersion)
            .sort(this.compareVersions.bind(this));
    }
    /**
     * Send migration notification
     */
    async sendNotification(result) {
        const notificationData = {
            type: result.success ? 'migration_success' : 'migration_failure',
            fromVersion: result.fromVersion,
            toVersion: result.toVersion,
            duration: result.duration,
            errors: result.errors,
            warnings: result.warnings
        };
        // In production, this would integrate with actual notification systems
        console.log('[MIGRATION NOTIFICATION]', JSON.stringify(notificationData, null, 2));
    }
    /**
     * Get current version
     */
    getCurrentVersion() {
        return this.currentVersion;
    }
    /**
     * Get version history
     */
    getVersionHistory() {
        return [...this.versionHistory];
    }
    /**
     * Get available migrations
     */
    getAvailableMigrationsList() {
        return Array.from(this.migrations.values()).map(m => ({
            version: m.version,
            description: m.description,
            breaking: m.breaking
        }));
    }
    /**
     * Check if migration is needed
     */
    isMigrationNeeded(targetVersion) {
        const target = targetVersion || this.getLatestVersion();
        return this.compareVersions(target, this.currentVersion) > 0;
    }
    /**
     * Add custom migration
     */
    addMigration(migration) {
        this.migrations.set(migration.version, migration);
    }
    /**
     * Remove migration
     */
    removeMigration(version) {
        return this.migrations.delete(version);
    }
    /**
     * Get migration system status
     */
    getStatus() {
        const latestVersion = this.getLatestVersion();
        return {
            currentVersion: this.currentVersion,
            latestVersion,
            migrationNeeded: this.isMigrationNeeded(latestVersion),
            totalMigrations: this.migrations.size,
            versionHistoryCount: this.versionHistory.length,
            rollbackEnabled: this.config.rollbackEnabled,
            autoMigrationEnabled: this.config.autoMigration
        };
    }
}
exports.ConfigurationMigrationManager = ConfigurationMigrationManager;
exports.default = ConfigurationMigrationManager;
//# sourceMappingURL=migration-versioning.js.map