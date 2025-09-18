"use strict";
/**
 * Security Vulnerability Gate Validator (QG-005)
 *
 * Implements security vulnerability gate with zero critical/high finding
 * enforcement and comprehensive security validation for quality gates.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.SecurityGateValidator = void 0;
const events_1 = require("events");
class SecurityGateValidator extends events_1.EventEmitter {
    constructor(thresholds) {
        super();
        this.vulnerabilityHistory = new Map();
        this.securityMetricsHistory = new Map();
        this.owaspTop10 = [
            'A01:2021-Broken Access Control',
            'A02:2021-Cryptographic Failures',
            'A03:2021-Injection',
            'A04:2021-Insecure Design',
            'A05:2021-Security Misconfiguration',
            'A06:2021-Vulnerable and Outdated Components',
            'A07:2021-Identification and Authentication Failures',
            'A08:2021-Software and Data Integrity Failures',
            'A09:2021-Security Logging and Monitoring Failures',
            'A10:2021-Server-Side Request Forgery'
        ];
        this.thresholds = thresholds;
    }
    /**
     * Validate security for quality gate
     */
    async validateSecurity(artifacts, context) {
        const violations = [];
        const recommendations = [];
        try {
            // Extract security data from artifacts
            const securityData = await this.extractSecurityData(artifacts, context);
            // Calculate security metrics
            const metrics = await this.calculateSecurityMetrics(securityData);
            // Perform vulnerability analysis
            const vulnerabilityViolations = await this.analyzeVulnerabilities(securityData);
            violations.push(...vulnerabilityViolations);
            // Check OWASP Top 10 compliance
            const owaspViolations = await this.checkOWASPCompliance(securityData);
            violations.push(...owaspViolations);
            // Validate authentication security
            const authViolations = this.validateAuthentication(securityData);
            violations.push(...authViolations);
            // Validate authorization security
            const authzViolations = this.validateAuthorization(securityData);
            violations.push(...authzViolations);
            // Validate encryption implementation
            const encryptionViolations = this.validateEncryption(securityData);
            violations.push(...encryptionViolations);
            // Validate logging and monitoring
            const loggingViolations = this.validateLogging(securityData);
            violations.push(...loggingViolations);
            // Check for blocking violations (critical/high)
            const blockers = violations.filter(v => v.severity === 'critical' || v.severity === 'high');
            // Determine pass/fail status
            const passed = this.determineSecurityGateStatus(violations, metrics);
            // Generate recommendations
            const securityRecommendations = this.generateSecurityRecommendations(violations, metrics);
            recommendations.push(...securityRecommendations);
            // Store historical data
            this.storeSecurityHistory(violations, metrics, context);
            const result = {
                metrics,
                violations,
                recommendations,
                passed,
                blockers
            };
            this.emit('security-validated', result);
            if (!passed) {
                this.emit('security-gate-failed', result);
            }
            if (blockers.length > 0) {
                this.emit('critical-vulnerability', { blockers, context });
            }
            return result;
        }
        catch (error) {
            const errorResult = {
                metrics: this.getDefaultSecurityMetrics(),
                violations: [{
                        id: `security-error-${Date.now()}`,
                        severity: 'critical',
                        category: 'system',
                        title: 'Security validation failed',
                        description: `Security validation system error: ${error.message}`,
                        location: 'security-gate',
                        recommendation: 'Fix security validation system',
                        autoRemediable: false,
                        estimatedFixTime: 60
                    }],
                recommendations: ['Fix security validation system'],
                passed: false,
                blockers: []
            };
            this.emit('security-error', errorResult);
            return errorResult;
        }
    }
    /**
     * Extract security data from artifacts
     */
    async extractSecurityData(artifacts, context) {
        const data = {};
        // Extract from SAST (Static Application Security Testing)
        const sastResults = artifacts.filter(a => a.type === 'sast');
        if (sastResults.length > 0) {
            data.sast = this.extractSASTData(sastResults);
        }
        // Extract from DAST (Dynamic Application Security Testing)
        const dastResults = artifacts.filter(a => a.type === 'dast');
        if (dastResults.length > 0) {
            data.dast = this.extractDASTData(dastResults);
        }
        // Extract from SCA (Software Composition Analysis)
        const scaResults = artifacts.filter(a => a.type === 'sca');
        if (scaResults.length > 0) {
            data.sca = this.extractSCAData(scaResults);
        }
        // Extract from Infrastructure scanning
        const infraScan = artifacts.filter(a => a.type === 'infrastructure-security');
        if (infraScan.length > 0) {
            data.infrastructure = this.extractInfraSecurityData(infraScan);
        }
        // Extract from Code quality security checks
        const codeQuality = artifacts.filter(a => a.type === 'code-security');
        if (codeQuality.length > 0) {
            data.codeQuality = this.extractCodeSecurityData(codeQuality);
        }
        // Extract from Compliance checks
        const compliance = artifacts.filter(a => a.type === 'compliance-security');
        if (compliance.length > 0) {
            data.compliance = this.extractComplianceSecurityData(compliance);
        }
        return data;
    }
    /**
     * Calculate comprehensive security metrics
     */
    async calculateSecurityMetrics(data) {
        // Calculate vulnerability metrics
        const vulnerabilities = this.calculateVulnerabilityMetrics(data);
        // Calculate compliance metrics
        const compliance = this.calculateComplianceMetrics(data);
        // Calculate authentication metrics
        const authentication = this.calculateAuthenticationMetrics(data);
        // Calculate authorization metrics
        const authorization = this.calculateAuthorizationMetrics(data);
        // Calculate encryption metrics
        const encryption = this.calculateEncryptionMetrics(data);
        // Calculate logging metrics
        const logging = this.calculateLoggingMetrics(data);
        // Calculate overall security score
        const overallScore = this.calculateOverallSecurityScore({
            vulnerabilities,
            compliance,
            authentication,
            authorization,
            encryption,
            logging
        });
        return {
            vulnerabilities,
            compliance,
            authentication,
            authorization,
            encryption,
            logging,
            overallScore
        };
    }
    /**
     * Calculate vulnerability metrics
     */
    calculateVulnerabilityMetrics(data) {
        const vulnerabilities = this.aggregateVulnerabilities(data);
        const byCategory = {};
        let critical = 0, high = 0, medium = 0, low = 0, info = 0;
        vulnerabilities.forEach(vuln => {
            // Count by severity
            switch (vuln.severity) {
                case 'critical':
                    critical++;
                    break;
                case 'high':
                    high++;
                    break;
                case 'medium':
                    medium++;
                    break;
                case 'low':
                    low++;
                    break;
                case 'info':
                    info++;
                    break;
            }
            // Count by category
            const category = vuln.category || 'unknown';
            byCategory[category] = (byCategory[category] || 0) + 1;
        });
        // Calculate trends (would use historical data in real implementation)
        const trends = {
            newVulnerabilities: vulnerabilities.length,
            fixedVulnerabilities: 0,
            regressionRate: 0
        };
        return {
            total: vulnerabilities.length,
            critical,
            high,
            medium,
            low,
            info,
            byCategory,
            trends
        };
    }
    /**
     * Aggregate vulnerabilities from all sources
     */
    aggregateVulnerabilities(data) {
        const vulnerabilities = [];
        // From SAST
        if (data.sast?.vulnerabilities) {
            vulnerabilities.push(...data.sast.vulnerabilities);
        }
        // From DAST
        if (data.dast?.vulnerabilities) {
            vulnerabilities.push(...data.dast.vulnerabilities);
        }
        // From SCA
        if (data.sca?.vulnerabilities) {
            vulnerabilities.push(...data.sca.vulnerabilities);
        }
        // From Infrastructure
        if (data.infrastructure?.vulnerabilities) {
            vulnerabilities.push(...data.infrastructure.vulnerabilities);
        }
        // From Code Quality
        if (data.codeQuality?.securityIssues) {
            vulnerabilities.push(...data.codeQuality.securityIssues);
        }
        return vulnerabilities;
    }
    /**
     * Calculate compliance metrics
     */
    calculateComplianceMetrics(data) {
        const owasp = this.calculateOWASPCompliance(data);
        const nist = this.calculateNISTCompliance(data);
        const pci = this.calculatePCICompliance(data);
        const gdpr = this.calculateGDPRCompliance(data);
        const iso27001 = this.calculateISO27001Compliance(data);
        return { owasp, nist, pci, gdpr, iso27001 };
    }
    /**
     * Calculate OWASP Top 10 compliance
     */
    calculateOWASPCompliance(data) {
        const top10Coverage = {};
        const violations = [];
        // Check each OWASP Top 10 category
        this.owaspTop10.forEach(category => {
            const hasViolation = this.checkOWASPCategory(category, data);
            top10Coverage[category] = !hasViolation;
            if (hasViolation) {
                violations.push(category);
            }
        });
        const coveredCount = Object.values(top10Coverage).filter(covered => covered).length;
        const score = (coveredCount / this.owaspTop10.length) * 100;
        return { score, top10Coverage, violations };
    }
    /**
     * Check specific OWASP category for violations
     */
    checkOWASPCategory(category, data) {
        const vulnerabilities = this.aggregateVulnerabilities(data);
        // Map OWASP categories to vulnerability types
        const categoryMappings = {
            'A01:2021-Broken Access Control': ['access-control', 'authorization', 'privilege-escalation'],
            'A02:2021-Cryptographic Failures': ['encryption', 'cryptography', 'weak-crypto'],
            'A03:2021-Injection': ['sql-injection', 'command-injection', 'ldap-injection'],
            'A04:2021-Insecure Design': ['insecure-design', 'threat-modeling'],
            'A05:2021-Security Misconfiguration': ['misconfiguration', 'default-config'],
            'A06:2021-Vulnerable and Outdated Components': ['outdated-components', 'vulnerable-dependencies'],
            'A07:2021-Identification and Authentication Failures': ['authentication', 'session-management'],
            'A08:2021-Software and Data Integrity Failures': ['integrity', 'supply-chain'],
            'A09:2021-Security Logging and Monitoring Failures': ['logging', 'monitoring'],
            'A10:2021-Server-Side Request Forgery': ['ssrf', 'request-forgery']
        };
        const relevantTypes = categoryMappings[category] || [];
        return vulnerabilities.some(vuln => relevantTypes.some(type => vuln.category?.toLowerCase().includes(type) ||
            vuln.title?.toLowerCase().includes(type)));
    }
    /**
     * Calculate other compliance metrics (simplified)
     */
    calculateNISTCompliance(data) {
        // Simplified NIST calculation
        const frameworkCoverage = {
            'Identify': 80,
            'Protect': 75,
            'Detect': 70,
            'Respond': 65,
            'Recover': 60
        };
        const totalScore = Object.values(frameworkCoverage).reduce((sum, score) => sum + score, 0);
        const averageScore = totalScore / Object.keys(frameworkCoverage).length;
        return {
            score: averageScore,
            frameworkCoverage,
            controlsImplemented: 85,
            totalControls: 100
        };
    }
    calculatePCICompliance(data) {
        const requirements = {
            'Install and maintain a firewall': true,
            'Do not use vendor-supplied defaults': true,
            'Protect stored cardholder data': false,
            'Encrypt transmission of cardholder data': true,
            'Protect all systems against malware': true,
            'Develop and maintain secure systems': false
        };
        const score = (Object.values(requirements).filter(req => req).length / Object.keys(requirements).length) * 100;
        return {
            score,
            requirements,
            dataProtection: false,
            networkSecurity: true
        };
    }
    calculateGDPRCompliance(data) {
        return {
            score: 75,
            dataProcessing: true,
            consent: true,
            rightToErasure: false,
            dataPortability: true
        };
    }
    calculateISO27001Compliance(data) {
        const controls = {
            'Information security policies': true,
            'Organization of information security': true,
            'Human resource security': false,
            'Asset management': true,
            'Access control': false
        };
        const score = (Object.values(controls).filter(ctrl => ctrl).length / Object.keys(controls).length) * 100;
        return {
            score,
            controls,
            riskAssessment: true,
            informationSecurity: false
        };
    }
    /**
     * Calculate authentication metrics
     */
    calculateAuthenticationMetrics(data) {
        const auth = data.compliance?.authentication || {};
        return {
            score: auth.score || 70,
            multiFactorAuth: auth.multiFactorAuth || false,
            passwordPolicies: auth.passwordPolicies || true,
            sessionManagement: auth.sessionManagement || true,
            accountLockout: auth.accountLockout || false,
            weakCredentials: auth.weakCredentials || 5
        };
    }
    /**
     * Calculate authorization metrics
     */
    calculateAuthorizationMetrics(data) {
        const authz = data.compliance?.authorization || {};
        return {
            score: authz.score || 65,
            accessControl: authz.accessControl || true,
            roleBasedAccess: authz.roleBasedAccess || false,
            privilegeEscalation: authz.privilegeEscalation || 2,
            unauthorizedAccess: authz.unauthorizedAccess || 1,
            dataLeakage: authz.dataLeakage || 0
        };
    }
    /**
     * Calculate encryption metrics
     */
    calculateEncryptionMetrics(data) {
        const encryption = data.compliance?.encryption || {};
        return {
            score: encryption.score || 80,
            dataAtRest: encryption.dataAtRest || true,
            dataInTransit: encryption.dataInTransit || true,
            keyManagement: encryption.keyManagement || false,
            cryptographicStrength: encryption.cryptographicStrength || 85,
            weakEncryption: encryption.weakEncryption || 1
        };
    }
    /**
     * Calculate logging metrics
     */
    calculateLoggingMetrics(data) {
        const logging = data.compliance?.logging || {};
        return {
            score: logging.score || 60,
            securityEvents: logging.securityEvents || false,
            auditTrail: logging.auditTrail || true,
            logIntegrity: logging.logIntegrity || false,
            logRetention: logging.logRetention || true,
            sensitiveDataLogging: logging.sensitiveDataLogging || 3
        };
    }
    /**
     * Calculate overall security score
     */
    calculateOverallSecurityScore(metrics) {
        const weights = {
            vulnerabilities: 0.30, // 30% - vulnerability assessment
            compliance: 0.25, // 25% - compliance frameworks
            authentication: 0.15, // 15% - authentication security
            authorization: 0.15, // 15% - authorization security
            encryption: 0.10, // 10% - encryption implementation
            logging: 0.05 // 5% - logging and monitoring
        };
        // Calculate vulnerability score (inverse of vulnerability count with severity weighting)
        const vulnScore = Math.max(0, 100 - (metrics.vulnerabilities.critical * 20 +
            metrics.vulnerabilities.high * 10 +
            metrics.vulnerabilities.medium * 5 +
            metrics.vulnerabilities.low * 1));
        // Calculate compliance score (average of all compliance frameworks)
        const complianceScores = [
            metrics.compliance.owasp.score,
            metrics.compliance.nist.score,
            metrics.compliance.pci.score,
            metrics.compliance.gdpr.score,
            metrics.compliance.iso27001.score
        ];
        const avgComplianceScore = complianceScores.reduce((sum, score) => sum + score, 0) / complianceScores.length;
        // Calculate weighted overall score
        const overallScore = (vulnScore * weights.vulnerabilities +
            avgComplianceScore * weights.compliance +
            metrics.authentication.score * weights.authentication +
            metrics.authorization.score * weights.authorization +
            metrics.encryption.score * weights.encryption +
            metrics.logging.score * weights.logging);
        return Math.round(overallScore);
    }
    /**
     * Analyze vulnerabilities and create violations
     */
    async analyzeVulnerabilities(data) {
        const violations = [];
        const vulnerabilities = this.aggregateVulnerabilities(data);
        for (const vuln of vulnerabilities) {
            const violation = {
                id: vuln.id || `vuln-${Date.now()}-${Math.random()}`,
                severity: vuln.severity || 'medium',
                category: vuln.category || 'unknown',
                title: vuln.title || 'Security vulnerability',
                description: vuln.description || 'Security vulnerability detected',
                cwe: vuln.cwe,
                cve: vuln.cve,
                location: vuln.location || 'unknown',
                recommendation: vuln.recommendation || 'Review and fix security issue',
                autoRemediable: vuln.autoRemediable || false,
                estimatedFixTime: vuln.estimatedFixTime || 30
            };
            violations.push(violation);
        }
        return violations;
    }
    /**
     * Check OWASP Top 10 compliance and create violations
     */
    async checkOWASPCompliance(data) {
        const violations = [];
        const owaspMetrics = this.calculateOWASPCompliance(data);
        for (const violation of owaspMetrics.violations) {
            violations.push({
                id: `owasp-${violation.replace(/[^a-zA-Z0-9]/g, '-')}`,
                severity: 'high',
                category: 'owasp',
                title: `OWASP Top 10 Violation: ${violation}`,
                description: `Application violates OWASP Top 10 category: ${violation}`,
                location: 'application',
                recommendation: `Address ${violation} vulnerabilities according to OWASP guidelines`,
                autoRemediable: false,
                estimatedFixTime: 120
            });
        }
        return violations;
    }
    /**
     * Validate authentication security
     */
    validateAuthentication(data) {
        const violations = [];
        const auth = data.compliance?.authentication || {};
        if (!auth.multiFactorAuth) {
            violations.push({
                id: 'auth-mfa-missing',
                severity: 'high',
                category: 'authentication',
                title: 'Multi-factor authentication not implemented',
                description: 'Application lacks multi-factor authentication implementation',
                location: 'authentication-system',
                recommendation: 'Implement multi-factor authentication for enhanced security',
                autoRemediable: false,
                estimatedFixTime: 240
            });
        }
        if (auth.weakCredentials > 0) {
            violations.push({
                id: 'auth-weak-credentials',
                severity: 'medium',
                category: 'authentication',
                title: 'Weak credentials detected',
                description: `${auth.weakCredentials} weak credentials found in the system`,
                location: 'user-accounts',
                recommendation: 'Enforce strong password policies and update weak credentials',
                autoRemediable: true,
                estimatedFixTime: 60
            });
        }
        return violations;
    }
    /**
     * Validate authorization security
     */
    validateAuthorization(data) {
        const violations = [];
        const authz = data.compliance?.authorization || {};
        if (authz.privilegeEscalation > 0) {
            violations.push({
                id: 'authz-privilege-escalation',
                severity: 'critical',
                category: 'authorization',
                title: 'Privilege escalation vulnerabilities detected',
                description: `${authz.privilegeEscalation} privilege escalation vulnerabilities found`,
                location: 'access-control-system',
                recommendation: 'Fix privilege escalation vulnerabilities immediately',
                autoRemediable: false,
                estimatedFixTime: 180
            });
        }
        if (!authz.roleBasedAccess) {
            violations.push({
                id: 'authz-rbac-missing',
                severity: 'medium',
                category: 'authorization',
                title: 'Role-based access control not implemented',
                description: 'Application lacks proper role-based access control',
                location: 'authorization-system',
                recommendation: 'Implement role-based access control system',
                autoRemediable: false,
                estimatedFixTime: 360
            });
        }
        return violations;
    }
    /**
     * Validate encryption implementation
     */
    validateEncryption(data) {
        const violations = [];
        const encryption = data.compliance?.encryption || {};
        if (!encryption.dataAtRest) {
            violations.push({
                id: 'encryption-data-at-rest',
                severity: 'high',
                category: 'encryption',
                title: 'Data at rest not encrypted',
                description: 'Sensitive data stored without encryption',
                location: 'data-storage',
                recommendation: 'Implement encryption for data at rest',
                autoRemediable: false,
                estimatedFixTime: 120
            });
        }
        if (!encryption.dataInTransit) {
            violations.push({
                id: 'encryption-data-in-transit',
                severity: 'critical',
                category: 'encryption',
                title: 'Data in transit not encrypted',
                description: 'Data transmitted without proper encryption',
                location: 'network-communication',
                recommendation: 'Implement TLS/SSL for all data transmission',
                autoRemediable: false,
                estimatedFixTime: 90
            });
        }
        if (encryption.weakEncryption > 0) {
            violations.push({
                id: 'encryption-weak-algorithms',
                severity: 'high',
                category: 'encryption',
                title: 'Weak encryption algorithms detected',
                description: `${encryption.weakEncryption} instances of weak encryption found`,
                location: 'cryptographic-implementations',
                recommendation: 'Replace weak encryption algorithms with strong alternatives',
                autoRemediable: false,
                estimatedFixTime: 150
            });
        }
        return violations;
    }
    /**
     * Validate logging and monitoring
     */
    validateLogging(data) {
        const violations = [];
        const logging = data.compliance?.logging || {};
        if (!logging.securityEvents) {
            violations.push({
                id: 'logging-security-events',
                severity: 'medium',
                category: 'logging',
                title: 'Security events not logged',
                description: 'Security-related events are not being logged',
                location: 'logging-system',
                recommendation: 'Implement comprehensive security event logging',
                autoRemediable: false,
                estimatedFixTime: 90
            });
        }
        if (logging.sensitiveDataLogging > 0) {
            violations.push({
                id: 'logging-sensitive-data',
                severity: 'high',
                category: 'logging',
                title: 'Sensitive data logged',
                description: `${logging.sensitiveDataLogging} instances of sensitive data in logs`,
                location: 'log-files',
                recommendation: 'Remove sensitive data from logs and implement data sanitization',
                autoRemediable: true,
                estimatedFixTime: 60
            });
        }
        return violations;
    }
    /**
     * Determine security gate pass/fail status
     */
    determineSecurityGateStatus(violations, metrics) {
        // Check critical/high violation thresholds
        const criticalCount = violations.filter(v => v.severity === 'critical').length;
        const highCount = violations.filter(v => v.severity === 'high').length;
        if (criticalCount > this.thresholds.criticalVulnerabilities) {
            return false;
        }
        if (highCount > this.thresholds.highVulnerabilities) {
            return false;
        }
        // Check overall security score
        if (metrics.overallScore < this.thresholds.minimumSecurityScore) {
            return false;
        }
        // Check medium vulnerabilities
        const mediumCount = violations.filter(v => v.severity === 'medium').length;
        if (mediumCount > this.thresholds.mediumVulnerabilities) {
            return false;
        }
        return true;
    }
    /**
     * Generate security recommendations
     */
    generateSecurityRecommendations(violations, metrics) {
        const recommendations = [];
        // High-priority recommendations based on violations
        const criticalViolations = violations.filter(v => v.severity === 'critical');
        if (criticalViolations.length > 0) {
            recommendations.push('Address critical security vulnerabilities immediately');
            recommendations.push('Consider emergency security review');
        }
        const highViolations = violations.filter(v => v.severity === 'high');
        if (highViolations.length > 0) {
            recommendations.push('Fix high severity security issues before deployment');
        }
        // OWASP-specific recommendations
        if (metrics.compliance.owasp.score < 80) {
            recommendations.push('Improve OWASP Top 10 compliance');
            recommendations.push('Implement OWASP security testing guidelines');
        }
        // Authentication recommendations
        if (metrics.authentication.score < 80) {
            recommendations.push('Strengthen authentication mechanisms');
            if (!metrics.authentication.multiFactorAuth) {
                recommendations.push('Implement multi-factor authentication');
            }
        }
        // Encryption recommendations
        if (metrics.encryption.score < 80) {
            recommendations.push('Improve encryption implementation');
            if (!metrics.encryption.dataAtRest) {
                recommendations.push('Implement data-at-rest encryption');
            }
            if (!metrics.encryption.dataInTransit) {
                recommendations.push('Ensure all data transmission is encrypted');
            }
        }
        // General security improvements
        if (metrics.overallScore < 85) {
            recommendations.push('Implement comprehensive security improvement program');
            recommendations.push('Regular security assessments and penetration testing');
        }
        return recommendations;
    }
    /**
     * Store security history for trending
     */
    storeSecurityHistory(violations, metrics, context) {
        const timestamp = new Date().toISOString();
        // Store violations
        this.vulnerabilityHistory.set(timestamp, violations);
        // Store metrics
        this.securityMetricsHistory.set(timestamp, metrics);
        // Keep only last 30 entries
        if (this.vulnerabilityHistory.size > 30) {
            const oldestKey = this.vulnerabilityHistory.keys().next().value;
            this.vulnerabilityHistory.delete(oldestKey);
        }
        if (this.securityMetricsHistory.size > 30) {
            const oldestKey = this.securityMetricsHistory.keys().next().value;
            this.securityMetricsHistory.delete(oldestKey);
        }
    }
    /**
     * Extract data from various artifact types
     */
    extractSASTData(artifacts) {
        return artifacts.reduce((acc, artifact) => ({
            ...acc,
            ...artifact.data
        }), {});
    }
    extractDASTData(artifacts) {
        return artifacts.reduce((acc, artifact) => ({
            ...acc,
            ...artifact.data
        }), {});
    }
    extractSCAData(artifacts) {
        return artifacts.reduce((acc, artifact) => ({
            ...acc,
            ...artifact.data
        }), {});
    }
    extractInfraSecurityData(artifacts) {
        return artifacts.reduce((acc, artifact) => ({
            ...acc,
            ...artifact.data
        }), {});
    }
    extractCodeSecurityData(artifacts) {
        return artifacts.reduce((acc, artifact) => ({
            ...acc,
            ...artifact.data
        }), {});
    }
    extractComplianceSecurityData(artifacts) {
        return artifacts.reduce((acc, artifact) => ({
            ...acc,
            ...artifact.data
        }), {});
    }
    /**
     * Get default security metrics for error cases
     */
    getDefaultSecurityMetrics() {
        return {
            vulnerabilities: {
                total: 0,
                critical: 0,
                high: 0,
                medium: 0,
                low: 0,
                info: 0,
                byCategory: {},
                trends: { newVulnerabilities: 0, fixedVulnerabilities: 0, regressionRate: 0 }
            },
            compliance: {
                owasp: { score: 0, top10Coverage: {}, violations: [] },
                nist: { score: 0, frameworkCoverage: {}, controlsImplemented: 0, totalControls: 0 },
                pci: { score: 0, requirements: {}, dataProtection: false, networkSecurity: false },
                gdpr: { score: 0, dataProcessing: false, consent: false, rightToErasure: false, dataPortability: false },
                iso27001: { score: 0, controls: {}, riskAssessment: false, informationSecurity: false }
            },
            authentication: {
                score: 0,
                multiFactorAuth: false,
                passwordPolicies: false,
                sessionManagement: false,
                accountLockout: false,
                weakCredentials: 0
            },
            authorization: {
                score: 0,
                accessControl: false,
                roleBasedAccess: false,
                privilegeEscalation: 0,
                unauthorizedAccess: 0,
                dataLeakage: 0
            },
            encryption: {
                score: 0,
                dataAtRest: false,
                dataInTransit: false,
                keyManagement: false,
                cryptographicStrength: 0,
                weakEncryption: 0
            },
            logging: {
                score: 0,
                securityEvents: false,
                auditTrail: false,
                logIntegrity: false,
                logRetention: false,
                sensitiveDataLogging: 0
            },
            overallScore: 0
        };
    }
    /**
     * Get current security status
     */
    async getCurrentStatus() {
        const history = Array.from(this.securityMetricsHistory.values());
        if (history.length > 0) {
            return history[history.length - 1];
        }
        return this.getDefaultSecurityMetrics();
    }
    /**
     * Get security trends
     */
    getSecurityTrends() {
        const history = Array.from(this.securityMetricsHistory.values());
        if (history.length < 2) {
            return { trend: 'insufficient-data' };
        }
        const recent = history.slice(-10);
        const overallScoreTrend = this.calculateTrend(recent.map(h => h.overallScore));
        const criticalVulnTrend = this.calculateTrend(recent.map(h => h.vulnerabilities.critical));
        const complianceTrend = this.calculateTrend(recent.map(h => h.compliance.owasp.score));
        return {
            overallScore: overallScoreTrend,
            criticalVulnerabilities: criticalVulnTrend,
            owaspCompliance: complianceTrend,
            overallTrend: (overallScoreTrend > 0 && criticalVulnTrend < 0 && complianceTrend > 0) ? 'improving' : 'degrading'
        };
    }
    /**
     * Calculate trend for a series of values
     */
    calculateTrend(values) {
        if (values.length < 2)
            return 0;
        const first = values[0];
        const last = values[values.length - 1];
        return ((last - first) / first) * 100;
    }
}
exports.SecurityGateValidator = SecurityGateValidator;
//# sourceMappingURL=SecurityGateValidator.js.map