/**
 * Princess Gate Quality System - Zero Tolerance Quality Gates
 * Part 3 of 3-Part Audit System: Princess Gate
 */

const TheaterDetector = require('./theater-detector');
const RealityValidator = require('./reality-validator');
const fs = require('fs');
const path = require('path');

class PrincessGate {
  constructor() {
    this.theaterDetector = new TheaterDetector();
    this.realityValidator = new RealityValidator();

    this.qualityGates = {
      THEATER_THRESHOLD: 60,    // Theater score must be >= 60
      REALITY_THRESHOLD: 70,    // Reality score must be >= 70
      CRITICAL_VIOLATIONS: 0,   // Zero critical violations allowed
      HIGH_VIOLATIONS: 2,       // Max 2 high severity violations
      TEST_COVERAGE: 80,        // Min 80% test coverage
      BUILD_SUCCESS: true       // Build must succeed
    };
  }

  async executeFullAudit(projectPath) {
    const auditResults = {
      timestamp: new Date().toISOString(),
      projectPath: projectPath,
      phase1_theater: null,
      phase2_reality: null,
      phase3_gate: null,
      overallStatus: 'UNKNOWN',
      princessDomains: {},
      recommendations: []
    };

    try {
      console.log('🚨 PRINCESS GATE: Initiating full quality audit...');

      // Phase 1: Theater Detection
      console.log('🎭 Phase 1: Theater Detection - Scanning for fake implementations...');
      auditResults.phase1_theater = await this._executeTheaterDetection(projectPath);

      // Phase 2: Reality Validation
      console.log('⚡ Phase 2: Reality Validation - Verifying actual functionality...');
      auditResults.phase2_reality = await this._executeRealityValidation(projectPath);

      // Phase 3: Princess Gate Decision
      console.log('👑 Phase 3: Princess Gate - Applying zero tolerance gates...');
      auditResults.phase3_gate = await this._executePrincessGate(auditResults);

      // Princess Domain Validation
      console.log('🏰 Princess Domain Validation - Checking all domains...');
      auditResults.princessDomains = await this._validatePrincessDomains(projectPath);

      // Overall Status
      auditResults.overallStatus = this._determineOverallStatus(auditResults);

      // Generate comprehensive report
      await this._generateAuditReport(auditResults);

      console.log(`🏁 Audit Complete - Status: ${auditResults.overallStatus}`);

    } catch (error) {
      console.error('❌ AUDIT FAILURE:', error.message);
      auditResults.overallStatus = 'SYSTEM_FAILURE';
      auditResults.error = error.message;
    }

    return auditResults;
  }

  async _executeTheaterDetection(projectPath) {
    const scanResults = await this.theaterDetector.scanDirectory(projectPath);
    const report = this.theaterDetector.generateReport(scanResults);

    const theaterAudit = {
      ...report,
      gateDecision: report.summary.theaterScore >= this.qualityGates.THEATER_THRESHOLD ? 'PASS' : 'FAIL',
      threshold: this.qualityGates.THEATER_THRESHOLD
    };

    // Log critical findings
    const criticalViolations = report.violations.filter(v => v.severity === 'CRITICAL');
    if (criticalViolations.length > 0) {
      console.log(`⚠️  CRITICAL: Found ${criticalViolations.length} mock implementations in production code`);
    }

    return theaterAudit;
  }

  async _executeRealityValidation(projectPath) {
    const validationResults = await this.realityValidator.validateProject(projectPath);

    const realityAudit = {
      ...validationResults,
      gateDecision: validationResults.realityScore >= this.qualityGates.REALITY_THRESHOLD ? 'PASS' : 'FAIL',
      threshold: this.qualityGates.REALITY_THRESHOLD
    };

    // Log critical findings
    const failedValidations = validationResults.validations.filter(v => v.status === 'FAIL');
    if (failedValidations.length > 0) {
      console.log(`⚠️  REALITY CHECK: ${failedValidations.length} validations failed`);
    }

    return realityAudit;
  }

  async _executePrincessGate(auditResults) {
    const gateResults = {
      gates: [],
      finalDecision: 'UNKNOWN',
      violationsBlocking: []
    };

    // Gate 1: Theater Score
    const theaterGate = {
      name: 'THEATER_ELIMINATION',
      required: this.qualityGates.THEATER_THRESHOLD,
      actual: auditResults.phase1_theater.summary.theaterScore,
      status: auditResults.phase1_theater.gateDecision,
      blocking: auditResults.phase1_theater.gateDecision === 'FAIL'
    };
    gateResults.gates.push(theaterGate);

    // Gate 2: Reality Score
    const realityGate = {
      name: 'REALITY_VALIDATION',
      required: this.qualityGates.REALITY_THRESHOLD,
      actual: auditResults.phase2_reality.realityScore,
      status: auditResults.phase2_reality.gateDecision,
      blocking: auditResults.phase2_reality.gateDecision === 'FAIL'
    };
    gateResults.gates.push(realityGate);

    // Gate 3: Critical Violations
    const criticalViolations = auditResults.phase1_theater.violations
      .filter(v => v.severity === 'CRITICAL').length;

    const criticalGate = {
      name: 'ZERO_CRITICAL_VIOLATIONS',
      required: this.qualityGates.CRITICAL_VIOLATIONS,
      actual: criticalViolations,
      status: criticalViolations <= this.qualityGates.CRITICAL_VIOLATIONS ? 'PASS' : 'FAIL',
      blocking: criticalViolations > this.qualityGates.CRITICAL_VIOLATIONS
    };
    gateResults.gates.push(criticalGate);

    // Gate 4: High Severity Violations
    const highViolations = auditResults.phase1_theater.violations
      .filter(v => v.severity === 'HIGH').length;

    const highGate = {
      name: 'LIMITED_HIGH_VIOLATIONS',
      required: `<= ${this.qualityGates.HIGH_VIOLATIONS}`,
      actual: highViolations,
      status: highViolations <= this.qualityGates.HIGH_VIOLATIONS ? 'PASS' : 'FAIL',
      blocking: highViolations > this.qualityGates.HIGH_VIOLATIONS
    };
    gateResults.gates.push(highGate);

    // Gate 5: Build Success
    const buildValidation = auditResults.phase2_reality.validations
      .find(v => v.type === 'BUILD');

    const buildGate = {
      name: 'BUILD_SUCCESS',
      required: 'PASS',
      actual: buildValidation ? buildValidation.status : 'UNKNOWN',
      status: buildValidation && buildValidation.status === 'PASS' ? 'PASS' : 'FAIL',
      blocking: !buildValidation || buildValidation.status !== 'PASS'
    };
    gateResults.gates.push(buildGate);

    // Determine final decision
    const blockingGates = gateResults.gates.filter(g => g.blocking);
    gateResults.finalDecision = blockingGates.length === 0 ? 'PASS' : 'FAIL';
    gateResults.violationsBlocking = blockingGates.map(g => g.name);

    return gateResults;
  }

  async _validatePrincessDomains(projectPath) {
    const domains = {
      DATA_PRINCESS: await this._validateDataDomain(projectPath),
      INTEGRATION_PRINCESS: await this._validateIntegrationDomain(projectPath),
      FRONTEND_PRINCESS: await this._validateFrontendDomain(projectPath),
      BACKEND_PRINCESS: await this._validateBackendDomain(projectPath),
      SECURITY_PRINCESS: await this._validateSecurityDomain(projectPath),
      DEVOPS_PRINCESS: await this._validateDevOpsDomain(projectPath)
    };

    return domains;
  }

  async _validateDataDomain(projectPath) {
    const domain = {
      name: 'DATA_PRINCESS',
      status: 'UNKNOWN',
      checks: [],
      score: 0
    };

    // Check for database files
    const dbFiles = ['*.sql', '*.db', '*.sqlite', 'migrations/*', 'models/*'];
    let foundDbFiles = 0;

    for (const pattern of dbFiles) {
      if (this._hasFilesMatching(projectPath, pattern)) {
        foundDbFiles++;
        domain.checks.push({
          check: `Database files found: ${pattern}`,
          status: 'PASS'
        });
      }
    }

    domain.score = (foundDbFiles / dbFiles.length) * 100;
    domain.status = domain.score >= 60 ? 'PASS' : 'NEEDS_WORK';

    return domain;
  }

  async _validateIntegrationDomain(projectPath) {
    const domain = {
      name: 'INTEGRATION_PRINCESS',
      status: 'UNKNOWN',
      checks: [],
      score: 0
    };

    // Check for API integration files
    const integrationIndicators = [
      'api/',
      'services/',
      'integrations/',
      'webhooks/',
      'external/'
    ];

    let foundIntegrations = 0;

    for (const indicator of integrationIndicators) {
      if (fs.existsSync(path.join(projectPath, indicator))) {
        foundIntegrations++;
        domain.checks.push({
          check: `Integration directory found: ${indicator}`,
          status: 'PASS'
        });
      }
    }

    domain.score = (foundIntegrations / integrationIndicators.length) * 100;
    domain.status = domain.score >= 40 ? 'PASS' : 'NEEDS_WORK';

    return domain;
  }

  async _validateFrontendDomain(projectPath) {
    const domain = {
      name: 'FRONTEND_PRINCESS',
      status: 'UNKNOWN',
      checks: [],
      score: 0
    };

    // Check for frontend files
    const frontendIndicators = [
      'public/',
      'static/',
      'assets/',
      'components/',
      'views/'
    ];

    let foundFrontend = 0;

    for (const indicator of frontendIndicators) {
      if (fs.existsSync(path.join(projectPath, indicator))) {
        foundFrontend++;
        domain.checks.push({
          check: `Frontend directory found: ${indicator}`,
          status: 'PASS'
        });
      }
    }

    domain.score = (foundFrontend / frontendIndicators.length) * 100;
    domain.status = domain.score >= 40 ? 'PASS' : 'NEEDS_WORK';

    return domain;
  }

  async _validateBackendDomain(projectPath) {
    const domain = {
      name: 'BACKEND_PRINCESS',
      status: 'UNKNOWN',
      checks: [],
      score: 0
    };

    // Check for backend files
    const backendIndicators = [
      'server.js',
      'app.js',
      'index.js',
      'main.py',
      'routes/',
      'controllers/'
    ];

    let foundBackend = 0;

    for (const indicator of backendIndicators) {
      if (fs.existsSync(path.join(projectPath, indicator))) {
        foundBackend++;
        domain.checks.push({
          check: `Backend file/directory found: ${indicator}`,
          status: 'PASS'
        });
      }
    }

    domain.score = (foundBackend / backendIndicators.length) * 100;
    domain.status = domain.score >= 50 ? 'PASS' : 'NEEDS_WORK';

    return domain;
  }

  async _validateSecurityDomain(projectPath) {
    const domain = {
      name: 'SECURITY_PRINCESS',
      status: 'UNKNOWN',
      checks: [],
      score: 0
    };

    // Check for security files
    const securityIndicators = [
      '.env.example',
      'auth/',
      'middleware/',
      'security/',
      '.semgrep.yml',
      '.bandit'
    ];

    let foundSecurity = 0;

    for (const indicator of securityIndicators) {
      if (fs.existsSync(path.join(projectPath, indicator))) {
        foundSecurity++;
        domain.checks.push({
          check: `Security file/directory found: ${indicator}`,
          status: 'PASS'
        });
      }
    }

    domain.score = (foundSecurity / securityIndicators.length) * 100;
    domain.status = domain.score >= 50 ? 'PASS' : 'NEEDS_WORK';

    return domain;
  }

  async _validateDevOpsDomain(projectPath) {
    const domain = {
      name: 'DEVOPS_PRINCESS',
      status: 'UNKNOWN',
      checks: [],
      score: 0
    };

    // Check for DevOps files
    const devopsIndicators = [
      '.github/',
      'docker-compose.yml',
      'Dockerfile',
      '.gitlab-ci.yml',
      'ci/',
      'scripts/'
    ];

    let foundDevOps = 0;

    for (const indicator of devopsIndicators) {
      if (fs.existsSync(path.join(projectPath, indicator))) {
        foundDevOps++;
        domain.checks.push({
          check: `DevOps file/directory found: ${indicator}`,
          status: 'PASS'
        });
      }
    }

    domain.score = (foundDevOps / devopsIndicators.length) * 100;
    domain.status = domain.score >= 50 ? 'PASS' : 'NEEDS_WORK';

    return domain;
  }

  _hasFilesMatching(projectPath, pattern) {
    try {
      // Simple pattern matching - could be enhanced with glob library
      const baseName = pattern.replace('/*', '').replace('*', '');
      return fs.existsSync(path.join(projectPath, baseName));
    } catch (error) {
      return false;
    }
  }

  _determineOverallStatus(auditResults) {
    if (auditResults.phase3_gate.finalDecision === 'FAIL') {
      return 'REJECTED';
    }

    const domainsPassing = Object.values(auditResults.princessDomains)
      .filter(d => d.status === 'PASS').length;

    const totalDomains = Object.keys(auditResults.princessDomains).length;
    const domainPassRate = domainsPassing / totalDomains;

    if (domainPassRate >= 0.8) {
      return 'APPROVED';
    } else if (domainPassRate >= 0.6) {
      return 'CONDITIONAL_APPROVAL';
    } else {
      return 'NEEDS_MAJOR_WORK';
    }
  }

  async _generateAuditReport(auditResults) {
    const reportPath = path.join(auditResults.projectPath, 'tests', 'quality', 'PRINCESS_GATE_AUDIT.json');

    const report = {
      ...auditResults,
      generatedAt: new Date().toISOString(),
      auditVersion: '1.0.0',
      qualityGates: this.qualityGates
    };

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`📋 Audit report saved: ${reportPath}`);

    // Also generate human-readable summary
    const summaryPath = path.join(auditResults.projectPath, 'tests', 'quality', 'AUDIT_SUMMARY.md');
    const summary = this._generateMarkdownSummary(auditResults);
    fs.writeFileSync(summaryPath, summary);
    console.log(`📄 Audit summary saved: ${summaryPath}`);
  }

  _generateMarkdownSummary(auditResults) {
    return `# Princess Gate Quality Audit Report

## Overall Status: ${auditResults.overallStatus}

**Audit Date:** ${auditResults.timestamp}
**Project:** ${auditResults.projectPath}

## Executive Summary

${auditResults.overallStatus === 'APPROVED' ? '✅ Project has passed all quality gates and is ready for production deployment.' :
  auditResults.overallStatus === 'CONDITIONAL_APPROVAL' ? '⚠️ Project has passed critical gates but has areas needing improvement.' :
  '❌ Project has failed quality gates and requires significant work before deployment.'}

## Phase Results

### Phase 1: Theater Detection
- **Score:** ${auditResults.phase1_theater.summary.theaterScore}/100
- **Status:** ${auditResults.phase1_theater.gateDecision}
- **Violations Found:** ${auditResults.phase1_theater.summary.violationsFound}

### Phase 2: Reality Validation
- **Score:** ${auditResults.phase2_reality.realityScore}/100
- **Status:** ${auditResults.phase2_reality.gateDecision}
- **Validations:** ${auditResults.phase2_reality.validations.length}

### Phase 3: Princess Gate
- **Final Decision:** ${auditResults.phase3_gate.finalDecision}
- **Blocking Issues:** ${auditResults.phase3_gate.violationsBlocking.length}

## Princess Domain Scores

${Object.entries(auditResults.princessDomains).map(([name, domain]) =>
  `- **${name}:** ${domain.score}/100 (${domain.status})`
).join('\n')}

## Critical Actions Required

${auditResults.phase3_gate.violationsBlocking.length > 0 ?
  auditResults.phase3_gate.violationsBlocking.map(violation => `- Fix: ${violation}`).join('\n') :
  'No critical actions required - all gates passed!'}

## Recommendations

${auditResults.phase1_theater.recommendations.map(rec =>
  `- **${rec.priority}:** ${rec.action} (${rec.count} instances)`
).join('\n')}

---
*Generated by Princess Gate Quality System v1.0.0*
`;
  }
}

module.exports = PrincessGate;