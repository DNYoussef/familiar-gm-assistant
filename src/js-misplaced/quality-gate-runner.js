#!/usr/bin/env node
/**
 * Quality Gate Runner - Main entry point for Princess Gate quality system
 * Orchestrates all quality validation drones
 */

const PrincessGate = require('./princess-gate');
const path = require('path');
const fs = require('fs');

class QualityGateRunner {
  constructor() {
    this.princessGate = new PrincessGate();
    this.projectPath = process.cwd();
  }

  async run(options = {}) {
    console.log('👑 QUALITY PRINCESS: Deploying drone hive for zero-theater validation...');
    console.log('🎯 Target Project:', this.projectPath);
    console.log('');

    const startTime = Date.now();

    try {
      // Execute full Princess Gate audit
      const auditResults = await this.princessGate.executeFullAudit(this.projectPath);

      // Display results
      this._displayResults(auditResults);

      // Handle exit codes based on results
      const exitCode = this._getExitCode(auditResults);

      const endTime = Date.now();
      const duration = ((endTime - startTime) / 1000).toFixed(2);

      console.log('');
      console.log(`⏱️  Audit completed in ${duration} seconds`);
      console.log(`🏁 Final Status: ${auditResults.overallStatus}`);
      console.log('');

      if (options.failOnReject && exitCode !== 0) {
        console.log('❌ Quality gates failed - terminating with error code');
        process.exit(exitCode);
      }

      return auditResults;

    } catch (error) {
      console.error('💥 QUALITY SYSTEM FAILURE:', error.message);
      console.error(error.stack);

      if (options.failOnReject) {
        process.exit(1);
      }

      return {
        overallStatus: 'SYSTEM_FAILURE',
        error: error.message
      };
    }
  }

  _displayResults(auditResults) {
    console.log('📊 AUDIT RESULTS');
    console.log('================');
    console.log('');

    // Phase 1: Theater Detection
    console.log('🎭 Phase 1: Theater Detection');
    console.log(`   Score: ${auditResults.phase1_theater.summary.theaterScore}/100`);
    console.log(`   Status: ${auditResults.phase1_theater.gateDecision}`);
    console.log(`   Violations: ${auditResults.phase1_theater.summary.violationsFound}`);

    if (auditResults.phase1_theater.violations.length > 0) {
      const criticalViolations = auditResults.phase1_theater.violations.filter(v => v.severity === 'CRITICAL');
      if (criticalViolations.length > 0) {
        console.log(`   🚨 CRITICAL VIOLATIONS: ${criticalViolations.length}`);
        criticalViolations.slice(0, 3).forEach(v => {
          console.log(`      - ${v.file}:${v.line} - ${v.violation}`);
        });
        if (criticalViolations.length > 3) {
          console.log(`      ... and ${criticalViolations.length - 3} more`);
        }
      }
    }
    console.log('');

    // Phase 2: Reality Validation
    console.log('⚡ Phase 2: Reality Validation');
    console.log(`   Score: ${auditResults.phase2_reality.realityScore}/100`);
    console.log(`   Status: ${auditResults.phase2_reality.gateDecision}`);

    const failedValidations = auditResults.phase2_reality.validations.filter(v => v.status === 'FAIL');
    if (failedValidations.length > 0) {
      console.log(`   ❌ Failed Validations: ${failedValidations.length}`);
      failedValidations.slice(0, 3).forEach(v => {
        console.log(`      - ${v.type}: ${v.message}`);
      });
    }
    console.log('');

    // Phase 3: Princess Gate
    console.log('👑 Phase 3: Princess Gate');
    console.log(`   Final Decision: ${auditResults.phase3_gate.finalDecision}`);

    if (auditResults.phase3_gate.violationsBlocking.length > 0) {
      console.log(`   🚫 Blocking Issues: ${auditResults.phase3_gate.violationsBlocking.length}`);
      auditResults.phase3_gate.violationsBlocking.forEach(violation => {
        console.log(`      - ${violation}`);
      });
    }

    console.log('');
    console.log('   Quality Gates:');
    auditResults.phase3_gate.gates.forEach(gate => {
      const icon = gate.status === 'PASS' ? '✅' : '❌';
      console.log(`   ${icon} ${gate.name}: ${gate.actual} (required: ${gate.required})`);
    });
    console.log('');

    // Princess Domains
    console.log('🏰 Princess Domain Status');
    Object.entries(auditResults.princessDomains).forEach(([name, domain]) => {
      const icon = domain.status === 'PASS' ? '✅' : domain.status === 'NEEDS_WORK' ? '⚠️' : '❌';
      console.log(`   ${icon} ${name}: ${domain.score}/100 (${domain.status})`);
    });
    console.log('');

    // Overall Status
    const statusIcon = this._getStatusIcon(auditResults.overallStatus);
    console.log(`${statusIcon} OVERALL STATUS: ${auditResults.overallStatus}`);
  }

  _getStatusIcon(status) {
    switch (status) {
      case 'APPROVED': return '🎉';
      case 'CONDITIONAL_APPROVAL': return '⚠️';
      case 'NEEDS_MAJOR_WORK': return '🔧';
      case 'REJECTED': return '❌';
      case 'SYSTEM_FAILURE': return '💥';
      default: return '❓';
    }
  }

  _getExitCode(auditResults) {
    switch (auditResults.overallStatus) {
      case 'APPROVED': return 0;
      case 'CONDITIONAL_APPROVAL': return 0; // Warning, but not failing
      case 'NEEDS_MAJOR_WORK': return 1;
      case 'REJECTED': return 2;
      case 'SYSTEM_FAILURE': return 3;
      default: return 1;
    }
  }

  // Individual drone commands
  async runTheaterScan() {
    console.log('🎭 Running Theater Detection Scan...');
    const scanResults = await this.princessGate.theaterDetector.scanDirectory(this.projectPath);
    const report = this.princessGate.theaterDetector.generateReport(scanResults);

    console.log(`Theater Score: ${report.summary.theaterScore}/100`);
    console.log(`Violations Found: ${report.summary.violationsFound}`);

    if (report.violations.length > 0) {
      console.log('\nTop Violations:');
      report.violations.slice(0, 5).forEach(v => {
        console.log(`  ${v.severity}: ${v.file}:${v.line} - ${v.violation}`);
      });
    }

    return report;
  }

  async runRealityCheck() {
    console.log('⚡ Running Reality Validation...');
    const validationResults = await this.princessGate.realityValidator.validateProject(this.projectPath);

    console.log(`Reality Score: ${validationResults.realityScore}/100`);
    console.log(`Status: ${validationResults.status}`);

    validationResults.validations.forEach(v => {
      const icon = v.status === 'PASS' ? '✅' : v.status === 'FAIL' ? '❌' : '⚠️';
      console.log(`  ${icon} ${v.type}: ${v.message}`);
    });

    return validationResults;
  }

  async runPrincessAudit() {
    console.log('👑 Running Full Princess Gate Audit...');
    return await this.run({ failOnReject: false });
  }
}

// CLI Interface
if (require.main === module) {
  const runner = new QualityGateRunner();
  const command = process.argv[2] || 'audit';

  (async () => {
    switch (command) {
      case 'theater':
        await runner.runTheaterScan();
        break;
      case 'reality':
        await runner.runRealityCheck();
        break;
      case 'audit':
      default:
        await runner.run({ failOnReject: true });
        break;
    }
  })().catch(error => {
    console.error('Command failed:', error.message);
    process.exit(1);
  });
}

module.exports = QualityGateRunner;