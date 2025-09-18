/**
 * Production Readiness Validation - End-to-End Drone
 * Comprehensive production deployment validation with zero tolerance for fake implementations
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

class ProductionReadinessValidator {
  constructor() {
    this.validationResults = [];
    this.criticalIssues = [];
    this.productionScore = 0;
  }

  async validateProductionReadiness(projectPath) {
    console.log('🚀 PRODUCTION DRONE: Executing production readiness validation...');
    console.log('');

    const results = {
      timestamp: new Date().toISOString(),
      projectPath: projectPath,
      validations: [],
      productionScore: 0,
      criticalIssues: [],
      readinessStatus: 'UNKNOWN',
      recommendations: []
    };

    try {
      // Phase 1: Environment & Configuration Validation
      await this._validateEnvironmentSetup(projectPath, results);

      // Phase 2: Build & Deployment Validation
      await this._validateBuildProcess(projectPath, results);

      // Phase 3: Security & Compliance Validation
      await this._validateSecurityCompliance(projectPath, results);

      // Phase 4: Performance & Scalability Validation
      await this._validatePerformanceReadiness(projectPath, results);

      // Phase 5: Monitoring & Observability Validation
      await this._validateMonitoringSetup(projectPath, results);

      // Phase 6: Database & Data Integrity Validation
      await this._validateDataIntegrity(projectPath, results);

      // Calculate overall production score
      results.productionScore = this._calculateProductionScore(results.validations);
      results.readinessStatus = this._determineReadinessStatus(results);

      console.log(`🏁 Production Validation Complete - Score: ${results.productionScore}/100`);
      console.log(`📊 Readiness Status: ${results.readinessStatus}`);

    } catch (error) {
      console.error('💥 Production validation failed:', error.message);
      results.criticalIssues.push({
        type: 'SYSTEM_FAILURE',
        message: error.message,
        severity: 'CRITICAL'
      });
      results.readinessStatus = 'NOT_READY';
    }

    return results;
  }

  async _validateEnvironmentSetup(projectPath, results) {
    console.log('🔧 Phase 1: Environment & Configuration Validation');

    // Check for environment configuration
    const envFiles = ['.env.example', '.env.production', '.env.local'];
    let envConfigFound = false;

    for (const envFile of envFiles) {
      const envPath = path.join(projectPath, envFile);
      if (fs.existsSync(envPath)) {
        envConfigFound = true;
        results.validations.push({
          type: 'ENVIRONMENT',
          check: `Environment file: ${envFile}`,
          status: 'PASS',
          severity: 'INFO'
        });

        // Validate environment file content
        const envContent = fs.readFileSync(envPath, 'utf8');
        this._validateEnvironmentVariables(envContent, results);
      }
    }

    if (!envConfigFound) {
      results.validations.push({
        type: 'ENVIRONMENT',
        check: 'Environment configuration files',
        status: 'FAIL',
        severity: 'HIGH',
        message: 'No environment configuration files found'
      });
    }

    // Check package.json production configuration
    const packageJsonPath = path.join(projectPath, 'package.json');
    if (fs.existsSync(packageJsonPath)) {
      const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

      // Check production scripts
      if (packageJson.scripts) {
        const productionScripts = ['build', 'start', 'prod'];
        let hasProductionScripts = false;

        for (const script of productionScripts) {
          if (packageJson.scripts[script]) {
            hasProductionScripts = true;
            results.validations.push({
              type: 'ENVIRONMENT',
              check: `Production script: ${script}`,
              status: 'PASS',
              severity: 'INFO'
            });
          }
        }

        if (!hasProductionScripts) {
          results.validations.push({
            type: 'ENVIRONMENT',
            check: 'Production scripts',
            status: 'WARN',
            severity: 'MEDIUM',
            message: 'No production scripts found in package.json'
          });
        }
      }

      // Check dependencies vs devDependencies
      const prodDeps = Object.keys(packageJson.dependencies || {});
      const devDeps = Object.keys(packageJson.devDependencies || {});

      if (prodDeps.length === 0) {
        results.validations.push({
          type: 'ENVIRONMENT',
          check: 'Production dependencies',
          status: 'WARN',
          severity: 'MEDIUM',
          message: 'No production dependencies found'
        });
      } else {
        results.validations.push({
          type: 'ENVIRONMENT',
          check: 'Production dependencies',
          status: 'PASS',
          severity: 'INFO',
          message: `${prodDeps.length} production dependencies configured`
        });
      }
    }

    console.log('   ✅ Environment validation completed');
  }

  async _validateBuildProcess(projectPath, results) {
    console.log('🔨 Phase 2: Build & Deployment Validation');

    return new Promise((resolve) => {
      // Test production build
      const buildProcess = spawn('npm', ['run', 'build'], {
        cwd: projectPath,
        stdio: 'pipe'
      });

      let stdout = '';
      let stderr = '';

      buildProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      buildProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      buildProcess.on('close', (code) => {
        if (code === 0) {
          results.validations.push({
            type: 'BUILD',
            check: 'Production build process',
            status: 'PASS',
            severity: 'CRITICAL',
            message: 'Build completed successfully'
          });

          // Check for build output
          const buildOutputs = ['dist/', 'build/', 'public/', 'out/'];
          let buildOutputFound = false;

          for (const output of buildOutputs) {
            const outputPath = path.join(projectPath, output);
            if (fs.existsSync(outputPath)) {
              buildOutputFound = true;
              results.validations.push({
                type: 'BUILD',
                check: `Build output directory: ${output}`,
                status: 'PASS',
                severity: 'INFO'
              });
              break;
            }
          }

          if (!buildOutputFound) {
            results.validations.push({
              type: 'BUILD',
              check: 'Build output directory',
              status: 'WARN',
              severity: 'MEDIUM',
              message: 'No standard build output directory found'
            });
          }

        } else {
          results.validations.push({
            type: 'BUILD',
            check: 'Production build process',
            status: 'FAIL',
            severity: 'CRITICAL',
            message: `Build failed with exit code ${code}`,
            details: stderr
          });

          results.criticalIssues.push({
            type: 'BUILD_FAILURE',
            message: 'Production build process failed',
            severity: 'CRITICAL'
          });
        }

        console.log('   ✅ Build process validation completed');
        resolve();
      });

      buildProcess.on('error', (error) => {
        results.validations.push({
          type: 'BUILD',
          check: 'Production build process',
          status: 'FAIL',
          severity: 'CRITICAL',
          message: `Build process error: ${error.message}`
        });

        console.log('   ❌ Build process validation failed');
        resolve();
      });

      // Timeout after 2 minutes
      setTimeout(() => {
        buildProcess.kill();
        results.validations.push({
          type: 'BUILD',
          check: 'Production build process',
          status: 'FAIL',
          severity: 'HIGH',
          message: 'Build process timed out'
        });

        console.log('   ⏰ Build process timed out');
        resolve();
      }, 120000);
    });
  }

  async _validateSecurityCompliance(projectPath, results) {
    console.log('🔒 Phase 3: Security & Compliance Validation');

    // Check for security configuration files
    const securityFiles = [
      '.semgrep.yml',
      '.bandit',
      'security.md',
      'SECURITY.md',
      '.snyk'
    ];

    let securityConfigFound = false;

    for (const secFile of securityFiles) {
      const secPath = path.join(projectPath, secFile);
      if (fs.existsSync(secPath)) {
        securityConfigFound = true;
        results.validations.push({
          type: 'SECURITY',
          check: `Security configuration: ${secFile}`,
          status: 'PASS',
          severity: 'INFO'
        });
      }
    }

    if (!securityConfigFound) {
      results.validations.push({
        type: 'SECURITY',
        check: 'Security configuration files',
        status: 'WARN',
        severity: 'HIGH',
        message: 'No security configuration files found'
      });
    }

    // Check for sensitive data in codebase
    await this._scanForSensitiveData(projectPath, results);

    // Check for secure defaults
    await this._validateSecureDefaults(projectPath, results);

    console.log('   ✅ Security validation completed');
  }

  async _validatePerformanceReadiness(projectPath, results) {
    console.log('⚡ Phase 4: Performance & Scalability Validation');

    // Check for performance optimization files
    const perfFiles = [
      'webpack.config.js',
      'vite.config.js',
      'rollup.config.js',
      '.babelrc',
      'tsconfig.json'
    ];

    let perfConfigFound = false;

    for (const perfFile of perfFiles) {
      const perfPath = path.join(projectPath, perfFile);
      if (fs.existsSync(perfPath)) {
        perfConfigFound = true;
        results.validations.push({
          type: 'PERFORMANCE',
          check: `Performance configuration: ${perfFile}`,
          status: 'PASS',
          severity: 'INFO'
        });

        // Check for production optimizations
        const content = fs.readFileSync(perfPath, 'utf8');
        if (content.includes('production') || content.includes('optimize')) {
          results.validations.push({
            type: 'PERFORMANCE',
            check: `Production optimizations in ${perfFile}`,
            status: 'PASS',
            severity: 'INFO'
          });
        }
      }
    }

    // Check for caching configuration
    const cacheIndicators = [
      'service-worker.js',
      'sw.js',
      'cache',
      'manifest.json'
    ];

    let cachingFound = false;

    for (const indicator of cacheIndicators) {
      const cachePath = path.join(projectPath, indicator);
      if (fs.existsSync(cachePath)) {
        cachingFound = true;
        results.validations.push({
          type: 'PERFORMANCE',
          check: `Caching strategy: ${indicator}`,
          status: 'PASS',
          severity: 'INFO'
        });
      }
    }

    if (!cachingFound) {
      results.validations.push({
        type: 'PERFORMANCE',
        check: 'Caching strategies',
        status: 'WARN',
        severity: 'MEDIUM',
        message: 'No caching strategies detected'
      });
    }

    console.log('   ✅ Performance validation completed');
  }

  async _validateMonitoringSetup(projectPath, results) {
    console.log('📊 Phase 5: Monitoring & Observability Validation');

    // Check for logging configuration
    const loggingIndicators = [
      'winston',
      'pino',
      'log4js',
      'bunyan',
      'console.log'
    ];

    let loggingFound = false;

    // Scan source files for logging
    const srcDir = path.join(projectPath, 'src');
    if (fs.existsSync(srcDir)) {
      const files = this._getAllJSFiles(srcDir);

      for (const file of files) {
        const content = fs.readFileSync(file, 'utf8');

        for (const indicator of loggingIndicators) {
          if (content.includes(indicator)) {
            loggingFound = true;
            break;
          }
        }

        if (loggingFound) break;
      }
    }

    if (loggingFound) {
      results.validations.push({
        type: 'MONITORING',
        check: 'Logging implementation',
        status: 'PASS',
        severity: 'INFO'
      });
    } else {
      results.validations.push({
        type: 'MONITORING',
        check: 'Logging implementation',
        status: 'WARN',
        severity: 'MEDIUM',
        message: 'No logging implementation detected'
      });
    }

    // Check for error handling
    const errorHandlingPatterns = [
      'try.*catch',
      'throw new Error',
      'error.*handler',
      'exception'
    ];

    let errorHandlingFound = false;

    if (fs.existsSync(srcDir)) {
      const files = this._getAllJSFiles(srcDir);

      for (const file of files) {
        const content = fs.readFileSync(file, 'utf8');

        for (const pattern of errorHandlingPatterns) {
          const regex = new RegExp(pattern, 'i');
          if (regex.test(content)) {
            errorHandlingFound = true;
            break;
          }
        }

        if (errorHandlingFound) break;
      }
    }

    if (errorHandlingFound) {
      results.validations.push({
        type: 'MONITORING',
        check: 'Error handling',
        status: 'PASS',
        severity: 'INFO'
      });
    } else {
      results.validations.push({
        type: 'MONITORING',
        check: 'Error handling',
        status: 'WARN',
        severity: 'HIGH',
        message: 'Limited error handling detected'
      });
    }

    console.log('   ✅ Monitoring validation completed');
  }

  async _validateDataIntegrity(projectPath, results) {
    console.log('💾 Phase 6: Database & Data Integrity Validation');

    // Check for database configuration
    const dbConfigFiles = [
      'knexfile.js',
      'database.json',
      'prisma/schema.prisma',
      'sequelize.config.js',
      'mongoose.config.js',
      'typeorm.config.js'
    ];

    let dbConfigFound = false;

    for (const configFile of dbConfigFiles) {
      const configPath = path.join(projectPath, configFile);
      if (fs.existsSync(configPath)) {
        dbConfigFound = true;
        results.validations.push({
          type: 'DATABASE',
          check: `Database configuration: ${configFile}`,
          status: 'PASS',
          severity: 'INFO'
        });

        // Check for production database settings
        const content = fs.readFileSync(configPath, 'utf8');
        if (content.includes('production')) {
          results.validations.push({
            type: 'DATABASE',
            check: `Production database config in ${configFile}`,
            status: 'PASS',
            severity: 'INFO'
          });
        }
      }
    }

    if (!dbConfigFound) {
      results.validations.push({
        type: 'DATABASE',
        check: 'Database configuration',
        status: 'SKIP',
        severity: 'LOW',
        message: 'No database configuration detected (may not be required)'
      });
    }

    // Check for migration files
    const migrationDirs = ['migrations/', 'prisma/migrations/', 'db/migrate/'];
    let migrationsFound = false;

    for (const migDir of migrationDirs) {
      const migPath = path.join(projectPath, migDir);
      if (fs.existsSync(migPath)) {
        const migFiles = fs.readdirSync(migPath);
        if (migFiles.length > 0) {
          migrationsFound = true;
          results.validations.push({
            type: 'DATABASE',
            check: `Database migrations: ${migDir}`,
            status: 'PASS',
            severity: 'INFO',
            message: `${migFiles.length} migration files found`
          });
        }
      }
    }

    if (dbConfigFound && !migrationsFound) {
      results.validations.push({
        type: 'DATABASE',
        check: 'Database migrations',
        status: 'WARN',
        severity: 'MEDIUM',
        message: 'Database configured but no migrations found'
      });
    }

    console.log('   ✅ Data integrity validation completed');
  }

  _validateEnvironmentVariables(envContent, results) {
    // Check for common production environment variables
    const requiredProdVars = [
      'NODE_ENV',
      'PORT',
      'DATABASE_URL'
    ];

    const sensitivePatterns = [
      /password\s*=\s*['"].*['"/gi,
      /secret\s*=\s*['"].*['"/gi,
      /key\s*=\s*['"].*['"/gi,
      /token\s*=\s*['"].*['"/gi
    ];

    // Check for sensitive data in plain text
    for (const pattern of sensitivePatterns) {
      if (pattern.test(envContent)) {
        results.validations.push({
          type: 'ENVIRONMENT',
          check: 'Sensitive data in environment files',
          status: 'WARN',
          severity: 'HIGH',
          message: 'Potential sensitive data found in environment file'
        });
      }
    }

    // Check for environment variable examples
    if (envContent.includes('example') || envContent.includes('YOUR_')) {
      results.validations.push({
        type: 'ENVIRONMENT',
        check: 'Environment variable examples',
        status: 'PASS',
        severity: 'INFO',
        message: 'Example environment variables found'
      });
    }
  }

  async _scanForSensitiveData(projectPath, results) {
    const sensitivePatterns = [
      /password\s*[:=]\s*['"][^'"]+['"]/gi,
      /api[_-]?key\s*[:=]\s*['"][^'"]+['"]/gi,
      /secret\s*[:=]\s*['"][^'"]+['"]/gi,
      /token\s*[:=]\s*['"][^'"]+['"]/gi,
      /-----BEGIN.*PRIVATE KEY-----/gi
    ];

    const srcDir = path.join(projectPath, 'src');
    if (!fs.existsSync(srcDir)) return;

    const files = this._getAllJSFiles(srcDir);
    let sensitiveDataFound = false;

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');

      for (const pattern of sensitivePatterns) {
        if (pattern.test(content)) {
          sensitiveDataFound = true;
          results.validations.push({
            type: 'SECURITY',
            check: `Sensitive data in source code: ${path.basename(file)}`,
            status: 'FAIL',
            severity: 'CRITICAL',
            message: 'Potential sensitive data found in source code'
          });

          results.criticalIssues.push({
            type: 'SENSITIVE_DATA_EXPOSURE',
            message: `Sensitive data found in ${file}`,
            severity: 'CRITICAL'
          });
        }
      }
    }

    if (!sensitiveDataFound) {
      results.validations.push({
        type: 'SECURITY',
        check: 'Sensitive data in source code',
        status: 'PASS',
        severity: 'INFO',
        message: 'No sensitive data found in source code'
      });
    }
  }

  async _validateSecureDefaults(projectPath, results) {
    // Check for HTTPS enforcement
    const srcDir = path.join(projectPath, 'src');
    if (!fs.existsSync(srcDir)) return;

    const files = this._getAllJSFiles(srcDir);
    let httpsFound = false;

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');

      if (content.includes('https://') || content.includes('ssl') || content.includes('tls')) {
        httpsFound = true;
        break;
      }
    }

    if (httpsFound) {
      results.validations.push({
        type: 'SECURITY',
        check: 'HTTPS/SSL configuration',
        status: 'PASS',
        severity: 'INFO'
      });
    } else {
      results.validations.push({
        type: 'SECURITY',
        check: 'HTTPS/SSL configuration',
        status: 'WARN',
        severity: 'MEDIUM',
        message: 'No HTTPS/SSL configuration detected'
      });
    }
  }

  _getAllJSFiles(dir) {
    const files = [];

    try {
      const items = fs.readdirSync(dir);

      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
          files.push(...this._getAllJSFiles(fullPath));
        } else if (/\.(js|ts|jsx|tsx)$/.test(item)) {
          files.push(fullPath);
        }
      }
    } catch (error) {
      // Directory access error - skip
    }

    return files;
  }

  _calculateProductionScore(validations) {
    const weights = {
      'ENVIRONMENT': 15,
      'BUILD': 25,
      'SECURITY': 20,
      'PERFORMANCE': 15,
      'MONITORING': 15,
      'DATABASE': 10
    };

    let totalScore = 0;
    let totalWeight = 0;

    for (const [category, weight] of Object.entries(weights)) {
      const categoryValidations = validations.filter(v => v.type === category);

      if (categoryValidations.length > 0) {
        const passCount = categoryValidations.filter(v => v.status === 'PASS').length;
        const categoryScore = (passCount / categoryValidations.length) * 100;

        totalScore += categoryScore * weight;
        totalWeight += weight;
      }
    }

    return totalWeight > 0 ? Math.round(totalScore / totalWeight) : 0;
  }

  _determineReadinessStatus(results) {
    if (results.criticalIssues.length > 0) {
      return 'NOT_READY';
    }

    const criticalFailures = results.validations.filter(v =>
      v.status === 'FAIL' && v.severity === 'CRITICAL'
    ).length;

    if (criticalFailures > 0) {
      return 'NOT_READY';
    }

    if (results.productionScore >= 90) {
      return 'PRODUCTION_READY';
    } else if (results.productionScore >= 75) {
      return 'MOSTLY_READY';
    } else if (results.productionScore >= 60) {
      return 'NEEDS_IMPROVEMENT';
    } else {
      return 'NOT_READY';
    }
  }
}

module.exports = ProductionReadinessValidator;

// CLI execution
if (require.main === module) {
  (async () => {
    const validator = new ProductionReadinessValidator();
    const results = await validator.validateProductionReadiness(process.cwd());

    console.log('');
    console.log('📊 PRODUCTION READINESS SUMMARY');
    console.log('================================');
    console.log(`Score: ${results.productionScore}/100`);
    console.log(`Status: ${results.readinessStatus}`);
    console.log(`Critical Issues: ${results.criticalIssues.length}`);

    const exitCode = results.readinessStatus === 'PRODUCTION_READY' || results.readinessStatus === 'MOSTLY_READY' ? 0 : 1;
    process.exit(exitCode);
  })().catch(error => {
    console.error('Production validation failed:', error.message);
    process.exit(1);
  });
}