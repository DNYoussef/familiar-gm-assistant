/**
 * Reality Validation System - Verify Actual Functionality
 * Part 2 of 3-Part Audit System: Reality Validation
 */

const fs = require('fs');
const { spawn } = require('child_process');
const axios = require('axios').default;

class RealityValidator {
  constructor() {
    this.validationResults = [];
    this.realityScore = 0;
  }

  async validateProject(projectPath) {
    const results = {
      timestamp: new Date().toISOString(),
      validations: [],
      realityScore: 0,
      status: 'UNKNOWN'
    };

    try {
      // 1. Validate package.json and dependencies
      await this._validateDependencies(projectPath, results);

      // 2. Validate test execution
      await this._validateTests(projectPath, results);

      // 3. Validate build process
      await this._validateBuild(projectPath, results);

      // 4. Validate API endpoints (if applicable)
      await this._validateAPIs(projectPath, results);

      // 5. Validate database connections (if applicable)
      await this._validateDatabase(projectPath, results);

      // Calculate reality score
      results.realityScore = this._calculateRealityScore(results.validations);
      results.status = results.realityScore >= 70 ? 'PASS' : 'FAIL';

    } catch (error) {
      results.validations.push({
        type: 'SYSTEM_ERROR',
        status: 'FAIL',
        message: `Validation system error: ${error.message}`,
        severity: 'CRITICAL'
      });
      results.realityScore = 0;
      results.status = 'FAIL';
    }

    return results;
  }

  async _validateDependencies(projectPath, results) {
    try {
      const packageJsonPath = `${projectPath}/package.json`;

      if (!fs.existsSync(packageJsonPath)) {
        results.validations.push({
          type: 'DEPENDENCIES',
          status: 'FAIL',
          message: 'No package.json found',
          severity: 'HIGH'
        });
        return;
      }

      const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

      // Check for production dependencies
      const prodDeps = Object.keys(packageJson.dependencies || {});
      const devDeps = Object.keys(packageJson.devDependencies || {});

      if (prodDeps.length === 0) {
        results.validations.push({
          type: 'DEPENDENCIES',
          status: 'WARN',
          message: 'No production dependencies found - may indicate incomplete implementation',
          severity: 'MEDIUM'
        });
      } else {
        results.validations.push({
          type: 'DEPENDENCIES',
          status: 'PASS',
          message: `Found ${prodDeps.length} production dependencies`,
          severity: 'INFO'
        });
      }

      // Validate node_modules exists
      if (fs.existsSync(`${projectPath}/node_modules`)) {
        results.validations.push({
          type: 'DEPENDENCIES',
          status: 'PASS',
          message: 'node_modules directory exists',
          severity: 'INFO'
        });
      } else {
        results.validations.push({
          type: 'DEPENDENCIES',
          status: 'FAIL',
          message: 'node_modules not found - run npm install',
          severity: 'HIGH'
        });
      }

    } catch (error) {
      results.validations.push({
        type: 'DEPENDENCIES',
        status: 'FAIL',
        message: `Dependency validation failed: ${error.message}`,
        severity: 'HIGH'
      });
    }
  }

  async _validateTests(projectPath, results) {
    return new Promise((resolve) => {
      // Try npm test first
      const testProcess = spawn('npm', ['test'], {
        cwd: projectPath,
        stdio: 'pipe'
      });

      let stdout = '';
      let stderr = '';

      testProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      testProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      testProcess.on('close', (code) => {
        if (code === 0) {
          results.validations.push({
            type: 'TESTS',
            status: 'PASS',
            message: 'All tests passed successfully',
            severity: 'INFO',
            details: { exitCode: code, output: stdout }
          });
        } else {
          results.validations.push({
            type: 'TESTS',
            status: 'FAIL',
            message: `Tests failed with exit code ${code}`,
            severity: 'CRITICAL',
            details: { exitCode: code, output: stderr }
          });
        }
        resolve();
      });

      // Timeout after 30 seconds
      setTimeout(() => {
        testProcess.kill();
        results.validations.push({
          type: 'TESTS',
          status: 'FAIL',
          message: 'Test execution timed out',
          severity: 'HIGH'
        });
        resolve();
      }, 30000);
    });
  }

  async _validateBuild(projectPath, results) {
    return new Promise((resolve) => {
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
            status: 'PASS',
            message: 'Build completed successfully',
            severity: 'INFO'
          });
        } else {
          results.validations.push({
            type: 'BUILD',
            status: 'FAIL',
            message: `Build failed with exit code ${code}`,
            severity: 'HIGH',
            details: { output: stderr }
          });
        }
        resolve();
      });

      buildProcess.on('error', (error) => {
        results.validations.push({
          type: 'BUILD',
          status: 'SKIP',
          message: 'No build script found',
          severity: 'LOW'
        });
        resolve();
      });

      // Timeout after 60 seconds
      setTimeout(() => {
        buildProcess.kill();
        results.validations.push({
          type: 'BUILD',
          status: 'FAIL',
          message: 'Build process timed out',
          severity: 'HIGH'
        });
        resolve();
      }, 60000);
    });
  }

  async _validateAPIs(projectPath, results) {
    try {
      // Look for API endpoint definitions
      const apiFiles = this._findAPIFiles(projectPath);

      if (apiFiles.length === 0) {
        results.validations.push({
          type: 'API',
          status: 'SKIP',
          message: 'No API endpoints detected',
          severity: 'LOW'
        });
        return;
      }

      // For each API file, try to validate endpoints
      for (const apiFile of apiFiles) {
        const endpoints = this._extractEndpoints(apiFile);

        for (const endpoint of endpoints) {
          try {
            // Simple connectivity test (if server is running)
            const response = await axios.get(`http://localhost:3000${endpoint}`, {
              timeout: 5000,
              validateStatus: () => true // Accept any status code
            });

            results.validations.push({
              type: 'API',
              status: 'PASS',
              message: `Endpoint ${endpoint} is accessible (${response.status})`,
              severity: 'INFO'
            });
          } catch (error) {
            results.validations.push({
              type: 'API',
              status: 'WARN',
              message: `Endpoint ${endpoint} not accessible - server may not be running`,
              severity: 'MEDIUM'
            });
          }
        }
      }

    } catch (error) {
      results.validations.push({
        type: 'API',
        status: 'FAIL',
        message: `API validation failed: ${error.message}`,
        severity: 'MEDIUM'
      });
    }
  }

  async _validateDatabase(projectPath, results) {
    try {
      // Look for database configuration files
      const dbConfigFiles = [
        'knexfile.js',
        'database.json',
        'sequelize.config.js',
        'mongoose.config.js'
      ];

      let hasDbConfig = false;

      for (const configFile of dbConfigFiles) {
        if (fs.existsSync(`${projectPath}/${configFile}`)) {
          hasDbConfig = true;
          results.validations.push({
            type: 'DATABASE',
            status: 'PASS',
            message: `Found database configuration: ${configFile}`,
            severity: 'INFO'
          });
          break;
        }
      }

      if (!hasDbConfig) {
        results.validations.push({
          type: 'DATABASE',
          status: 'SKIP',
          message: 'No database configuration detected',
          severity: 'LOW'
        });
      }

    } catch (error) {
      results.validations.push({
        type: 'DATABASE',
        status: 'FAIL',
        message: `Database validation failed: ${error.message}`,
        severity: 'MEDIUM'
      });
    }
  }

  _findAPIFiles(projectPath) {
    const apiFiles = [];

    const searchDirs = ['src', 'routes', 'api', 'controllers'];

    for (const dir of searchDirs) {
      const dirPath = `${projectPath}/${dir}`;
      if (fs.existsSync(dirPath)) {
        this._findFilesRecursively(dirPath, /\.(js|ts)$/, apiFiles);
      }
    }

    return apiFiles;
  }

  _findFilesRecursively(dir, pattern, results) {
    const items = fs.readdirSync(dir);

    for (const item of items) {
      const fullPath = `${dir}/${item}`;
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        this._findFilesRecursively(fullPath, pattern, results);
      } else if (pattern.test(item)) {
        results.push(fullPath);
      }
    }
  }

  _extractEndpoints(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const endpoints = [];

    // Simple regex to find route definitions
    const routePatterns = [
      /app\.(get|post|put|delete)\(['"`]([^'"`]+)['"`]/g,
      /router\.(get|post|put|delete)\(['"`]([^'"`]+)['"`]/g,
      /\.route\(['"`]([^'"`]+)['"`]/g
    ];

    for (const pattern of routePatterns) {
      let match;
      pattern.lastIndex = 0;

      while ((match = pattern.exec(content)) !== null) {
        const endpoint = match[2] || match[1];
        if (endpoint && !endpoints.includes(endpoint)) {
          endpoints.push(endpoint);
        }
      }
    }

    return endpoints;
  }

  _calculateRealityScore(validations) {
    let totalScore = 0;
    let totalWeight = 0;

    const weights = {
      'DEPENDENCIES': 20,
      'TESTS': 40,
      'BUILD': 25,
      'API': 10,
      'DATABASE': 5
    };

    for (const [type, weight] of Object.entries(weights)) {
      const typeValidations = validations.filter(v => v.type === type);

      if (typeValidations.length > 0) {
        const passCount = typeValidations.filter(v => v.status === 'PASS').length;
        const typeScore = (passCount / typeValidations.length) * 100;

        totalScore += typeScore * weight;
        totalWeight += weight;
      }
    }

    return totalWeight > 0 ? Math.round(totalScore / totalWeight) : 0;
  }
}

module.exports = RealityValidator;