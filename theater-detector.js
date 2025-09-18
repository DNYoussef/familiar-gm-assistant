/**
 * Theater Detection System - Zero Tolerance for Mock/Stub Implementations
 * Part 1 of 3-Part Audit System: Theater Detection
 */

const fs = require('fs');
const path = require('path');

class TheaterDetector {
  constructor() {
    this.violations = [];
    this.mockPatterns = [
      // Mock implementations
      /mock[A-Z]\w+/g,
      /fake[A-Z]\w+/g,
      /stub[A-Z]\w+/g,
      /dummy[A-Z]\w+/g,

      // Incomplete implementations
      /TODO.*implement/gi,
      /FIXME.*mock/gi,
      /throw new Error\(['"]not implemented/gi,
      /return null.*\/\/.*mock/gi,
      /return undefined.*\/\/.*stub/gi,

      // Test doubles in production code
      /\.mock\(/g,
      /\.stub\(/g,
      /sinon\./g,
      /jest\.mock/g,

      // Hardcoded test data
      /test@example\.com/g,
      /localhost:\d+/g,
      /127\.0\.0\.1/g,
      /example\.com/g,

      // Performance theater indicators
      /console\.log.*performance/gi,
      /setTimeout.*mock/gi,
      /Promise\.resolve.*fake/gi
    ];
  }

  async scanDirectory(dirPath, excludeDirs = ['node_modules', '.git', 'tests', '__tests__']) {
    const results = {
      totalFiles: 0,
      violationsFound: 0,
      theaterScore: 0,
      violations: []
    };

    await this._scanRecursively(dirPath, excludeDirs, results);

    // Calculate theater score (0-100, higher is better)
    results.theaterScore = Math.max(0, 100 - (results.violationsFound * 10));

    return results;
  }

  async _scanRecursively(dirPath, excludeDirs, results) {
    const items = fs.readdirSync(dirPath);

    for (const item of items) {
      const fullPath = path.join(dirPath, item);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory() && !excludeDirs.includes(item)) {
        await this._scanRecursively(fullPath, excludeDirs, results);
      } else if (stat.isFile() && this._isSourceFile(item)) {
        results.totalFiles++;
        await this._scanFile(fullPath, results);
      }
    }
  }

  _isSourceFile(filename) {
    const extensions = ['.js', '.ts', '.jsx', '.tsx', '.py', '.java', '.cs', '.cpp'];
    return extensions.some(ext => filename.endsWith(ext));
  }

  async _scanFile(filePath, results) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');

      for (let lineNum = 0; lineNum < lines.length; lineNum++) {
        const line = lines[lineNum];

        for (const pattern of this.mockPatterns) {
          pattern.lastIndex = 0; // Reset regex state
          const match = pattern.exec(line);

          if (match) {
            results.violationsFound++;
            results.violations.push({
              file: filePath,
              line: lineNum + 1,
              violation: match[0],
              context: line.trim(),
              severity: this._getSeverity(match[0]),
              type: this._getViolationType(match[0])
            });
          }
        }
      }
    } catch (error) {
      console.error(`Error scanning file ${filePath}:`, error.message);
    }
  }

  _getSeverity(violation) {
    if (/mock|fake|stub|dummy/i.test(violation)) return 'CRITICAL';
    if (/TODO|FIXME/i.test(violation)) return 'HIGH';
    if (/test@|example\.com|localhost/i.test(violation)) return 'MEDIUM';
    return 'LOW';
  }

  _getViolationType(violation) {
    if (/mock|fake|stub|dummy/i.test(violation)) return 'MOCK_IMPLEMENTATION';
    if (/TODO|FIXME/i.test(violation)) return 'INCOMPLETE_IMPLEMENTATION';
    if (/test@|example\.com|localhost/i.test(violation)) return 'HARDCODED_TEST_DATA';
    if (/console\.log|setTimeout.*mock/i.test(violation)) return 'PERFORMANCE_THEATER';
    return 'OTHER';
  }

  generateReport(scanResults) {
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        totalFiles: scanResults.totalFiles,
        violationsFound: scanResults.violationsFound,
        theaterScore: scanResults.theaterScore,
        status: scanResults.theaterScore >= 60 ? 'PASS' : 'FAIL'
      },
      violations: scanResults.violations,
      recommendations: this._generateRecommendations(scanResults)
    };

    return report;
  }

  _generateRecommendations(scanResults) {
    const recommendations = [];

    const mockViolations = scanResults.violations.filter(v => v.type === 'MOCK_IMPLEMENTATION');
    if (mockViolations.length > 0) {
      recommendations.push({
        priority: 'CRITICAL',
        action: 'Replace all mock implementations with real implementations',
        count: mockViolations.length,
        files: [...new Set(mockViolations.map(v => v.file))]
      });
    }

    const incompleteViolations = scanResults.violations.filter(v => v.type === 'INCOMPLETE_IMPLEMENTATION');
    if (incompleteViolations.length > 0) {
      recommendations.push({
        priority: 'HIGH',
        action: 'Complete all TODO and FIXME implementations',
        count: incompleteViolations.length,
        files: [...new Set(incompleteViolations.map(v => v.file))]
      });
    }

    const testDataViolations = scanResults.violations.filter(v => v.type === 'HARDCODED_TEST_DATA');
    if (testDataViolations.length > 0) {
      recommendations.push({
        priority: 'MEDIUM',
        action: 'Replace hardcoded test data with production-appropriate values',
        count: testDataViolations.length,
        files: [...new Set(testDataViolations.map(v => v.file))]
      });
    }

    return recommendations;
  }
}

module.exports = TheaterDetector;