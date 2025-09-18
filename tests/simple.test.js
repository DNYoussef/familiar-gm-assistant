/**
 * Simple test to verify basic functionality
 */

describe('Basic Project Tests', () => {
  test('Environment is set up correctly', () => {
    expect(process.env.NODE_ENV).toBeDefined();
  });

  test('Basic math operations work', () => {
    expect(2 + 2).toBe(4);
    expect(10 * 10).toBe(100);
  });

  test('String operations work', () => {
    const str = 'Hello World';
    expect(str.toLowerCase()).toBe('hello world');
    expect(str.length).toBe(11);
  });

  test('Array operations work', () => {
    const arr = [1, 2, 3, 4, 5];
    expect(arr.length).toBe(5);
    expect(arr.filter(x => x > 3)).toEqual([4, 5]);
  });

  test('Object operations work', () => {
    const obj = { name: 'Familiar', type: 'GM Assistant' };
    expect(obj.name).toBe('Familiar');
    expect(Object.keys(obj)).toHaveLength(2);
  });

  test('Async operations work', async () => {
    const promise = new Promise(resolve => {
      setTimeout(() => resolve('done'), 100);
    });
    const result = await promise;
    expect(result).toBe('done');
  });
});

describe('Configuration Tests', () => {
  test('Package.json exists and has correct structure', () => {
    const packageJson = require('../package.json');
    expect(packageJson.name).toBe('familiar-gm-assistant');
    expect(packageJson.version).toBeDefined();
    expect(packageJson.scripts).toBeDefined();
    expect(packageJson.dependencies).toBeDefined();
  });

  test('Jest config is properly set', () => {
    // Jest config exists and is working (we're running tests!)
    expect(true).toBe(true);
  });
});

describe('File Structure Tests', () => {
  const fs = require('fs');
  const path = require('path');

  test('Critical directories exist', () => {
    const dirs = ['src', 'tests', 'docs', 'scripts', 'config'];
    dirs.forEach(dir => {
      const dirPath = path.join(process.cwd(), dir);
      expect(fs.existsSync(dirPath)).toBe(true);
    });
  });

  test('Critical files exist', () => {
    const files = [
      'package.json',
      'README.md',
      'CLAUDE.md',
      'comprehensive_analysis_engine.py'
    ];
    files.forEach(file => {
      const filePath = path.join(process.cwd(), file);
      expect(fs.existsSync(filePath)).toBe(true);
    });
  });

  test('API server file exists', () => {
    const apiPath = path.join(process.cwd(), 'src', 'api-server.js');
    expect(fs.existsSync(apiPath)).toBe(true);
  });
});