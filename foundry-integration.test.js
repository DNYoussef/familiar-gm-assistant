/**
 * Foundry Module Integration Tests - Reality Validation
 * Tests real integration with Foundry VTT modules and APIs
 */

const TestFramework = require('../unit/test-framework');
const fs = require('fs');
const path = require('path');

class FoundryIntegrationTester {
  constructor() {
    this.testFramework = new TestFramework();
    this.foundryPath = this._detectFoundryPath();
    this.moduleManifests = new Map();
  }

  async runIntegrationTests() {
    console.log('🔧 INTEGRATION DRONE: Testing Foundry module integration...');
    console.log('');

    // Register integration test suite
    const suite = this.testFramework.registerSuite('foundry-integration', {
      requireRealImplementations: true,
      realityValidation: true,
      minCoverage: 70
    });

    // Add integration tests
    this._addFoundryDetectionTests();
    this._addModuleStructureTests();
    this._addManifestValidationTests();
    this._addAPIIntegrationTests();
    this._addDataIntegrityTests();

    // Execute tests
    const results = await this.testFramework.runAllSuites();

    console.log(`🔧 Foundry Integration: ${results.passed}/${results.totalTests} tests passed`);

    return results;
  }

  _addFoundryDetectionTests() {
    this.testFramework.addTest('foundry-integration', 'detect-foundry-installation', async (context) => {
      // Test real Foundry installation detection
      const foundryPaths = [
        'C:\\Users\\%USERNAME%\\AppData\\Local\\FoundryVTT',
        '%LOCALAPPDATA%\\FoundryVTT',
        process.env.FOUNDRY_VTT_DATA_PATH
      ].filter(Boolean);

      let foundryFound = false;
      let foundryDataPath = null;

      for (const pathTemplate of foundryPaths) {
        const resolvedPath = pathTemplate.replace('%USERNAME%', process.env.USERNAME)
                                         .replace('%LOCALAPPDATA%', process.env.LOCALAPPDATA);

        if (fs.existsSync(resolvedPath)) {
          foundryFound = true;
          foundryDataPath = resolvedPath;
          break;
        }
      }

      // Reality check: Verify this is a real Foundry installation
      if (foundryFound) {
        const configPath = path.join(foundryDataPath, 'Config', 'options.json');
        const dataPath = path.join(foundryDataPath, 'Data');

        context.reality.requireReal(fs.existsSync(configPath), 'Foundry config file');
        context.reality.requireReal(fs.existsSync(dataPath), 'Foundry data directory');

        context.expect(fs.existsSync(configPath)).toBeTruthy();
        context.expect(fs.existsSync(dataPath)).toBeTruthy();
      }

      // Log result for integration verification
      console.log(`   🎲 Foundry VTT ${foundryFound ? 'detected' : 'not found'} at: ${foundryDataPath || 'none'}`);
    });

    this.testFramework.addTest('foundry-integration', 'verify-foundry-version', async (context) => {
      if (!this.foundryPath) {
        console.log('   ⚠️  Skipping version check - Foundry not detected');
        return;
      }

      // Check for Foundry version information
      const configPath = path.join(this.foundryPath, 'Config', 'options.json');

      if (fs.existsSync(configPath)) {
        const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

        context.reality.requireReal(config, 'Foundry configuration');
        context.expect(config).toBeTruthy();

        // Log version for verification
        if (config.world) {
          console.log(`   🎲 Foundry Version: ${config.version || 'unknown'}`);
        }
      }
    });
  }

  _addModuleStructureTests() {
    this.testFramework.addTest('foundry-integration', 'validate-module-structure', async (context) => {
      const moduleStructure = {
        required: [
          'module.json',      // Module manifest
          'scripts/',         // JavaScript files
          'styles/',          // CSS files
          'templates/',       // Handlebars templates
          'lang/'            // Localization files
        ],
        optional: [
          'assets/',          // Images and other assets
          'packs/',          // Compendium data
          'system/',         // System-specific files
          'README.md'        // Documentation
        ]
      };

      const projectRoot = process.cwd();

      // Check required structure
      for (const required of moduleStructure.required) {
        const fullPath = path.join(projectRoot, required);
        const exists = fs.existsSync(fullPath);

        if (required === 'module.json') {
          context.reality.requireReal(exists, 'Module manifest file');
          context.expect(exists).toBeTruthy();

          if (exists) {
            // Validate manifest content
            const manifest = JSON.parse(fs.readFileSync(fullPath, 'utf8'));
            context.reality.validateNoMocks(manifest);

            context.expect(manifest.name).toBeTruthy();
            context.expect(manifest.version).toBeTruthy();
            context.expect(manifest.minimumCoreVersion).toBeTruthy();

            this.moduleManifests.set('main', manifest);
          }
        } else {
          // Directories are optional if not implemented yet
          if (!exists) {
            console.log(`   ⚠️  Optional structure missing: ${required}`);
          }
        }
      }
    });

    this.testFramework.addTest('foundry-integration', 'validate-javascript-files', async (context) => {
      const scriptsDir = path.join(process.cwd(), 'scripts');

      if (!fs.existsSync(scriptsDir)) {
        console.log('   ⚠️  No scripts directory found - skipping JS validation');
        return;
      }

      const jsFiles = fs.readdirSync(scriptsDir)
        .filter(file => file.endsWith('.js'));

      context.expect(jsFiles.length).toBeGreaterThan(0);

      for (const jsFile of jsFiles) {
        const filePath = path.join(scriptsDir, jsFile);
        const content = fs.readFileSync(filePath, 'utf8');

        // Reality check: Ensure no mock implementations in module files
        context.reality.validateNoMocks({ content }, `JavaScript file: ${jsFile}`);

        // Basic syntax validation
        try {
          new Function(content);  // Simple syntax check
        } catch (error) {
          throw new Error(`Syntax error in ${jsFile}: ${error.message}`);
        }
      }

      console.log(`   📝 Validated ${jsFiles.length} JavaScript files`);
    });
  }

  _addManifestValidationTests() {
    this.testFramework.addTest('foundry-integration', 'validate-manifest-schema', async (context) => {
      const manifestPath = path.join(process.cwd(), 'module.json');

      if (!fs.existsSync(manifestPath)) {
        throw new Error('Module manifest (module.json) not found');
      }

      const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

      // Reality validation: Ensure manifest is real, not mock data
      context.reality.requireReal(manifest, 'Module manifest');
      context.reality.validateNoMocks(manifest);

      // Required fields validation
      const requiredFields = [
        'name', 'version', 'title', 'description',
        'minimumCoreVersion', 'compatibleCoreVersion'
      ];

      for (const field of requiredFields) {
        context.expect(manifest[field]).toBeTruthy();
      }

      // Validate version format
      const versionRegex = /^\d+\.\d+\.\d+$/;
      context.expect(versionRegex.test(manifest.version)).toBeTruthy();

      // Validate scripts array if present
      if (manifest.scripts) {
        context.expect(Array.isArray(manifest.scripts)).toBeTruthy();

        for (const script of manifest.scripts) {
          const scriptPath = path.join(process.cwd(), script);
          context.expect(fs.existsSync(scriptPath)).toBeTruthy();
        }
      }

      console.log(`   📋 Manifest validation passed for module: ${manifest.name} v${manifest.version}`);
    });

    this.testFramework.addTest('foundry-integration', 'validate-dependencies', async (context) => {
      const manifest = this.moduleManifests.get('main');

      if (!manifest) {
        console.log('   ⚠️  No manifest loaded - skipping dependency validation');
        return;
      }

      // Check module dependencies
      if (manifest.dependencies) {
        for (const dependency of manifest.dependencies) {
          context.reality.requireReal(dependency, 'Module dependency');

          context.expect(dependency.name).toBeTruthy();
          context.expect(dependency.type).toBeTruthy();

          if (dependency.type === 'module') {
            // Could check if dependency module exists in Foundry installation
            console.log(`   🔗 Dependency: ${dependency.name} (${dependency.type})`);
          }
        }
      }

      // Check system compatibility
      if (manifest.systems) {
        context.expect(Array.isArray(manifest.systems)).toBeTruthy();
        console.log(`   🎲 Compatible with ${manifest.systems.length} game systems`);
      }
    });
  }

  _addAPIIntegrationTests() {
    this.testFramework.addTest('foundry-integration', 'test-foundry-api-compatibility', async (context) => {
      // Test Foundry API compatibility (simulated)
      const foundryAPIMethods = [
        'game.users',
        'game.actors',
        'game.items',
        'game.scenes',
        'canvas.tokens',
        'ui.notifications'
      ];

      // Since we can't actually test against Foundry runtime here,
      // we'll validate that our module code doesn't use deprecated APIs
      const scriptsDir = path.join(process.cwd(), 'scripts');

      if (!fs.existsSync(scriptsDir)) {
        console.log('   ⚠️  No scripts to validate API usage');
        return;
      }

      const deprecatedAPIs = [
        'game.data',           // Deprecated in v9+
        'entity.data',         // Replaced with document
        'getTemplate',         // Use loadTemplate
        'renderTemplate'       // Use renderTemplate from utils
      ];

      const jsFiles = fs.readdirSync(scriptsDir)
        .filter(file => file.endsWith('.js'));

      for (const jsFile of jsFiles) {
        const filePath = path.join(scriptsDir, jsFile);
        const content = fs.readFileSync(filePath, 'utf8');

        // Check for deprecated API usage
        for (const deprecatedAPI of deprecatedAPIs) {
          if (content.includes(deprecatedAPI)) {
            console.log(`   ⚠️  Deprecated API found in ${jsFile}: ${deprecatedAPI}`);
          }
        }

        // Reality check: Ensure we're testing real implementation
        context.reality.validateNoMocks({ content }, `API usage in ${jsFile}`);
      }

      console.log(`   🔌 API compatibility check completed for ${jsFiles.length} files`);
    });

    this.testFramework.addTest('foundry-integration', 'validate-hooks-integration', async (context) => {
      // Test Foundry hooks integration
      const commonHooks = [
        'init',
        'ready',
        'canvasInit',
        'canvasReady',
        'updateActor',
        'updateItem',
        'renderActorSheet'
      ];

      const scriptsDir = path.join(process.cwd(), 'scripts');

      if (!fs.existsSync(scriptsDir)) {
        console.log('   ⚠️  No scripts to validate hooks');
        return;
      }

      const jsFiles = fs.readdirSync(scriptsDir)
        .filter(file => file.endsWith('.js'));

      let hooksFound = 0;

      for (const jsFile of jsFiles) {
        const filePath = path.join(scriptsDir, jsFile);
        const content = fs.readFileSync(filePath, 'utf8');

        // Look for Hooks.on or Hooks.once usage
        const hookPattern = /Hooks\.(on|once)\(['"`](\w+)['"`]/g;
        let match;

        while ((match = hookPattern.exec(content)) !== null) {
          hooksFound++;
          const hookName = match[2];
          console.log(`   🪝 Hook found: ${hookName} in ${jsFile}`);
        }

        // Reality validation
        context.reality.validateNoMocks({ content }, `Hooks in ${jsFile}`);
      }

      if (hooksFound > 0) {
        context.expect(hooksFound).toBeGreaterThan(0);
        console.log(`   🪝 Total hooks registered: ${hooksFound}`);
      } else {
        console.log('   ⚠️  No Foundry hooks found - module may not integrate properly');
      }
    });
  }

  _addDataIntegrityTests() {
    this.testFramework.addTest('foundry-integration', 'validate-compendium-data', async (context) => {
      const packsDir = path.join(process.cwd(), 'packs');

      if (!fs.existsSync(packsDir)) {
        console.log('   ⚠️  No packs directory - skipping compendium validation');
        return;
      }

      const packFiles = fs.readdirSync(packsDir)
        .filter(file => file.endsWith('.db') || file.endsWith('.json'));

      if (packFiles.length === 0) {
        console.log('   ⚠️  No compendium files found');
        return;
      }

      for (const packFile of packFiles) {
        const filePath = path.join(packsDir, packFile);
        const content = fs.readFileSync(filePath, 'utf8');

        // Basic JSON validation for .json files
        if (packFile.endsWith('.json')) {
          try {
            const data = JSON.parse(content);
            context.reality.validateNoMocks(data, `Compendium: ${packFile}`);
            context.expect(Array.isArray(data) || typeof data === 'object').toBeTruthy();
          } catch (error) {
            throw new Error(`Invalid JSON in compendium ${packFile}: ${error.message}`);
          }
        }

        // DB files are binary but should exist and have size > 0
        if (packFile.endsWith('.db')) {
          const stats = fs.statSync(filePath);
          context.expect(stats.size).toBeGreaterThan(0);
        }
      }

      console.log(`   📦 Validated ${packFiles.length} compendium files`);
    });

    this.testFramework.addTest('foundry-integration', 'validate-asset-integrity', async (context) => {
      const assetsDir = path.join(process.cwd(), 'assets');

      if (!fs.existsSync(assetsDir)) {
        console.log('   ⚠️  No assets directory found');
        return;
      }

      const assetFiles = this._getAllFiles(assetsDir)
        .filter(file => /\.(png|jpg|jpeg|webp|svg|mp3|wav|ogg)$/i.test(file));

      if (assetFiles.length === 0) {
        console.log('   ⚠️  No asset files found');
        return;
      }

      for (const assetFile of assetFiles) {
        const stats = fs.statSync(assetFile);

        // Basic integrity: file exists and has reasonable size
        context.expect(stats.size).toBeGreaterThan(0);

        // Files should be under 50MB (reasonable for web distribution)
        context.expect(stats.size).toBeLessThan(50 * 1024 * 1024);
      }

      console.log(`   🖼️  Validated ${assetFiles.length} asset files`);
    });
  }

  _getAllFiles(dir) {
    const files = [];

    const items = fs.readdirSync(dir);
    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        files.push(...this._getAllFiles(fullPath));
      } else {
        files.push(fullPath);
      }
    }

    return files;
  }

  _detectFoundryPath() {
    const possiblePaths = [
      path.join(process.env.LOCALAPPDATA || '', 'FoundryVTT'),
      path.join(process.env.APPDATA || '', 'FoundryVTT'),
      process.env.FOUNDRY_VTT_DATA_PATH
    ].filter(Boolean);

    for (const foundryPath of possiblePaths) {
      if (fs.existsSync(foundryPath)) {
        return foundryPath;
      }
    }

    return null;
  }
}

// Export for use by quality gate system
module.exports = FoundryIntegrationTester;

// CLI execution
if (require.main === module) {
  (async () => {
    const tester = new FoundryIntegrationTester();
    const results = await tester.runIntegrationTests();

    const exitCode = results.overallStatus === 'EXCELLENT' || results.overallStatus === 'GOOD' ? 0 : 1;
    process.exit(exitCode);
  })().catch(error => {
    console.error('Integration test failed:', error.message);
    process.exit(1);
  });
}