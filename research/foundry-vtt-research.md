# Foundry VTT Module Development Research

## Executive Summary

This research document provides comprehensive insights into Foundry VTT module development for the Familiar project, focusing on module architecture, UI overlay techniques, WebGL compatibility, API hooks, and best practices for PF2e integration. The findings will inform the development of a non-blocking familiar companion interface.

## 1. Module Architecture and Manifest Requirements

### 1.1 Core Module Structure

Foundry VTT modules require a specific architecture with the following key components:

**Module Identifier Requirements:**
- All lowercase string with no special characters
- Use hyphens (not underscores) to separate terms
- Must match exactly with module directory name
- Example: `familiar-companion` (NOT `familiar_companion`)

**Essential Manifest Fields (module.json):**
```json
{
  "id": "familiar-companion",
  "title": "Familiar Companion",
  "description": "Intelligent familiar companion interface for PF2e",
  "version": "1.0.0",
  "compatibility": {
    "minimum": "12",
    "verified": "13",
    "maximum": "13"
  },
  "esmodules": ["scripts/main.js"],
  "url": "https://github.com/username/familiar-companion",
  "manifest": "https://github.com/username/familiar-companion/releases/latest/download/module.json",
  "download": "https://github.com/username/familiar-companion/releases/latest/download/familiar-companion.zip",
  "license": "MIT",
  "authors": [{"name": "Author Name"}],
  "system": ["pf2e"]
}
```

**Critical Manifest Guidelines:**
- Use semantic versioning (string format, not float)
- Stable URLs for manifest and download locations
- GitHub releases/latest/download pattern recommended
- License field required for distribution
- System dependency array for PF2e integration

### 1.2 Module Structure and Organization

```
familiar-companion/
├── module.json           # Module manifest
├── scripts/
│   ├── main.js          # Entry point (ES6 module)
│   ├── familiar-ui.js   # UI overlay components
│   └── pf2e-integration.js
├── styles/
│   └── familiar.css     # Custom styling
├── templates/
│   └── familiar-panel.hbs
├── lang/
│   └── en.json          # Localization
└── assets/
    └── icons/
```

### 1.3 Data Management and Flags

**Document Flags for Data Persistence:**
```javascript
// Store familiar data in actor flags
await actor.update({
  "flags.familiar-companion.familiarData": {
    name: "Whiskers",
    type: "cat",
    personality: "curious",
    preferences: ["fish", "warm spots"]
  }
});

// Retrieve data
const familiarData = actor.getFlag("familiar-companion", "familiarData");
```

## 2. Canvas Integration and UI Overlay Techniques

### 2.1 Non-Blocking UI Integration Approaches

**HeadsUpDisplay (HUD) Integration:**
The primary method for creating non-blocking overlays that don't interfere with gameplay:

```javascript
// Add UI element to HUD overlay
class FamiliarHUD extends Application {
  static get defaultOptions() {
    return mergeObject(super.defaultOptions, {
      id: "familiar-hud",
      template: "modules/familiar-companion/templates/familiar-panel.hbs",
      popOut: false,
      minimizable: false,
      resizable: false,
      classes: ["familiar-hud"],
      left: 10,
      top: 100,
      width: 300,
      height: 200
    });
  }
}
```

### 2.2 Canvas Layer Integration

**Custom Layer Creation:**
```javascript
// Register custom layer in init hook
Hooks.once("init", () => {
  CONFIG.Canvas.layers.familiar = {
    layerClass: FamiliarLayer,
    group: "interface"
  };
});

class FamiliarLayer extends CanvasLayer {
  constructor() {
    super();
    this.familiar = null;
  }

  async draw() {
    super.draw();
    // Draw familiar UI elements without blocking gameplay
  }
}
```

### 2.3 Grid Highlighting for Familiar Actions

**Non-Blocking Grid Interactions:**
```javascript
// Highlight grid spaces for familiar movement/actions
function highlightFamiliarRange(familiar, range) {
  const highlightId = `familiar-${familiar.id}`;
  canvas.interface.grid.addHighlightLayer(highlightId);

  const config = {
    x: familiar.x,
    y: familiar.y,
    color: 0x00ff00,
    border: null,
    alpha: 0.25,
    shape: null
  };

  canvas.interface.grid.highlightPosition(highlightId, config);
}

// Clear highlights when action complete
function clearFamiliarHighlights(familiarId) {
  canvas.interface.grid.clearHighlightLayer(`familiar-${familiarId}`);
}
```

## 3. WebGL Compatibility Considerations

### 3.1 PixiJS Integration and Performance

**Current PixiJS Status:**
- Foundry VTT uses PixiJS for WebGL-powered canvas rendering
- PixiJS v8 migration postponed due to compatibility challenges
- WebGL 2.0 requirement for optimal performance
- Hardware acceleration must be enabled

**WebGL Requirements:**
- Dedicated GPU supporting WebGL 2.0
- Hardware acceleration enabled in browser
- Minimum 8GB RAM recommended
- Chrome/Chromium-based browser up-to-date

### 3.2 Performance Optimization Strategies

**Memory Management:**
```javascript
// Optimize texture usage for familiar sprites
class FamiliarSprite extends PIXI.Sprite {
  constructor(texture) {
    super(texture);
    // Use texture caching to reduce memory overhead
    this.texture.baseTexture.scaleMode = PIXI.SCALE_MODES.LINEAR;
  }

  destroy() {
    // Proper cleanup to prevent memory leaks
    super.destroy({ children: true, texture: false, baseTexture: false });
  }
}
```

**Rendering Optimization:**
- Batch draw calls when rendering multiple familiar elements
- Use object pooling for frequently created/destroyed objects
- Implement level-of-detail (LOD) for distant familiar sprites
- Cache static UI elements as textures

### 3.3 Canvas API Stability Considerations

**Important Notes:**
- Canvas API has major breaking changes between versions
- Not very stable - requires adaptation each version
- Most module developers don't need deep canvas manipulation
- Use established layers and HUD system when possible

## 4. API Hooks and Event System

### 4.1 Core Hook Architecture

**Essential Hook Types:**
```javascript
// Lifecycle hooks
Hooks.once("init", () => {
  // Initialize module configuration
  CONFIG.familiarCompanion = {
    debug: false,
    autoShow: true
  };
});

Hooks.once("ready", () => {
  // Module fully loaded and ready
  game.familiarCompanion = new FamiliarManager();
});

// Document lifecycle hooks
Hooks.on("createActor", (actor, data, options, userId) => {
  if (actor.type === "familiar") {
    // Handle familiar creation
    game.familiarCompanion.onFamiliarCreated(actor);
  }
});

Hooks.on("updateActor", (actor, changes, options, userId) => {
  if (actor.type === "familiar") {
    // Handle familiar updates
    game.familiarCompanion.onFamiliarUpdated(actor, changes);
  }
});
```

### 4.2 Custom Hook Implementation

**Creating Module-Specific Hooks:**
```javascript
// Custom hook for familiar AI actions
class FamiliarManager {
  async performAction(familiar, action) {
    // Call hook before action
    const allowed = Hooks.call("familiarPreAction", familiar, action);
    if (allowed === false) return;

    // Perform action
    const result = await this.executeAction(familiar, action);

    // Call hook after action
    Hooks.callAll("familiarAction", familiar, action, result);

    return result;
  }
}

// Other modules can register for these hooks
Hooks.on("familiarAction", (familiar, action, result) => {
  console.log(`Familiar ${familiar.name} performed ${action.type}`);
});
```

### 4.3 PF2e System Integration Hooks

**PF2e-Specific Events:**
```javascript
// Listen for PF2e system hooks
Hooks.on("pf2e.rollInitiative", (combat, initiative) => {
  // Handle familiar initiative in combat
  game.familiarCompanion.handleCombatTurn(combat);
});

Hooks.on("pf2e.rollDamage", (actor, damage) => {
  // React to damage for familiar AI
  if (actor.type === "familiar") {
    game.familiarCompanion.handleDamage(actor, damage);
  }
});
```

## 5. Best Practices from Successful Modules

### 5.1 Module Development Best Practices

**Code Organization:**
```javascript
// Namespace your code properly
window.FamiliarCompanion = {
  MODULE_NAME: "familiar-companion",
  MODULE_TITLE: "Familiar Companion",

  log: (message, level = "info") => {
    console[level](`${FamiliarCompanion.MODULE_TITLE} | ${message}`);
  },

  getSetting: (key) => {
    return game.settings.get(FamiliarCompanion.MODULE_NAME, key);
  }
};

// Use libWrapper for core function patches
Hooks.once("init", () => {
  if (typeof libWrapper === "function") {
    libWrapper.register(
      "familiar-companion",
      "Actor.prototype.rollInitiative",
      function(wrapped, ...args) {
        // Custom familiar initiative logic
        if (this.type === "familiar") {
          return this.rollFamiliarInitiative(...args);
        }
        return wrapped(...args);
      },
      "WRAPPER"
    );
  }
});
```

### 5.2 PF2e Integration Best Practices

**PF2e Content Module Structure:**
```javascript
// Register familiar types with PF2e system
Hooks.once("init", () => {
  CONFIG.PF2E.familiarTypes = {
    ...CONFIG.PF2E.familiarTypes,
    "ai-companion": {
      label: "AI Companion",
      intelligence: "high",
      communication: "telepathic"
    }
  };
});

// Integrate with PF2e action system
class FamiliarAction extends game.pf2e.actions.BaseAction {
  constructor() {
    super({
      actionType: "familiar",
      actionCost: 1,
      traits: ["familiar", "mental"]
    });
  }
}
```

### 5.3 Performance Optimization

**Memory and Performance Guidelines:**
- Keep module size under 5MB for optimal loading
- Use WebP format for images when possible
- Implement lazy loading for non-essential features
- Cache frequently accessed data in memory
- Use requestAnimationFrame for smooth animations

**Asset Optimization:**
```javascript
// Preload essential assets
Hooks.once("ready", async () => {
  const assets = [
    "modules/familiar-companion/assets/familiar-icon.webp",
    "modules/familiar-companion/assets/ui-background.webp"
  ];

  await Promise.all(assets.map(path =>
    new Promise(resolve => {
      const img = new Image();
      img.onload = resolve;
      img.src = path;
    })
  ));
});
```

### 5.4 User Experience Best Practices

**Settings Management:**
```javascript
// Register module settings
Hooks.once("init", () => {
  game.settings.register("familiar-companion", "showUI", {
    name: "Show Familiar UI",
    hint: "Display the familiar companion interface",
    scope: "client",
    config: true,
    type: Boolean,
    default: true,
    onChange: value => {
      if (value) {
        game.familiarCompanion.showUI();
      } else {
        game.familiarCompanion.hideUI();
      }
    }
  });

  game.settings.register("familiar-companion", "aiPersonality", {
    name: "AI Personality",
    hint: "Choose the personality type for your familiar",
    scope: "world",
    config: true,
    type: String,
    choices: {
      "curious": "Curious",
      "protective": "Protective",
      "playful": "Playful",
      "wise": "Wise"
    },
    default: "curious"
  });
});
```

## 6. Implementation Recommendations for Familiar Project

### 6.1 Architecture Recommendations

1. **Use HUD Overlay Approach**: Implement the familiar interface as a HUD overlay to avoid blocking gameplay
2. **Modular Design**: Separate AI logic, UI rendering, and PF2e integration into distinct modules
3. **Event-Driven Architecture**: Use hooks extensively for loose coupling between components
4. **Performance-First**: Implement lazy loading and caching for optimal performance

### 6.2 UI Integration Strategy

```javascript
// Recommended familiar UI implementation
class FamiliarCompanionUI extends Application {
  constructor(familiar) {
    super();
    this.familiar = familiar;
    this.isMinimized = false;
  }

  static get defaultOptions() {
    return mergeObject(super.defaultOptions, {
      id: "familiar-companion-ui",
      classes: ["familiar-companion"],
      template: "modules/familiar-companion/templates/familiar-ui.hbs",
      width: 320,
      height: 240,
      left: 20,
      top: 80,
      resizable: true,
      minimizable: true,
      title: "Familiar Companion"
    });
  }

  activateListeners(html) {
    super.activateListeners(html);

    // Non-blocking interaction handlers
    html.find('.familiar-action').click(this._onFamiliarAction.bind(this));
    html.find('.familiar-chat').click(this._onFamiliarChat.bind(this));
  }

  async _onFamiliarAction(event) {
    const action = event.currentTarget.dataset.action;
    await game.familiarCompanion.performAction(this.familiar, action);
  }
}
```

### 6.3 PF2e Integration Approach

1. **System Compatibility**: Target PF2e system specifically with compatibility flags
2. **Action Integration**: Integrate with PF2e's action economy system
3. **Character Sheet Integration**: Add familiar companion panel to character sheets
4. **Combat Integration**: Handle familiar actions during combat turns

### 6.4 Technical Specifications

**Minimum Technical Requirements:**
- Foundry VTT v12+ compatibility
- PF2e system dependency
- WebGL 2.0 support required
- Modern browser with hardware acceleration

**Recommended Development Stack:**
- ES6 modules for modern JavaScript
- Handlebars templates for UI
- CSS Grid/Flexbox for responsive layout
- libWrapper for core function patches
- Socket.io for multiplayer synchronization

## 7. Key Research Citations

This research synthesized information from the following primary sources:

1. **Official Foundry VTT Documentation**
   - Module Development Guide (https://foundryvtt.com/article/module-development/)
   - Canvas Layers Architecture (https://foundryvtt.com/article/canvas-layers/)
   - API Documentation (https://foundryvtt.com/api/)

2. **Community Resources**
   - Foundry VTT Community Wiki Development Section
   - Package Development Best Practices Checklist
   - PIXI Integration Guidelines

3. **Performance Optimization Research**
   - Hardware Requirements and Optimization Guide
   - WebGL Compatibility Studies
   - Module Performance Analysis

4. **PF2e System Integration**
   - PF2e Content Module Creation Guide
   - System-Specific Hook Documentation
   - Best Practices from Successful PF2e Modules

This research provides the foundation for developing a sophisticated, non-blocking familiar companion interface that integrates seamlessly with Foundry VTT and the PF2e system while maintaining optimal performance and user experience.