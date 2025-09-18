/**
 * Familiar GM Assistant - Core Architecture Design
 * Phase 2: Core Architecture (Development Princess + Coordination Princess)
 * CRITICAL PATH: Blocks all Group B parallel execution
 */

/**
 * Main Familiar Application using Foundry ApplicationV2 framework
 * Manages the raven familiar UI and coordinates all system components
 */
export class FamiliarApplication extends Application {
  static DEFAULT_OPTIONS = {
    id: "familiar-gm-assistant",
    classes: ["familiar-app"],
    tag: "form",
    window: {
      positioned: true,
      minimizable: true,
      resizable: false,
      title: "familiar.title"
    },
    position: {
      width: 340,
      height: 450,
      top: 100,
      left: window.innerWidth - 380  // Bottom-right positioning
    },
    form: {
      handler: FamiliarApplication.#onSubmitForm,
      submitOnChange: false,
      closeOnSubmit: false
    }
  };

  constructor(options = {}) {
    super(options);

    // Initialize core system components
    this.ragSystem = new FamiliarRAGSystem();
    this.monsterGenerator = new PF2eMonsterGenerator();
    this.artGenerator = new CreatureArtGenerator();
    this.chatProcessor = new ChatCommandProcessor();
    this.sessionManager = new SessionManager();

    // UI state management
    this.isMinimized = false;
    this.activeTab = 'chat';
    this.conversationHistory = [];

    // Performance tracking
    this.performanceTracker = new PerformanceTracker();
  }

  /**
   * Prepare rendering context for the familiar UI
   */
  async _prepareContext(options) {
    const context = await super._prepareContext(options);

    return foundry.utils.mergeObject(context, {
      // UI state
      isMinimized: this.isMinimized,
      activeTab: this.activeTab,

      // System status
      ragStatus: await this.ragSystem.getStatus(),
      sessionInfo: this.sessionManager.getCurrentSession(),

      // User preferences
      userSettings: this.getUserSettings(),

      // Recent conversation
      recentMessages: this.conversationHistory.slice(-5),

      // Performance metrics
      responseTime: this.performanceTracker.getAverageResponseTime(),
      sessionCost: this.sessionManager.getSessionCost()
    });
  }

  /**
   * Define the HTML template for the familiar interface
   */
  static PARTS = {
    form: {
      template: "modules/familiar-gm-assistant/templates/familiar-app.hbs"
    }
  };

  /**
   * Handle form submission (chat commands and interactions)
   */
  static async #onSubmitForm(event, form, formData) {
    const app = this;
    const command = formData.object.command;

    if (!command) return;

    try {
      // Track performance
      const startTime = performance.now();

      // Process command through chat processor
      const response = await app.chatProcessor.processCommand(command, {
        user: game.user.id,
        character: game.user.character,
        scene: canvas.scene?.id
      });

      // Update conversation history
      app.addToConversation('user', command);
      app.addToConversation('familiar', response.content);

      // Display response in UI
      await app.displayResponse(response);

      // Track response time
      const responseTime = performance.now() - startTime;
      app.performanceTracker.recordResponse(responseTime);
      app.sessionManager.recordAPIUsage(response.cost);

    } catch (error) {
      console.error('Familiar command processing error:', error);
      await app.displayError('Sorry, I encountered an error processing your request.');
    }

    // Clear input
    form.querySelector('input[name="command"]').value = '';
  }

  /**
   * Add message to conversation history
   */
  addToConversation(role, content) {
    this.conversationHistory.push({
      role,
      content,
      timestamp: new Date().toISOString()
    });

    // Keep only last 20 messages for memory management
    if (this.conversationHistory.length > 20) {
      this.conversationHistory = this.conversationHistory.slice(-20);
    }
  }

  /**
   * Display response in the familiar UI
   */
  async displayResponse(response) {
    const messageContainer = this.element.querySelector('.familiar-messages');

    const messageElement = document.createElement('div');
    messageElement.classList.add('familiar-message', 'familiar-response');

    // Handle different response types
    switch (response.type) {
      case 'rule':
        messageElement.innerHTML = this.formatRuleResponse(response);
        break;
      case 'monster':
        messageElement.innerHTML = await this.formatMonsterResponse(response);
        break;
      case 'encounter':
        messageElement.innerHTML = this.formatEncounterResponse(response);
        break;
      case 'art':
        messageElement.innerHTML = await this.formatArtResponse(response);
        break;
      default:
        messageElement.innerHTML = this.formatTextResponse(response);
    }

    messageContainer.appendChild(messageElement);
    messageContainer.scrollTop = messageContainer.scrollHeight;

    // Update UI to show new message
    this.render();
  }

  /**
   * Format rule lookup responses
   */
  formatRuleResponse(response) {
    const { content, citations, relatedRules } = response;

    let html = `<div class="rule-response">
      <h4>${content.title}</h4>
      <p>${content.description}</p>`;

    if (content.mechanics) {
      html += `<div class="rule-mechanics">
        <h5>Mechanics:</h5>
        <ul>${content.mechanics.map(m => `<li>${m}</li>`).join('')}</ul>
      </div>`;
    }

    if (citations && citations.length > 0) {
      html += `<div class="citations">
        <small><strong>Source:</strong> ${citations.join(', ')}</small>
      </div>`;
    }

    if (relatedRules && relatedRules.length > 0) {
      html += `<div class="related-rules">
        <h6>Related Rules:</h6>
        <ul>${relatedRules.map(r =>
          `<li><a href="#" class="rule-link" data-rule="${r.id}">${r.title}</a></li>`
        ).join('')}</ul>
      </div>`;
    }

    return html + '</div>';
  }

  /**
   * Format monster generation responses
   */
  async formatMonsterResponse(response) {
    const { monster, artwork } = response;

    let html = `<div class="monster-response">
      <div class="monster-header">
        <h4>${monster.name}</h4>
        <span class="monster-level">Level ${monster.level}</span>
      </div>`;

    // Add artwork if available
    if (artwork) {
      html += `<div class="monster-artwork">
        <img src="${artwork.url}" alt="${monster.name}"
             style="max-width: 200px; border-radius: 8px;">
      </div>`;
    }

    // Basic stats
    html += `<div class="monster-stats">
      <div class="stat-block">
        <strong>AC:</strong> ${monster.ac} |
        <strong>HP:</strong> ${monster.hp} |
        <strong>Speed:</strong> ${this.formatSpeed(monster.speeds)}
      </div>

      <div class="abilities">
        <strong>Abilities:</strong>
        STR ${monster.abilities.strength} |
        DEX ${monster.abilities.dexterity} |
        CON ${monster.abilities.constitution} |
        INT ${monster.abilities.intelligence} |
        WIS ${monster.abilities.wisdom} |
        CHA ${monster.abilities.charisma}
      </div>`;

    // Traits
    if (monster.traits && monster.traits.length > 0) {
      html += `<div class="monster-traits">
        <strong>Traits:</strong> ${monster.traits.join(', ')}
      </div>`;
    }

    // Actions
    if (monster.actions && monster.actions.length > 0) {
      html += `<div class="monster-actions">
        <h6>Actions:</h6>
        <ul>`;

      monster.actions.forEach(action => {
        html += `<li><strong>${action.name}</strong> ${action.description}</li>`;
      });

      html += `</ul></div>`;
    }

    // Export to Foundry button
    html += `<div class="monster-actions">
      <button type="button" class="export-monster" data-monster="${encodeURIComponent(JSON.stringify(monster))}">
        Export to Foundry
      </button>
    </div>`;

    return html + '</div>';
  }

  /**
   * Format encounter building responses
   */
  formatEncounterResponse(response) {
    const { encounter, difficulty, xpBudget } = response;

    let html = `<div class="encounter-response">
      <div class="encounter-header">
        <h4>Generated Encounter</h4>
        <span class="difficulty ${difficulty.toLowerCase()}">${difficulty} Difficulty</span>
      </div>

      <div class="encounter-budget">
        <strong>XP Budget:</strong> ${xpBudget.used}/${xpBudget.total}
        (${Math.round((xpBudget.used/xpBudget.total)*100)}%)
      </div>

      <div class="encounter-creatures">
        <h6>Creatures:</h6>
        <ul>`;

    encounter.creatures.forEach(creature => {
      html += `<li>
        <strong>${creature.quantity}x ${creature.name}</strong>
        (Level ${creature.level}, ${creature.xp} XP each)
        <button type="button" class="add-to-scene" data-creature="${creature.id}">
          Add to Scene
        </button>
      </li>`;
    });

    html += `</ul></div>

      <div class="encounter-tactics">
        <h6>Tactical Notes:</h6>
        <p>${encounter.tactics}</p>
      </div>
    </div>`;

    return html;
  }

  /**
   * Format art generation responses
   */
  async formatArtResponse(response) {
    const { image, description, processing } = response;

    if (processing) {
      return `<div class="art-response processing">
        <div class="loading-spinner"></div>
        <p>Generating artwork... This may take a moment.</p>
      </div>`;
    }

    return `<div class="art-response">
      <div class="generated-art">
        <img src="${image.url}" alt="${description.title}"
             style="max-width: 300px; border-radius: 8px;">
      </div>

      <div class="art-description">
        <h6>${description.title}</h6>
        <p>${description.detailed}</p>
      </div>

      <div class="art-actions">
        <button type="button" class="save-image" data-url="${image.url}">
          Save to Assets
        </button>
        <button type="button" class="regenerate-art" data-description="${encodeURIComponent(description.prompt)}">
          Regenerate
        </button>
      </div>
    </div>`;
  }

  /**
   * Handle UI events and interactions
   */
  activateListeners(html) {
    super.activateListeners(html);

    // Chat command submission
    html.find('form').on('submit', this._onSubmitForm.bind(this));

    // Minimize/maximize toggle
    html.find('.familiar-toggle').on('click', this._onToggleMinimize.bind(this));

    // Tab switching
    html.find('.tab-button').on('click', this._onTabSwitch.bind(this));

    // Monster export
    html.find('.export-monster').on('click', this._onExportMonster.bind(this));

    // Creature addition to scene
    html.find('.add-to-scene').on('click', this._onAddToScene.bind(this));

    // Art actions
    html.find('.save-image').on('click', this._onSaveImage.bind(this));
    html.find('.regenerate-art').on('click', this._onRegenerateArt.bind(this));

    // Rule links
    html.find('.rule-link').on('click', this._onRuleLink.bind(this));

    // Settings button
    html.find('.familiar-settings').on('click', this._onSettings.bind(this));

    // Help system
    html.find('.familiar-help').on('click', this._onHelp.bind(this));
  }

  /**
   * Toggle minimize state
   */
  async _onToggleMinimize(event) {
    this.isMinimized = !this.isMinimized;

    if (this.isMinimized) {
      this.element.classList.add('minimized');
      this.setPosition({ height: 60 });
    } else {
      this.element.classList.remove('minimized');
      this.setPosition({ height: 450 });
    }

    await this.render();
  }

  /**
   * Switch active tab
   */
  async _onTabSwitch(event) {
    const tab = event.currentTarget.dataset.tab;
    this.activeTab = tab;
    await this.render();
  }

  /**
   * Export monster to Foundry
   */
  async _onExportMonster(event) {
    const monsterData = JSON.parse(decodeURIComponent(event.currentTarget.dataset.monster));

    try {
      const actor = await this.createFoundryActor(monsterData);
      ui.notifications.info(`${monsterData.name} has been created as an actor.`);
    } catch (error) {
      console.error('Failed to export monster:', error);
      ui.notifications.error('Failed to export monster to Foundry.');
    }
  }

  /**
   * Add creature to current scene
   */
  async _onAddToScene(event) {
    const creatureId = event.currentTarget.dataset.creature;

    try {
      // Implementation depends on Foundry scene management
      const token = await this.addTokenToScene(creatureId);
      ui.notifications.info(`Creature added to scene.`);
    } catch (error) {
      console.error('Failed to add to scene:', error);
      ui.notifications.error('Failed to add creature to scene.');
    }
  }

  /**
   * Initialize familiar system on Foundry ready
   */
  static initialize() {
    // Register Handlebars helpers
    this.registerHandlebarsHelpers();

    // Register socket handlers for multiplayer sync
    this.registerSocketHandlers();

    // Register chat commands
    this.registerChatCommands();

    // Initialize performance monitoring
    this.initializePerformanceMonitoring();

    console.log('Familiar GM Assistant initialized successfully');
  }

  /**
   * Register Handlebars template helpers
   */
  static registerHandlebarsHelpers() {
    Handlebars.registerHelper('formatSpeed', function(speeds) {
      if (!speeds) return 'Unknown';

      let speedText = `${speeds.land || 0} feet`;
      if (speeds.fly) speedText += `, fly ${speeds.fly} feet`;
      if (speeds.swim) speedText += `, swim ${speeds.swim} feet`;
      if (speeds.climb) speedText += `, climb ${speeds.climb} feet`;

      return speedText;
    });

    Handlebars.registerHelper('abilityModifier', function(score) {
      return Math.floor((score - 10) / 2);
    });

    Handlebars.registerHelper('formatCost', function(cost) {
      return `$${cost.toFixed(4)}`;
    });
  }

  /**
   * Register socket handlers for multiplayer synchronization
   */
  static registerSocketHandlers() {
    game.socket.on('module.familiar-gm-assistant', (data) => {
      switch (data.type) {
        case 'encounter-shared':
          this.handleSharedEncounter(data.encounter);
          break;
        case 'monster-created':
          this.handleMonsterCreated(data.monster);
          break;
      }
    });
  }

  /**
   * Register chat commands
   */
  static registerChatCommands() {
    const commands = [
      'familiar', 'fam', 'rules', 'monster', 'encounter', 'art', 'help'
    ];

    commands.forEach(cmd => {
      game.messages.registerCommand({
        command: cmd,
        callback: (args) => this.handleChatCommand(cmd, args)
      });
    });
  }

  /**
   * Get user settings for the familiar
   */
  getUserSettings() {
    return {
      apiKeys: {
        openai: !!game.settings.get('familiar-gm-assistant', 'openai-key'),
        anthropic: !!game.settings.get('familiar-gm-assistant', 'anthropic-key'),
        gemini: !!game.settings.get('familiar-gm-assistant', 'gemini-key')
      },
      preferences: {
        defaultModel: game.settings.get('familiar-gm-assistant', 'default-model'),
        artStyle: game.settings.get('familiar-gm-assistant', 'art-style'),
        responseLength: game.settings.get('familiar-gm-assistant', 'response-length'),
        showCosts: game.settings.get('familiar-gm-assistant', 'show-costs')
      }
    };
  }

  /**
   * Create Foundry actor from monster data
   */
  async createFoundryActor(monsterData) {
    const actorData = {
      name: monsterData.name,
      type: 'npc',
      system: {
        details: {
          level: { value: monsterData.level },
          creature: { value: monsterData.creature_type || 'creature' }
        },
        attributes: {
          hp: { value: monsterData.hp, max: monsterData.hp },
          ac: { value: monsterData.ac }
        },
        abilities: monsterData.abilities,
        saves: monsterData.saves,
        traits: {
          value: monsterData.traits || []
        }
      }
    };

    return await Actor.create(actorData);
  }

  /**
   * Performance tracking and monitoring
   */
  static initializePerformanceMonitoring() {
    // Track module load time
    const loadStart = performance.now();

    Hooks.once('ready', () => {
      const loadTime = performance.now() - loadStart;
      console.log(`Familiar GM Assistant loaded in ${loadTime.toFixed(2)}ms`);
    });
  }
}

// Utility helper functions
export const FamiliarUtils = {
  /**
   * Format response time for display
   */
  formatResponseTime(ms) {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  },

  /**
   * Calculate encounter difficulty
   */
  calculateEncounterDifficulty(creatures, partyLevel, partySize = 4) {
    const totalXP = creatures.reduce((sum, creature) => sum + creature.xp, 0);
    const xpThresholds = {
      trivial: partyLevel * partySize * 10,
      low: partyLevel * partySize * 15,
      moderate: partyLevel * partySize * 20,
      severe: partyLevel * partySize * 30,
      extreme: partyLevel * partySize * 40
    };

    if (totalXP <= xpThresholds.trivial) return 'Trivial';
    if (totalXP <= xpThresholds.low) return 'Low';
    if (totalXP <= xpThresholds.moderate) return 'Moderate';
    if (totalXP <= xpThresholds.severe) return 'Severe';
    return 'Extreme';
  },

  /**
   * Validate PF2e stat block completeness
   */
  validateStatBlock(monster) {
    const required = ['name', 'level', 'ac', 'hp', 'abilities', 'saves'];
    const missing = required.filter(field => !monster[field]);

    return {
      isValid: missing.length === 0,
      missing: missing,
      completeness: (required.length - missing.length) / required.length
    };
  }
};

export default FamiliarApplication;