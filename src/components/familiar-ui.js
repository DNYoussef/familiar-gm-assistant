/**
 * Foundry UI Module Implementation
 * Phase 3.1: Foundry UI Module (Development Princess)
 * ApplicationV2 framework with raven familiar interface
 */

import { BaseComponent, ErrorHandler, globalCache } from '../base-components.js';

/**
 * Familiar UI Component
 * Manages the raven familiar visual interface and user interactions
 */
export class FamiliarUI extends BaseComponent {
  constructor(config = {}) {
    super('FamiliarUI', config);

    // UI state
    this.isVisible = true;
    this.isMinimized = false;
    this.activeTab = 'chat';
    this.ravenSprite = null;

    // Animation states
    this.animationStates = {
      idle: 'idle',
      thinking: 'thinking',
      speaking: 'speaking',
      working: 'working'
    };
    this.currentAnimation = this.animationStates.idle;

    // Message queue for UI updates
    this.messageQueue = [];
    this.isProcessingMessage = false;
  }

  /**
   * Initialize the familiar UI component
   */
  async initialize() {
    try {
      const startTime = performance.now();

      // Create UI elements
      await this.createFamiliarContainer();
      await this.createRavenSprite();
      await this.createChatInterface();
      await this.setupEventListeners();

      // Position UI in bottom-right corner
      this.positionUI();

      // Load saved UI state
      await this.loadUIState();

      this.isInitialized = true;

      const duration = performance.now() - startTime;
      this.recordOperation('initialize', duration, true);
      this.logger.info('Familiar UI initialized successfully', { duration });

      return true;
    } catch (error) {
      this.logger.error('Failed to initialize Familiar UI', error);
      return false;
    }
  }

  /**
   * Create the main familiar container
   */
  async createFamiliarContainer() {
    const container = document.createElement('div');
    container.id = 'familiar-container';
    container.classList.add('familiar-container');

    container.innerHTML = `
      <div class="familiar-window" id="familiar-window">
        <!-- Familiar Header -->
        <div class="familiar-header">
          <div class="familiar-title">
            <i class="fas fa-crow"></i>
            <span>Familiar</span>
          </div>
          <div class="familiar-controls">
            <button class="familiar-btn" id="familiar-minimize" title="Minimize">
              <i class="fas fa-minus"></i>
            </button>
            <button class="familiar-btn" id="familiar-settings" title="Settings">
              <i class="fas fa-cog"></i>
            </button>
            <button class="familiar-btn" id="familiar-help" title="Help">
              <i class="fas fa-question"></i>
            </button>
          </div>
        </div>

        <!-- Tab Navigation -->
        <div class="familiar-tabs" id="familiar-tabs">
          <button class="tab-btn active" data-tab="chat">
            <i class="fas fa-comments"></i> Chat
          </button>
          <button class="tab-btn" data-tab="monsters">
            <i class="fas fa-dragon"></i> Monsters
          </button>
          <button class="tab-btn" data-tab="encounters">
            <i class="fas fa-sword"></i> Encounters
          </button>
          <button class="tab-btn" data-tab="art">
            <i class="fas fa-palette"></i> Art
          </button>
        </div>

        <!-- Tab Content -->
        <div class="familiar-content">
          <!-- Chat Tab -->
          <div class="tab-content active" id="tab-chat">
            <div class="familiar-messages" id="familiar-messages">
              <div class="familiar-intro">
                <div class="raven-container" id="raven-container">
                  <!-- Raven sprite will be inserted here -->
                </div>
                <p>Greetings, I'm your familiar assistant. Ask me about Pathfinder 2e rules, monsters, encounters, or request artwork!</p>
                <div class="quick-commands">
                  <button class="quick-cmd" data-cmd="/familiar help">Help</button>
                  <button class="quick-cmd" data-cmd="/familiar rules">Rules</button>
                  <button class="quick-cmd" data-cmd="/familiar monster">Monsters</button>
                </div>
              </div>
            </div>

            <div class="familiar-input-area">
              <form id="familiar-chat-form">
                <div class="input-group">
                  <input
                    type="text"
                    id="familiar-input"
                    placeholder="Ask me about PF2e rules, monsters, encounters..."
                    autocomplete="off"
                    maxlength="500"
                  >
                  <button type="submit" id="familiar-send">
                    <i class="fas fa-paper-plane"></i>
                  </button>
                </div>
              </form>

              <div class="input-hints">
                <small>Try: "/familiar rules grapple" or "/familiar monster level 5"</small>
              </div>
            </div>
          </div>

          <!-- Monsters Tab -->
          <div class="tab-content" id="tab-monsters">
            <div class="monster-generator">
              <h4><i class="fas fa-dragon"></i> Monster Generator</h4>

              <form id="monster-form" class="generator-form">
                <div class="form-row">
                  <div class="form-group">
                    <label for="monster-level">Level (1-25)</label>
                    <input type="number" id="monster-level" min="1" max="25" value="1">
                  </div>
                  <div class="form-group">
                    <label for="monster-type">Creature Type</label>
                    <select id="monster-type">
                      <option value="">Random</option>
                      <option value="beast">Beast</option>
                      <option value="humanoid">Humanoid</option>
                      <option value="undead">Undead</option>
                      <option value="dragon">Dragon</option>
                      <option value="fiend">Fiend</option>
                      <option value="fey">Fey</option>
                      <option value="elemental">Elemental</option>
                    </select>
                  </div>
                </div>

                <div class="form-row">
                  <div class="form-group">
                    <label for="monster-role">Role</label>
                    <select id="monster-role">
                      <option value="">Any</option>
                      <option value="brute">Brute</option>
                      <option value="skirmisher">Skirmisher</option>
                      <option value="soldier">Soldier</option>
                      <option value="spellcaster">Spellcaster</option>
                    </select>
                  </div>
                  <div class="form-group">
                    <label for="monster-environment">Environment</label>
                    <select id="monster-environment">
                      <option value="">Any</option>
                      <option value="forest">Forest</option>
                      <option value="dungeon">Dungeon</option>
                      <option value="urban">Urban</option>
                      <option value="mountain">Mountain</option>
                      <option value="desert">Desert</option>
                    </select>
                  </div>
                </div>

                <button type="submit" class="generate-btn">
                  <i class="fas fa-magic"></i> Generate Monster
                </button>
              </form>

              <div id="monster-results" class="results-area"></div>
            </div>
          </div>

          <!-- Encounters Tab -->
          <div class="tab-content" id="tab-encounters">
            <div class="encounter-builder">
              <h4><i class="fas fa-sword"></i> Encounter Builder</h4>

              <form id="encounter-form" class="generator-form">
                <div class="form-row">
                  <div class="form-group">
                    <label for="party-level">Party Level</label>
                    <input type="number" id="party-level" min="1" max="25" value="1">
                  </div>
                  <div class="form-group">
                    <label for="party-size">Party Size</label>
                    <input type="number" id="party-size" min="1" max="8" value="4">
                  </div>
                </div>

                <div class="form-row">
                  <div class="form-group">
                    <label for="encounter-difficulty">Difficulty</label>
                    <select id="encounter-difficulty">
                      <option value="trivial">Trivial</option>
                      <option value="low">Low</option>
                      <option value="moderate" selected>Moderate</option>
                      <option value="severe">Severe</option>
                      <option value="extreme">Extreme</option>
                    </select>
                  </div>
                  <div class="form-group">
                    <label for="encounter-theme">Theme</label>
                    <input type="text" id="encounter-theme" placeholder="e.g. bandits, undead, beasts">
                  </div>
                </div>

                <button type="submit" class="generate-btn">
                  <i class="fas fa-dice"></i> Build Encounter
                </button>
              </form>

              <div id="encounter-results" class="results-area"></div>
            </div>
          </div>

          <!-- Art Tab -->
          <div class="tab-content" id="tab-art">
            <div class="art-generator">
              <h4><i class="fas fa-palette"></i> Creature Art Generator</h4>

              <form id="art-form" class="generator-form">
                <div class="form-group">
                  <label for="art-creature">Creature Name or Description</label>
                  <input type="text" id="art-creature" placeholder="e.g. fire dragon, orc warrior, elven wizard">
                </div>

                <div class="form-row">
                  <div class="form-group">
                    <label for="art-style">Art Style</label>
                    <select id="art-style">
                      <option value="fantasy">Fantasy</option>
                      <option value="realistic">Realistic</option>
                      <option value="cartoon">Cartoon</option>
                      <option value="sketch">Sketch</option>
                      <option value="digital-art">Digital Art</option>
                    </select>
                  </div>
                  <div class="form-group">
                    <label for="art-mood">Mood</label>
                    <select id="art-mood">
                      <option value="neutral">Neutral</option>
                      <option value="menacing">Menacing</option>
                      <option value="heroic">Heroic</option>
                      <option value="mysterious">Mysterious</option>
                      <option value="whimsical">Whimsical</option>
                    </select>
                  </div>
                </div>

                <button type="submit" class="generate-btn">
                  <i class="fas fa-brush"></i> Generate Art
                </button>
              </form>

              <div id="art-results" class="results-area"></div>
            </div>
          </div>
        </div>

        <!-- Status Bar -->
        <div class="familiar-status">
          <div class="status-left">
            <span id="status-text">Ready</span>
          </div>
          <div class="status-right">
            <span id="response-time">--ms</span>
            <span id="session-cost">$0.00</span>
          </div>
        </div>
      </div>

      <!-- Minimized View -->
      <div class="familiar-minimized" id="familiar-minimized" style="display: none;">
        <div class="raven-sprite-mini" id="raven-sprite-mini">
          <i class="fas fa-crow"></i>
        </div>
        <div class="notification-badge" id="notification-badge" style="display: none;">
          <span id="notification-count">0</span>
        </div>
      </div>
    `;

    // Add to DOM
    document.body.appendChild(container);
    this.container = container;
  }

  /**
   * Create animated raven sprite
   */
  async createRavenSprite() {
    const ravenContainer = document.getElementById('raven-container');

    // Create SVG raven sprite with animations
    const ravenSVG = `
      <svg class="raven-sprite" viewBox="0 0 100 100" width="60" height="60">
        <defs>
          <style>
            .raven-body { fill: #1a1a1a; }
            .raven-wing { fill: #0f0f0f; }
            .raven-eye { fill: #ff4444; }
            .raven-beak { fill: #ffa500; }

            .idle .raven-wing {
              animation: gentle-flap 3s ease-in-out infinite;
            }
            .thinking .raven-wing {
              animation: thinking-flap 1s ease-in-out infinite;
            }
            .speaking .raven-wing {
              animation: excited-flap 0.5s ease-in-out infinite;
            }
            .working .raven-wing {
              animation: working-flap 0.8s linear infinite;
            }

            @keyframes gentle-flap {
              0%, 100% { transform: rotateX(0deg); }
              50% { transform: rotateX(-10deg); }
            }

            @keyframes thinking-flap {
              0%, 100% { transform: rotateX(0deg) rotateY(0deg); }
              25% { transform: rotateX(-5deg) rotateY(-3deg); }
              75% { transform: rotateX(-5deg) rotateY(3deg); }
            }

            @keyframes excited-flap {
              0%, 100% { transform: rotateX(0deg); }
              50% { transform: rotateX(-15deg); }
            }

            @keyframes working-flap {
              0% { transform: rotateX(0deg); }
              25% { transform: rotateX(-10deg); }
              50% { transform: rotateX(0deg); }
              75% { transform: rotateX(-10deg); }
              100% { transform: rotateX(0deg); }
            }
          </style>
        </defs>

        <!-- Raven Body -->
        <ellipse class="raven-body" cx="50" cy="60" rx="15" ry="20"/>

        <!-- Raven Head -->
        <circle class="raven-body" cx="50" cy="35" r="12"/>

        <!-- Wings -->
        <ellipse class="raven-wing" cx="40" cy="55" rx="8" ry="15" transform="rotate(-20 40 55)"/>
        <ellipse class="raven-wing" cx="60" cy="55" rx="8" ry="15" transform="rotate(20 60 55)"/>

        <!-- Eyes -->
        <circle class="raven-eye" cx="47" cy="32" r="2"/>
        <circle class="raven-eye" cx="53" cy="32" r="2"/>

        <!-- Beak -->
        <polygon class="raven-beak" points="50,40 45,43 50,45"/>

        <!-- Tail -->
        <ellipse class="raven-body" cx="50" cy="78" rx="12" ry="6"/>

        <!-- Feet -->
        <line stroke="#ffa500" stroke-width="2" x1="45" y1="78" x2="45" y2="82"/>
        <line stroke="#ffa500" stroke-width="2" x1="55" y1="78" x2="55" y2="82"/>
      </svg>
    `;

    ravenContainer.innerHTML = ravenSVG;
    this.ravenSprite = ravenContainer.querySelector('.raven-sprite');

    // Set initial animation state
    this.setRavenAnimation(this.animationStates.idle);
  }

  /**
   * Create chat interface elements
   */
  async createChatInterface() {
    // Chat interface is already created in HTML template
    // Set up message handling
    this.messagesContainer = document.getElementById('familiar-messages');
    this.inputElement = document.getElementById('familiar-input');
    this.chatForm = document.getElementById('familiar-chat-form');
  }

  /**
   * Setup event listeners for UI interactions
   */
  async setupEventListeners() {
    // Minimize/maximize
    document.getElementById('familiar-minimize').addEventListener('click', () => {
      this.toggleMinimize();
    });

    // Restore from minimized
    document.getElementById('familiar-minimized').addEventListener('click', () => {
      this.toggleMinimize();
    });

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.switchTab(e.target.dataset.tab);
      });
    });

    // Chat form submission
    this.chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleChatSubmission();
    });

    // Quick command buttons
    document.querySelectorAll('.quick-cmd').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const command = e.target.dataset.cmd;
        this.inputElement.value = command;
        this.handleChatSubmission();
      });
    });

    // Monster form
    document.getElementById('monster-form').addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleMonsterGeneration();
    });

    // Encounter form
    document.getElementById('encounter-form').addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleEncounterGeneration();
    });

    // Art form
    document.getElementById('art-form').addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleArtGeneration();
    });

    // Settings and help buttons
    document.getElementById('familiar-settings').addEventListener('click', () => {
      this.openSettings();
    });

    document.getElementById('familiar-help').addEventListener('click', () => {
      this.showHelp();
    });

    // Window resize handler
    window.addEventListener('resize', () => {
      this.positionUI();
    });
  }

  /**
   * Position UI in bottom-right corner
   */
  positionUI() {
    if (!this.container) return;

    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    const containerWidth = 360;
    const containerHeight = 500;

    const rightOffset = 20;
    const bottomOffset = 20;

    this.container.style.position = 'fixed';
    this.container.style.right = `${rightOffset}px`;
    this.container.style.bottom = `${bottomOffset}px`;
    this.container.style.width = `${containerWidth}px`;
    this.container.style.height = `${containerHeight}px`;
    this.container.style.zIndex = '999';
  }

  /**
   * Toggle minimize/maximize state
   */
  toggleMinimize() {
    const window = document.getElementById('familiar-window');
    const minimized = document.getElementById('familiar-minimized');

    this.isMinimized = !this.isMinimized;

    if (this.isMinimized) {
      window.style.display = 'none';
      minimized.style.display = 'flex';
      this.container.style.width = '80px';
      this.container.style.height = '80px';
    } else {
      window.style.display = 'flex';
      minimized.style.display = 'none';
      this.container.style.width = '360px';
      this.container.style.height = '500px';
    }

    this.saveUIState();
  }

  /**
   * Switch active tab
   */
  switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
      content.classList.remove('active');
    });
    document.getElementById(`tab-${tabName}`).classList.add('active');

    this.activeTab = tabName;
    this.saveUIState();
  }

  /**
   * Set raven animation state
   */
  setRavenAnimation(state) {
    if (this.ravenSprite) {
      // Remove all animation classes
      Object.values(this.animationStates).forEach(animState => {
        this.ravenSprite.classList.remove(animState);
      });

      // Add new animation class
      this.ravenSprite.classList.add(state);
      this.currentAnimation = state;
    }
  }

  /**
   * Handle chat form submission
   */
  async handleChatSubmission() {
    const input = this.inputElement.value.trim();
    if (!input) return;

    try {
      // Clear input and show thinking state
      this.inputElement.value = '';
      this.setRavenAnimation(this.animationStates.thinking);
      this.updateStatus('Processing...');

      // Add user message to chat
      this.addMessage('user', input);

      // Emit command event for processing
      this.emit('command', { command: input, source: 'chat' });

    } catch (error) {
      this.logger.error('Chat submission error', error);
      this.addMessage('error', 'Sorry, I encountered an error processing your request.');
      this.setRavenAnimation(this.animationStates.idle);
      this.updateStatus('Error');
    }
  }

  /**
   * Add message to chat interface
   */
  addMessage(type, content, metadata = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `message-${type}`);

    const timestamp = new Date().toLocaleTimeString();

    let messageHTML = '';

    switch (type) {
      case 'user':
        messageHTML = `
          <div class="message-content">
            <div class="message-header">
              <span class="message-author">You</span>
              <span class="message-time">${timestamp}</span>
            </div>
            <div class="message-text">${content}</div>
          </div>
        `;
        break;

      case 'familiar':
        messageHTML = `
          <div class="message-content">
            <div class="message-header">
              <i class="fas fa-crow"></i>
              <span class="message-author">Familiar</span>
              <span class="message-time">${timestamp}</span>
            </div>
            <div class="message-text">${content}</div>
            ${metadata.cost ? `<div class="message-cost">Cost: $${metadata.cost.toFixed(4)}</div>` : ''}
          </div>
        `;
        break;

      case 'error':
        messageHTML = `
          <div class="message-content error">
            <div class="message-header">
              <i class="fas fa-exclamation-triangle"></i>
              <span class="message-author">System</span>
              <span class="message-time">${timestamp}</span>
            </div>
            <div class="message-text">${content}</div>
          </div>
        `;
        break;
    }

    messageDiv.innerHTML = messageHTML;
    this.messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  /**
   * Handle monster generation form
   */
  async handleMonsterGeneration() {
    const level = document.getElementById('monster-level').value;
    const type = document.getElementById('monster-type').value;
    const role = document.getElementById('monster-role').value;
    const environment = document.getElementById('monster-environment').value;

    const criteria = {
      level: parseInt(level),
      creatureType: type || undefined,
      role: role || undefined,
      environment: environment || undefined
    };

    this.setRavenAnimation(this.animationStates.working);
    this.updateStatus('Generating monster...');

    this.emit('monster-generation', { criteria, source: 'form' });
  }

  /**
   * Handle encounter generation form
   */
  async handleEncounterGeneration() {
    const partyLevel = document.getElementById('party-level').value;
    const partySize = document.getElementById('party-size').value;
    const difficulty = document.getElementById('encounter-difficulty').value;
    const theme = document.getElementById('encounter-theme').value;

    const partyInfo = {
      level: parseInt(partyLevel),
      size: parseInt(partySize)
    };

    const criteria = {
      difficulty,
      theme: theme || undefined
    };

    this.setRavenAnimation(this.animationStates.working);
    this.updateStatus('Building encounter...');

    this.emit('encounter-generation', { partyInfo, criteria, source: 'form' });
  }

  /**
   * Handle art generation form
   */
  async handleArtGeneration() {
    const creature = document.getElementById('art-creature').value;
    const style = document.getElementById('art-style').value;
    const mood = document.getElementById('art-mood').value;

    if (!creature.trim()) {
      this.addMessage('error', 'Please enter a creature name or description.');
      return;
    }

    const artRequest = {
      creature: creature.trim(),
      style,
      mood
    };

    this.setRavenAnimation(this.animationStates.working);
    this.updateStatus('Generating artwork...');

    this.emit('art-generation', { artRequest, source: 'form' });
  }

  /**
   * Update status text and animation
   */
  updateStatus(text, responseTime = null, cost = null) {
    const statusElement = document.getElementById('status-text');
    const responseTimeElement = document.getElementById('response-time');
    const costElement = document.getElementById('session-cost');

    if (statusElement) statusElement.textContent = text;
    if (responseTime && responseTimeElement) {
      responseTimeElement.textContent = `${responseTime}ms`;
    }
    if (cost && costElement) {
      costElement.textContent = `$${cost.toFixed(4)}`;
    }
  }

  /**
   * Display response in appropriate tab/interface
   */
  displayResponse(response) {
    try {
      // Reset to idle animation
      this.setRavenAnimation(this.animationStates.idle);
      this.updateStatus('Ready', response.responseTime, response.cost);

      switch (response.type) {
        case 'rule':
        case 'help':
          this.displayChatResponse(response);
          break;
        case 'monster':
          this.displayMonsterResponse(response);
          break;
        case 'encounter':
          this.displayEncounterResponse(response);
          break;
        case 'art':
          this.displayArtResponse(response);
          break;
        default:
          this.displayChatResponse(response);
      }

    } catch (error) {
      this.logger.error('Error displaying response', error);
      this.addMessage('error', 'Failed to display response.');
    }
  }

  /**
   * Display chat response
   */
  displayChatResponse(response) {
    this.addMessage('familiar', response.content, {
      cost: response.cost,
      responseTime: response.responseTime
    });
  }

  /**
   * Display monster response
   */
  displayMonsterResponse(response) {
    const resultsContainer = document.getElementById('monster-results');
    const monster = response.monster;

    const monsterHTML = `
      <div class="generated-monster">
        <div class="monster-header">
          <h5>${monster.name}</h5>
          <span class="monster-level">Level ${monster.level}</span>
        </div>

        <div class="monster-stats">
          <div class="stat-row">
            <strong>AC:</strong> ${monster.ac} |
            <strong>HP:</strong> ${monster.hp} |
            <strong>Speed:</strong> ${monster.speeds.land || 25} feet
          </div>

          <div class="abilities-row">
            <strong>STR</strong> ${monster.abilities.strength} |
            <strong>DEX</strong> ${monster.abilities.dexterity} |
            <strong>CON</strong> ${monster.abilities.constitution} |
            <strong>INT</strong> ${monster.abilities.intelligence} |
            <strong>WIS</strong> ${monster.abilities.wisdom} |
            <strong>CHA</strong> ${monster.abilities.charisma}
          </div>
        </div>

        <div class="monster-actions">
          <button class="action-btn export-monster" data-monster='${JSON.stringify(monster)}'>
            <i class="fas fa-download"></i> Export to Foundry
          </button>
          <button class="action-btn add-to-scene" data-monster='${JSON.stringify(monster)}'>
            <i class="fas fa-plus"></i> Add to Scene
          </button>
        </div>

        <div class="response-meta">
          <small>Generated in ${response.responseTime}ms | Cost: $${response.cost.toFixed(4)}</small>
        </div>
      </div>
    `;

    resultsContainer.innerHTML = monsterHTML;

    // Setup action button listeners
    this.setupMonsterActionButtons(resultsContainer);

    // Switch to monsters tab if not already active
    if (this.activeTab !== 'monsters') {
      this.switchTab('monsters');
    }
  }

  /**
   * Setup monster action button listeners
   */
  setupMonsterActionButtons(container) {
    container.querySelectorAll('.export-monster').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const monsterData = JSON.parse(e.target.dataset.monster);
        this.emit('export-monster', monsterData);
      });
    });

    container.querySelectorAll('.add-to-scene').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const monsterData = JSON.parse(e.target.dataset.monster);
        this.emit('add-to-scene', monsterData);
      });
    });
  }

  /**
   * Display encounter response
   */
  displayEncounterResponse(response) {
    const resultsContainer = document.getElementById('encounter-results');
    const encounter = response.encounter;

    let creaturesHTML = '';
    encounter.creatures.forEach(creature => {
      creaturesHTML += `
        <div class="encounter-creature">
          <span class="creature-count">${creature.quantity}x</span>
          <span class="creature-name">${creature.name}</span>
          <span class="creature-level">(Level ${creature.level})</span>
          <button class="mini-btn add-creature" data-creature='${JSON.stringify(creature)}'>
            <i class="fas fa-plus"></i>
          </button>
        </div>
      `;
    });

    const encounterHTML = `
      <div class="generated-encounter">
        <div class="encounter-header">
          <h5>Generated Encounter</h5>
          <span class="encounter-difficulty ${response.balance.difficulty.toLowerCase()}">
            ${response.balance.difficulty} Difficulty
          </span>
        </div>

        <div class="encounter-budget">
          <strong>XP Budget:</strong> ${response.balance.xpBudget.used}/${response.balance.xpBudget.total}
          (${Math.round(response.balance.xpBudget.efficiency * 100)}% efficiency)
        </div>

        <div class="encounter-creatures">
          <h6>Creatures:</h6>
          ${creaturesHTML}
        </div>

        <div class="encounter-tactics">
          <h6>Tactical Notes:</h6>
          <p>${encounter.tactics}</p>
        </div>

        <div class="encounter-actions">
          <button class="action-btn export-encounter" data-encounter='${JSON.stringify(encounter)}'>
            <i class="fas fa-download"></i> Export Encounter
          </button>
        </div>

        <div class="response-meta">
          <small>Generated in ${response.responseTime}ms | Cost: $${response.cost.toFixed(4)}</small>
        </div>
      </div>
    `;

    resultsContainer.innerHTML = encounterHTML;

    // Setup action button listeners
    this.setupEncounterActionButtons(resultsContainer);

    // Switch to encounters tab if not already active
    if (this.activeTab !== 'encounters') {
      this.switchTab('encounters');
    }
  }

  /**
   * Setup encounter action button listeners
   */
  setupEncounterActionButtons(container) {
    container.querySelectorAll('.add-creature').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const creatureData = JSON.parse(e.target.dataset.creature);
        this.emit('add-creature-to-scene', creatureData);
      });
    });

    container.querySelectorAll('.export-encounter').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const encounterData = JSON.parse(e.target.dataset.encounter);
        this.emit('export-encounter', encounterData);
      });
    });
  }

  /**
   * Display art response
   */
  displayArtResponse(response) {
    const resultsContainer = document.getElementById('art-results');

    if (response.processing) {
      resultsContainer.innerHTML = `
        <div class="art-processing">
          <div class="loading-spinner"></div>
          <p>Generating artwork... This may take a moment.</p>
        </div>
      `;
      return;
    }

    const artHTML = `
      <div class="generated-art">
        <div class="art-image">
          <img src="${response.image.url}" alt="${response.description.title}"
               style="max-width: 100%; border-radius: 8px;">
        </div>

        <div class="art-description">
          <h6>${response.description.title}</h6>
          <p>${response.description.detailed}</p>
        </div>

        <div class="art-actions">
          <button class="action-btn save-art" data-url="${response.image.url}" data-name="${response.description.title}">
            <i class="fas fa-save"></i> Save to Assets
          </button>
          <button class="action-btn regenerate-art" data-prompt="${encodeURIComponent(response.description.prompt)}">
            <i class="fas fa-redo"></i> Regenerate
          </button>
        </div>

        <div class="response-meta">
          <small>Generated in ${response.responseTime}ms | Cost: $${response.cost.total.toFixed(4)}</small>
        </div>
      </div>
    `;

    resultsContainer.innerHTML = artHTML;

    // Setup action button listeners
    this.setupArtActionButtons(resultsContainer);

    // Switch to art tab if not already active
    if (this.activeTab !== 'art') {
      this.switchTab('art');
    }
  }

  /**
   * Setup art action button listeners
   */
  setupArtActionButtons(container) {
    container.querySelectorAll('.save-art').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const url = e.target.dataset.url;
        const name = e.target.dataset.name;
        this.emit('save-art', { url, name });
      });
    });

    container.querySelectorAll('.regenerate-art').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const prompt = decodeURIComponent(e.target.dataset.prompt);
        this.emit('regenerate-art', { prompt });
      });
    });
  }

  /**
   * Open settings dialog
   */
  openSettings() {
    this.emit('open-settings');
  }

  /**
   * Show help information
   */
  showHelp() {
    this.addMessage('familiar', `
      <div class="help-content">
        <h6>Available Commands:</h6>
        <ul>
          <li><code>/familiar rules [topic]</code> - Look up PF2e rules</li>
          <li><code>/familiar monster [level] [type]</code> - Generate a monster</li>
          <li><code>/familiar encounter [party level] [difficulty]</code> - Build an encounter</li>
          <li><code>/familiar art [creature]</code> - Generate creature artwork</li>
          <li><code>/familiar help</code> - Show this help message</li>
        </ul>

        <h6>Tips:</h6>
        <ul>
          <li>Use the tabs above for guided generation</li>
          <li>All generated content can be exported to Foundry</li>
          <li>Response times and costs are shown for each query</li>
        </ul>
      </div>
    `, { cost: 0 });
  }

  /**
   * Load saved UI state
   */
  async loadUIState() {
    try {
      const saved = globalCache.get('ui-state');
      if (saved) {
        this.isMinimized = saved.isMinimized || false;
        this.activeTab = saved.activeTab || 'chat';

        if (this.isMinimized) {
          this.toggleMinimize();
        }

        if (this.activeTab !== 'chat') {
          this.switchTab(this.activeTab);
        }
      }
    } catch (error) {
      this.logger.warn('Failed to load UI state', error);
    }
  }

  /**
   * Save current UI state
   */
  async saveUIState() {
    try {
      const state = {
        isMinimized: this.isMinimized,
        activeTab: this.activeTab,
        timestamp: Date.now()
      };

      globalCache.set('ui-state', state);
    } catch (error) {
      this.logger.warn('Failed to save UI state', error);
    }
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    try {
      // Save current state
      await this.saveUIState();

      // Remove event listeners
      // (Event listeners are automatically removed when DOM elements are removed)

      // Remove from DOM
      if (this.container && this.container.parentNode) {
        this.container.parentNode.removeChild(this.container);
      }

      this.isInitialized = false;
      this.logger.info('Familiar UI cleaned up successfully');

    } catch (error) {
      this.logger.error('Error during UI cleanup', error);
    }
  }

  /**
   * Health check
   */
  async healthCheck() {
    return {
      status: this.isInitialized ? 'healthy' : 'not-initialized',
      ui: {
        visible: this.isVisible,
        minimized: this.isMinimized,
        activeTab: this.activeTab,
        containerExists: !!this.container
      },
      sprite: {
        exists: !!this.ravenSprite,
        animation: this.currentAnimation
      }
    };
  }
}

export default FamiliarUI;