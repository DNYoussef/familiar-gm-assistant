/**
 * API Contracts and Interface Definitions
 * Phase 2: Core Architecture - Development Princess Domain
 * CRITICAL: Defines all interfaces between system components
 */

/**
 * RAG System API Contract
 * Defines interface between Frontend UI and RAG Backend
 */
export const RAGSystemContract = {
  /**
   * Query the PF2e knowledge base
   * @param {string} query - User query text
   * @param {Object} context - Query context (user, scene, etc.)
   * @returns {Promise<RAGResponse>}
   */
  async queryKnowledge(query, context = {}) {
    return {
      type: 'rule' | 'monster' | 'spell' | 'equipment' | 'class',
      content: {
        title: String,
        description: String,
        mechanics: Array<String>,
        examples: Array<String>
      },
      citations: Array<{
        source: String,
        page: Number,
        url: String
      }>,
      relatedContent: Array<{
        id: String,
        title: String,
        type: String
      }>,
      confidence: Number, // 0-1
      responseTime: Number,
      cost: Number
    };
  },

  /**
   * Get system status and health
   */
  async getStatus() {
    return {
      knowledgeBase: {
        status: 'ready' | 'loading' | 'error',
        lastUpdated: Date,
        totalDocuments: Number,
        totalVectors: Number
      },
      cache: {
        hitRate: Number,
        size: Number,
        maxSize: Number
      },
      performance: {
        averageResponseTime: Number,
        totalQueries: Number,
        errorRate: Number
      }
    };
  }
};

/**
 * Monster Generation API Contract
 * Defines interface for PF2e creature generation
 */
export const MonsterGenerationContract = {
  /**
   * Generate a PF2e creature
   * @param {Object} criteria - Generation criteria
   * @returns {Promise<MonsterResponse>}
   */
  async generateMonster(criteria) {
    // Input validation
    const validatedCriteria = {
      level: Number, // 1-25
      creatureType: String, // 'beast', 'humanoid', 'undead', etc.
      traits: Array<String>,
      role: 'brute' | 'skirmisher' | 'sniper' | 'soldier' | 'spellcaster',
      environment: String,
      rarity: 'common' | 'uncommon' | 'rare' | 'unique',
      alignment: String // PF2e alignment system
    };

    return {
      monster: {
        // Basic Information
        name: String,
        level: Number,
        creatureType: String,
        size: 'tiny' | 'small' | 'medium' | 'large' | 'huge' | 'gargantuan',
        alignment: String,
        traits: Array<String>,
        rarity: String,

        // Defensive Stats
        ac: Number,
        hp: Number,
        saves: {
          fortitude: Number,
          reflex: Number,
          will: Number
        },
        resistances: Array<{type: String, value: Number}>,
        immunities: Array<String>,
        weaknesses: Array<{type: String, value: Number}>,

        // Abilities
        abilities: {
          strength: Number,
          dexterity: Number,
          constitution: Number,
          intelligence: Number,
          wisdom: Number,
          charisma: Number
        },

        // Skills and Movement
        skills: Array<{name: String, bonus: Number}>,
        speeds: {
          land: Number,
          fly: Number,
          swim: Number,
          climb: Number,
          burrow: Number
        },
        languages: Array<String>,
        senses: Array<String>,

        // Combat Abilities
        attacks: Array<{
          name: String,
          type: 'melee' | 'ranged',
          bonus: Number,
          damage: String,
          damageType: String,
          traits: Array<String>,
          range: Number
        }>,

        spells: Array<{
          tradition: String,
          level: Number,
          dc: Number,
          spells: Array<{
            level: Number,
            name: String,
            uses: Number | 'at-will' | 'constant'
          }>
        }>,

        // Special Abilities
        specialAbilities: Array<{
          name: String,
          type: 'passive' | 'reaction' | 'free' | 'action' | 'two-actions' | 'three-actions',
          description: String,
          frequency: String,
          trigger: String
        }>
      },

      // Generation metadata
      generationData: {
        seed: String,
        model: String,
        responseTime: Number,
        cost: Number,
        validation: {
          isValid: Boolean,
          warnings: Array<String>,
          errors: Array<String>
        }
      }
    };
  },

  /**
   * Validate monster stat block against PF2e rules
   */
  async validateMonster(monster) {
    return {
      isValid: Boolean,
      score: Number, // 0-100
      issues: Array<{
        type: 'error' | 'warning' | 'info',
        field: String,
        message: String,
        suggestion: String
      }>,
      recommendations: Array<String>
    };
  }
};

/**
 * Art Generation API Contract
 * Two-phase AI art generation system
 */
export const ArtGenerationContract = {
  /**
   * Generate creature artwork
   * @param {Object} creature - Monster data or description
   * @returns {Promise<ArtResponse>}
   */
  async generateArt(creature) {
    return {
      image: {
        url: String,
        width: Number,
        height: Number,
        format: 'png' | 'jpg' | 'webp',
        size: Number // bytes
      },

      description: {
        title: String,
        detailed: String, // AI-generated description
        prompt: String, // Used for image generation
        style: String,
        mood: String
      },

      generation: {
        phase1Model: String, // Description generation model
        phase2Model: String, // Image generation model
        responseTime: Number,
        cost: {
          description: Number,
          image: Number,
          total: Number
        },
        attempts: Number,
        quality: Number // 0-1
      },

      processing: Boolean // True while generating
    };
  },

  /**
   * Edit or regenerate existing artwork
   */
  async editArt(imageUrl, editInstructions) {
    return {
      originalUrl: String,
      editedUrl: String,
      changes: Array<String>,
      cost: Number,
      responseTime: Number
    };
  }
};

/**
 * Encounter Builder API Contract
 * Balanced encounter generation for PF2e
 */
export const EncounterBuilderContract = {
  /**
   * Build balanced encounter
   * @param {Object} partyInfo - Party composition and level
   * @param {Object} criteria - Encounter criteria
   * @returns {Promise<EncounterResponse>}
   */
  async buildEncounter(partyInfo, criteria) {
    // Input validation
    const validatedPartyInfo = {
      level: Number, // Average party level
      size: Number, // Number of players
      composition: Array<{
        class: String,
        level: Number,
        role: 'tank' | 'dps' | 'support' | 'utility'
      }>
    };

    const validatedCriteria = {
      difficulty: 'trivial' | 'low' | 'moderate' | 'severe' | 'extreme',
      environment: String,
      theme: String,
      maxCreatures: Number,
      budget: Number // XP budget
    };

    return {
      encounter: {
        creatures: Array<{
          id: String,
          name: String,
          level: Number,
          quantity: Number,
          xp: Number,
          role: String,
          placement: String,
          tactics: String
        }>,

        environment: {
          terrain: String,
          hazards: Array<String>,
          cover: String,
          lighting: String
        },

        tactics: String, // Tactical advice for GM
        scaling: {
          easier: String, // How to make easier
          harder: String  // How to make harder
        }
      },

      balance: {
        difficulty: String,
        xpBudget: {
          total: Number,
          used: Number,
          efficiency: Number
        },
        actionEconomy: {
          creatureActions: Number,
          partyActions: Number,
          balance: Number // -1 to 1
        },
        threatAssessment: {
          primaryThreats: Array<String>,
          weaknesses: Array<String>,
          counterplay: Array<String>
        }
      }
    };
  }
};

/**
 * Session Management API Contract
 * Manages user sessions, costs, and performance
 */
export const SessionManagementContract = {
  /**
   * Start new session
   */
  async startSession(userId) {
    return {
      sessionId: String,
      startTime: Date,
      userId: String,
      initialBudget: Number,
      settings: Object
    };
  },

  /**
   * Get current session info
   */
  getCurrentSession() {
    return {
      sessionId: String,
      duration: Number, // milliseconds
      queries: Number,
      costs: {
        total: Number,
        byModel: Object<String, Number>,
        byFeature: Object<String, Number>
      },
      performance: {
        averageResponseTime: Number,
        cacheHitRate: Number,
        errorRate: Number
      },
      usage: {
        rules: Number,
        monsters: Number,
        encounters: Number,
        art: Number
      }
    };
  },

  /**
   * Record API usage and costs
   */
  recordAPIUsage(feature, model, cost, responseTime) {
    // Updates session statistics
  },

  /**
   * End session and generate report
   */
  async endSession() {
    return {
      sessionId: String,
      duration: Number,
      summary: {
        totalQueries: Number,
        totalCost: Number,
        averageResponseTime: Number,
        featuresUsed: Array<String>,
        errorCount: Number
      },
      recommendations: Array<String>
    };
  }
};

/**
 * Chat Command Processing Contract
 * Processes and routes chat commands
 */
export const ChatProcessingContract = {
  /**
   * Process chat command
   * @param {string} command - User command
   * @param {Object} context - Execution context
   * @returns {Promise<CommandResponse>}
   */
  async processCommand(command, context) {
    return {
      type: 'rule' | 'monster' | 'encounter' | 'art' | 'help' | 'error',
      content: Object, // Response content based on type
      metadata: {
        command: String,
        processingTime: Number,
        model: String,
        cost: Number,
        cacheHit: Boolean
      }
    };
  },

  /**
   * Get available commands and help
   */
  getCommands() {
    return {
      commands: Array<{
        name: String,
        aliases: Array<String>,
        description: String,
        usage: String,
        examples: Array<String>
      }>,
      categories: Array<{
        name: String,
        commands: Array<String>
      }>
    };
  }
};

/**
 * Performance Tracking Contract
 * Monitors system performance and costs
 */
export const PerformanceTrackingContract = {
  /**
   * Record response time
   */
  recordResponse(responseTime, feature, success = true) {
    // Updates performance metrics
  },

  /**
   * Get performance metrics
   */
  getMetrics() {
    return {
      responseTime: {
        average: Number,
        median: Number,
        p95: Number,
        p99: Number
      },
      throughput: {
        queriesPerMinute: Number,
        peakQPM: Number
      },
      reliability: {
        uptime: Number,
        errorRate: Number,
        successRate: Number
      },
      costs: {
        totalSession: Number,
        averagePerQuery: Number,
        byFeature: Object<String, Number>
      }
    };
  },

  /**
   * Check if performance targets are met
   */
  validatePerformanceTargets() {
    return {
      responseTime: {
        target: 2000, // ms
        actual: Number,
        met: Boolean
      },
      costPerSession: {
        target: 0.015, // $
        actual: Number,
        met: Boolean
      },
      errorRate: {
        target: 0.05, // 5%
        actual: Number,
        met: Boolean
      }
    };
  }
};

/**
 * Foundry Integration Contract
 * Handles Foundry VTT specific functionality
 */
export const FoundryIntegrationContract = {
  /**
   * Export monster to Foundry Actor
   */
  async exportToActor(monsterData) {
    return {
      actorId: String,
      success: Boolean,
      warnings: Array<String>
    };
  },

  /**
   * Add tokens to current scene
   */
  async addToScene(actorId, position = {}) {
    return {
      tokenId: String,
      success: Boolean,
      position: {x: Number, y: Number}
    };
  },

  /**
   * Check Foundry system compatibility
   */
  checkCompatibility() {
    return {
      foundryVersion: String,
      compatible: Boolean,
      pf2eSystem: Boolean,
      requiredModules: Array<{
        name: String,
        installed: Boolean,
        required: Boolean
      }>
    };
  }
};

/**
 * Component Interface Registry
 * Central registry for all system interfaces
 */
export const ComponentRegistry = {
  contracts: {
    rag: RAGSystemContract,
    monster: MonsterGenerationContract,
    art: ArtGenerationContract,
    encounter: EncounterBuilderContract,
    session: SessionManagementContract,
    chat: ChatProcessingContract,
    performance: PerformanceTrackingContract,
    foundry: FoundryIntegrationContract
  },

  /**
   * Validate component implements required contract
   */
  validateComponent(component, contractName) {
    const contract = this.contracts[contractName];
    if (!contract) return false;

    // Validate all required methods exist
    const methods = Object.getOwnPropertyNames(contract).filter(
      name => typeof contract[name] === 'function'
    );

    return methods.every(method => typeof component[method] === 'function');
  },

  /**
   * Get contract documentation
   */
  getContractDocs(contractName) {
    return {
      name: contractName,
      methods: this.getContractMethods(contractName),
      types: this.getContractTypes(contractName)
    };
  }
};

export default ComponentRegistry;