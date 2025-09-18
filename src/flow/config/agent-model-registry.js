/**
 * SPEK Agent Model Registry
 * Automatic AI model assignment based on agent capabilities and requirements
 */

const AIModel = {
  GPT5: 'gpt-5',
  GPT5_CODEX: 'gpt-5-codex',
  GEMINI_PRO: 'gemini-2.5-pro',
  GEMINI_FLASH: 'gemini-2.5-flash',
  CLAUDE_OPUS: 'claude-opus-4.1',
  CLAUDE_SONNET: 'claude-sonnet-4'
};

const ReasoningComplexity = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high'
};

/**
 * Agent Model Configuration Registry
 * Maps each agent to optimal AI model based on capability requirements
 */
const AGENT_MODEL_REGISTRY = {

  // =============================================================================
  // BROWSER AUTOMATION & VISUAL AGENTS → GPT-5 CODEX
  // =============================================================================
  'frontend-developer': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.CLAUDE_SONNET,
    sequentialThinking: false,
    contextThreshold: 50000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['browser_automation', 'screenshot_capture', 'ui_testing', 'visual_validation'],
    mcpServers: ['claude-flow', 'memory', 'github', 'playwright', 'figma'],
    rationale: 'Codex can spin up browser, take screenshots, iterate on UI implementation + Figma for design integration'
  },

  'ui-designer': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: false,
    contextThreshold: 30000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['visual_design', 'browser_testing', 'screenshot_validation'],
    mcpServers: ['claude-flow', 'memory', 'playwright', 'figma', 'puppeteer'],
    rationale: 'Visual feedback loop with browser automation + design systems integration'
  },

  'mobile-dev': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 60000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['mobile_testing', 'device_simulation', 'gesture_control', 'multimodal_input'],
    mcpServers: ['claude-flow', 'memory', 'github', 'playwright', 'puppeteer'],
    rationale: 'Mobile development with device simulation + GitHub integration'
  },

  'rapid-prototyper': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.CLAUDE_SONNET,
    sequentialThinking: false,
    contextThreshold: 40000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['rapid_iteration', 'visual_testing', 'browser_automation'],
    mcpServers: ['claude-flow', 'memory', 'playwright', 'figma'],
    rationale: 'Rapid prototyping with visual validation + design tools'
  },

  // =============================================================================
  // LARGE CONTEXT & RESEARCH AGENTS → GEMINI 2.5 PRO (1M TOKENS)
  // =============================================================================
  'researcher': {
    primaryModel: AIModel.GEMINI_PRO,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 500000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['large_context_analysis', 'web_search', 'comprehensive_research'],
    mcpServers: ['claude-flow', 'memory', 'deepwiki', 'firecrawl', 'ref', 'context7'],
    rationale: 'Large context research + web scraping and documentation tools'
  },

  'research-agent': {
    primaryModel: AIModel.GEMINI_PRO,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 800000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['deep_research', 'pattern_analysis', 'large_file_processing'],
    mcpServers: ['claude-flow', 'memory', 'deepwiki', 'firecrawl', 'ref', 'context7'],
    rationale: 'Deep research with comprehensive web access and documentation'
  },

  'specification': {
    primaryModel: AIModel.GEMINI_PRO,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 300000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['requirements_analysis', 'large_spec_processing', 'context_synthesis'],
    mcpServers: ['claude-flow', 'memory', 'deepwiki', 'ref', 'context7', 'markitdown'],
    rationale: 'Specification analysis + documentation and markdown tools'
  },

  'architecture': {
    primaryModel: AIModel.GEMINI_PRO,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 400000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['system_design', 'architectural_analysis', 'pattern_recognition'],
    mcpServers: ['claude-flow', 'memory', 'deepwiki', 'ref', 'context7'],
    rationale: 'Architecture design with documentation and reference access'
  },

  'system-architect': {
    primaryModel: AIModel.GEMINI_PRO,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 600000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['enterprise_architecture', 'system_integration', 'large_scale_design'],
    mcpServers: ['claude-flow', 'memory', 'deepwiki', 'ref', 'context7'],
    rationale: 'Enterprise architecture with comprehensive documentation access'
  },

  // =============================================================================
  // AUTONOMOUS CODING & COMPLEX IMPLEMENTATION → GPT-5 CODEX
  // =============================================================================
  'coder': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 100000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['autonomous_coding', 'long_sessions', 'test_execution', 'iterative_development'],
    mcpServers: ['claude-flow', 'memory', 'github', 'filesystem'],
    rationale: 'Autonomous coding with GitHub integration and file operations'
  },

  'sparc-coder': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 80000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['sparc_methodology', 'tdd_implementation', 'autonomous_development'],
    mcpServers: ['claude-flow', 'memory', 'github', 'filesystem'],
    rationale: 'SPARC methodology with GitHub integration and autonomous development'
  },

  'backend-dev': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 70000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['api_development', 'database_integration', 'autonomous_testing'],
    mcpServers: ['claude-flow', 'memory', 'github', 'filesystem'],
    rationale: 'Backend API development with GitHub integration and testing'
  },

  'ml-developer': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: false,
    contextThreshold: 90000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['ml_implementation', 'data_processing', 'model_training'],
    mcpServers: ['claude-flow', 'memory', 'github', 'filesystem'],
    rationale: 'ML development with data processing and model management'
  },

  'cicd-engineer': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.CLAUDE_SONNET,
    sequentialThinking: false,
    contextThreshold: 50000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['pipeline_automation', 'github_integration', 'testing_automation'],
    mcpServers: ['claude-flow', 'memory', 'github', 'filesystem'],
    rationale: 'CI/CD automation with comprehensive GitHub integration'
  },

  // =============================================================================
  // QUALITY ASSURANCE & CODE REVIEW → CLAUDE OPUS 4.1 (72.7% SWE-BENCH)
  // =============================================================================
  'reviewer': {
    primaryModel: AIModel.CLAUDE_OPUS,
    fallbackModel: AIModel.GPT5_CODEX,
    sequentialThinking: false,
    contextThreshold: 80000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['code_review', 'quality_analysis', 'security_review', 'architectural_validation'],
    mcpServers: ['claude-flow', 'memory', 'github', 'eva'],
    rationale: 'Superior code review with GitHub integration and evaluation tools'
  },

  'code-analyzer': {
    primaryModel: AIModel.CLAUDE_OPUS,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: false,
    contextThreshold: 100000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['static_analysis', 'pattern_detection', 'code_quality_assessment'],
    mcpServers: ['claude-flow', 'memory', 'eva'],
    rationale: 'Advanced code analysis with performance evaluation tools'
  },

  'security-manager': {
    primaryModel: AIModel.CLAUDE_OPUS,
    fallbackModel: AIModel.GPT5,
    sequentialThinking: false,
    contextThreshold: 60000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['security_analysis', 'vulnerability_detection', 'compliance_checking'],
    mcpServers: ['claude-flow', 'memory', 'eva'],
    rationale: 'Security analysis with comprehensive evaluation capabilities'
  },

  'tester': {
    primaryModel: AIModel.CLAUDE_OPUS,
    fallbackModel: AIModel.GPT5_CODEX,
    sequentialThinking: false,
    contextThreshold: 70000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['test_design', 'quality_validation', 'tdd_implementation'],
    mcpServers: ['claude-flow', 'memory', 'github', 'playwright', 'eva'],
    rationale: 'Testing with browser automation and comprehensive evaluation'
  },

  'production-validator': {
    primaryModel: AIModel.CLAUDE_OPUS,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: false,
    contextThreshold: 90000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['production_readiness', 'quality_gates', 'compliance_validation'],
    mcpServers: ['claude-flow', 'memory', 'eva'],
    rationale: 'Production validation with comprehensive quality evaluation'
  },

  'reality-checker': {
    primaryModel: AIModel.CLAUDE_OPUS,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: false,
    contextThreshold: 50000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['validation', 'evidence_analysis', 'theater_detection'],
    mcpServers: ['claude-flow', 'memory', 'eva'],
    rationale: 'Reality validation with performance and evidence analysis'
  },

  // =============================================================================
  // COORDINATION & ORCHESTRATION → CLAUDE SONNET 4 + SEQUENTIAL THINKING
  // =============================================================================
  'sparc-coord': {
    primaryModel: AIModel.CLAUDE_SONNET,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: true,
    contextThreshold: 60000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['methodology_coordination', 'phase_orchestration', 'agent_coordination'],
    mcpServers: ['claude-flow', 'memory', 'sequential-thinking', 'plane'],
    rationale: 'SPARC coordination with sequential thinking and project management'
  },

  'hierarchical-coordinator': {
    primaryModel: AIModel.CLAUDE_SONNET,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: true,
    contextThreshold: 80000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['multi_agent_coordination', 'hierarchical_management', 'task_delegation'],
    mcpServers: ['claude-flow', 'memory', 'sequential-thinking', 'plane'],
    rationale: 'Hierarchical coordination with enhanced reasoning and task management'
  },

  'mesh-coordinator': {
    primaryModel: AIModel.CLAUDE_SONNET,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: true,
    contextThreshold: 70000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['peer_coordination', 'distributed_management', 'consensus_building'],
    mcpServers: ['claude-flow', 'memory', 'sequential-thinking', 'plane'],
    rationale: 'Mesh coordination with sequential reasoning and project integration'
  },

  'adaptive-coordinator': {
    primaryModel: AIModel.CLAUDE_SONNET,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: true,
    contextThreshold: 75000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['adaptive_management', 'dynamic_coordination', 'optimization'],
    mcpServers: ['claude-flow', 'memory', 'sequential-thinking', 'plane'],
    rationale: 'Adaptive coordination with optimization and project management'
  },

  'task-orchestrator': {
    primaryModel: AIModel.CLAUDE_SONNET,
    fallbackModel: AIModel.GEMINI_FLASH,
    sequentialThinking: true,
    contextThreshold: 50000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['task_management', 'workflow_orchestration', 'resource_allocation'],
    mcpServers: ['claude-flow', 'memory', 'sequential-thinking', 'plane'],
    rationale: 'Task orchestration with structured thinking and project management'
  },

  'memory-coordinator': {
    primaryModel: AIModel.CLAUDE_SONNET,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: true,
    contextThreshold: 60000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['memory_management', 'knowledge_coordination', 'state_management'],
    mcpServers: ['claude-flow', 'memory', 'sequential-thinking'],
    rationale: 'Memory coordination with enhanced reasoning and state management'
  },

  // =============================================================================
  // FAST OPERATIONS & ROUTINE TASKS → GEMINI 2.5 FLASH + SEQUENTIAL THINKING
  // =============================================================================
  'planner': {
    primaryModel: AIModel.GEMINI_FLASH,
    fallbackModel: AIModel.CLAUDE_SONNET,
    sequentialThinking: true,
    contextThreshold: 40000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['project_planning', 'resource_allocation', 'timeline_management'],
    mcpServers: ['claude-flow', 'memory', 'sequential-thinking', 'plane'],
    rationale: 'Cost-effective planning with enhanced reasoning and project management'
  },

  'refinement': {
    primaryModel: AIModel.GEMINI_FLASH,
    fallbackModel: AIModel.CLAUDE_SONNET,
    sequentialThinking: true,
    contextThreshold: 30000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['iterative_improvement', 'optimization', 'enhancement'],
    mcpServers: ['claude-flow', 'memory', 'sequential-thinking'],
    rationale: 'Iterative refinement with structured reasoning and memory'
  },

  'pr-manager': {
    primaryModel: AIModel.GEMINI_FLASH,
    fallbackModel: AIModel.GPT5,
    sequentialThinking: true,
    contextThreshold: 35000,
    reasoningComplexity: ReasoningComplexity.LOW,
    capabilities: ['pull_request_management', 'github_integration', 'review_coordination'],
    mcpServers: ['claude-flow', 'memory', 'github', 'sequential-thinking'],
    rationale: 'PR management with GitHub integration and enhanced reasoning'
  },

  'issue-tracker': {
    primaryModel: AIModel.GEMINI_FLASH,
    fallbackModel: AIModel.GPT5,
    sequentialThinking: true,
    contextThreshold: 25000,
    reasoningComplexity: ReasoningComplexity.LOW,
    capabilities: ['issue_management', 'bug_tracking', 'project_coordination'],
    mcpServers: ['claude-flow', 'memory', 'github', 'sequential-thinking', 'plane'],
    rationale: 'Issue tracking with GitHub and project management integration'
  },

  'performance-benchmarker': {
    primaryModel: AIModel.GEMINI_FLASH,
    fallbackModel: AIModel.CLAUDE_SONNET,
    sequentialThinking: true,
    contextThreshold: 30000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['performance_testing', 'benchmark_analysis', 'optimization'],
    mcpServers: ['claude-flow', 'memory', 'eva', 'sequential-thinking'],
    rationale: 'Performance analysis with evaluation tools and structured reasoning'
  },

  // =============================================================================
  // SPECIALIZED GITHUB INTEGRATION → GPT-5 CODEX (NATIVE GITHUB INTEGRATION)
  // =============================================================================
  'github-modes': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.GEMINI_FLASH,
    sequentialThinking: false,
    contextThreshold: 40000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['github_integration', 'repository_management', 'workflow_automation'],
    mcpServers: ['claude-flow', 'memory', 'github'],
    rationale: 'Comprehensive GitHub integration with native Codex capabilities'
  },

  'workflow-automation': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.CLAUDE_SONNET,
    sequentialThinking: false,
    contextThreshold: 45000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['github_actions', 'automation', 'ci_cd_integration'],
    mcpServers: ['claude-flow', 'memory', 'github', 'filesystem'],
    rationale: 'Workflow automation with GitHub Actions and file management'
  },

  'code-review-swarm': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 60000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['collaborative_review', 'github_integration', 'automated_feedback'],
    mcpServers: ['claude-flow', 'memory', 'github', 'eva'],
    rationale: 'Collaborative code review with GitHub integration and evaluation'
  },

  // =============================================================================
  // GENERAL PURPOSE & BALANCED REQUIREMENTS → GPT-5 STANDARD
  // =============================================================================
  'swarm-init': {
    primaryModel: AIModel.GPT5,
    fallbackModel: AIModel.CLAUDE_SONNET,
    sequentialThinking: false,
    contextThreshold: 35000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['swarm_initialization', 'topology_setup', 'agent_spawning'],
    mcpServers: ['claude-flow', 'memory'],
    rationale: 'Swarm initialization with flow coordination and memory management'
  },

  'smart-agent': {
    primaryModel: AIModel.GPT5,
    fallbackModel: AIModel.CLAUDE_SONNET,
    sequentialThinking: false,
    contextThreshold: 40000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['intelligent_routing', 'dynamic_selection', 'optimization'],
    mcpServers: ['claude-flow', 'memory'],
    rationale: 'Intelligent agent coordination with flow management and memory'
  },

  // =============================================================================
  // ADDITIONAL SPECIALIZED AGENTS
  // =============================================================================
  'api-docs': {
    primaryModel: AIModel.GEMINI_PRO,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 200000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['api_documentation', 'openapi_specs', 'technical_writing'],
    mcpServers: ['claude-flow', 'memory', 'ref', 'markitdown'],
    rationale: 'API documentation with large context and markdown tools'
  },

  'perf-analyzer': {
    primaryModel: AIModel.CLAUDE_OPUS,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: false,
    contextThreshold: 80000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['performance_analysis', 'bottleneck_detection', 'optimization'],
    mcpServers: ['claude-flow', 'memory', 'eva'],
    rationale: 'Performance analysis with comprehensive evaluation tools'
  },

  'migration-planner': {
    primaryModel: AIModel.GEMINI_PRO,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: true,
    contextThreshold: 300000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['migration_planning', 'legacy_analysis', 'risk_assessment'],
    mcpServers: ['claude-flow', 'memory', 'sequential-thinking', 'deepwiki'],
    rationale: 'Migration planning with large context and structured reasoning'
  },

  'release-manager': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.CLAUDE_SONNET,
    sequentialThinking: false,
    contextThreshold: 60000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['release_management', 'version_control', 'deployment_automation'],
    mcpServers: ['claude-flow', 'memory', 'github', 'filesystem'],
    rationale: 'Release management with GitHub integration and automation'
  },

  'project-board-sync': {
    primaryModel: AIModel.GEMINI_FLASH,
    fallbackModel: AIModel.CLAUDE_SONNET,
    sequentialThinking: true,
    contextThreshold: 40000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['project_management', 'board_sync', 'task_coordination'],
    mcpServers: ['claude-flow', 'memory', 'plane', 'github', 'sequential-thinking'],
    rationale: 'Project board synchronization with Plane and GitHub integration'
  },

  'repo-architect': {
    primaryModel: AIModel.GEMINI_PRO,
    fallbackModel: AIModel.CLAUDE_OPUS,
    sequentialThinking: false,
    contextThreshold: 400000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['repository_structure', 'codebase_organization', 'architecture_design'],
    mcpServers: ['claude-flow', 'memory', 'deepwiki', 'github'],
    rationale: 'Repository architecture with large context and documentation tools'
  },

  'multi-repo-swarm': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: false,
    contextThreshold: 100000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['multi_repository', 'cross_repo_coordination', 'distributed_development'],
    mcpServers: ['claude-flow', 'memory', 'github', 'deepwiki'],
    rationale: 'Multi-repository coordination with GitHub and documentation access'
  },

  'tdd-london-swarm': {
    primaryModel: AIModel.CLAUDE_OPUS,
    fallbackModel: AIModel.GPT5_CODEX,
    sequentialThinking: false,
    contextThreshold: 80000,
    reasoningComplexity: ReasoningComplexity.HIGH,
    capabilities: ['tdd_methodology', 'london_school', 'mock_driven_development'],
    mcpServers: ['claude-flow', 'memory', 'github', 'eva'],
    rationale: 'TDD London School methodology with superior testing capabilities'
  },

  'base-template-generator': {
    primaryModel: AIModel.GPT5_CODEX,
    fallbackModel: AIModel.GEMINI_PRO,
    sequentialThinking: false,
    contextThreshold: 70000,
    reasoningComplexity: ReasoningComplexity.MEDIUM,
    capabilities: ['template_generation', 'boilerplate_creation', 'project_scaffolding'],
    mcpServers: ['claude-flow', 'memory', 'filesystem', 'markitdown'],
    rationale: 'Template generation with file operations and documentation'
  }
};

/**
 * Get optimal model configuration for an agent
 * @param {string} agentType - The type of agent
 * @returns {object} Model configuration or default
 */
function getAgentModelConfig(agentType) {
  const config = AGENT_MODEL_REGISTRY[agentType];

  if (!config) {
    // Default configuration for unregistered agents
    return {
      primaryModel: AIModel.GPT5,
      fallbackModel: AIModel.CLAUDE_SONNET,
      sequentialThinking: false,
      contextThreshold: 30000,
      reasoningComplexity: ReasoningComplexity.MEDIUM,
      capabilities: ['general_purpose'],
      mcpServers: ['claude-flow', 'memory'],
      rationale: 'Default configuration for unregistered agent type'
    };
  }

  return config;
}

/**
 * Check if agent should use sequential thinking
 * @param {string} agentType - The type of agent
 * @returns {boolean} Whether to enable sequential thinking
 */
function shouldUseSequentialThinking(agentType) {
  const config = getAgentModelConfig(agentType);
  return config.sequentialThinking;
}

/**
 * Get all agents that use a specific model
 * @param {string} model - The AI model
 * @returns {array} List of agent types using this model
 */
function getAgentsByModel(model) {
  return Object.entries(AGENT_MODEL_REGISTRY)
    .filter(([agentType, config]) => config.primaryModel === model)
    .map(([agentType, config]) => agentType);
}

/**
 * Get performance capabilities for an agent
 * @param {string} agentType - The type of agent
 * @returns {array} List of capabilities
 */
function getAgentCapabilities(agentType) {
  const config = getAgentModelConfig(agentType);
  return config.capabilities || [];
}

module.exports = {
  AIModel,
  ReasoningComplexity,
  AGENT_MODEL_REGISTRY,
  getAgentModelConfig,
  shouldUseSequentialThinking,
  getAgentsByModel,
  getAgentCapabilities
};