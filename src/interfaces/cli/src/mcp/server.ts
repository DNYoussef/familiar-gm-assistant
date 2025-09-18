import express from 'express';
import cors from 'cors';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import * as fs from 'fs-extra';
import axios from 'axios';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google-ai/generativelanguage';
import { CohereClient } from 'cohere-ai';
import Replicate from 'replicate';

export interface MCPServerConfig {
  port: number;
  host: string;
  configFile?: string;
  enableContext7: boolean;
}

export interface AIProvider {
  name: string;
  enabled: boolean;
  apiKey: string;
  model: string;
  client?: any;
}

export interface Context7Config {
  enabled: boolean;
  endpoint: string;
  features: string[];
  apiKey?: string;
}

/**
 * MCP Server for Connascence AI Integration
 * 
 * Provides unified interface for multiple AI providers with Context7 integration
 * for accurate, up-to-date API configuration and rate limiting
 */
export class MCPServer {
  private app: express.Application;
  private server: any;
  private wss: WebSocketServer;
  private config: any;
  private providers: Map<string, AIProvider> = new Map();
  private context7: Context7Config | null = null;

  constructor(private serverConfig: MCPServerConfig) {
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();
  }

  async start(): Promise<void> {
    await this.loadConfiguration();
    await this.initializeProviders();
    await this.initializeContext7();

    this.server = createServer(this.app);
    this.wss = new WebSocketServer({ server: this.server });

    this.setupWebSocket();

    return new Promise((resolve, reject) => {
      this.server.listen(this.serverConfig.port, this.serverConfig.host, () => {
        console.log(`MCP Server running on http://${this.serverConfig.host}:${this.serverConfig.port}`);
        resolve();
      }).on('error', reject);
    });
  }

  async stop(): Promise<void> {
    if (this.wss) {
      this.wss.close();
    }
    if (this.server) {
      this.server.close();
    }
  }

  private setupMiddleware(): void {
    this.app.use(cors());
    this.app.use(express.json());
    
    // Request logging
    this.app.use((req, res, next) => {
      console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
      next();
    });
  }

  private setupRoutes(): void {
    // Health check
    this.app.get('/api/health', (req, res) => {
      res.json({ 
        status: 'ok', 
        providers: Array.from(this.providers.keys()),
        context7: this.context7?.enabled || false,
        timestamp: new Date().toISOString()
      });
    });

    // AI fix endpoint
    this.app.post('/api/ai/fix', async (req, res) => {
      try {
        const { finding, context, options } = req.body;
        const fix = await this.generateFix(finding, context, options);
        res.json({ fix });
      } catch (error) {
        console.error('Fix generation error:', error);
        res.status(500).json({ error: 'Fix generation failed' });
      }
    });

    // AI suggestions endpoint
    this.app.post('/api/ai/suggestions', async (req, res) => {
      try {
        const { finding, context, options } = req.body;
        const suggestions = await this.generateSuggestions(finding, context, options);
        res.json({ suggestions });
      } catch (error) {
        console.error('Suggestions generation error:', error);
        res.status(500).json({ error: 'Suggestions generation failed' });
      }
    });

    // AI explanation endpoint
    this.app.post('/api/ai/explain', async (req, res) => {
      try {
        const { finding, options } = req.body;
        const explanation = await this.generateExplanation(finding, options);
        res.json({ explanation });
      } catch (error) {
        console.error('Explanation generation error:', error);
        res.status(500).json({ error: 'Explanation generation failed' });
      }
    });

    // Provider configuration endpoint
    this.app.get('/api/providers', (req, res) => {
      const providerInfo = Array.from(this.providers.entries()).map(([name, provider]) => ({
        name,
        enabled: provider.enabled,
        model: provider.model,
        hasApiKey: !!provider.apiKey
      }));
      res.json({ providers: providerInfo });
    });

    // Update provider configuration
    this.app.post('/api/providers/:name/configure', async (req, res) => {
      try {
        const { name } = req.params;
        const { enabled, apiKey, model } = req.body;

        if (!this.providers.has(name)) {
          return res.status(404).json({ error: 'Provider not found' });
        }

        await this.updateProvider(name, { enabled, apiKey, model });
        res.json({ message: 'Provider updated successfully' });
      } catch (error) {
        console.error('Provider update error:', error);
        res.status(500).json({ error: 'Provider update failed' });
      }
    });

    // Context7 integration endpoints
    this.app.get('/api/context7/status', (req, res) => {
      res.json({ 
        context7: this.context7,
        available: !!this.context7?.enabled
      });
    });

    this.app.post('/api/context7/refresh-apis', async (req, res) => {
      try {
        if (!this.context7?.enabled) {
          return res.status(400).json({ error: 'Context7 not enabled' });
        }

        const updatedApis = await this.refreshContext7APIs();
        res.json({ apis: updatedApis });
      } catch (error) {
        console.error('Context7 API refresh error:', error);
        res.status(500).json({ error: 'Failed to refresh APIs' });
      }
    });
  }

  private setupWebSocket(): void {
    this.wss.on('connection', (ws) => {
      console.log('WebSocket connection established');

      ws.on('message', async (data) => {
        try {
          const message = JSON.parse(data.toString());
          await this.handleWebSocketMessage(ws, message);
        } catch (error) {
          ws.send(JSON.stringify({ error: 'Invalid message format' }));
        }
      });

      ws.on('close', () => {
        console.log('WebSocket connection closed');
      });
    });
  }

  private async handleWebSocketMessage(ws: any, message: any): Promise<void> {
    const { type, id, data } = message;

    try {
      switch (type) {
        case 'chat':
          const response = await this.handleChatMessage(data);
          ws.send(JSON.stringify({ type: 'chat_response', id, data: response }));
          break;

        case 'stream_fix':
          await this.streamFix(ws, id, data);
          break;

        default:
          ws.send(JSON.stringify({ type: 'error', id, error: 'Unknown message type' }));
      }
    } catch (error) {
      ws.send(JSON.stringify({ type: 'error', id, error: error instanceof Error ? error.message : 'Unknown error' }));
    }
  }

  private async loadConfiguration(): Promise<void> {
    const configFile = this.serverConfig.configFile || 'mcp-config.json';
    
    try {
      this.config = await fs.readJson(configFile);
    } catch (error) {
      console.warn(`Could not load config file ${configFile}, using defaults`);
      this.config = this.getDefaultConfig();
    }
  }

  private getDefaultConfig(): any {
    return {
      ai: {
        providers: {
          openai: { enabled: false, model: 'gpt-4' },
          anthropic: { enabled: false, model: 'claude-3-sonnet-20240229' },
          google: { enabled: false, model: 'gemini-pro' },
          cohere: { enabled: false, model: 'command' },
          replicate: { enabled: false, model: 'meta/llama-2-70b-chat' }
        },
        context7: {
          enabled: this.serverConfig.enableContext7,
          endpoint: 'https://api.context7.ai/v1',
          features: ['api-discovery', 'rate-limiting', 'cost-tracking']
        }
      }
    };
  }

  private async initializeProviders(): Promise<void> {
    const providers = this.config.ai?.providers || {};

    for (const [name, config] of Object.entries(providers)) {
      const providerConfig = config as any;
      
      const provider: AIProvider = {
        name,
        enabled: providerConfig.enabled || false,
        apiKey: process.env[`${name.toUpperCase()}_API_KEY`] || providerConfig.apiKey || '',
        model: providerConfig.model || 'default',
        client: null
      };

      if (provider.enabled && provider.apiKey) {
        provider.client = await this.createProviderClient(name, provider);
      }

      this.providers.set(name, provider);
    }
  }

  private async createProviderClient(name: string, provider: AIProvider): Promise<any> {
    switch (name) {
      case 'openai':
        return new OpenAI({ apiKey: provider.apiKey });

      case 'anthropic':
        return new Anthropic({ apiKey: provider.apiKey });

      case 'google':
        return new GoogleGenerativeAI(provider.apiKey);

      case 'cohere':
        return new CohereClient({ token: provider.apiKey });

      case 'replicate':
        return new Replicate({ auth: provider.apiKey });

      default:
        throw new Error(`Unknown provider: ${name}`);
    }
  }

  private async initializeContext7(): Promise<void> {
    const context7Config = this.config.ai?.context7;
    
    if (context7Config?.enabled) {
      this.context7 = {
        enabled: true,
        endpoint: context7Config.endpoint || 'https://api.context7.ai/v1',
        features: context7Config.features || ['api-discovery'],
        apiKey: process.env.CONTEXT7_API_KEY || context7Config.apiKey
      };

      // Initialize Context7 connection
      if (this.context7.apiKey) {
        try {
          await this.testContext7Connection();
          console.log('Context7 integration initialized');
        } catch (error) {
          console.warn('Context7 connection failed:', error);
          this.context7.enabled = false;
        }
      }
    }
  }

  private async testContext7Connection(): Promise<void> {
    if (!this.context7?.enabled || !this.context7.apiKey) {
      throw new Error('Context7 not properly configured');
    }

    const response = await axios.get(`${this.context7.endpoint}/health`, {
      headers: {
        'Authorization': `Bearer ${this.context7.apiKey}`,
        'Content-Type': 'application/json'
      }
    });

    if (response.status !== 200) {
      throw new Error('Context7 health check failed');
    }
  }

  private async refreshContext7APIs(): Promise<any[]> {
    if (!this.context7?.enabled || !this.context7.apiKey) {
      throw new Error('Context7 not enabled');
    }

    const response = await axios.get(`${this.context7.endpoint}/apis/discover`, {
      headers: {
        'Authorization': `Bearer ${this.context7.apiKey}`,
        'Content-Type': 'application/json'
      },
      params: {
        category: 'ai-models',
        include_rates: true,
        include_limits: true
      }
    });

    return response.data.apis || [];
  }

  private async generateFix(finding: any, context: string, options: any): Promise<any> {
    const provider = this.getAvailableProvider();
    if (!provider) {
      throw new Error('No AI providers available');
    }

    // Check if we have enhanced pipeline data
    const enhancedContext = await this.getEnhancedPipelineContext(finding);
    const prompt = this.buildFixPrompt(finding, context, options, enhancedContext);
    const response = await this.callProvider(provider, prompt);

    return {
      patch: response.code || '',
      confidence: response.confidence || 75,
      description: response.description || 'AI-generated fix',
      safety: response.safety || 'caution',
      enhanced_recommendations: enhancedContext.smart_recommendations || [],
      cross_phase_analysis: enhancedContext.correlations || []
    };
  }

  private async getEnhancedPipelineContext(finding: any): Promise<any> {
    try {
      const { spawn } = require('child_process');
      const path = require('path');
      
      // Get file path from finding object
      const filePath = finding.file || finding.path || process.cwd();
      
      // Find the enhanced unified analyzer path
      const analyzerBasePath = process.env.CONNASCENCE_ANALYZER_PATH || '../../../analyzer';
      const analyzerPath = path.resolve(__dirname, analyzerBasePath, 'unified_analyzer.py');
      
      // Build enhanced analysis command with cross-phase features enabled
      const args = [
        analyzerPath,
        '--path', filePath,
        '--format', 'json',
        '--enable-correlations',
        '--enable-audit-trail', 
        '--enable-smart-recommendations',
        '--policy', 'standard' // Default policy for MCP context
      ];
      
      return new Promise((resolve, reject) => {
        const process = spawn('python', args, {
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: path.dirname(analyzerPath),
          timeout: 30000 // 30 second timeout
        });
        
        let stdout = '';
        let stderr = '';
        
        process.stdout.on('data', (data) => {
          stdout += data.toString();
        });
        
        process.stderr.on('data', (data) => {
          stderr += data.toString();
        });
        
        process.on('close', (code) => {
          if (code !== 0) {
            console.warn(`Enhanced pipeline process failed with code ${code}: ${stderr}`);
            // Return minimal context on failure but don't fail completely
            resolve({
              smart_recommendations: [],
              correlations: [],
              audit_trail: [],
              cross_phase_analysis: false,
              error: `Pipeline exit code: ${code}`
            });
            return;
          }
          
          try {
            const result = JSON.parse(stdout);
            
            // Extract and format enhanced context for MCP use
            const enhancedContext = {
              smart_recommendations: result.smart_recommendations || [],
              correlations: result.correlations || [],
              audit_trail: result.audit_trail || [],
              cross_phase_analysis: result.cross_phase_analysis || false,
              canonical_policy: result.canonical_policy,
              components_used: result.components_used || {},
              policy_config: result.policy_config || {}
            };
            
            console.log(`Enhanced pipeline context retrieved: ${enhancedContext.correlations.length} correlations, ${enhancedContext.smart_recommendations.length} recommendations`);
            resolve(enhancedContext);
            
          } catch (parseError) {
            console.warn(`Failed to parse enhanced pipeline JSON output: ${parseError}`);
            console.warn(`Raw stdout: ${stdout.substring(0, 500)}...`);
            
            // Return partial context on parse failure
            resolve({
              smart_recommendations: [],
              correlations: [],
              audit_trail: [],
              cross_phase_analysis: false,
              parse_error: parseError.message
            });
          }
        });
        
        process.on('error', (error) => {
          console.warn(`Failed to spawn enhanced pipeline process: ${error}`);
          resolve({
            smart_recommendations: [],
            correlations: [],
            audit_trail: [],
            cross_phase_analysis: false,
            spawn_error: error.message
          });
        });
        
        // Handle timeout explicitly
        const timeoutId = setTimeout(() => {
          console.warn('Enhanced pipeline analysis timed out after 30 seconds');
          process.kill('SIGTERM');
          resolve({
            smart_recommendations: [],
            correlations: [],
            audit_trail: [],
            cross_phase_analysis: false,
            timeout: true
          });
        }, 30000);
        
        process.on('close', () => {
          clearTimeout(timeoutId);
        });
      });
      
    } catch (error) {
      console.warn('Enhanced pipeline context setup error:', error);
      return {
        smart_recommendations: [],
        correlations: [],
        audit_trail: [],
        cross_phase_analysis: false,
        setup_error: error.message
      };
    }
  }

  private async generateSuggestions(finding: any, context: string, options: any): Promise<any[]> {
    const provider = this.getAvailableProvider();
    if (!provider) {
      return [];
    }

    // Get enhanced pipeline context for better suggestions
    const enhancedContext = await this.getEnhancedPipelineContext(finding);
    const prompt = this.buildSuggestionsPrompt(finding, context, options, enhancedContext);
    const response = await this.callProvider(provider, prompt);

    // Enhance suggestions with pipeline context
    const suggestions = response.suggestions || [];
    if (enhancedContext?.smart_recommendations?.length > 0) {
      // Add smart recommendations as additional suggestions
      const smartSuggestions = enhancedContext.smart_recommendations.slice(0, 2).map((rec: any, index: number) => ({
        technique: `Smart Rec: ${rec.category || 'Architectural'}`,
        description: rec.description,
        confidence: 85,
        complexity: rec.effort?.toLowerCase() || 'medium',
        risk: rec.priority === 'high' ? 'low' : 'medium',
        source: 'enhanced_pipeline'
      }));
      
      suggestions.push(...smartSuggestions);
    }

    return suggestions;
  }

  private async generateExplanation(finding: any, options: any): Promise<any> {
    const provider = this.getAvailableProvider();
    if (!provider) {
      throw new Error('No AI providers available');
    }

    const prompt = this.buildExplanationPrompt(finding, options);
    const response = await this.callProvider(provider, prompt);

    return {
      explanation: response.explanation || 'AI-generated explanation',
      impact: response.impact || 'Potential code quality impact',
      recommendation: response.recommendation || 'Consider refactoring'
    };
  }

  private getAvailableProvider(): AIProvider | null {
    for (const provider of this.providers.values()) {
      if (provider.enabled && provider.client) {
        return provider;
      }
    }
    return null;
  }

  private async callProvider(provider: AIProvider, prompt: string): Promise<any> {
    // Rate limiting with Context7 if available
    if (this.context7?.enabled) {
      await this.checkContext7RateLimit(provider.name);
    }

    switch (provider.name) {
      case 'openai':
        return this.callOpenAI(provider, prompt);
      case 'anthropic':
        return this.callAnthropic(provider, prompt);
      case 'google':
        return this.callGoogle(provider, prompt);
      case 'cohere':
        return this.callCohere(provider, prompt);
      case 'replicate':
        return this.callReplicate(provider, prompt);
      default:
        throw new Error(`Provider ${provider.name} not implemented`);
    }
  }

  private async callOpenAI(provider: AIProvider, prompt: string): Promise<any> {
    const response = await provider.client.chat.completions.create({
      model: provider.model,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.7,
      max_tokens: 1000
    });

    return this.parseAIResponse(response.choices[0]?.message?.content || '');
  }

  private async callAnthropic(provider: AIProvider, prompt: string): Promise<any> {
    const response = await provider.client.messages.create({
      model: provider.model,
      max_tokens: 1000,
      messages: [{ role: 'user', content: prompt }]
    });

    return this.parseAIResponse(response.content[0]?.text || '');
  }

  private async callGoogle(provider: AIProvider, prompt: string): Promise<any> {
    const response = await provider.client.generateText({
      model: `models/${provider.model}`,
      prompt: { text: prompt },
      temperature: 0.7,
      candidateCount: 1,
      maxOutputTokens: 1000
    });

    return this.parseAIResponse(response.candidates?.[0]?.output || '');
  }

  private async callCohere(provider: AIProvider, prompt: string): Promise<any> {
    const response = await provider.client.generate({
      model: provider.model,
      prompt: prompt,
      max_tokens: 1000,
      temperature: 0.7
    });

    return this.parseAIResponse(response.generations[0]?.text || '');
  }

  private async callReplicate(provider: AIProvider, prompt: string): Promise<any> {
    const response = await provider.client.run(provider.model, {
      input: {
        prompt: prompt,
        max_new_tokens: 1000,
        temperature: 0.7
      }
    });

    return this.parseAIResponse(Array.isArray(response) ? response.join('') : response);
  }

  private parseAIResponse(text: string): any {
    // Simple parsing - in production, you'd want more sophisticated parsing
    try {
      // Try to extract JSON if present
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
    } catch (e) {
      // Fall back to text parsing
    }

    return {
      code: text.includes('```') ? text.match(/```[\s\S]*?```/)?.[0]?.replace(/```/g, '') : '',
      description: text.split('\n')[0] || text.substring(0, 100),
      confidence: 75
    };
  }

  private async checkContext7RateLimit(provider: string): Promise<void> {
    if (!this.context7?.enabled || !this.context7.apiKey) {
      return;
    }

    try {
      const response = await axios.post(`${this.context7.endpoint}/rate-limit/check`, {
        provider,
        tokens: 1000 // Estimated token usage
      }, {
        headers: {
          'Authorization': `Bearer ${this.context7.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.data.allowed) {
        throw new Error(`Rate limit exceeded for ${provider}`);
      }
    } catch (error) {
      console.warn('Context7 rate limit check failed:', error);
    }
  }

  private buildFixPrompt(finding: any, context: string, options: any, enhancedContext?: any): string {
    let prompt = `You are a senior software architect with access to advanced cross-phase connascence analysis. Generate an optimal fix for this violation:

Type: ${finding.type}
Message: ${finding.message}
Severity: ${finding.severity}
File: ${finding.file || 'unknown'}
Line: ${finding.line || 'unknown'}

Context:
${context}`;

    // Add enhanced pipeline context with detailed recommendations
    if (enhancedContext && (enhancedContext.correlations?.length > 0 || enhancedContext.smart_recommendations?.length > 0)) {
      prompt += `\n\n=== ENHANCED ARCHITECTURAL ANALYSIS ===`;
      
      // Add cross-phase correlations with architectural insights
      if (enhancedContext.correlations?.length > 0) {
        prompt += `\n\nCross-Phase Correlations (${enhancedContext.correlations.length} found):`;
        enhancedContext.correlations.slice(0, 5).forEach((corr: any, index: number) => {
          const score = (corr.correlation_score * 100).toFixed(1);
          const priority = corr.priority || 'medium';
          prompt += `\n${index + 1}. [${priority.toUpperCase()}] ${corr.analyzer1} <-> ${corr.analyzer2} (${score}% correlation)`;
          prompt += `\n   Impact: ${corr.description}`;
          if (corr.remediation_impact) {
            prompt += `\n   Remediation: ${corr.remediation_impact}`;
          }
          if (corr.affected_files?.length > 0) {
            prompt += `\n   Affects: ${corr.affected_files.slice(0, 3).join(', ')}${corr.affected_files.length > 3 ? '...' : ''}`;
          }
        });
      }
      
      // Add smart architectural recommendations with detailed context
      if (enhancedContext.smart_recommendations?.length > 0) {
        prompt += `\n\nSmart Architectural Recommendations (${enhancedContext.smart_recommendations.length} generated):`;
        enhancedContext.smart_recommendations.slice(0, 4).forEach((rec: any, index: number) => {
          const priority = rec.priority || 'medium';
          const category = rec.category || 'General';
          const impact = rec.impact || 'unknown';
          const effort = rec.effort || 'unknown';
          
          prompt += `\n${index + 1}. [${priority.toUpperCase()} PRIORITY] ${category}`;
          prompt += `\n   Recommendation: ${rec.description}`;
          prompt += `\n   Impact: ${impact} | Effort: ${effort}`;
          
          if (rec.rationale) {
            prompt += `\n   Rationale: ${rec.rationale}`;
          }
          if (rec.implementation_notes) {
            prompt += `\n   Implementation: ${rec.implementation_notes}`;
          }
        });
      }
      
      // Add policy and component information
      if (enhancedContext.canonical_policy) {
        prompt += `\n\nActive Policy: ${enhancedContext.canonical_policy}`;
      }
      if (enhancedContext.components_used) {
        const activeComponents = Object.entries(enhancedContext.components_used)
          .filter(([_, enabled]) => enabled)
          .map(([component, _]) => component);
        if (activeComponents.length > 0) {
          prompt += `\nActive Analyzers: ${activeComponents.join(', ')}`;
        }
      }
    }

    prompt += `\n\n=== REQUIREMENTS ===
- Generate working code that eliminates the connascence violation
- Preserve all functionality while reducing coupling strength
- Apply architectural patterns identified in enhanced analysis
- Consider cross-phase correlations to avoid introducing new violations
- Leverage smart recommendations for optimal architectural decisions
- Include detailed confidence assessment (0-100)
- Assess safety level: safe (no risk), caution (minor risk), unsafe (breaking change risk)
${enhancedContext?.correlations?.length > 0 ? '- Account for cross-analyzer correlations to prevent cascade effects' : ''}
${enhancedContext?.smart_recommendations?.length > 0 ? '- Align with smart architectural recommendations for long-term maintainability' : ''}

Response format:
{
  "code": "// Complete fixed code with comments explaining architectural decisions",
  "description": "Detailed explanation of changes and architectural improvements",
  "confidence": 85,
  "safety": "safe",
  "architectural_pattern": "Design pattern or principle applied"${enhancedContext?.correlations?.length > 0 || enhancedContext?.smart_recommendations?.length > 0 ? ',\n  "enhanced_rationale": "How enhanced analysis influenced this solution",\n  "correlation_impact": "How this fix addresses cross-phase correlations",\n  "recommendation_alignment": "Which smart recommendations this implements"' : ''}
}`;

    return prompt;
  }

  private buildSuggestionsPrompt(finding: any, context: string, options: any, enhancedContext?: any): string {
    let prompt = `Generate comprehensive refactoring suggestions for this connascence violation using advanced architectural analysis:

Type: ${finding.type}
Message: ${finding.message}
Severity: ${finding.severity}
File: ${finding.file || 'unknown'}

Context:
${context}`;

    // Add enhanced analysis context for better suggestions
    if (enhancedContext && (enhancedContext.correlations?.length > 0 || enhancedContext.smart_recommendations?.length > 0)) {
      prompt += `\n\n=== ENHANCED ARCHITECTURAL CONTEXT ===`;
      
      if (enhancedContext.correlations?.length > 0) {
        prompt += `\nCross-Phase Correlations:`;
        enhancedContext.correlations.slice(0, 3).forEach((corr: any, index: number) => {
          const score = (corr.correlation_score * 100).toFixed(1);
          prompt += `\n[U+2022] ${corr.analyzer1} <-> ${corr.analyzer2} (${score}%): ${corr.description}`;
        });
      }
      
      if (enhancedContext.smart_recommendations?.length > 0) {
        prompt += `\nSmart Architectural Recommendations:`;
        enhancedContext.smart_recommendations.slice(0, 3).forEach((rec: any, index: number) => {
          prompt += `\n[U+2022] [${rec.priority || 'medium'}] ${rec.category}: ${rec.description}`;
        });
      }
    }

    prompt += `

Generate 4-6 different refactoring approaches considering:
- Traditional connascence reduction techniques
- Modern architectural patterns
- Cross-phase impact analysis
${enhancedContext?.correlations?.length > 0 ? '- Correlation cascade effects' : ''}
${enhancedContext?.smart_recommendations?.length > 0 ? '- Smart architectural recommendations alignment' : ''}

For each suggestion provide:
- Technique name (clear, descriptive)
- Detailed description with implementation steps
- Confidence level (0-100)
- Complexity assessment (low/medium/high)
- Risk level (low/medium/high)
- Expected impact on coupling
${enhancedContext ? '- Architectural pattern category' : ''}

Response format:
{
  "suggestions": [
    {
      "technique": "Extract Constant",
      "description": "Move magic literal to named constant to reduce connascence of literal",
      "confidence": 90,
      "complexity": "low",
      "risk": "low",
      "coupling_impact": "reduces connascence of literal"${enhancedContext ? ',\n      "architectural_pattern": "Information Hiding",\n      "enhanced_rationale": "Aligns with smart recommendations for separation of concerns"' : ''}
    }
  ]
}`;

    return prompt;
  }

  private buildExplanationPrompt(finding: any, options: any): string {
    return `Explain this connascence violation in detail:

Type: ${finding.type}
Message: ${finding.message}
Severity: ${finding.severity}

Provide:
- Clear explanation of what connascence means here
- Why this specific violation impacts code quality
- Recommended approaches to fix it
- Examples of better patterns

Response format:
{
  "explanation": "detailed explanation",
  "impact": "impact on code quality",
  "recommendation": "how to fix it"
}`;
  }

  private async updateProvider(name: string, updates: any): Promise<void> {
    const provider = this.providers.get(name);
    if (!provider) {
      throw new Error(`Provider ${name} not found`);
    }

    if (updates.enabled !== undefined) {
      provider.enabled = updates.enabled;
    }
    if (updates.apiKey) {
      provider.apiKey = updates.apiKey;
    }
    if (updates.model) {
      provider.model = updates.model;
    }

    // Reinitialize client if enabled and has API key
    if (provider.enabled && provider.apiKey) {
      provider.client = await this.createProviderClient(name, provider);
    } else {
      provider.client = null;
    }

    this.providers.set(name, provider);

    // Update config file
    if (!this.config.ai) this.config.ai = {};
    if (!this.config.ai.providers) this.config.ai.providers = {};
    if (!this.config.ai.providers[name]) this.config.ai.providers[name] = {};

    this.config.ai.providers[name] = {
      ...this.config.ai.providers[name],
      enabled: provider.enabled,
      model: provider.model
    };

    // Don't save API key to config file for security
    const configFile = this.serverConfig.configFile || 'mcp-config.json';
    await fs.writeJson(configFile, this.config, { spaces: 2 });
  }

  private async handleChatMessage(data: any): Promise<string> {
    const { message, context } = data;
    const provider = this.getAvailableProvider();
    
    if (!provider) {
      return "No AI providers are currently configured. Please configure an API key in the AI Configuration sidebar.";
    }

    const prompt = `You are a helpful coding assistant specializing in connascence analysis. 
    
User question: ${message}
Context: ${JSON.stringify(context)}

Provide a helpful, accurate response about connascence, code quality, or refactoring advice.`;

    const response = await this.callProvider(provider, prompt);
    return response.description || response.explanation || "I couldn't generate a response. Please try again.";
  }

  private async streamFix(ws: any, id: string, data: any): Promise<void> {
    // Stream the fix generation process to the client
    ws.send(JSON.stringify({ type: 'stream_start', id }));
    
    try {
      ws.send(JSON.stringify({ type: 'stream_progress', id, message: 'Analyzing violation...' }));
      
      const fix = await this.generateFix(data.finding, data.context, data.options);
      
      ws.send(JSON.stringify({ type: 'stream_progress', id, message: 'Generated fix successfully' }));
      ws.send(JSON.stringify({ type: 'stream_complete', id, data: fix }));
      
    } catch (error) {
      ws.send(JSON.stringify({ type: 'stream_error', id, error: error instanceof Error ? error.message : 'Unknown error' }));
    }
  }
}