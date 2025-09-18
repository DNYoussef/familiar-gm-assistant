/**
 * RAG System Configuration for PF2e Knowledge Base
 * Research Princess Domain - Phase 1.2 Implementation
 */

export const RAGConfig = {
  // Archives of Nethys Scraping Configuration
  archivesOfNethys: {
    baseUrl: 'https://2e.aonprd.com',
    endpoints: {
      rules: '/Rules.aspx',
      bestiary: '/Monsters.aspx',
      spells: '/Spells.aspx',
      equipment: '/Equipment.aspx',
      classes: '/Classes.aspx'
    },
    scraping: {
      rateLimit: 1000, // ms between requests
      maxConcurrent: 3,
      retryAttempts: 3,
      userAgent: 'Familiar-GM-Assistant/1.0 (Educational Use)'
    },
    compliance: {
      respectRobotsTxt: true,
      attributionRequired: true,
      nonCommercialUse: true
    }
  },

  // Neo4j Knowledge Graph Configuration
  neo4j: {
    uri: process.env.NEO4J_URI || 'bolt://localhost:7687',
    user: process.env.NEO4J_USER || 'neo4j',
    password: process.env.NEO4J_PASSWORD,
    database: 'pf2e-knowledge',
    schema: {
      nodes: [
        'Rule', 'Spell', 'Monster', 'Class', 'Feat', 'Equipment',
        'Condition', 'Trait', 'Action', 'Skill'
      ],
      relationships: [
        'REFERENCES', 'REQUIRES', 'MODIFIES', 'GRANTS',
        'APPLIES_TO', 'INTERACTS_WITH', 'SUPERSEDES'
      ]
    }
  },

  // Pinecone Vector Database Configuration
  pinecone: {
    apiKey: process.env.PINECONE_API_KEY,
    environment: process.env.PINECONE_ENVIRONMENT || 'us-west1-gcp',
    indexName: 'pf2e-knowledge-vectors',
    dimensions: 1536, // OpenAI text-embedding-ada-002
    metric: 'cosine',
    podType: 'p1.x1',
    replicas: 1,
    shardsPerReplica: 1
  },

  // Embedding Configuration
  embeddings: {
    provider: 'openai',
    model: 'text-embedding-ada-002',
    batchSize: 100,
    maxTokens: 8191,
    dimensions: 1536
  },

  // Caching Configuration
  cache: {
    provider: 'redis',
    host: process.env.REDIS_HOST || 'localhost',
    port: process.env.REDIS_PORT || 6379,
    ttl: 3600, // 1 hour
    maxMemory: '100mb',
    evictionPolicy: 'allkeys-lru',
    keyPrefix: 'familiar:rag:'
  },

  // Query Processing Configuration
  queryProcessing: {
    maxQueryLength: 500,
    topK: 10, // Vector search results
    rerank: true,
    rerankThreshold: 0.75,
    contextWindow: 4000,
    overlapTokens: 200
  },

  // Content Processing
  contentProcessing: {
    chunkSize: 512,
    chunkOverlap: 50,
    minChunkSize: 100,
    maxChunkSize: 1000,
    separators: ['\n\n', '\n', '. ', ' '],
    preprocessors: ['stripHTML', 'normalizeWhitespace', 'extractStructure']
  },

  // Performance Optimization
  performance: {
    batchProcessing: true,
    parallelProcessing: 4,
    memoryLimit: '2GB',
    diskCache: true,
    compressionLevel: 6
  }
};

export default RAGConfig;