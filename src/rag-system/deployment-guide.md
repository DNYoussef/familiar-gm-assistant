# Familiar RAG System Deployment Guide

## Overview

This guide covers deploying the Hybrid RAG System for the Familiar project, enabling intelligent Pathfinder 2e rules assistance with GraphRAG + Vector search capabilities.

## Quick Start

### 1. Prerequisites

```bash
# Node.js 18+ required
node --version  # Should be >= 18.0.0

# Install dependencies
npm install

# Copy environment configuration
cp .env.example .env
```

### 2. Database Setup

#### Neo4j (Graph Database)
```bash
# Using Docker
docker run \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:latest

# Or install locally: https://neo4j.com/download/
```

#### Pinecone (Vector Database)
```bash
# 1. Sign up at https://www.pinecone.io/
# 2. Create index: 'pathfinder-2e-rules'
# 3. Dimension: 1536 (for OpenAI embeddings)
# 4. Metric: cosine
```

#### Redis (Caching)
```bash
# Using Docker
docker run -p 6379:6379 redis:alpine

# Or install locally: https://redis.io/download
```

### 3. Configuration

Edit `.env` with your API keys and database URLs:

```env
# Required for basic functionality
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
NEO4J_PASSWORD=...

# Optional for cost optimization
GEMINI_API_KEY=...
CLAUDE_API_KEY=...
```

### 4. Data Ingestion

```bash
# Scrape Archives of Nethys (sample data)
npm run scrape-nethys

# Build knowledge graph
npm run build-graph

# Index vector data
npm run index-data

# Validate system
npm run validate-system
```

### 5. Start System

```bash
# Development mode
npm run dev

# Production mode
npm start
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Familiar RAG System                      │
├─────────────────────────────────────────────────────────────┤
│  Query Classification → Optimal RAG Selection               │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Vector    │  │    Graph    │  │     Archives        │  │
│  │    RAG      │  │     RAG     │  │   Connector         │  │
│  │             │  │             │  │                     │  │
│  │ Pinecone +  │  │ Neo4j +     │  │ Web Scraping +      │  │
│  │ OpenAI      │  │ Cypher      │  │ Rate Limiting       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │           Cost Optimizer (<$0.015/session)             │  │
│  │  • Model Selection  • Context Optimization             │  │
│  │  • Token Management • Budget Tracking                  │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## API Usage

### Basic Query
```javascript
const { HybridRAGSystem } = require('./hybrid-rag-core');

const rag = new HybridRAGSystem({
    costLimit: 0.015,
    neo4jUri: process.env.NEO4J_URI,
    pineconeApiKey: process.env.PINECONE_API_KEY
});

const result = await rag.processQuery(
    "How does the Shield spell interact with Armor Class?",
    { characterLevel: 5 }
);

console.log(result.answer);
console.log(result.sources);
```

### Advanced Configuration
```javascript
const rag = new HybridRAGSystem({
    costLimit: 0.010,              // Lower cost limit
    cacheEnabled: true,            // Enable caching
    fallbackToVector: true,        // Fallback strategy
    performanceThreshold: 1500,    // 1.5s max response

    // Database connections
    neo4jUri: 'neo4j://localhost:7687',
    pineconeApiKey: 'pc-...',
    indexName: 'pathfinder-2e-rules'
});
```

## Integration with Familiar

### Foundry VTT Module Integration
```javascript
// In your Foundry module
import { HybridRAGSystem } from './rag-system/hybrid-rag-core.js';

class FamiliarGMAssistant {
    constructor() {
        this.rag = new HybridRAGSystem({
            costLimit: 0.008,  // Conservative for production
            cacheEnabled: true
        });
    }

    async answerRulesQuestion(question, context) {
        try {
            const result = await this.rag.processQuery(question, {
                characterLevel: context.level,
                playerCount: context.party?.length
            });

            return {
                answer: result.answer,
                sources: result.sources,
                confidence: result.confidence,
                cost: result.performance.costEstimate
            };
        } catch (error) {
            console.error('RAG query failed:', error);
            return { answer: 'Unable to process question', error: true };
        }
    }
}
```

### Chat Interface Integration
```javascript
// Raven familiar chat handler
async function handleRavenQuery(userMessage) {
    const result = await familiar.answerRulesQuestion(userMessage, {
        level: game.user.character?.level || 1,
        party: game.users.contents.filter(u => u.active)
    });

    if (result.error) {
        return "I'm having trouble accessing the rules database right now.";
    }

    return {
        message: result.answer,
        sources: result.sources.map(s => ({
            title: s.title,
            page: s.pageNumber,
            book: s.bookReference,
            url: s.url
        })),
        confidence: result.confidence
    };
}
```

## Performance Optimization

### Cost Management
- **Target**: <$0.015 per session
- **Strategy**: Gemini Flash for 70% of queries, GPT-4 for complex reasoning
- **Caching**: 80%+ cache hit rate for common queries
- **Monitoring**: Real-time cost tracking with circuit breakers

### Response Time Optimization
- **Target**: <2 seconds average response time
- **Vector Search**: <500ms for simple lookups
- **Graph Traversal**: <1.5s for complex reasoning
- **Caching**: <100ms for cached responses

### Scalability
```javascript
// Connection pooling for high load
const rag = new HybridRAGSystem({
    neo4jConfig: {
        maxConnectionPoolSize: 50,
        connectionTimeout: 5000
    },
    pineconeConfig: {
        maxRetries: 3,
        timeout: 10000
    }
});
```

## Monitoring & Maintenance

### Health Checks
```bash
# System health
curl http://localhost:3000/health

# Performance metrics
curl http://localhost:3000/metrics

# Cost tracking
curl http://localhost:3000/api/costs/session
```

### Log Analysis
```bash
# View system logs
docker logs familiar-rag-system

# Query performance
grep "Query performance" logs/app.log

# Cost tracking
grep "Cost exceeded" logs/app.log
```

### Data Updates
```bash
# Update knowledge graph (weekly)
npm run scrape-nethys
npm run build-graph

# Rebuild vector index (when SRD updates)
npm run index-data

# Clear cache (after updates)
redis-cli FLUSHALL
```

## Troubleshooting

### Common Issues

#### High Costs
```javascript
// Check cost breakdown
const summary = rag.costOptimizer.getSessionSummary();
console.log(summary.costs.breakdown);

// Enable aggressive optimization
rag.costOptimizer.setOptimization('preferCheaperModels', true);
```

#### Slow Responses
```javascript
// Check performance metrics
const metrics = rag.getPerformanceMetrics();
console.log('Average response time:', metrics.averageResponseTime);

// Optimize context length
rag.config.maxContextLength = 2000;
```

#### Connection Issues
```bash
# Test database connections
npm run validate-system

# Check Redis connection
redis-cli ping

# Test Neo4j connection
cypher-shell -u neo4j -p your_password "MATCH (n) RETURN count(n) LIMIT 1"
```

## Security Considerations

### API Key Management
- Store all keys in environment variables
- Use different keys for development/production
- Rotate keys regularly
- Monitor usage for anomalies

### Data Privacy
- No user data stored in knowledge graph
- Query logs scrubbed of personal information
- HTTPS/TLS for all external connections

### Rate Limiting
```javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // limit each IP to 100 requests per windowMs
});

app.use('/api/', limiter);
```

## Production Deployment

### Docker Deployment
```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
EXPOSE 3000

CMD ["npm", "start"]
```

### Environment Variables
```bash
# Production settings
NODE_ENV=production
LOG_LEVEL=warn
ENABLE_METRICS=true
SESSION_COST_LIMIT=0.010
```

### Monitoring Setup
```yaml
# docker-compose.yml
version: '3.8'
services:
  rag-system:
    build: .
    environment:
      - NODE_ENV=production
    ports:
      - "3000:3000"

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  neo4j:
    image: neo4j:latest
    environment:
      - NEO4J_AUTH=neo4j/production_password
    ports:
      - "7687:7687"
```

## Support

- **Issues**: Report at [GitHub Issues](https://github.com/familiar-project/rag-system/issues)
- **Documentation**: [Wiki](https://github.com/familiar-project/rag-system/wiki)
- **Community**: [Discord](https://discord.gg/familiar-project)