# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Familiar** is an AI-powered Game Master assistant for Foundry VTT using a Queen-Princess-Drone swarm architecture with 21 specialized agents across 6 domains. The system integrates GraphRAG + Vector search for Pathfinder 2e rules assistance, content generation, and AI artwork creation.

**Current Status**: Loop 3 Infrastructure & Quality Phase (83.3% Production Ready)
- Loop 1 ✅: Planning reduced risk from 15% to 2.8%
- Loop 2 ✅: Development deployed 21 agents, validated core systems as authentic
- Loop 3 🔄: CI/CD pipeline, MECE task distribution, theater elimination

## Essential Commands

### Development & Testing
```bash
# Development
npm start                    # Start API server (src/api-server.js)
npm run dev                  # Development mode with nodemon
npm run build:production     # Full production build with optimization

# Testing - JavaScript/Node.js
npm test                     # Run all Jest tests
npm run test:unit            # Unit tests only
npm run test:integration     # Integration tests only
npm run test:e2e             # End-to-end tests
npm run test:coverage        # With coverage report

# Testing - Python Components
python -m pytest tests/      # Run Python test suite
python comprehensive_analysis_engine.py --mode theater-detection
python comprehensive_analysis_engine.py --mode production-assessment

# Quality & Validation
npm run lint                 # ESLint for JavaScript
npm run lint:fix             # Auto-fix linting issues
npm run typecheck            # Generate JSDoc documentation
npm run security-scan        # Security audit

# Performance
npm run performance:benchmark              # Benchmark performance
npm run performance:production-validate    # Validate production performance
```

### Queen-Princess-Drone Commands
```bash
# Queen Orchestrator for automated fixes
npm run queen:development-fixes    # Fix development domain issues
npm run queen:quality-fixes         # Theater elimination
npm run queen:coordination-fixes   # Workflow implementation

# Theater Detection & Analysis
npm run analyzer:theater-detection      # Detect performance theater
npm run analyzer:production-assessment  # Production readiness check
npm run analyzer:quality-scan           # Comprehensive quality scan
```

### Foundry VTT Deployment
```bash
npm run package:foundry-module   # Package module for Foundry
npm run deploy:foundry           # Deploy to Foundry modules
npm run deploy:api-server        # Deploy API server
npm run deploy:docs              # Deploy documentation
```

## High-Level Architecture

### Core System Components

1. **Multi-Agent Swarm Architecture**
   - **Queen Coordinator**: `src/coordination/queen_coordinator.py` - Central orchestration
   - **6 Princess Domains**: Development, Integration, Infrastructure, Quality, Research, Coordination
   - **21 Specialized Drones**: Domain-specific task execution agents
   - **Theater Detection**: Zero-tolerance validation in `src/security/enterprise_theater_detection.py`

2. **GraphRAG + Vector Search Engine**
   - **Neo4j Integration**: Graph database for Pathfinder 2e rules relationships
   - **Hybrid RAG Core**: Combines vector and graph search for <2s response times
   - **Multi-hop Reasoning**: Complex rule interaction resolution
   - **Archives of Nethys**: Primary data source integration

3. **API Infrastructure**
   - **Express.js Server**: `src/api-server.js` - Main API entry point
   - **WebSocket Support**: Real-time communication with Foundry VTT
   - **Performance Targets**: <2s response, <$0.10/session cost
   - **Security**: Helmet.js, rate limiting, CORS configured

4. **Foundry VTT Integration**
   - **Raven Familiar UI**: Animated sprite interface system
   - **Module Framework**: v11+ compatibility layer
   - **Canvas System**: Non-intrusive overlay integration
   - **Chat Interface**: WebSocket-based real-time messaging

5. **Quality Framework**
   - **Theater Detection**: `comprehensive_analysis_engine.py` with 95% accuracy
   - **NASA POT10 Compliance**: Defense-grade code standards (92% compliant)
   - **Performance Regression**: `tests/regression/performance_regression_suite.py`
   - **Six Sigma Telemetry**: `src/enterprise/telemetry/six_sigma.py`

### Critical Files & Patterns

- **Analysis Engine**: `comprehensive_analysis_engine.py` - 25,640 LOC unified analysis system
- **Queen Orchestrator**: `QueenDebugOrchestrator.js` - Automated fix coordination
- **Test Orchestrator**: `tests/unified_test_orchestrator.py` - Test suite management
- **Security Validation**: `src/security/dfars_*` - Defense industry compliance modules
- **Configuration**: `jest.config.js` for JavaScript, `config/policy/` for Python components

### Mixed Technology Stack

The codebase uses both **JavaScript/Node.js** (API server, Foundry integration) and **Python** (analysis engine, AI coordination, quality tools):
- JavaScript: Express API, WebSocket, Foundry module
- Python: Theater detection, Queen-Princess coordination, NASA compliance analysis
- Configuration: npm for JS dependencies, Python modules for analysis tools

### Testing Strategy

- **JavaScript Tests**: Jest framework with unit/integration/e2e separation
- **Python Tests**: pytest with compliance validation and theater detection
- **Coverage Targets**: 80% branches/functions/lines/statements
- **Performance Baselines**: Regression suite maintains Phase 3 achievements

## Development Guidelines

### File Organization
- `/src` - Mixed JS/Python source code
- `/tests` - Test files (Jest for JS, pytest for Python)
- `/docs` - Documentation
- `/scripts` - Python utility scripts
- `/config` - Configuration files
- `.claude/.artifacts` - QA outputs and analysis results

### Quality Gates
- NASA Compliance: >=90% (currently 92%)
- Theater Detection Score: >=60/100 for authenticity
- Test Coverage: >=80% all metrics
- Security: Zero critical/high findings
- Performance: <2s response, <$0.10 cost targets

### Key Development Patterns
- Use Queen-Princess-Drone hierarchy for task distribution
- Validate all implementations with theater detection
- Maintain NASA POT10 compliance (functions <60 lines)
- Follow MECE (Mutually Exclusive, Collectively Exhaustive) task division
- Implement comprehensive error handling and logging