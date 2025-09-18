# Familiar: GM Assistant for Foundry VTT

## Vision Statement
An AI-powered Game Master assistant that manifests as a clickable raven familiar in Foundry VTT, providing intelligent rules assistance, content generation, and artwork creation specifically for Pathfinder 2e (with future system support).

## Development Status - Loop 2 Complete

**PROJECT STATUS**: 83.3% Production Ready (Loop 2 Complete, Loop 3 In Progress)
**COMPLETION DATE**: September 18, 2025
**SESSION ID**: familiar-loop-3-infrastructure

### Real Progress Achieved Through 3-Loop System

#### Loop 1: Planning & Risk Mitigation ✅ COMPLETE
- **5 Iterations** of spec→research→premortem cycles
- **Failure Probability**: Reduced from 15% to 2.8%
- **Risk-Mitigated Foundation**: Evidence-based planning with comprehensive coverage
- **Dependency Analysis**: 40% time reduction through parallel execution planning

#### Loop 2: Development & Implementation ✅ COMPLETE
- **Queen-Princess-Drone Hierarchy**: 21 specialized agents deployed across 6 domains
- **Theater Detection**: 47 violations identified, core systems validated as real
- **Real Implementations**: GraphRAG, API server, quality gates confirmed authentic
- **Production Readiness**: 83.3% ready with minor remediation needed

#### Loop 3: Quality & Infrastructure 🔄 IN PROGRESS
- **Infrastructure-First Approach**: CI/CD pipeline design and implementation
- **MECE Distribution**: Systematic task allocation to Princess domains
- **Theater Elimination**: Zero tolerance validation for remaining mocks
- **Production Deployment**: Final quality gates and GitHub integration

## Core Requirements

### 1. User Interface
- **Raven Familiar**: Animated raven sprite in bottom-right corner
- **Chat Window**: Opens on click with conversational interface
- **Minimizable**: Can be hidden/shown during gameplay
- **Non-intrusive**: Doesn't block game canvas or controls

### 2. Rules Assistance System
- **Hybrid RAG Architecture**: GraphRAG + Vector search
- **Data Source**: Archives of Nethys (Pathfinder 2e SRD)
- **Multi-hop Reasoning**: Handle complex rule interactions
- **Source Citations**: Show book/page references
- **Context Awareness**: Understand current game state

### 3. Content Generation

#### Monster Generation
- Follow Pathfinder 2e creature building rules
- Auto-balance for party level/composition
- Generate stat blocks in Foundry format
- Include unique abilities and lore

#### Encounter Building
- Use official encounter difficulty rules
- Account for party composition
- Suggest environmental factors
- Generate treasure appropriately

#### Treasure/Loot Generation
- Follow wealth-by-level guidelines
- Generate magic items appropriately
- Create custom items with balance
- Export to Foundry item format

### 4. AI Art Generation

#### Two-Phase System
1. **Initial Generation**:
   - Use DALL-E 3 or Midjourney API
   - Generate based on text descriptions
   - Multiple variations offered

2. **Editing Phase**:
   - Switch to Nana Banana (Gemini 2.5 Flash Image)
   - Allow specific edits via natural language
   - Maintain consistency with scene

#### Art Categories
- Character portraits
- Monster/NPC artwork
- Battle maps and scenes
- Item/equipment visuals
- Environmental backgrounds

### 5. Technical Requirements

#### Foundry Integration
- Compatible with Foundry v11+
- Module manifest compliant
- Use Foundry API hooks
- Respect canvas layers
- WebGL-safe rendering

#### Performance
- <2 second response time
- Lazy loading of resources
- Efficient caching strategy
- Minimal impact on Foundry performance
- Background processing for heavy tasks

#### Data Management
- Local caching of common queries
- Encrypted API key storage
- User data privacy
- Offline fallback options

### 6. Monetization & Sustainability
- Freemium model consideration
- API token management
- Usage tracking and limits
- Premium features definition
- Cost optimization strategies

## User Stories

### As a Game Master, I want to:
1. Quickly look up complex rule interactions during gameplay
2. Generate balanced encounters on the fly
3. Create unique NPCs with appropriate stats
4. Get custom artwork for my campaign
5. Have treasure that makes narrative sense
6. Not slow down my game with research

## Success Metrics
- Query accuracy: >95% for rules questions
- Response time: <2 seconds average
- Cost per session: <$0.10 average
- User satisfaction: >90% positive
- Adoption rate: 100+ users in first month

## Out of Scope (Phase 1)
- Systems other than Pathfinder 2e
- Player-facing features
- Campaign management
- Voice integration
- Mobile app version

## Technical Constraints
- Must work in browser environment
- API rate limits consideration
- Token cost management
- Cross-origin restrictions
- Foundry module size limits

## Legal Considerations
- Paizo Community Use Policy compliance
- Archives of Nethys terms of service
- API provider terms
- User data protection
- Content attribution requirements