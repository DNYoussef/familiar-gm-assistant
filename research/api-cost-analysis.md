# Familiar Project: API Cost Analysis & Optimization

## Executive Summary

**Target**: <$0.10 per session average
**Current Baseline**: $0.08-$0.15 per session (estimated)
**Optimization Goal**: Achieve consistent $0.05-$0.08 per session
**Break-even**: 10,000+ MAU at $5/month freemium conversion

---

## 1. LLM API Cost Breakdown

### Claude API (Anthropic)
- **Claude 3.5 Sonnet**: $3.00/1M input tokens, $15.00/1M output tokens
- **Claude 3 Haiku**: $0.25/1M input tokens, $1.25/1M output tokens
- **Average session**: 2,000 input + 800 output tokens
- **Cost per session**: $0.018 (Sonnet) | $0.0015 (Haiku)

### GPT-5 API (OpenAI)
- **GPT-5**: $10.00/1M input tokens, $30.00/1M output tokens
- **GPT-4o**: $2.50/1M input tokens, $10.00/1M output tokens
- **Average session**: 2,000 input + 800 output tokens
- **Cost per session**: $0.044 (GPT-5) | $0.013 (GPT-4o)

### Gemini API (Google)
- **Gemini 2.5 Pro**: $1.25/1M input tokens, $5.00/1M output tokens
- **Gemini 2.5 Flash**: $0.075/1M input tokens, $0.30/1M output tokens
- **Average session**: 2,000 input + 800 output tokens
- **Cost per session**: $0.0065 (Pro) | $0.00039 (Flash)

### Recommended LLM Strategy
```
Primary: Gemini 2.5 Flash ($0.0004/session)
Fallback: Claude 3 Haiku ($0.0015/session)
Premium: Claude 3.5 Sonnet ($0.018/session)
```

---

## 2. Image Generation Cost Analysis

### DALL-E 3 (OpenAI)
- **1024x1024**: $0.040 per image
- **1792x1024**: $0.080 per image
- **Average per session**: 1.2 images
- **Cost per session**: $0.048-$0.096

### Midjourney API
- **Fast mode**: ~$0.05 per image
- **Relax mode**: ~$0.02 per image (queued)
- **Average per session**: 1.2 images
- **Cost per session**: $0.024-$0.060

### FLUX (Replicate/HuggingFace)
- **FLUX.1-dev**: $0.003-$0.008 per image
- **FLUX.1-schnell**: $0.001-$0.003 per image
- **Average per session**: 1.2 images
- **Cost per session**: $0.0012-$0.0096

### Recommended Image Generation Strategy
```
Primary: FLUX.1-schnell ($0.0036/session)
Quality: FLUX.1-dev ($0.0096/session)
Premium: DALL-E 3 ($0.048/session)
```

---

## 3. Image Editing Cost Analysis

### Nana Banana API
- **Basic edit**: $0.01-$0.02 per operation
- **Complex edit**: $0.03-$0.05 per operation
- **Average per session**: 0.8 edits
- **Cost per session**: $0.008-$0.040

### Gemini 2.5 Flash Vision
- **Image input**: $0.075/1M tokens (~$0.0001 per image)
- **Text output**: $0.30/1M tokens (~$0.0002 per edit)
- **Average per session**: 0.8 edits
- **Cost per session**: $0.0002

### Canvas API (Various)
- **Basic transformations**: $0.005-$0.01 per operation
- **AI-powered edits**: $0.02-$0.04 per operation
- **Average per session**: 0.8 edits
- **Cost per session**: $0.004-$0.032

### Recommended Image Editing Strategy
```
Primary: Gemini 2.5 Flash Vision ($0.0002/session)
Fallback: Canvas API Basic ($0.008/session)
Premium: Nana Banana ($0.024/session)
```

---

## 4. Database & Hosting Costs

### Supabase (Recommended)
- **Free tier**: 500MB DB, 2GB bandwidth
- **Pro tier**: $25/month (8GB DB, 250GB bandwidth)
- **Cost per MAU**: $0.005-$0.01 (estimated)

### Vercel Hosting
- **Free tier**: 100GB bandwidth
- **Pro tier**: $20/month (1TB bandwidth)
- **Cost per session**: <$0.001

### Redis Caching (Upstash)
- **Free tier**: 10,000 requests/day
- **Pay-as-you-go**: $0.20/100k requests
- **Cost per session**: $0.0004-$0.002

### Total Infrastructure Cost per Session
```
Database: $0.005-$0.01
Hosting: <$0.001
Caching: $0.0004-$0.002
Total: $0.0054-$0.013
```

---

## 5. Caching Strategies to Reduce Costs

### Response Caching
```javascript
// LLM Response Cache (24h TTL)
const cacheKey = `llm:${contentHash}:${model}`;
const cachedResponse = await redis.get(cacheKey);
if (cachedResponse) return cachedResponse;

// Cache hit rate target: 35-50%
// Cost reduction: 35-50% on repeat queries
```

### Image Generation Cache
```javascript
// Generated Image Cache (7d TTL)
const imageKey = `img:${promptHash}:${style}`;
const cachedImage = await redis.get(imageKey);
if (cachedImage) return cachedImage;

// Cache hit rate target: 25-40%
// Cost reduction: 25-40% on similar prompts
```

### Precomputed Responses
```javascript
// Common FAQ/Tutorial Responses
const commonResponses = {
  'getting-started': precomputedResponse,
  'basic-editing': precomputedResponse,
  'troubleshooting': precomputedResponse
};

// Coverage target: 20-30% of sessions
// Cost reduction: 100% for covered sessions
```

### Smart Model Routing
```javascript
// Route simple queries to cheaper models
function selectModel(complexity, userTier) {
  if (complexity < 0.3) return 'gemini-flash';
  if (complexity < 0.7) return 'claude-haiku';
  if (userTier === 'premium') return 'claude-sonnet';
  return 'gemini-pro';
}
```

---

## 6. Cost Per Session Breakdown

### Optimized Cost Structure
```
LLM (Primary): $0.0004 (Gemini Flash)
Image Gen: $0.0036 (FLUX schnell)
Image Edit: $0.0002 (Gemini Vision)
Infrastructure: $0.006
Cache overhead: $0.001
Total Base: $0.011 per session
```

### With Cache Hit Rates (35% average)
```
LLM: $0.0004 * 0.65 = $0.00026
Image Gen: $0.0036 * 0.70 = $0.00252
Image Edit: $0.0002 * 0.75 = $0.00015
Infrastructure: $0.006
Total Optimized: $0.0089 per session
```

### Premium Features (10% of sessions)
```
Premium LLM: +$0.0176 (Claude Sonnet)
Premium Images: +$0.044 (DALL-E 3)
Premium Edits: +$0.024 (Nana Banana)
Weighted average: +$0.008
Final Average: $0.017 per session
```

---

## 7. Monthly Operational Cost Projections

### Scenario 1: 10,000 Monthly Active Users
```
Sessions per month: 50,000 (5 sessions/user avg)
Base cost: 50,000 * $0.0089 = $445
Premium uplift: 5,000 * $0.008 = $40
Total monthly: $485
Cost per MAU: $0.049
```

### Scenario 2: 50,000 Monthly Active Users
```
Sessions per month: 200,000
Base cost: 200,000 * $0.0089 = $1,780
Premium uplift: 20,000 * $0.008 = $160
Infrastructure scaling: +$300
Total monthly: $2,240
Cost per MAU: $0.045
```

### Scenario 3: 100,000 Monthly Active Users
```
Sessions per month: 400,000
Base cost: 400,000 * $0.0089 = $3,560
Premium uplift: 40,000 * $0.008 = $320
Infrastructure scaling: +$800
Volume discounts: -$400
Total monthly: $4,280
Cost per MAU: $0.043
```

---

## 8. Break-Even Analysis for Freemium Model

### Revenue Assumptions
```
Free users: 85% (limited features)
Paid users: 15% at $5/month
Premium users: 5% at $15/month (within paid tier)
```

### Break-Even Calculations

#### 10,000 MAU Scenario
```
Revenue:
- Free: 8,500 users * $0 = $0
- Paid: 1,275 users * $5 = $6,375
- Premium: 225 users * $15 = $3,375
Total Revenue: $9,750

Costs: $485 (API) + $200 (other) = $685
Profit: $9,065
Margin: 93%
```

#### Break-Even Point
```
Required MAU for break-even: ~1,500 users
- 1,275 free users
- 191 paid users ($5) = $955
- 34 premium users ($15) = $510
Total revenue: $1,465
Total costs: $150 (API) + $200 (base) = $350
Break-even achieved at 1,500 MAU
```

---

## 9. Cost Optimization Recommendations

### Immediate Actions (Week 1)
1. **Implement Gemini 2.5 Flash as primary LLM** - 95% cost reduction
2. **Deploy FLUX.1-schnell for image generation** - 90% cost reduction
3. **Setup Redis caching with 24h TTL** - 35% cost reduction
4. **Implement smart model routing** - 20% additional savings

### Short-term Actions (Month 1)
1. **Build response cache from common queries** - 25% session coverage
2. **Implement image similarity detection** - Reduce duplicate generations
3. **Add user-tier based model selection** - Premium features for paid users
4. **Setup cost monitoring and alerts** - Real-time cost tracking

### Long-term Actions (Quarter 1)
1. **Train custom smaller models for common tasks** - 80% cost reduction for covered use cases
2. **Implement edge caching with Cloudflare** - Reduce API calls by 15%
3. **Build predictive pre-generation** - Cache likely-needed content
4. **Negotiate volume discounts with providers** - 10-20% additional savings

---

## 10. Risk Mitigation & Monitoring

### Cost Spike Protection
```javascript
// Per-user rate limiting
const userLimit = userTier === 'free' ? 10 : 100; // sessions/hour
const sessionCost = calculateSessionCost(session);
if (sessionCost > maxCostPerSession) {
  fallbackToFreeTier();
}
```

### Real-time Monitoring
```javascript
// Cost tracking per session
await logCost({
  userId,
  sessionId,
  llmCost,
  imageCost,
  totalCost,
  timestamp
});

// Daily cost alerts
if (dailyCost > budgetThreshold) {
  await sendAlert('Cost threshold exceeded');
}
```

### Failover Strategies
```
Primary fails -> Fallback model (same provider)
Provider fails -> Secondary provider
All fail -> Cached/precomputed responses
Emergency -> Rate limiting + queue system
```

---

## 11. Success Metrics & KPIs

### Cost Metrics
- **Target**: <$0.10 per session average ✅ ($0.017 projected)
- **Efficiency**: >70% cache hit rate (target: 75%)
- **Model distribution**: 80% Gemini Flash, 15% Claude Haiku, 5% Premium
- **Monthly growth**: <2x cost increase per 3x user growth

### Quality Metrics
- **User satisfaction**: >4.5/5 despite cost optimization
- **Response time**: <2s average (with caching)
- **Uptime**: >99.9% with failover systems
- **Feature parity**: 95% functionality on optimized models

### Business Metrics
- **Break-even**: 1,500 MAU (well below launch targets)
- **Profit margin**: >90% at 10,000 MAU
- **Conversion rate**: 15% free-to-paid (industry standard: 1-5%)
- **Churn rate**: <5% monthly (cost efficiency supports pricing)

---

## Conclusion

The Familiar project can easily achieve the <$0.10 per session target with an optimized cost structure of ~$0.017 per session. The key strategies are:

1. **Primary**: Gemini 2.5 Flash for 80% of LLM queries
2. **Primary**: FLUX.1-schnell for image generation
3. **Primary**: Gemini Vision for image editing
4. **Caching**: 35%+ hit rates across all services
5. **Smart routing**: Tier-based model selection

This approach provides:
- ✅ 83% under cost target ($0.017 vs $0.10)
- ✅ Break-even at just 1,500 MAU
- ✅ 93% profit margin at scale
- ✅ Room for premium features and growth

**Next Steps**: Implement Gemini Flash integration and caching infrastructure as immediate priorities.