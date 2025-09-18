/**
 * Cache Manager for Familiar RAG System
 * Implements intelligent caching for query performance optimization
 */

const Redis = require('redis');
const crypto = require('crypto');

class CacheManager {
    constructor(config = {}) {
        this.config = {
            url: config.redisUrl || process.env.REDIS_URL || 'redis://localhost:6379',
            password: config.redisPassword || process.env.REDIS_PASSWORD,
            ttl: config.ttl || 1800, // 30 minutes default
            maxMemory: config.maxMemory || '256mb',
            keyPrefix: config.keyPrefix || 'familiar:rag:',
            enabled: config.enabled !== false,
            ...config
        };

        this.client = null;
        this.connected = false;
        this.stats = {
            hits: 0,
            misses: 0,
            sets: 0,
            deletes: 0,
            errors: 0
        };

        if (this.config.enabled) {
            this.initialize();
        }
    }

    /**
     * Initialize Redis connection
     */
    async initialize() {
        try {
            this.client = Redis.createClient({
                url: this.config.url,
                password: this.config.password,
                retry_strategy: (options) => {
                    if (options.error && options.error.code === 'ECONNREFUSED') {
                        return new Error('Redis server refuses connection');
                    }
                    if (options.total_retry_time > 1000 * 60 * 60) {
                        return new Error('Retry time exhausted');
                    }
                    if (options.attempt > 10) {
                        return undefined;
                    }
                    return Math.min(options.attempt * 100, 3000);
                }
            });

            this.client.on('error', (err) => {
                console.error('Redis error:', err);
                this.stats.errors++;
                this.connected = false;
            });

            this.client.on('connect', () => {
                console.log('Redis connected');
                this.connected = true;
            });

            this.client.on('ready', () => {
                console.log('Redis ready');
                this.setupMemoryPolicy();
            });

            await this.client.connect();

        } catch (error) {
            console.error('Failed to initialize Redis:', error);
            this.config.enabled = false;
        }
    }

    /**
     * Setup Redis memory policy for optimal caching
     */
    async setupMemoryPolicy() {
        try {
            // Set memory policy to evict least recently used keys
            await this.client.configSet('maxmemory-policy', 'allkeys-lru');
            await this.client.configSet('maxmemory', this.config.maxMemory);
        } catch (error) {
            console.warn('Failed to set Redis memory policy:', error);
        }
    }

    /**
     * Generate cache key for query
     */
    generateKey(query, context = {}) {
        const normalizedQuery = query.toLowerCase().trim().replace(/\s+/g, ' ');
        const contextString = JSON.stringify(context, Object.keys(context).sort());
        const combined = `${normalizedQuery}:${contextString}`;
        const hash = crypto.createHash('sha256').update(combined).digest('hex').substring(0, 16);
        return `${this.config.keyPrefix}query:${hash}`;
    }

    /**
     * Get cached result
     */
    async get(query, context = {}) {
        if (!this.config.enabled || !this.connected) {
            this.stats.misses++;
            return null;
        }

        try {
            const key = this.generateKey(query, context);
            const cached = await this.client.get(key);

            if (cached) {
                this.stats.hits++;
                const result = JSON.parse(cached);

                // Update access time for LRU
                await this.client.expire(key, this.config.ttl);

                return {
                    ...result,
                    fromCache: true,
                    cachedAt: result.metadata?.cachedAt
                };
            } else {
                this.stats.misses++;
                return null;
            }

        } catch (error) {
            console.error('Cache get error:', error);
            this.stats.errors++;
            this.stats.misses++;
            return null;
        }
    }

    /**
     * Set cache result
     */
    async set(query, context = {}, result, customTtl = null) {
        if (!this.config.enabled || !this.connected) {
            return false;
        }

        try {
            const key = this.generateKey(query, context);
            const ttl = customTtl || this.config.ttl;

            // Add cache metadata
            const cacheableResult = {
                ...result,
                metadata: {
                    ...result.metadata,
                    cachedAt: new Date().toISOString(),
                    cacheKey: key,
                    ttl: ttl
                }
            };

            await this.client.setEx(key, ttl, JSON.stringify(cacheableResult));
            this.stats.sets++;

            // Cache query classification for faster routing
            if (result.queryType) {
                await this.cacheQueryClassification(query, result.queryType, result.complexity);
            }

            return true;

        } catch (error) {
            console.error('Cache set error:', error);
            this.stats.errors++;
            return false;
        }
    }

    /**
     * Cache query classification results
     */
    async cacheQueryClassification(query, queryType, complexity) {
        try {
            const classificationKey = `${this.config.keyPrefix}classification:${this.hashQuery(query)}`;
            const classification = { queryType, complexity, cachedAt: new Date().toISOString() };

            await this.client.setEx(classificationKey, this.config.ttl * 2, JSON.stringify(classification));
        } catch (error) {
            console.warn('Failed to cache query classification:', error);
        }
    }

    /**
     * Get cached query classification
     */
    async getQueryClassification(query) {
        if (!this.config.enabled || !this.connected) {
            return null;
        }

        try {
            const classificationKey = `${this.config.keyPrefix}classification:${this.hashQuery(query)}`;
            const cached = await this.client.get(classificationKey);

            return cached ? JSON.parse(cached) : null;
        } catch (error) {
            console.warn('Failed to get cached query classification:', error);
            return null;
        }
    }

    /**
     * Cache frequently accessed entities
     */
    async cacheEntity(entityType, entityId, data, ttl = null) {
        if (!this.config.enabled || !this.connected) {
            return false;
        }

        try {
            const key = `${this.config.keyPrefix}entity:${entityType}:${entityId}`;
            const entityTtl = ttl || (this.config.ttl * 4); // Entities cached longer

            await this.client.setEx(key, entityTtl, JSON.stringify({
                ...data,
                cachedAt: new Date().toISOString()
            }));

            return true;
        } catch (error) {
            console.error('Entity cache error:', error);
            return false;
        }
    }

    /**
     * Get cached entity
     */
    async getEntity(entityType, entityId) {
        if (!this.config.enabled || !this.connected) {
            return null;
        }

        try {
            const key = `${this.config.keyPrefix}entity:${entityType}:${entityId}`;
            const cached = await this.client.get(key);

            return cached ? JSON.parse(cached) : null;
        } catch (error) {
            console.error('Entity get error:', error);
            return null;
        }
    }

    /**
     * Cache embeddings to reduce API calls
     */
    async cacheEmbedding(text, embedding) {
        if (!this.config.enabled || !this.connected) {
            return false;
        }

        try {
            const textHash = this.hashQuery(text);
            const key = `${this.config.keyPrefix}embedding:${textHash}`;

            await this.client.setEx(key, this.config.ttl * 6, JSON.stringify({
                text: text,
                embedding: embedding,
                cachedAt: new Date().toISOString()
            }));

            return true;
        } catch (error) {
            console.error('Embedding cache error:', error);
            return false;
        }
    }

    /**
     * Get cached embedding
     */
    async getEmbedding(text) {
        if (!this.config.enabled || !this.connected) {
            return null;
        }

        try {
            const textHash = this.hashQuery(text);
            const key = `${this.config.keyPrefix}embedding:${textHash}`;
            const cached = await this.client.get(key);

            if (cached) {
                const result = JSON.parse(cached);
                return result.embedding;
            }

            return null;
        } catch (error) {
            console.error('Embedding get error:', error);
            return null;
        }
    }

    /**
     * Invalidate cache by pattern
     */
    async invalidate(pattern = '*') {
        if (!this.config.enabled || !this.connected) {
            return 0;
        }

        try {
            const keys = await this.client.keys(`${this.config.keyPrefix}${pattern}`);

            if (keys.length > 0) {
                const deleted = await this.client.del(keys);
                this.stats.deletes += deleted;
                return deleted;
            }

            return 0;
        } catch (error) {
            console.error('Cache invalidation error:', error);
            return 0;
        }
    }

    /**
     * Clear all cache
     */
    async clear() {
        return await this.invalidate('*');
    }

    /**
     * Get cache statistics
     */
    getStats() {
        const total = this.stats.hits + this.stats.misses;
        const hitRate = total > 0 ? (this.stats.hits / total) * 100 : 0;

        return {
            ...this.stats,
            hitRate: hitRate,
            totalRequests: total,
            enabled: this.config.enabled,
            connected: this.connected
        };
    }

    /**
     * Get detailed cache info
     */
    async getInfo() {
        if (!this.config.enabled || !this.connected) {
            return { enabled: false };
        }

        try {
            const info = await this.client.info('memory');
            const keyspace = await this.client.info('keyspace');

            return {
                enabled: true,
                connected: this.connected,
                memory: this.parseMemoryInfo(info),
                keyspace: this.parseKeyspaceInfo(keyspace),
                stats: this.getStats()
            };
        } catch (error) {
            console.error('Failed to get cache info:', error);
            return { enabled: false, error: error.message };
        }
    }

    /**
     * Warm up cache with common queries
     */
    async warmUp(commonQueries = []) {
        if (!this.config.enabled || !this.connected) {
            return false;
        }

        console.log(`Warming up cache with ${commonQueries.length} common queries`);

        for (const queryData of commonQueries) {
            try {
                // Pre-cache query classifications
                await this.cacheQueryClassification(
                    queryData.query,
                    queryData.type,
                    queryData.complexity
                );

                // Pre-cache entities if provided
                if (queryData.entities) {
                    for (const [entityType, entities] of Object.entries(queryData.entities)) {
                        for (const entity of entities) {
                            await this.cacheEntity(entityType, entity.id, entity.data);
                        }
                    }
                }

            } catch (error) {
                console.warn(`Failed to warm up query: ${queryData.query}`, error);
            }
        }

        return true;
    }

    /**
     * Utility methods
     */
    hashQuery(query) {
        return crypto.createHash('md5').update(query.toLowerCase().trim()).digest('hex');
    }

    parseMemoryInfo(info) {
        const lines = info.split('\r\n');
        const memory = {};

        lines.forEach(line => {
            if (line.includes('used_memory:')) {
                memory.used = line.split(':')[1];
            }
            if (line.includes('used_memory_peak:')) {
                memory.peak = line.split(':')[1];
            }
        });

        return memory;
    }

    parseKeyspaceInfo(info) {
        const lines = info.split('\r\n');
        const keyspace = {};

        lines.forEach(line => {
            if (line.startsWith('db0:')) {
                const stats = line.split(':')[1];
                const parts = stats.split(',');
                parts.forEach(part => {
                    const [key, value] = part.split('=');
                    keyspace[key] = parseInt(value) || value;
                });
            }
        });

        return keyspace;
    }

    /**
     * Cleanup and close connections
     */
    async close() {
        if (this.client && this.connected) {
            await this.client.quit();
            this.connected = false;
        }
    }

    /**
     * Health check
     */
    async healthCheck() {
        if (!this.config.enabled) {
            return { status: 'disabled' };
        }

        try {
            const pong = await this.client.ping();
            return {
                status: pong === 'PONG' ? 'healthy' : 'unhealthy',
                connected: this.connected,
                stats: this.getStats()
            };
        } catch (error) {
            return {
                status: 'unhealthy',
                error: error.message,
                connected: false
            };
        }
    }
}

module.exports = { CacheManager };