/**
 * Base Components and Utilities
 * Phase 2: Core Architecture - Foundation Classes
 * Provides common functionality for all system components
 */

/**
 * Base Component Class
 * All system components inherit from this base class
 */
export class BaseComponent {
  constructor(name, config = {}) {
    this.name = name;
    this.config = config;
    this.isInitialized = false;
    this.metrics = new ComponentMetrics(name);
    this.logger = new ComponentLogger(name);
    this.eventEmitter = new ComponentEventEmitter();
  }

  /**
   * Initialize component - must be implemented by subclasses
   */
  async initialize() {
    throw new Error(`Component ${this.name} must implement initialize() method`);
  }

  /**
   * Cleanup resources - must be implemented by subclasses
   */
  async cleanup() {
    throw new Error(`Component ${this.name} must implement cleanup() method`);
  }

  /**
   * Health check - must be implemented by subclasses
   */
  async healthCheck() {
    throw new Error(`Component ${this.name} must implement healthCheck() method`);
  }

  /**
   * Get component status
   */
  getStatus() {
    return {
      name: this.name,
      initialized: this.isInitialized,
      metrics: this.metrics.getSummary(),
      health: this.getHealth(),
      lastActivity: this.metrics.lastActivity
    };
  }

  /**
   * Get component health status
   */
  getHealth() {
    return {
      status: this.isInitialized ? 'healthy' : 'uninitialized',
      uptime: this.metrics.uptime,
      errorRate: this.metrics.errorRate,
      lastError: this.metrics.lastError
    };
  }

  /**
   * Record operation metrics
   */
  recordOperation(operation, duration, success = true, metadata = {}) {
    this.metrics.recordOperation(operation, duration, success, metadata);
    this.logger.logOperation(operation, duration, success, metadata);
  }

  /**
   * Emit component event
   */
  emit(event, data) {
    this.eventEmitter.emit(`${this.name}:${event}`, data);
  }

  /**
   * Listen for component events
   */
  on(event, handler) {
    this.eventEmitter.on(`${this.name}:${event}`, handler);
  }
}

/**
 * Component Metrics Tracking
 * Tracks performance and usage metrics for components
 */
class ComponentMetrics {
  constructor(componentName) {
    this.componentName = componentName;
    this.startTime = Date.now();
    this.operations = new Map();
    this.errors = [];
    this.lastActivity = null;
  }

  /**
   * Record operation metrics
   */
  recordOperation(operation, duration, success, metadata) {
    this.lastActivity = Date.now();

    if (!this.operations.has(operation)) {
      this.operations.set(operation, {
        count: 0,
        totalTime: 0,
        successCount: 0,
        errorCount: 0,
        avgTime: 0,
        minTime: Infinity,
        maxTime: 0
      });
    }

    const stats = this.operations.get(operation);
    stats.count++;
    stats.totalTime += duration;
    stats.avgTime = stats.totalTime / stats.count;
    stats.minTime = Math.min(stats.minTime, duration);
    stats.maxTime = Math.max(stats.maxTime, duration);

    if (success) {
      stats.successCount++;
    } else {
      stats.errorCount++;
      this.errors.push({
        operation,
        timestamp: Date.now(),
        metadata
      });

      // Keep only last 100 errors
      if (this.errors.length > 100) {
        this.errors = this.errors.slice(-100);
      }
    }
  }

  /**
   * Get metrics summary
   */
  getSummary() {
    const summary = {
      component: this.componentName,
      uptime: Date.now() - this.startTime,
      totalOperations: 0,
      totalErrors: this.errors.length,
      operations: {}
    };

    for (const [operation, stats] of this.operations) {
      summary.totalOperations += stats.count;
      summary.operations[operation] = {
        count: stats.count,
        avgTime: Math.round(stats.avgTime * 100) / 100,
        minTime: stats.minTime === Infinity ? 0 : stats.minTime,
        maxTime: stats.maxTime,
        successRate: stats.count > 0 ? stats.successCount / stats.count : 0,
        errorRate: stats.count > 0 ? stats.errorCount / stats.count : 0
      };
    }

    summary.errorRate = summary.totalOperations > 0 ?
      summary.totalErrors / summary.totalOperations : 0;

    return summary;
  }

  /**
   * Get recent errors
   */
  getRecentErrors(limit = 10) {
    return this.errors.slice(-limit);
  }

  get uptime() {
    return Date.now() - this.startTime;
  }

  get errorRate() {
    const totalOps = Array.from(this.operations.values())
      .reduce((sum, stats) => sum + stats.count, 0);
    return totalOps > 0 ? this.errors.length / totalOps : 0;
  }

  get lastError() {
    return this.errors.length > 0 ? this.errors[this.errors.length - 1] : null;
  }
}

/**
 * Component Logger
 * Provides consistent logging across all components
 */
class ComponentLogger {
  constructor(componentName) {
    this.componentName = componentName;
    this.logLevel = 'info';
  }

  log(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      component: this.componentName,
      level,
      message,
      data
    };

    // Use Foundry's logging if available, otherwise console
    if (typeof console !== 'undefined') {
      const logMethod = console[level] || console.log;
      logMethod(`[${timestamp}] [${this.componentName}] ${message}`, data || '');
    }

    // Emit log event for centralized logging
    this.emit('log', logEntry);
  }

  logOperation(operation, duration, success, metadata) {
    const level = success ? 'debug' : 'error';
    const status = success ? 'SUCCESS' : 'FAILED';
    this.log(level, `Operation ${operation} ${status} in ${duration}ms`, metadata);
  }

  debug(message, data) { this.log('debug', message, data); }
  info(message, data) { this.log('info', message, data); }
  warn(message, data) { this.log('warn', message, data); }
  error(message, data) { this.log('error', message, data); }

  emit(event, data) {
    // Simple event emission for logging
    if (typeof window !== 'undefined' && window.dispatchEvent) {
      window.dispatchEvent(new CustomEvent(`familiar:${event}`, { detail: data }));
    }
  }
}

/**
 * Component Event Emitter
 * Simple event system for component communication
 */
class ComponentEventEmitter {
  constructor() {
    this.events = new Map();
  }

  on(event, handler) {
    if (!this.events.has(event)) {
      this.events.set(event, []);
    }
    this.events.get(event).push(handler);
  }

  off(event, handler) {
    if (this.events.has(event)) {
      const handlers = this.events.get(event);
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.events.has(event)) {
      this.events.get(event).forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Event handler error for ${event}:`, error);
        }
      });
    }
  }
}

/**
 * Error Handler Utility
 * Provides consistent error handling across components
 */
export class ErrorHandler {
  static async handleAsync(operation, context = '') {
    try {
      return await operation();
    } catch (error) {
      console.error(`Error in ${context}:`, error);

      // Create user-friendly error response
      return {
        success: false,
        error: {
          type: error.name || 'UnknownError',
          message: error.message || 'An unknown error occurred',
          code: error.code || 'UNKNOWN',
          context,
          timestamp: new Date().toISOString()
        }
      };
    }
  }

  static handleSync(operation, context = '') {
    try {
      return operation();
    } catch (error) {
      console.error(`Error in ${context}:`, error);
      return null;
    }
  }

  static createError(type, message, code = null, details = null) {
    const error = new Error(message);
    error.name = type;
    error.code = code;
    error.details = details;
    error.timestamp = new Date().toISOString();
    return error;
  }
}

/**
 * Configuration Manager
 * Manages configuration for all components
 */
export class ConfigManager {
  constructor() {
    this.config = new Map();
    this.defaults = new Map();
  }

  /**
   * Set default configuration for a component
   */
  setDefaults(componentName, defaults) {
    this.defaults.set(componentName, defaults);
  }

  /**
   * Get configuration for a component
   */
  getConfig(componentName, key = null) {
    const componentConfig = this.config.get(componentName) || {};
    const componentDefaults = this.defaults.get(componentName) || {};

    const mergedConfig = { ...componentDefaults, ...componentConfig };

    return key ? mergedConfig[key] : mergedConfig;
  }

  /**
   * Set configuration for a component
   */
  setConfig(componentName, key, value) {
    if (!this.config.has(componentName)) {
      this.config.set(componentName, {});
    }

    if (typeof key === 'object') {
      // Setting entire config object
      this.config.set(componentName, { ...this.config.get(componentName), ...key });
    } else {
      // Setting single key
      this.config.get(componentName)[key] = value;
    }
  }

  /**
   * Load configuration from storage
   */
  async loadConfig() {
    try {
      // Load from Foundry settings or localStorage
      if (typeof game !== 'undefined' && game.settings) {
        // Load from Foundry settings
        const savedConfig = game.settings.get('familiar-gm-assistant', 'component-config') || {};
        this.config = new Map(Object.entries(savedConfig));
      } else if (typeof localStorage !== 'undefined') {
        // Fallback to localStorage
        const savedConfig = JSON.parse(localStorage.getItem('familiar-config') || '{}');
        this.config = new Map(Object.entries(savedConfig));
      }
    } catch (error) {
      console.warn('Failed to load configuration:', error);
    }
  }

  /**
   * Save configuration to storage
   */
  async saveConfig() {
    try {
      const configObject = Object.fromEntries(this.config);

      if (typeof game !== 'undefined' && game.settings) {
        // Save to Foundry settings
        await game.settings.set('familiar-gm-assistant', 'component-config', configObject);
      } else if (typeof localStorage !== 'undefined') {
        // Fallback to localStorage
        localStorage.setItem('familiar-config', JSON.stringify(configObject));
      }
    } catch (error) {
      console.warn('Failed to save configuration:', error);
    }
  }
}

/**
 * Cache Manager
 * Provides caching functionality for components
 */
export class CacheManager {
  constructor(maxSize = 100, ttl = 3600000) { // 1 hour default TTL
    this.cache = new Map();
    this.maxSize = maxSize;
    this.ttl = ttl;
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Get item from cache
   */
  get(key) {
    const item = this.cache.get(key);

    if (!item) {
      this.misses++;
      return null;
    }

    if (Date.now() > item.expires) {
      this.cache.delete(key);
      this.misses++;
      return null;
    }

    this.hits++;
    item.lastAccessed = Date.now();
    return item.value;
  }

  /**
   * Set item in cache
   */
  set(key, value, customTTL = null) {
    // Evict if at max size
    if (this.cache.size >= this.maxSize) {
      this.evictLRU();
    }

    const ttl = customTTL || this.ttl;
    this.cache.set(key, {
      value,
      created: Date.now(),
      expires: Date.now() + ttl,
      lastAccessed: Date.now()
    });
  }

  /**
   * Check if key exists in cache
   */
  has(key) {
    return this.get(key) !== null;
  }

  /**
   * Remove item from cache
   */
  delete(key) {
    return this.cache.delete(key);
  }

  /**
   * Clear entire cache
   */
  clear() {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Evict least recently used item
   */
  evictLRU() {
    let lruKey = null;
    let lruTime = Date.now();

    for (const [key, item] of this.cache) {
      if (item.lastAccessed < lruTime) {
        lruTime = item.lastAccessed;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.cache.delete(lruKey);
    }
  }

  /**
   * Get cache statistics
   */
  getStats() {
    const total = this.hits + this.misses;
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? this.hits / total : 0,
      memoryUsage: this.getMemoryUsage()
    };
  }

  /**
   * Estimate memory usage (rough approximation)
   */
  getMemoryUsage() {
    let size = 0;
    for (const [key, item] of this.cache) {
      size += JSON.stringify(key).length;
      size += JSON.stringify(item.value).length;
      size += 64; // Overhead estimate
    }
    return size;
  }
}

/**
 * Validation Utilities
 * Common validation functions for all components
 */
export class ValidationUtils {
  /**
   * Validate PF2e level (1-25)
   */
  static validateLevel(level) {
    const num = parseInt(level);
    return {
      isValid: num >= 1 && num <= 25,
      value: num,
      error: num < 1 || num > 25 ? 'Level must be between 1 and 25' : null
    };
  }

  /**
   * Validate PF2e creature type
   */
  static validateCreatureType(type) {
    const validTypes = [
      'aberration', 'animal', 'astral', 'beast', 'celestial', 'construct',
      'dragon', 'elemental', 'fey', 'fiend', 'fungus', 'giant', 'humanoid',
      'monitor', 'ooze', 'plant', 'spirit', 'undead'
    ];

    return {
      isValid: validTypes.includes(type?.toLowerCase()),
      value: type?.toLowerCase(),
      error: !validTypes.includes(type?.toLowerCase()) ?
        `Invalid creature type. Valid types: ${validTypes.join(', ')}` : null
    };
  }

  /**
   * Validate ability score (1-30 typical range)
   */
  static validateAbilityScore(score) {
    const num = parseInt(score);
    return {
      isValid: num >= 1 && num <= 30,
      value: num,
      modifier: Math.floor((num - 10) / 2),
      error: num < 1 || num > 30 ? 'Ability score must be between 1 and 30' : null
    };
  }

  /**
   * Validate required fields in object
   */
  static validateRequired(obj, requiredFields) {
    const missing = requiredFields.filter(field =>
      obj[field] === undefined || obj[field] === null || obj[field] === ''
    );

    return {
      isValid: missing.length === 0,
      missing,
      error: missing.length > 0 ? `Missing required fields: ${missing.join(', ')}` : null
    };
  }

  /**
   * Sanitize user input
   */
  static sanitizeInput(input, maxLength = 500) {
    if (typeof input !== 'string') {
      return '';
    }

    return input
      .trim()
      .slice(0, maxLength)
      .replace(/[<>]/g, '') // Basic HTML removal
      .replace(/javascript:/gi, ''); // Basic script removal
  }
}

/**
 * Performance Utils
 * Utilities for performance monitoring and optimization
 */
export class PerformanceUtils {
  /**
   * Measure execution time of async function
   */
  static async measureAsync(operation, name = 'operation') {
    const start = performance.now();
    try {
      const result = await operation();
      const duration = performance.now() - start;
      return { result, duration, success: true, name };
    } catch (error) {
      const duration = performance.now() - start;
      return { error, duration, success: false, name };
    }
  }

  /**
   * Measure execution time of sync function
   */
  static measure(operation, name = 'operation') {
    const start = performance.now();
    try {
      const result = operation();
      const duration = performance.now() - start;
      return { result, duration, success: true, name };
    } catch (error) {
      const duration = performance.now() - start;
      return { error, duration, success: false, name };
    }
  }

  /**
   * Debounce function calls
   */
  static debounce(func, delay) {
    let timeoutId;
    return (...args) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func.apply(null, args), delay);
    };
  }

  /**
   * Throttle function calls
   */
  static throttle(func, limit) {
    let inThrottle;
    return (...args) => {
      if (!inThrottle) {
        func.apply(null, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }
}

// Global configuration manager instance
export const globalConfig = new ConfigManager();

// Global cache manager instance
export const globalCache = new CacheManager();

export default {
  BaseComponent,
  ErrorHandler,
  ConfigManager,
  CacheManager,
  ValidationUtils,
  PerformanceUtils,
  globalConfig,
  globalCache
};