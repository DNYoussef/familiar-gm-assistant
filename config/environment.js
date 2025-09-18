/**
 * Environment Configuration Management
 * Infrastructure Princess - Environment Consistency System
 */

const environments = {
  development: {
    name: 'development',
    port: 3000,
    database: {
      host: 'localhost',
      port: 5432,
      name: 'familiar_dev',
      ssl: false,
      pool: { min: 2, max: 10 }
    },
    redis: {
      host: 'localhost',
      port: 6379,
      db: 0
    },
    api: {
      baseUrl: 'http://localhost:3000',
      timeout: 5000,
      rateLimit: 100
    },
    logging: {
      level: 'debug',
      format: 'combined',
      file: 'logs/development.log'
    },
    monitoring: {
      enabled: true,
      interval: 30000,
      metrics: ['cpu', 'memory', 'response_time']
    },
    features: {
      hotReload: true,
      debugMode: true,
      mockData: true,
      profiling: true
    }
  },

  staging: {
    name: 'staging',
    port: process.env.PORT || 3000,
    database: {
      host: process.env.DB_HOST,
      port: process.env.DB_PORT || 5432,
      name: process.env.DB_NAME,
      username: process.env.DB_USERNAME,
      password: process.env.DB_PASSWORD,
      ssl: true,
      pool: { min: 5, max: 20 }
    },
    redis: {
      host: process.env.REDIS_HOST,
      port: process.env.REDIS_PORT || 6379,
      password: process.env.REDIS_PASSWORD,
      db: 1
    },
    api: {
      baseUrl: process.env.API_BASE_URL,
      timeout: 10000,
      rateLimit: 1000
    },
    logging: {
      level: 'info',
      format: 'json',
      file: 'logs/staging.log',
      rotation: {
        maxFiles: 10,
        maxSize: '10m'
      }
    },
    monitoring: {
      enabled: true,
      interval: 15000,
      metrics: ['cpu', 'memory', 'response_time', 'error_rate'],
      alerts: {
        email: process.env.ALERT_EMAIL,
        threshold: {
          cpu: 80,
          memory: 85,
          error_rate: 5
        }
      }
    },
    features: {
      hotReload: false,
      debugMode: false,
      mockData: false,
      profiling: true
    },
    security: {
      helmet: true,
      cors: {
        origin: process.env.ALLOWED_ORIGINS?.split(',') || [],
        credentials: true
      },
      rateLimit: {
        windowMs: 15 * 60 * 1000,
        max: 100
      }
    }
  },

  production: {
    name: 'production',
    port: process.env.PORT || 3000,
    database: {
      host: process.env.DB_HOST,
      port: process.env.DB_PORT || 5432,
      name: process.env.DB_NAME,
      username: process.env.DB_USERNAME,
      password: process.env.DB_PASSWORD,
      ssl: {
        require: true,
        rejectUnauthorized: false
      },
      pool: {
        min: 10,
        max: 50,
        idle: 10000,
        acquire: 60000
      }
    },
    redis: {
      host: process.env.REDIS_HOST,
      port: process.env.REDIS_PORT || 6379,
      password: process.env.REDIS_PASSWORD,
      db: 0,
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3
    },
    api: {
      baseUrl: process.env.API_BASE_URL,
      timeout: 30000,
      rateLimit: 10000,
      compression: true,
      etag: true
    },
    logging: {
      level: 'warn',
      format: 'json',
      file: 'logs/production.log',
      rotation: {
        maxFiles: 30,
        maxSize: '50m'
      },
      transport: {
        type: 'winston-elasticsearch',
        host: process.env.ELASTICSEARCH_HOST,
        index: 'familiar-logs'
      }
    },
    monitoring: {
      enabled: true,
      interval: 10000,
      metrics: ['cpu', 'memory', 'response_time', 'error_rate', 'throughput'],
      alerts: {
        email: process.env.ALERT_EMAIL,
        slack: process.env.SLACK_WEBHOOK,
        threshold: {
          cpu: 70,
          memory: 80,
          error_rate: 2,
          response_time: 1000
        }
      },
      healthCheck: {
        enabled: true,
        interval: 5000,
        endpoints: ['/health', '/metrics']
      }
    },
    features: {
      hotReload: false,
      debugMode: false,
      mockData: false,
      profiling: false
    },
    security: {
      helmet: {
        contentSecurityPolicy: {
          directives: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'", "'unsafe-inline'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            imgSrc: ["'self'", "data:", "https:"]
          }
        }
      },
      cors: {
        origin: process.env.ALLOWED_ORIGINS?.split(',') || [],
        credentials: true
      },
      rateLimit: {
        windowMs: 15 * 60 * 1000,
        max: 1000,
        standardHeaders: true,
        legacyHeaders: false
      },
      jwt: {
        secret: process.env.JWT_SECRET,
        expiresIn: '15m',
        refreshTokenExpiry: '7d'
      }
    },
    cache: {
      ttl: 3600,
      max: 1000,
      updateAgeOnGet: true
    },
    clustering: {
      enabled: true,
      workers: process.env.WORKER_COUNT || 'auto'
    }
  }
};

class EnvironmentManager {
  constructor() {
    this.currentEnv = process.env.NODE_ENV || 'development';
    this.config = environments[this.currentEnv];

    if (!this.config) {
      throw new Error(`Unknown environment: ${this.currentEnv}`);
    }
  }

  get(path) {
    return path.split('.').reduce((obj, key) => obj?.[key], this.config);
  }

  isDevelopment() {
    return this.currentEnv === 'development';
  }

  isStaging() {
    return this.currentEnv === 'staging';
  }

  isProduction() {
    return this.currentEnv === 'production';
  }

  validate() {
    const required = {
      production: [
        'DATABASE_URL',
        'REDIS_URL',
        'JWT_SECRET',
        'API_BASE_URL'
      ],
      staging: [
        'DB_HOST',
        'DB_NAME',
        'REDIS_HOST'
      ]
    };

    const envVars = required[this.currentEnv] || [];
    const missing = envVars.filter(varName => !process.env[varName]);

    if (missing.length > 0) {
      throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }

    return true;
  }

  getSecrets() {
    return {
      database: {
        password: process.env.DB_PASSWORD,
        connectionString: process.env.DATABASE_URL
      },
      redis: {
        password: process.env.REDIS_PASSWORD,
        url: process.env.REDIS_URL
      },
      jwt: {
        secret: process.env.JWT_SECRET,
        refreshSecret: process.env.JWT_REFRESH_SECRET
      },
      external: {
        apiKey: process.env.EXTERNAL_API_KEY,
        webhookSecret: process.env.WEBHOOK_SECRET
      }
    };
  }
}

module.exports = {
  environments,
  EnvironmentManager,
  current: new EnvironmentManager()
};