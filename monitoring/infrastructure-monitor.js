/**
 * Infrastructure Monitoring System
 * Infrastructure Princess - System Health and Performance Monitoring
 */

const os = require('os');
const fs = require('fs').promises;
const path = require('path');
const EventEmitter = require('events');

class InfrastructureMonitor extends EventEmitter {
  constructor(config = {}) {
    super();

    this.config = {
      interval: config.interval || 30000, // 30 seconds
      thresholds: {
        cpu: config.cpuThreshold || 80,
        memory: config.memoryThreshold || 85,
        disk: config.diskThreshold || 90,
        responseTime: config.responseTimeThreshold || 1000
      },
      alertCooldown: config.alertCooldown || 300000, // 5 minutes
      logPath: config.logPath || 'logs/monitoring.log',
      metricsPath: config.metricsPath || 'monitoring/metrics',
      ...config
    };

    this.metrics = {
      system: {},
      application: {},
      network: {},
      alerts: []
    };

    this.lastAlerts = new Map();
    this.isRunning = false;
    this.intervalId = null;
  }

  async start() {
    if (this.isRunning) {
      console.log('🔄 Monitor already running');
      return;
    }

    console.log('🚀 Starting Infrastructure Monitor...');

    await this.ensureDirectories();

    this.isRunning = true;
    this.intervalId = setInterval(() => {
      this.collectMetrics();
    }, this.config.interval);

    // Initial collection
    await this.collectMetrics();

    console.log(`✅ Infrastructure Monitor started (interval: ${this.config.interval}ms)`);
    this.emit('started');
  }

  async stop() {
    if (!this.isRunning) {
      console.log('⏹️ Monitor not running');
      return;
    }

    console.log('🛑 Stopping Infrastructure Monitor...');

    this.isRunning = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    console.log('✅ Infrastructure Monitor stopped');
    this.emit('stopped');
  }

  async collectMetrics() {
    const timestamp = new Date().toISOString();

    try {
      // System metrics
      const systemMetrics = await this.collectSystemMetrics();
      this.metrics.system = { ...systemMetrics, timestamp };

      // Application metrics
      const appMetrics = await this.collectApplicationMetrics();
      this.metrics.application = { ...appMetrics, timestamp };

      // Network metrics
      const networkMetrics = await this.collectNetworkMetrics();
      this.metrics.network = { ...networkMetrics, timestamp };

      // Check thresholds and generate alerts
      await this.checkThresholds();

      // Log metrics
      await this.logMetrics();

      // Store metrics for historical analysis
      await this.storeMetrics();

      this.emit('metrics', this.metrics);

    } catch (error) {
      console.error('❌ Error collecting metrics:', error);
      this.emit('error', error);
    }
  }

  async collectSystemMetrics() {
    // CPU Usage
    const cpus = os.cpus();
    const cpuUsage = this.calculateCPUUsage();

    // Memory Usage
    const totalMemory = os.totalmem();
    const freeMemory = os.freemem();
    const usedMemory = totalMemory - freeMemory;
    const memoryUsagePercent = (usedMemory / totalMemory) * 100;

    // Load Average
    const loadAverage = os.loadavg();

    // Uptime
    const uptime = os.uptime();

    // Disk Usage
    const diskUsage = await this.getDiskUsage();

    return {
      cpu: {
        cores: cpus.length,
        model: cpus[0].model,
        usage: cpuUsage,
        loadAverage: {
          '1m': loadAverage[0],
          '5m': loadAverage[1],
          '15m': loadAverage[2]
        }
      },
      memory: {
        total: totalMemory,
        free: freeMemory,
        used: usedMemory,
        usagePercent: memoryUsagePercent
      },
      disk: diskUsage,
      uptime: uptime,
      platform: os.platform(),
      arch: os.arch(),
      hostname: os.hostname()
    };
  }

  async collectApplicationMetrics() {
    const processMemory = process.memoryUsage();
    const processUptime = process.uptime();

    // Node.js specific metrics
    const heapUsed = processMemory.heapUsed;
    const heapTotal = processMemory.heapTotal;
    const heapUsagePercent = (heapUsed / heapTotal) * 100;

    // Event loop lag (approximation)
    const eventLoopLag = await this.measureEventLoopLag();

    // Active handles and requests
    const activeHandles = process._getActiveHandles ? process._getActiveHandles().length : 0;
    const activeRequests = process._getActiveRequests ? process._getActiveRequests().length : 0;

    return {
      process: {
        pid: process.pid,
        uptime: processUptime,
        version: process.version,
        memory: {
          heapUsed,
          heapTotal,
          heapUsagePercent,
          external: processMemory.external,
          rss: processMemory.rss
        },
        eventLoopLag,
        activeHandles,
        activeRequests
      }
    };
  }

  async collectNetworkMetrics() {
    const networkInterfaces = os.networkInterfaces();

    // Extract network statistics
    const interfaces = {};
    for (const [name, iface] of Object.entries(networkInterfaces)) {
      interfaces[name] = iface.map(addr => ({
        address: addr.address,
        family: addr.family,
        internal: addr.internal
      }));
    }

    return {
      interfaces
    };
  }

  calculateCPUUsage() {
    // Simple CPU usage calculation
    // This is a simplified version - in production, you'd want more sophisticated monitoring
    const cpus = os.cpus();
    let totalIdle = 0;
    let totalTick = 0;

    cpus.forEach(cpu => {
      for (const type in cpu.times) {
        totalTick += cpu.times[type];
      }
      totalIdle += cpu.times.idle;
    });

    const idle = totalIdle / cpus.length;
    const total = totalTick / cpus.length;
    const usage = 100 - (100 * idle / total);

    return Math.round(usage * 100) / 100;
  }

  async getDiskUsage() {
    try {
      const stats = await fs.statfs ? fs.statfs(process.cwd()) : null;

      if (stats) {
        const total = stats.blocks * stats.blksize;
        const free = stats.bavail * stats.blksize;
        const used = total - free;
        const usagePercent = (used / total) * 100;

        return {
          total,
          free,
          used,
          usagePercent: Math.round(usagePercent * 100) / 100
        };
      }
    } catch (error) {
      // Fallback for systems without statfs
    }

    return {
      total: 0,
      free: 0,
      used: 0,
      usagePercent: 0,
      error: 'Unable to retrieve disk usage'
    };
  }

  async measureEventLoopLag() {
    return new Promise((resolve) => {
      const start = process.hrtime.bigint();
      setImmediate(() => {
        const lag = Number(process.hrtime.bigint() - start) / 1000000; // Convert to milliseconds
        resolve(Math.round(lag * 100) / 100);
      });
    });
  }

  async checkThresholds() {
    const alerts = [];
    const now = Date.now();

    // CPU threshold
    if (this.metrics.system.cpu?.usage > this.config.thresholds.cpu) {
      const alertKey = 'cpu_high';
      if (!this.lastAlerts.has(alertKey) ||
          (now - this.lastAlerts.get(alertKey)) > this.config.alertCooldown) {

        alerts.push({
          type: 'cpu_high',
          severity: 'warning',
          message: `CPU usage is ${this.metrics.system.cpu.usage.toFixed(1)}% (threshold: ${this.config.thresholds.cpu}%)`,
          timestamp: new Date().toISOString(),
          value: this.metrics.system.cpu.usage,
          threshold: this.config.thresholds.cpu
        });

        this.lastAlerts.set(alertKey, now);
      }
    }

    // Memory threshold
    if (this.metrics.system.memory?.usagePercent > this.config.thresholds.memory) {
      const alertKey = 'memory_high';
      if (!this.lastAlerts.has(alertKey) ||
          (now - this.lastAlerts.get(alertKey)) > this.config.alertCooldown) {

        alerts.push({
          type: 'memory_high',
          severity: 'warning',
          message: `Memory usage is ${this.metrics.system.memory.usagePercent.toFixed(1)}% (threshold: ${this.config.thresholds.memory}%)`,
          timestamp: new Date().toISOString(),
          value: this.metrics.system.memory.usagePercent,
          threshold: this.config.thresholds.memory
        });

        this.lastAlerts.set(alertKey, now);
      }
    }

    // Disk threshold
    if (this.metrics.system.disk?.usagePercent > this.config.thresholds.disk) {
      const alertKey = 'disk_high';
      if (!this.lastAlerts.has(alertKey) ||
          (now - this.lastAlerts.get(alertKey)) > this.config.alertCooldown) {

        alerts.push({
          type: 'disk_high',
          severity: 'critical',
          message: `Disk usage is ${this.metrics.system.disk.usagePercent.toFixed(1)}% (threshold: ${this.config.thresholds.disk}%)`,
          timestamp: new Date().toISOString(),
          value: this.metrics.system.disk.usagePercent,
          threshold: this.config.thresholds.disk
        });

        this.lastAlerts.set(alertKey, now);
      }
    }

    if (alerts.length > 0) {
      this.metrics.alerts = alerts;
      for (const alert of alerts) {
        this.emit('alert', alert);
        console.warn(`🚨 ALERT: ${alert.message}`);
      }
    }
  }

  async logMetrics() {
    const logEntry = {
      timestamp: new Date().toISOString(),
      system: {
        cpu: this.metrics.system.cpu?.usage,
        memory: this.metrics.system.memory?.usagePercent,
        disk: this.metrics.system.disk?.usagePercent,
        uptime: this.metrics.system.uptime
      },
      application: {
        heapUsage: this.metrics.application.process?.memory?.heapUsagePercent,
        eventLoopLag: this.metrics.application.process?.eventLoopLag,
        activeHandles: this.metrics.application.process?.activeHandles
      },
      alerts: this.metrics.alerts
    };

    const logLine = JSON.stringify(logEntry) + '\n';

    try {
      await fs.appendFile(this.config.logPath, logLine);
    } catch (error) {
      console.error('❌ Error writing to log file:', error);
    }
  }

  async storeMetrics() {
    const filename = `metrics-${new Date().toISOString().split('T')[0]}.json`;
    const filepath = path.join(this.config.metricsPath, filename);

    try {
      // Load existing data or create new
      let dailyMetrics = [];
      try {
        const existing = await fs.readFile(filepath, 'utf8');
        dailyMetrics = JSON.parse(existing);
      } catch {
        // File doesn't exist, start with empty array
      }

      // Add current metrics
      dailyMetrics.push({
        timestamp: new Date().toISOString(),
        ...this.metrics
      });

      // Keep only last 24 hours of data (assuming 30s intervals = 2880 data points)
      if (dailyMetrics.length > 2880) {
        dailyMetrics = dailyMetrics.slice(-2880);
      }

      await fs.writeFile(filepath, JSON.stringify(dailyMetrics, null, 2));

    } catch (error) {
      console.error('❌ Error storing metrics:', error);
    }
  }

  async ensureDirectories() {
    const dirs = [
      path.dirname(this.config.logPath),
      this.config.metricsPath
    ];

    for (const dir of dirs) {
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        if (error.code !== 'EEXIST') {
          console.error(`❌ Error creating directory ${dir}:`, error);
        }
      }
    }
  }

  getHealthStatus() {
    const cpu = this.metrics.system.cpu?.usage || 0;
    const memory = this.metrics.system.memory?.usagePercent || 0;
    const disk = this.metrics.system.disk?.usagePercent || 0;

    const status = {
      healthy: true,
      timestamp: new Date().toISOString(),
      scores: {
        cpu: cpu < this.config.thresholds.cpu ? 'good' : 'warning',
        memory: memory < this.config.thresholds.memory ? 'good' : 'warning',
        disk: disk < this.config.thresholds.disk ? 'good' : 'critical'
      },
      metrics: {
        cpu: `${cpu.toFixed(1)}%`,
        memory: `${memory.toFixed(1)}%`,
        disk: `${disk.toFixed(1)}%`
      }
    };

    status.healthy = Object.values(status.scores).every(score => score === 'good');

    return status;
  }

  getCurrentMetrics() {
    return this.metrics;
  }
}

// Export for use in other modules
module.exports = InfrastructureMonitor;

// CLI usage
if (require.main === module) {
  const monitor = new InfrastructureMonitor({
    interval: process.env.MONITOR_INTERVAL || 30000,
    cpuThreshold: process.env.CPU_THRESHOLD || 80,
    memoryThreshold: process.env.MEMORY_THRESHOLD || 85,
    diskThreshold: process.env.DISK_THRESHOLD || 90
  });

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\n🛑 Received SIGINT, shutting down gracefully...');
    await monitor.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    console.log('\n🛑 Received SIGTERM, shutting down gracefully...');
    await monitor.stop();
    process.exit(0);
  });

  // Event handlers
  monitor.on('alert', (alert) => {
    console.log(`🚨 ${alert.severity.toUpperCase()}: ${alert.message}`);
  });

  monitor.on('error', (error) => {
    console.error('❌ Monitor error:', error);
  });

  // Start monitoring
  monitor.start().catch(error => {
    console.error('❌ Failed to start monitor:', error);
    process.exit(1);
  });
}