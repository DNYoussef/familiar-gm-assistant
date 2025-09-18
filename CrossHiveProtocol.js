"use strict";
/**
 * Cross-Hive Protocol - Inter-Princess Communication System
 * Enables secure, efficient communication between princess domains
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.CrossHiveProtocol = void 0;
const events_1 = require("events");
const crypto = __importStar(require("crypto"));
const ContextDNA_1 = require("../../context/ContextDNA");
const ContextValidator_1 = require("../../context/ContextValidator");
class CrossHiveProtocol extends events_1.EventEmitter {
    constructor(princesses, consensus) {
        super();
        this.princesses = princesses;
        this.consensus = consensus;
        this.channels = new Map();
        this.messageQueues = new Map();
        this.messageHistory = new Map();
        this.maxRetries = 3;
        this.messageTimeout = 5000;
        this.heartbeatInterval = 10000;
        this.maxMessageHistory = 1000;
        this.protocolVersion = '1.0.0';
        this.contextDNA = new ContextDNA_1.ContextDNA();
        this.validator = new ContextValidator_1.ContextValidator();
        this.protocolState = {
            synchronized: false,
            lastSyncTime: new Date(),
            versionVector: new Map(),
            pendingAcks: new Map()
        };
        this.initializeProtocol();
    }
    /**
     * Initialize the cross-hive communication protocol
     */
    async initializeProtocol() {
        // Create default channels
        this.createDefaultChannels();
        // Initialize message queues for each princess
        this.initializeMessageQueues();
        // Start heartbeat mechanism
        this.startHeartbeat();
        // Initialize version vectors
        this.initializeVersionVectors();
        // Perform initial synchronization
        await this.synchronizeHives();
        this.emit('protocol:initialized');
    }
    /**
     * Send a message to a target princess or broadcast
     */
    async sendMessage(source, target, payload, options = {}) {
        const message = {
            id: this.generateMessageId(),
            type: options.type || 'request',
            source,
            target,
            payload,
            timestamp: Date.now(),
            signature: await this.signMessage(source, payload),
            priority: options.priority || 'medium',
            requiresAck: options.requiresAck ?? true,
            ttl: options.ttl || 5,
            hops: [source]
        };
        // Validate message
        if (!await this.validateMessage(message)) {
            throw new Error('Message validation failed');
        }
        // Route message
        if (target === 'all') {
            await this.broadcastMessage(message, options.channel);
        }
        else {
            await this.routeMessage(message, options.channel);
        }
        // Store in history and persistence queue
        this.addToHistory(message);
        this.addToPersistenceQueue(message);
        // Handle acknowledgment if required
        if (message.requiresAck) {
            this.protocolState.pendingAcks.set(message.id, message);
            this.scheduleAckTimeout(message);
        }
        return message.id;
    }
    /**
     * Broadcast message to all princesses
     */
    async broadcastMessage(message, channelName) {
        const channel = channelName
            ? this.channels.get(channelName)
            : this.channels.get('broadcast');
        if (!channel) {
            throw new Error(`Channel ${channelName} not found`);
        }
        // Send to all participants except source
        const targets = Array.from(channel.participants)
            .filter(p => p !== message.source);
        const broadcasts = targets.map(async (target) => {
            try {
                await this.deliverMessage(message, target);
            }
            catch (error) {
                console.error(`Broadcast to ${target} failed:`, error);
                this.handleDeliveryFailure(message, target);
            }
        });
        await Promise.allSettled(broadcasts);
        this.emit('message:broadcast', message);
    }
    /**
     * Route message to specific target
     */
    async routeMessage(message, channelName) {
        const channel = this.selectOptimizedChannel(message, channelName);
        if (!channel.participants.has(message.target)) {
            throw new Error(`Target ${message.target} not in channel ${channel.name}`);
        }
        try {
            await this.deliverMessage(message, message.target);
            this.emit('message:sent', message);
        }
        catch (error) {
            console.error(`Failed to route message to ${message.target}:`, error);
            this.handleDeliveryFailure(message, message.target);
        }
    }
    /**
     * Deliver message to target princess
     */
    async deliverMessage(message, target) {
        const queue = this.messageQueues.get(target);
        if (!queue) {
            throw new Error(`No queue for princess ${target}`);
        }
        // Add to queue
        queue.queue.push(message);
        // Process queue if not already processing
        if (!queue.processing) {
            await this.processMessageQueue(target);
        }
    }
    /**
     * Process message queue for a princess
     */
    async processMessageQueue(princessId) {
        const queue = this.messageQueues.get(princessId);
        if (!queue || queue.queue.length === 0)
            return;
        queue.processing = true;
        while (queue.queue.length > 0) {
            const message = queue.queue.shift();
            try {
                // Check TTL
                if (message.ttl <= 0) {
                    console.warn(`Message ${message.id} expired (TTL=0)`);
                    continue;
                }
                // Get princess instance
                const princess = this.princesses.get(princessId);
                if (!princess) {
                    throw new Error(`Princess ${princessId} not found`);
                }
                // Deliver based on message type
                await this.handleMessageDelivery(message, princess);
                // Update metrics
                queue.lastContact = new Date();
                // Send acknowledgment if required
                if (message.requiresAck && message.source !== princessId) {
                    await this.sendAcknowledgment(message);
                }
            }
            catch (error) {
                console.error(`Failed to process message for ${princessId}:`, error);
                await this.retryMessage(message, princessId);
            }
        }
        queue.processing = false;
    }
    /**
     * Handle message delivery based on type
     */
    async handleMessageDelivery(message, princess) {
        switch (message.type) {
            case 'request':
                const response = await princess.handleRequest(message.payload);
                if (response) {
                    await this.sendMessage(princess.id, message.source, response, { type: 'response', requiresAck: false });
                }
                break;
            case 'response':
                await princess.handleResponse(message.payload);
                break;
            case 'broadcast':
                await princess.handleBroadcast(message.payload);
                break;
            case 'sync':
                await this.handleSyncMessage(message, princess);
                break;
            case 'heartbeat':
                this.updateHeartbeat(message.source);
                break;
        }
    }
    /**
     * Send acknowledgment for received message
     */
    async sendAcknowledgment(message) {
        await this.sendMessage(message.target, message.source, { ack: message.id, timestamp: Date.now() }, { type: 'response', requiresAck: false, priority: message.priority });
    }
    /**
     * Handle synchronization message
     */
    async handleSyncMessage(message, princess) {
        const syncData = message.payload;
        // Update version vector
        if (syncData.versionVector) {
            this.mergeVersionVectors(syncData.versionVector);
        }
        // Sync context if provided
        if (syncData.context) {
            await princess.handleSync(syncData.context);
        }
        // Update protocol state
        this.protocolState.synchronized = true;
        this.protocolState.lastSyncTime = new Date();
    }
    /**
     * Retry failed message delivery
     */
    async retryMessage(message, target) {
        const queue = this.messageQueues.get(target);
        if (!queue)
            return;
        const retryCount = queue.retryCount.get(message.id) || 0;
        if (retryCount < this.maxRetries) {
            queue.retryCount.set(message.id, retryCount + 1);
            // Exponential backoff
            const delay = Math.pow(2, retryCount) * 1000;
            setTimeout(() => {
                message.ttl--; // Decrement TTL
                queue.queue.push(message);
                this.processMessageQueue(target);
            }, delay);
        }
        else {
            console.error(`Max retries exceeded for message ${message.id} to ${target}`);
            this.emit('message:failed', { message, target });
        }
    }
    /**
     * Handle delivery failure
     */
    handleDeliveryFailure(message, target) {
        // Remove from pending acks
        this.protocolState.pendingAcks.delete(message.id);
        // Emit failure event
        this.emit('delivery:failed', { message, target });
        // Attempt recovery through consensus if available
        if (this.consensus) {
            this.consensus.propose(message.source, 'recovery', {
                failedMessage: message,
                target,
                reason: 'delivery_failure'
            });
        }
    }
    /**
     * Create default communication channels
     */
    createDefaultChannels() {
        // Broadcast channel for all princesses
        this.createChannel('broadcast', {
            type: 'broadcast',
            encrypted: false,
            reliability: 'best-effort',
            participants: Array.from(this.princesses.keys())
        });
        // Consensus channel for critical decisions
        this.createChannel('consensus', {
            type: 'consensus',
            encrypted: true,
            reliability: 'exactly-once',
            participants: Array.from(this.princesses.keys())
        });
        // Direct channels between princess pairs
        const princessIds = Array.from(this.princesses.keys());
        for (let i = 0; i < princessIds.length; i++) {
            for (let j = i + 1; j < princessIds.length; j++) {
                const channelName = `${princessIds[i]}-${princessIds[j]}`;
                this.createChannel(channelName, {
                    type: 'direct',
                    encrypted: true,
                    reliability: 'at-least-once',
                    participants: [princessIds[i], princessIds[j]]
                });
            }
        }
    }
    /**
     * Create a new communication channel
     */
    createChannel(name, options) {
        const channel = {
            id: crypto.randomBytes(16).toString('hex'),
            name,
            participants: new Set(options.participants),
            type: options.type,
            encrypted: options.encrypted,
            reliability: options.reliability,
            maxMessageSize: options.maxMessageSize || 1048576, // 1MB default
            rateLimit: options.rateLimit || 100 // 100 msg/s default
        };
        this.channels.set(name, channel);
        this.emit('channel:created', channel);
        return channel;
    }
    /**
     * Select appropriate channel for message
     */
    selectChannel(message, preferredChannel) {
        if (preferredChannel && this.channels.has(preferredChannel)) {
            return this.channels.get(preferredChannel);
        }
        // Select based on message priority and type
        if (message.priority === 'critical' || message.type === 'sync') {
            return this.channels.get('consensus');
        }
        // Try direct channel
        const directChannel = `${message.source}-${message.target}`;
        const reverseChannel = `${message.target}-${message.source}`;
        if (this.channels.has(directChannel)) {
            return this.channels.get(directChannel);
        }
        else if (this.channels.has(reverseChannel)) {
            return this.channels.get(reverseChannel);
        }
        // Default to broadcast channel
        return this.channels.get('broadcast');
    }
    /**
     * Initialize message queues for princesses
     */
    initializeMessageQueues() {
        for (const princessId of this.princesses.keys()) {
            this.messageQueues.set(princessId, {
                princess: princessId,
                queue: [],
                processing: false,
                retryCount: new Map(),
                lastContact: new Date()
            });
        }
    }
    /**
     * Initialize version vectors for consistency
     */
    initializeVersionVectors() {
        for (const princessId of this.princesses.keys()) {
            this.protocolState.versionVector.set(princessId, 0);
        }
    }
    /**
     * Start heartbeat mechanism
     */
    startHeartbeat() {
        this.heartbeatTimer = setInterval(async () => {
            for (const princessId of this.princesses.keys()) {
                await this.sendMessage('protocol', princessId, { timestamp: Date.now(), version: this.protocolVersion }, { type: 'heartbeat', requiresAck: false, priority: 'low' });
            }
            // Check for dead princesses
            this.checkPrincessHealth();
        }, this.heartbeatInterval);
    }
    /**
     * Check princess health based on last contact
     */
    checkPrincessHealth() {
        const deadlineMs = Date.now() - (this.heartbeatInterval * 3);
        for (const [princessId, queue] of this.messageQueues) {
            if (queue.lastContact.getTime() < deadlineMs) {
                console.warn(`Princess ${princessId} appears to be unresponsive`);
                this.emit('princess:unresponsive', { princess: princessId });
                // Attempt recovery
                this.attemptPrincessRecovery(princessId);
            }
        }
    }
    /**
     * Attempt to recover unresponsive princess
     */
    async attemptPrincessRecovery(princessId) {
        // Send high-priority sync message
        try {
            await this.sendMessage('protocol', princessId, { recovery: true, timestamp: Date.now() }, { type: 'sync', priority: 'critical', requiresAck: true });
        }
        catch (error) {
            console.error(`Failed to recover princess ${princessId}:`, error);
            // Notify consensus system if available
            if (this.consensus) {
                await this.consensus.propose('protocol', 'escalation', { unresponsivePrincess: princessId });
            }
        }
    }
    /**
     * Synchronize state across all hives
     */
    async synchronizeHives() {
        const syncData = {
            versionVector: Object.fromEntries(this.protocolState.versionVector),
            timestamp: Date.now(),
            protocol: this.protocolVersion
        };
        await this.sendMessage('protocol', 'all', syncData, { type: 'sync', priority: 'high' });
        this.protocolState.synchronized = true;
        this.protocolState.lastSyncTime = new Date();
        this.emit('hives:synchronized');
    }
    /**
     * Merge version vectors for consistency
     */
    mergeVersionVectors(incoming) {
        for (const [princess, version] of Object.entries(incoming)) {
            const current = this.protocolState.versionVector.get(princess) || 0;
            this.protocolState.versionVector.set(princess, Math.max(current, version));
        }
    }
    /**
     * Update heartbeat timestamp
     */
    updateHeartbeat(princessId) {
        const queue = this.messageQueues.get(princessId);
        if (queue) {
            queue.lastContact = new Date();
        }
    }
    /**
     * Schedule timeout for acknowledgment
     */
    scheduleAckTimeout(message) {
        setTimeout(() => {
            if (this.protocolState.pendingAcks.has(message.id)) {
                console.warn(`Acknowledgment timeout for message ${message.id}`);
                this.protocolState.pendingAcks.delete(message.id);
                this.emit('ack:timeout', message);
            }
        }, this.messageTimeout);
    }
    /**
     * Process received acknowledgment
     */
    processAcknowledgment(messageId) {
        if (this.protocolState.pendingAcks.has(messageId)) {
            this.protocolState.pendingAcks.delete(messageId);
            this.emit('ack:received', messageId);
        }
    }
    /**
     * Helper functions
     */
    generateMessageId() {
        return `msg_${Date.now()}_${crypto.randomBytes(8).toString('hex')}`;
    }
    async signMessage(source, payload) {
        const data = JSON.stringify({ source, payload, timestamp: Date.now() });
        return crypto.createHash('sha256').update(data).digest('hex');
    }
    async validateMessage(message) {
        // Check message structure
        if (!message.id || !message.source || !message.target) {
            return false;
        }
        // Verify signature
        const expectedSig = await this.signMessage(message.source, message.payload);
        if (message.signature !== expectedSig) {
            console.warn(`Invalid signature for message ${message.id}`);
            return false;
        }
        // Check message size
        const size = JSON.stringify(message).length;
        const channel = this.selectOptimizedChannel(message);
        if (size > channel.maxMessageSize) {
            console.warn(`Message ${message.id} exceeds size limit`);
            return false;
        }
        return true;
    }
    addToHistory(message) {
        this.messageHistory.set(message.id, message);
        // Prune old messages
        if (this.messageHistory.size > this.maxMessageHistory) {
            const oldestMessages = Array.from(this.messageHistory.entries())
                .sort((a, b) => a[1].timestamp - b[1].timestamp)
                .slice(0, this.messageHistory.size - this.maxMessageHistory);
            for (const [id] of oldestMessages) {
                this.messageHistory.delete(id);
            }
        }
    }
    /**
     * Get protocol metrics
     */
    getMetrics() {
        let sent = 0;
        let received = 0;
        let dropped = 0;
        for (const message of this.messageHistory.values()) {
            if (message.source === 'protocol')
                sent++;
            else
                received++;
        }
        const channelUtilization = new Map();
        for (const [name, channel] of this.channels) {
            // Calculate utilization based on queue sizes
            let utilization = 0;
            for (const participant of channel.participants) {
                const queue = this.messageQueues.get(participant);
                if (queue) {
                    utilization += queue.queue.length;
                }
            }
            channelUtilization.set(name, utilization / channel.participants.size);
        }
        return {
            messagesSent: sent,
            messagesReceived: received,
            messagesDropped: dropped,
            averageLatency: this.calculateAverageLatency(),
            channelUtilization,
            protocolVersion: this.protocolVersion
        };
    }
    calculateAverageLatency() {
        // This would track actual message round-trip times
        // Placeholder for now
        return 50;
    }
    /**
     * Cleanup and shutdown
     */
    shutdown() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
        }
        this.emit('protocol:shutdown');
    }
}
exports.CrossHiveProtocol = CrossHiveProtocol;
