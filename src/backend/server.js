/**
 * Familiar GM Assistant - RAG Backend Server
 * Provides Pathfinder 2e rules assistance through AI and vector search
 */

const express = require('express');
const cors = require('cors');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');

// Import service modules
const { PathfinderRulesService } = require('./services/pathfinder-rules-service');
const { VectorSearchService } = require('./services/vector-search-service');
const { ChatService } = require('./services/chat-service');
const { ContextService } = require('./services/context-service');

class FamiliarBackendServer {
    constructor(options = {}) {
        this.port = options.port || 3001;
        this.app = express();
        this.server = http.createServer(this.app);
        this.io = socketIo(this.server, {
            cors: {
                origin: "*",
                methods: ["GET", "POST"]
            }
        });

        // Services
        this.rulesService = null;
        this.vectorService = null;
        this.chatService = null;
        this.contextService = null;

        this.setupMiddleware();
        this.setupRoutes();
        this.setupSocketHandlers();
    }

    /**
     * Setup Express middleware
     */
    setupMiddleware() {
        // CORS for Foundry VTT
        this.app.use(cors({
            origin: ['http://localhost:30000', 'http://127.0.0.1:30000'], // Foundry default ports
            credentials: true
        }));

        this.app.use(express.json({ limit: '50mb' }));
        this.app.use(express.urlencoded({ extended: true }));

        // Request logging
        this.app.use((req, res, next) => {
            console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
            next();
        });
    }

    /**
     * Setup API routes
     */
    setupRoutes() {
        // Health check
        this.app.get('/api/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: Date.now(),
                services: {
                    rules: this.rulesService ? 'ready' : 'initializing',
                    vector: this.vectorService ? 'ready' : 'initializing',
                    chat: this.chatService ? 'ready' : 'initializing'
                }
            });
        });

        // Chat endpoint
        this.app.post('/api/chat', async (req, res) => {
            try {
                const { message, context, conversationHistory } = req.body;

                if (!message) {
                    return res.status(400).json({ error: 'Message is required' });
                }

                const response = await this.chatService.processMessage({
                    message,
                    context: context || {},
                    history: conversationHistory || []
                });

                res.json(response);
            } catch (error) {
                console.error('Chat API Error:', error);
                res.status(500).json({
                    error: 'Internal server error',
                    response: 'I encountered an error processing your request. Please try again.',
                    type: 'error'
                });
            }
        });

        // Rules search endpoint
        this.app.post('/api/rules/search', async (req, res) => {
            try {
                const { query, category, limit } = req.body;

                const results = await this.rulesService.searchRules({
                    query,
                    category,
                    limit: limit || 5
                });

                res.json({ results });
            } catch (error) {
                console.error('Rules Search Error:', error);
                res.status(500).json({ error: 'Failed to search rules' });
            }
        });

        // Spell lookup endpoint
        this.app.get('/api/spells/:name', async (req, res) => {
            try {
                const spellName = req.params.name;
                const spell = await this.rulesService.getSpell(spellName);

                if (!spell) {
                    return res.status(404).json({ error: 'Spell not found' });
                }

                res.json({ spell });
            } catch (error) {
                console.error('Spell Lookup Error:', error);
                res.status(500).json({ error: 'Failed to lookup spell' });
            }
        });

        // Combat rules endpoint
        this.app.post('/api/combat/resolve', async (req, res) => {
            try {
                const { action, context } = req.body;

                const resolution = await this.rulesService.resolveCombatAction({
                    action,
                    context
                });

                res.json({ resolution });
            } catch (error) {
                console.error('Combat Resolution Error:', error);
                res.status(500).json({ error: 'Failed to resolve combat action' });
            }
        });

        // Vector similarity endpoint
        this.app.post('/api/vector/similar', async (req, res) => {
            try {
                const { text, limit } = req.body;

                const similar = await this.vectorService.findSimilar(text, limit || 5);
                res.json({ similar });
            } catch (error) {
                console.error('Vector Similarity Error:', error);
                res.status(500).json({ error: 'Failed to find similar content' });
            }
        });

        // Context analysis endpoint
        this.app.post('/api/context/analyze', async (req, res) => {
            try {
                const { context } = req.body;

                const analysis = await this.contextService.analyzeContext(context);
                res.json({ analysis });
            } catch (error) {
                console.error('Context Analysis Error:', error);
                res.status(500).json({ error: 'Failed to analyze context' });
            }
        });

        // Static files (if needed for development)
        this.app.use('/static', express.static(path.join(__dirname, '../ui')));

        // 404 handler
        this.app.use((req, res) => {
            res.status(404).json({ error: 'Endpoint not found' });
        });

        // Error handler
        this.app.use((error, req, res, next) => {
            console.error('Unhandled Error:', error);
            res.status(500).json({ error: 'Internal server error' });
        });
    }

    /**
     * Setup Socket.IO handlers
     */
    setupSocketHandlers() {
        this.io.on('connection', (socket) => {
            console.log('Client connected:', socket.id);

            // Handle real-time chat
            socket.on('chat-message', async (data) => {
                try {
                    // Emit thinking indicator
                    socket.emit('familiar-thinking');

                    const response = await this.chatService.processMessage({
                        message: data.message,
                        context: data.context || {},
                        history: data.history || []
                    });

                    // Emit response
                    socket.emit('familiar-response', response);
                } catch (error) {
                    console.error('Socket Chat Error:', error);
                    socket.emit('familiar-response', {
                        response: 'I encountered an error processing your request.',
                        type: 'error'
                    });
                }
            });

            // Handle context updates
            socket.on('context-update', (context) => {
                socket.emit('familiar-context-update', context);
            });

            // Handle disconnection
            socket.on('disconnect', () => {
                console.log('Client disconnected:', socket.id);
            });
        });
    }

    /**
     * Initialize all services
     */
    async initializeServices() {
        try {
            console.log('Initializing Familiar Backend Services...');

            // Initialize Pathfinder Rules Service
            this.rulesService = new PathfinderRulesService();
            await this.rulesService.initialize();

            // Initialize Vector Search Service
            this.vectorService = new VectorSearchService();
            await this.vectorService.initialize();

            // Initialize Context Service
            this.contextService = new ContextService();
            await this.contextService.initialize();

            // Initialize Chat Service (depends on other services)
            this.chatService = new ChatService({
                rulesService: this.rulesService,
                vectorService: this.vectorService,
                contextService: this.contextService
            });
            await this.chatService.initialize();

            console.log('All services initialized successfully');
        } catch (error) {
            console.error('Service initialization failed:', error);
            throw error;
        }
    }

    /**
     * Start the server
     */
    async start() {
        try {
            await this.initializeServices();

            this.server.listen(this.port, () => {
                console.log(`
===========================================
🐦‍⬛ Familiar GM Assistant Backend Started
===========================================
Server: http://localhost:${this.port}
API: http://localhost:${this.port}/api
Socket.IO: ws://localhost:${this.port}
Health: http://localhost:${this.port}/api/health

Services Status:
✓ Pathfinder Rules Service
✓ Vector Search Service
✓ Chat Service
✓ Context Service
===========================================
                `);
            });

            // Graceful shutdown
            process.on('SIGTERM', () => this.shutdown());
            process.on('SIGINT', () => this.shutdown());

        } catch (error) {
            console.error('Failed to start server:', error);
            process.exit(1);
        }
    }

    /**
     * Shutdown server gracefully
     */
    async shutdown() {
        console.log('Shutting down Familiar Backend...');

        this.server.close(() => {
            console.log('Server closed');
            process.exit(0);
        });

        // Close services
        if (this.vectorService) {
            await this.vectorService.close();
        }
        if (this.rulesService) {
            await this.rulesService.close();
        }
    }

    /**
     * Get server statistics
     */
    getStats() {
        return {
            uptime: process.uptime(),
            memory: process.memoryUsage(),
            connections: this.io.engine.clientsCount,
            services: {
                rules: this.rulesService?.getStats(),
                vector: this.vectorService?.getStats(),
                chat: this.chatService?.getStats()
            }
        };
    }
}

// Create and start server if run directly
if (require.main === module) {
    const server = new FamiliarBackendServer();
    server.start().catch(console.error);
}

module.exports = { FamiliarBackendServer };