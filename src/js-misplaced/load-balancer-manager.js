"use strict";
/**
 * Load Balancer Manager - Real Infrastructure Integration
 *
 * Provides genuine load balancer integration for traffic switching
 * Supports Nginx, HAProxy, AWS ALB, and Kubernetes Ingress
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
exports.LoadBalancerManager = void 0;
class LoadBalancerManager {
    constructor(environment) {
        this.config = this.detectLoadBalancerType(environment);
    }
    /**
     * Update traffic weights between blue and green environments
     */
    async updateWeights(weights) {
        const totalWeight = weights.blue + weights.green;
        if (Math.abs(totalWeight - 100) > 0.1) {
            throw new Error(`Invalid traffic weights: ${weights.blue}% + ${weights.green}% != 100%`);
        }
        switch (this.config.type) {
            case 'nginx':
                await this.updateNginxWeights(weights);
                break;
            case 'haproxy':
                await this.updateHAProxyWeights(weights);
                break;
            case 'aws-alb':
                await this.updateAWSALBWeights(weights);
                break;
            case 'k8s-ingress':
                await this.updateKubernetesWeights(weights);
                break;
            default:
                throw new Error(`Unsupported load balancer type: ${this.config.type}`);
        }
        // Wait for configuration propagation
        await this.waitForPropagation();
    }
    /**
     * Verify current traffic distribution
     */
    async verifyTrafficDistribution(expectedGreenPercentage) {
        try {
            // Sample traffic for 30 seconds to verify distribution
            const samples = await this.sampleTrafficDistribution(30000, 100);
            const actualGreenPercentage = (samples.greenRequests / samples.totalRequests) * 100;
            const tolerance = 5; // 5% tolerance
            const withinTolerance = Math.abs(actualGreenPercentage - expectedGreenPercentage) <= tolerance;
            return {
                success: withinTolerance,
                actualDistribution: {
                    blue: 100 - actualGreenPercentage,
                    green: actualGreenPercentage
                },
                error: withinTolerance ? undefined :
                    `Traffic distribution mismatch: expected ${expectedGreenPercentage}%, actual ${actualGreenPercentage.toFixed(1)}%`
            };
        }
        catch (error) {
            return {
                success: false,
                error: `Traffic verification failed: ${error.message}`
            };
        }
    }
    /**
     * Update Nginx upstream weights
     */
    async updateNginxWeights(weights) {
        const nginxConfig = `
upstream backend {
    server blue.internal:8080 weight=${weights.blue};
    server green.internal:8080 weight=${weights.green};
}`;
        try {
            // Real Nginx API call
            const response = await fetch(`${this.config.endpoint}/api/6/http/upstreams/backend/servers`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.config.credentials?.accessKey}`
                },
                body: JSON.stringify({
                    servers: [
                        { server: 'blue.internal:8080', weight: weights.blue },
                        { server: 'green.internal:8080', weight: weights.green }
                    ]
                })
            });
            if (!response.ok) {
                throw new Error(`Nginx update failed: ${response.status} ${response.statusText}`);
            }
            console.log(`Nginx weights updated: blue=${weights.blue}%, green=${weights.green}%`);
        }
        catch (error) {
            console.error('Nginx weight update failed:', error);
            throw error;
        }
    }
    /**
     * Update HAProxy server weights
     */
    async updateHAProxyWeights(weights) {
        try {
            // HAProxy stats socket commands
            const commands = [
                `set weight backend/blue ${weights.blue}`,
                `set weight backend/green ${weights.green}`
            ];
            for (const command of commands) {
                const response = await fetch(`${this.config.endpoint}/stats`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'text/plain'
                    },
                    body: command
                });
                if (!response.ok) {
                    throw new Error(`HAProxy command failed: ${command}`);
                }
            }
            console.log(`HAProxy weights updated: blue=${weights.blue}%, green=${weights.green}%`);
        }
        catch (error) {
            console.error('HAProxy weight update failed:', error);
            throw error;
        }
    }
    /**
     * Update AWS Application Load Balancer weights
     */
    async updateAWSALBWeights(weights) {
        try {
            // AWS SDK v3 ELBv2 client
            const elbv2 = await Promise.resolve().then(() => __importStar(require('@aws-sdk/client-elastic-load-balancing-v2')));
            const client = new elbv2.ElasticLoadBalancingV2Client({
                region: this.config.credentials?.region,
                credentials: {
                    accessKeyId: this.config.credentials?.accessKey,
                    secretAccessKey: this.config.credentials?.secretKey
                }
            });
            const command = new elbv2.ModifyTargetGroupCommand({
                TargetGroupArn: this.config.endpoint,
                HealthCheckPath: '/health',
                Targets: [
                    { Id: 'blue-target-group', Weight: weights.blue },
                    { Id: 'green-target-group', Weight: weights.green }
                ]
            });
            await client.send(command);
            console.log(`AWS ALB weights updated: blue=${weights.blue}%, green=${weights.green}%`);
        }
        catch (error) {
            console.error('AWS ALB weight update failed:', error);
            throw error;
        }
    }
    /**
     * Update Kubernetes Ingress weights
     */
    async updateKubernetesWeights(weights) {
        try {
            const k8s = await Promise.resolve().then(() => __importStar(require('@kubernetes/client-node')));
            const kc = new k8s.KubeConfig();
            kc.loadFromDefault();
            const k8sApi = kc.makeApiClient(k8s.NetworkingV1Api);
            // Update Ingress with traffic splitting annotations
            const ingressPatch = {
                metadata: {
                    annotations: {
                        'nginx.ingress.kubernetes.io/canary': 'true',
                        'nginx.ingress.kubernetes.io/canary-weight': weights.green.toString(),
                        'nginx.ingress.kubernetes.io/canary-by-header': 'deployment-target'
                    }
                }
            };
            await k8sApi.patchNamespacedIngress('app-ingress', 'default', ingressPatch, undefined, undefined, undefined, undefined, { headers: { 'Content-Type': 'application/merge-patch+json' } });
            console.log(`Kubernetes Ingress weights updated: blue=${weights.blue}%, green=${weights.green}%`);
        }
        catch (error) {
            console.error('Kubernetes weight update failed:', error);
            throw error;
        }
    }
    /**
     * Sample traffic distribution to verify routing
     */
    async sampleTrafficDistribution(duration, sampleCount) {
        let greenRequests = 0;
        let blueRequests = 0;
        const interval = duration / sampleCount;
        for (let i = 0; i < sampleCount; i++) {
            try {
                const response = await fetch(`${this.config.endpoint}/health`, {
                    headers: { 'X-Request-ID': `sample-${i}` }
                });
                const targetHeader = response.headers.get('X-Target-Environment');
                if (targetHeader === 'green') {
                    greenRequests++;
                }
                else {
                    blueRequests++;
                }
            }
            catch (error) {
                // Count as blue request if green is unreachable
                blueRequests++;
            }
            if (i < sampleCount - 1) {
                await new Promise(resolve => setTimeout(resolve, interval));
            }
        }
        return {
            totalRequests: greenRequests + blueRequests,
            greenRequests,
            blueRequests
        };
    }
    /**
     * Wait for configuration propagation across load balancer nodes
     */
    async waitForPropagation() {
        // Different load balancers have different propagation times
        const propagationTime = {
            'nginx': 2000, // 2 seconds
            'haproxy': 1000, // 1 second
            'aws-alb': 10000, // 10 seconds (AWS eventual consistency)
            'k8s-ingress': 5000 // 5 seconds
        };
        const waitTime = propagationTime[this.config.type] || 5000;
        await new Promise(resolve => setTimeout(resolve, waitTime));
    }
    /**
     * Detect load balancer type from environment
     */
    detectLoadBalancerType(environment) {
        // Detection logic based on environment configuration
        if (environment.platform === 'kubernetes') {
            return {
                type: 'k8s-ingress',
                endpoint: environment.ingressEndpoint || 'http://localhost:8080'
            };
        }
        if (environment.platform === 'aws') {
            return {
                type: 'aws-alb',
                endpoint: environment.albArn,
                credentials: {
                    accessKey: environment.awsAccessKey,
                    secretKey: environment.awsSecretKey,
                    region: environment.awsRegion || 'us-east-1'
                }
            };
        }
        if (environment.loadBalancer === 'nginx') {
            return {
                type: 'nginx',
                endpoint: environment.nginxEndpoint || 'http://localhost:8081',
                credentials: {
                    accessKey: environment.nginxApiKey
                }
            };
        }
        if (environment.loadBalancer === 'haproxy') {
            return {
                type: 'haproxy',
                endpoint: environment.haproxyEndpoint || 'http://localhost:8082'
            };
        }
        // Default to nginx if no specific configuration
        return {
            type: 'nginx',
            endpoint: 'http://localhost:8081'
        };
    }
}
exports.LoadBalancerManager = LoadBalancerManager;
//# sourceMappingURL=load-balancer-manager.js.map