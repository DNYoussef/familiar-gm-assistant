"use strict";
/**
 * Automated Remediation Orchestrator
 * Implements automated remediation workflows with escalation and tracking
 *
 * Component of Task EC-006: Real-time compliance monitoring with automated remediation workflows
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.RemediationOrchestrator = void 0;
const events_1 = require("events");
class RemediationOrchestrator extends events_1.EventEmitter {
    constructor(config) {
        super();
        this.remediationPlans = new Map();
        this.activeExecutions = new Map();
        this.executionHistory = [];
        this.config = config;
        this.actionExecutor = new ActionExecutor();
        this.approvalManager = new ApprovalManager(config);
        this.rollbackManager = new RollbackManager();
        this.initializeRemediationOrchestrator();
    }
    /**
     * Initialize remediation orchestrator
     */
    initializeRemediationOrchestrator() {
        this.setupDefaultRemediationPlans();
        this.emit('orchestrator_initialized', { timestamp: new Date() });
    }
    /**
     * Setup default remediation plans
     */
    setupDefaultRemediationPlans() {
        const defaultPlans = [
            {
                id: 'compliance-drift-standard',
                name: 'Standard Compliance Drift Remediation',
                description: 'Addresses compliance score degradation through automated controls',
                type: 'compliance_drift',
                severity: 'medium',
                trigger: 'automatic',
                steps: [
                    {
                        id: 'assess-drift',
                        name: 'Assess Compliance Drift',
                        description: 'Analyze the extent and cause of compliance drift',
                        type: 'validation',
                        action: {
                            type: 'validation_check',
                            implementation: 'ComplianceAssessment.analyzeDrift',
                            parameters: { includeRootCause: true, detailedAnalysis: true },
                            timeout: 300000,
                            authentication: { type: 'service_account', credentials: 'compliance_service', scope: ['read'], timeout: 60000 },
                            logging: true
                        },
                        parameters: {},
                        timeout: 300000,
                        retryCount: 0,
                        maxRetries: 2,
                        onSuccess: ['update-controls'],
                        onFailure: ['escalate-manual'],
                        validationCriteria: [
                            { type: 'status', condition: 'assessment.completed', expectedValue: true, timeout: 300000 }
                        ],
                        status: 'pending'
                    },
                    {
                        id: 'update-controls',
                        name: 'Update Control Configurations',
                        description: 'Apply corrective control configurations',
                        type: 'configuration',
                        action: {
                            type: 'config_update',
                            implementation: 'ControlManager.updateConfigurations',
                            parameters: { applyBestPractices: true, validateChanges: true },
                            timeout: 600000,
                            authentication: { type: 'service_account', credentials: 'config_service', scope: ['write'], timeout: 60000 },
                            logging: true
                        },
                        parameters: {},
                        timeout: 600000,
                        retryCount: 0,
                        maxRetries: 1,
                        onSuccess: ['validate-remediation'],
                        onFailure: ['rollback-changes'],
                        rollbackAction: {
                            type: 'config_update',
                            implementation: 'ControlManager.rollbackConfigurations',
                            parameters: { preserveAuditTrail: true },
                            timeout: 300000,
                            authentication: { type: 'service_account', credentials: 'config_service', scope: ['write'], timeout: 60000 },
                            logging: true
                        },
                        validationCriteria: [
                            { type: 'config', condition: 'controls.updated', expectedValue: true, timeout: 600000 }
                        ],
                        status: 'pending'
                    },
                    {
                        id: 'validate-remediation',
                        name: 'Validate Remediation Success',
                        description: 'Verify that compliance drift has been corrected',
                        type: 'validation',
                        action: {
                            type: 'validation_check',
                            implementation: 'ComplianceValidator.validateRemediation',
                            parameters: { thresholdCheck: true, trendAnalysis: true },
                            timeout: 300000,
                            authentication: { type: 'service_account', credentials: 'validation_service', scope: ['read'], timeout: 60000 },
                            logging: true
                        },
                        parameters: {},
                        timeout: 300000,
                        retryCount: 0,
                        maxRetries: 3,
                        onSuccess: ['notify-success'],
                        onFailure: ['escalate-manual'],
                        validationCriteria: [
                            { type: 'metric', condition: 'compliance_score > baseline', expectedValue: true, timeout: 300000 }
                        ],
                        status: 'pending'
                    },
                    {
                        id: 'notify-success',
                        name: 'Notify Successful Remediation',
                        description: 'Send notification of successful remediation',
                        type: 'notification',
                        action: {
                            type: 'notification',
                            implementation: 'NotificationService.send',
                            parameters: { template: 'remediation_success', includeMetrics: true },
                            timeout: 60000,
                            authentication: { type: 'api_key', credentials: 'notification_api_key', scope: ['send'], timeout: 30000 },
                            logging: true
                        },
                        parameters: {},
                        timeout: 60000,
                        retryCount: 0,
                        maxRetries: 2,
                        onSuccess: [],
                        onFailure: [],
                        validationCriteria: [],
                        status: 'pending'
                    },
                    {
                        id: 'escalate-manual',
                        name: 'Escalate for Manual Intervention',
                        description: 'Escalate issue for manual review and intervention',
                        type: 'notification',
                        action: {
                            type: 'notification',
                            implementation: 'EscalationService.escalate',
                            parameters: { priority: 'high', includeDiagnostics: true },
                            timeout: 120000,
                            authentication: { type: 'service_account', credentials: 'escalation_service', scope: ['escalate'], timeout: 60000 },
                            logging: true
                        },
                        parameters: {},
                        timeout: 120000,
                        retryCount: 0,
                        maxRetries: 1,
                        onSuccess: [],
                        onFailure: [],
                        validationCriteria: [],
                        status: 'pending'
                    }
                ],
                prerequisites: ['monitoring_active', 'controls_accessible'],
                dependencies: ['compliance_assessment_service', 'control_management_service'],
                estimatedDuration: 1800000, // 30 minutes
                riskAssessment: {
                    impactLevel: 'medium',
                    likelihood: 0.8,
                    mitigationFactors: ['automated_rollback', 'validation_checks'],
                    residualRisk: 'Low risk of service disruption',
                    businessImpact: 'Minimal impact on business operations'
                },
                approvalWorkflow: {
                    required: false,
                    approvers: [],
                    approvalType: 'any',
                    timeout: 0,
                    escalationPath: [],
                    autoApprovalConditions: ['severity <= medium', 'business_hours']
                },
                rollbackPlan: {
                    automatic: true,
                    trigger: 'failure',
                    steps: [],
                    verificationRequired: true
                },
                status: 'approved',
                created: new Date(),
                lastUpdated: new Date(),
                executedBy: 'system'
            }
        ];
        defaultPlans.forEach(plan => {
            this.remediationPlans.set(plan.id, plan);
        });
        this.emit('remediation_plans_configured', { count: defaultPlans.length });
    }
    /**
     * Create remediation plan for compliance drift
     */
    async createRemediationPlan(params) {
        const planId = `remediation-${params.type}-${Date.now()}`;
        // Select appropriate template based on type
        const templateId = this.selectPlanTemplate(params.type, params.severity);
        const template = this.remediationPlans.get(templateId);
        if (!template) {
            throw new Error(`No remediation template found for type: ${params.type}`);
        }
        // Create customized plan
        const plan = {
            ...template,
            id: planId,
            name: `${params.framework} ${params.type} Remediation`,
            description: `Automated remediation for ${params.type} in ${params.framework} framework`,
            severity: params.severity,
            steps: this.customizeRemediationSteps(template.steps, params),
            created: new Date(),
            lastUpdated: new Date(),
            status: 'draft'
        };
        this.remediationPlans.set(planId, plan);
        this.emit('remediation_plan_created', {
            planId,
            type: params.type,
            severity: params.severity,
            framework: params.framework
        });
        return plan;
    }
    /**
     * Create emergency remediation plan
     */
    async createEmergencyPlan(params) {
        const planId = `emergency-${Date.now()}`;
        const emergencyPlan = {
            id: planId,
            name: `Emergency: ${params.control} Failure`,
            description: `Emergency remediation for critical control failure`,
            type: 'control_failure',
            severity: 'critical',
            trigger: 'automatic',
            priority: 1,
            bypassApproval: true,
            immediateExecution: true,
            notificationEscalation: ['security_team_lead', 'ciso'],
            businessContinuityImpact: params.impact,
            steps: [
                {
                    id: 'immediate-containment',
                    name: 'Immediate Containment',
                    description: 'Contain the control failure impact',
                    type: 'script',
                    action: {
                        type: 'script_execution',
                        implementation: 'EmergencyResponse.containFailure',
                        parameters: { control: params.control, framework: params.framework },
                        timeout: 180000,
                        authentication: { type: 'service_account', credentials: 'emergency_service', scope: ['admin'], timeout: 60000 },
                        logging: true
                    },
                    parameters: {},
                    timeout: 180000,
                    retryCount: 0,
                    maxRetries: 1,
                    onSuccess: ['assess-impact'],
                    onFailure: ['escalate-immediate'],
                    validationCriteria: [
                        { type: 'status', condition: 'containment.successful', expectedValue: true, timeout: 180000 }
                    ],
                    status: 'pending'
                },
                {
                    id: 'assess-impact',
                    name: 'Assess Impact',
                    description: 'Assess the full impact of the control failure',
                    type: 'validation',
                    action: {
                        type: 'validation_check',
                        implementation: 'ImpactAssessment.assessControlFailure',
                        parameters: { comprehensive: true, includeDownstream: true },
                        timeout: 300000,
                        authentication: { type: 'service_account', credentials: 'assessment_service', scope: ['read'], timeout: 60000 },
                        logging: true
                    },
                    parameters: {},
                    timeout: 300000,
                    retryCount: 0,
                    maxRetries: 2,
                    onSuccess: ['implement-fix'],
                    onFailure: ['escalate-immediate'],
                    validationCriteria: [
                        { type: 'status', condition: 'assessment.completed', expectedValue: true, timeout: 300000 }
                    ],
                    status: 'pending'
                }
            ],
            prerequisites: [],
            dependencies: ['emergency_response_service'],
            estimatedDuration: 600000, // 10 minutes
            riskAssessment: {
                impactLevel: 'critical',
                likelihood: 1.0,
                mitigationFactors: ['immediate_execution', 'expert_escalation'],
                residualRisk: 'High risk requires immediate action',
                businessImpact: 'Critical business impact possible'
            },
            approvalWorkflow: {
                required: false,
                approvers: [],
                approvalType: 'any',
                timeout: 0,
                escalationPath: ['security_team_lead', 'ciso'],
                autoApprovalConditions: ['emergency_plan']
            },
            rollbackPlan: {
                automatic: false,
                trigger: 'manual',
                steps: [],
                verificationRequired: true
            },
            status: 'approved',
            created: new Date(),
            lastUpdated: new Date(),
            executedBy: 'emergency_system'
        };
        this.remediationPlans.set(planId, emergencyPlan);
        this.emit('emergency_plan_created', {
            planId,
            control: params.control,
            framework: params.framework,
            priority: emergencyPlan.priority
        });
        return emergencyPlan;
    }
    /**
     * Create risk mitigation plan
     */
    async createRiskMitigation(params) {
        const planId = `risk-mitigation-${Date.now()}`;
        const riskMitigation = {
            id: planId,
            name: `Risk Mitigation: ${params.risk}`,
            description: `Mitigation plan for ${params.level} level risk`,
            type: 'security_incident',
            severity: params.level,
            trigger: 'automatic',
            riskId: `risk-${Date.now()}`,
            mitigationStrategy: 'reduce',
            controlsToImplement: params.controls,
            monitoringRequirements: ['continuous_monitoring', 'alerting', 'periodic_assessment'],
            effectivenessMeasures: ['risk_score_reduction', 'control_effectiveness', 'incident_frequency'],
            steps: [
                {
                    id: 'implement-controls',
                    name: 'Implement Risk Controls',
                    description: 'Deploy additional controls to mitigate identified risk',
                    type: 'configuration',
                    action: {
                        type: 'config_update',
                        implementation: 'RiskControlManager.deployControls',
                        parameters: { controls: params.controls, frameworks: params.frameworks },
                        timeout: 900000,
                        authentication: { type: 'service_account', credentials: 'risk_service', scope: ['write'], timeout: 60000 },
                        logging: true
                    },
                    parameters: {},
                    timeout: 900000,
                    retryCount: 0,
                    maxRetries: 2,
                    onSuccess: ['validate-effectiveness'],
                    onFailure: ['escalate-risk-team'],
                    validationCriteria: [
                        { type: 'config', condition: 'controls.deployed', expectedValue: true, timeout: 900000 }
                    ],
                    status: 'pending'
                }
            ],
            prerequisites: ['risk_analysis_complete', 'controls_available'],
            dependencies: ['risk_control_manager', 'monitoring_service'],
            estimatedDuration: 1800000, // 30 minutes
            riskAssessment: {
                impactLevel: params.level,
                likelihood: 0.6,
                mitigationFactors: ['additional_controls', 'monitoring_enhancement'],
                residualRisk: 'Reduced risk with additional controls',
                businessImpact: 'Minimal impact with proper mitigation'
            },
            approvalWorkflow: {
                required: params.level === 'critical',
                approvers: params.level === 'critical' ? ['risk_manager', 'ciso'] : [],
                approvalType: 'all',
                timeout: 3600000, // 1 hour
                escalationPath: ['risk_team_lead'],
                autoApprovalConditions: ['level != critical']
            },
            rollbackPlan: {
                automatic: true,
                trigger: 'failure',
                steps: [],
                verificationRequired: true
            },
            status: 'draft',
            created: new Date(),
            lastUpdated: new Date(),
            executedBy: 'system'
        };
        this.remediationPlans.set(planId, riskMitigation);
        this.emit('risk_mitigation_created', {
            planId,
            risk: params.risk,
            level: params.level,
            frameworks: params.frameworks.length
        });
        return riskMitigation;
    }
    /**
     * Execute remediation plan
     */
    async executeRemediation(plan) {
        const executionId = `execution-${plan.id}-${Date.now()}`;
        const execution = {
            planId: plan.id,
            executionId,
            startTime: new Date(),
            status: 'initializing',
            currentStep: undefined,
            completedSteps: [],
            failedSteps: [],
            skippedSteps: [],
            logs: [],
            metrics: {
                totalSteps: plan.steps.length,
                completedSteps: 0,
                failedSteps: 0,
                duration: 0,
                efficiency: 0,
                successRate: 0
            },
            approvals: []
        };
        this.activeExecutions.set(executionId, execution);
        try {
            this.emit('remediation_execution_started', { planId: plan.id, executionId });
            // Check if approval is required
            if (plan.approvalWorkflow.required && !this.isAutoApproved(plan)) {
                await this.requestApproval(execution, plan);
                if (execution.status === 'cancelled') {
                    return execution;
                }
            }
            // Execute steps
            execution.status = 'running';
            for (const step of plan.steps) {
                execution.currentStep = step.id;
                this.addExecutionLog(execution, 'info', step.id, `Starting step: ${step.name}`);
                const stepResult = await this.executeRemediationStep(step, execution, plan);
                if (stepResult.success) {
                    execution.completedSteps.push(step.id);
                    execution.metrics.completedSteps++;
                    this.addExecutionLog(execution, 'info', step.id, `Completed step: ${step.name}`);
                    // Continue to next steps based on onSuccess
                    if (step.onSuccess.length === 0) {
                        break; // End of workflow
                    }
                }
                else {
                    execution.failedSteps.push(step.id);
                    execution.metrics.failedSteps++;
                    this.addExecutionLog(execution, 'error', step.id, `Failed step: ${step.name} - ${stepResult.error}`);
                    // Handle failure based on onFailure
                    if (step.onFailure.length > 0) {
                        const nextSteps = step.onFailure;
                        // Execute failure handling steps...
                    }
                    else {
                        // No failure handling defined, consider rollback
                        if (plan.rollbackPlan.automatic && plan.rollbackPlan.trigger === 'failure') {
                            await this.executeRollback(execution, plan);
                        }
                        break;
                    }
                }
            }
            // Calculate final metrics
            execution.endTime = new Date();
            execution.metrics.duration = execution.endTime.getTime() - execution.startTime.getTime();
            execution.metrics.successRate = execution.metrics.completedSteps / execution.metrics.totalSteps;
            execution.metrics.efficiency = execution.metrics.successRate * (1 - (execution.metrics.duration / plan.estimatedDuration));
            // Determine final status
            if (execution.metrics.successRate === 1) {
                execution.status = 'completed';
            }
            else if (execution.metrics.completedSteps > 0) {
                execution.status = 'failed'; // Partial completion is considered failed
            }
            else {
                execution.status = 'failed';
            }
            this.addExecutionLog(execution, 'info', undefined, `Remediation ${execution.status}: ${execution.metrics.completedSteps}/${execution.metrics.totalSteps} steps completed`);
            // Move to history
            this.executionHistory.push(execution);
            this.activeExecutions.delete(executionId);
            this.emit('remediation_execution_completed', {
                planId: plan.id,
                executionId,
                status: execution.status,
                successRate: execution.metrics.successRate
            });
            return execution;
        }
        catch (error) {
            execution.status = 'failed';
            execution.endTime = new Date();
            this.addExecutionLog(execution, 'error', undefined, `Execution failed: ${error.message}`);
            this.executionHistory.push(execution);
            this.activeExecutions.delete(executionId);
            this.emit('remediation_execution_failed', { planId: plan.id, executionId, error: error.message });
            throw new Error(`Remediation execution failed: ${error.message}`);
        }
    }
    /**
     * Execute emergency remediation
     */
    async executeEmergencyRemediation(plan) {
        // Emergency plans bypass normal approval and execute immediately
        this.emit('emergency_remediation_triggered', { planId: plan.id, priority: plan.priority });
        return await this.executeRemediation(plan);
    }
    /**
     * Execute risk mitigation
     */
    async executeRiskMitigation(plan) {
        this.emit('risk_mitigation_triggered', { planId: plan.id, risk: plan.riskId });
        return await this.executeRemediation(plan);
    }
    /**
     * Execute individual remediation step
     */
    async executeRemediationStep(step, execution, plan) {
        try {
            step.startTime = new Date();
            step.status = 'running';
            const result = await this.actionExecutor.executeAction(step.action, step.parameters);
            // Validate step completion
            const validationResult = await this.validateStepCompletion(step, result);
            step.endTime = new Date();
            step.output = result;
            if (validationResult.valid) {
                step.status = 'completed';
                return { success: true, output: result };
            }
            else {
                step.status = 'failed';
                step.error = validationResult.error;
                return { success: false, error: validationResult.error };
            }
        }
        catch (error) {
            step.endTime = new Date();
            step.status = 'failed';
            step.error = error.message;
            // Retry logic
            if (step.retryCount < step.maxRetries) {
                step.retryCount++;
                step.status = 'pending';
                this.addExecutionLog(execution, 'warn', step.id, `Retrying step: ${step.name} (attempt ${step.retryCount + 1})`);
                // Wait before retry
                await new Promise(resolve => setTimeout(resolve, 5000));
                return await this.executeRemediationStep(step, execution, plan);
            }
            return { success: false, error: error.message };
        }
    }
    /**
     * Validate step completion
     */
    async validateStepCompletion(step, result) {
        for (const criteria of step.validationCriteria) {
            try {
                const isValid = await this.evaluateValidationCriteria(criteria, result);
                if (!isValid) {
                    return { valid: false, error: `Validation failed: ${criteria.condition}` };
                }
            }
            catch (error) {
                return { valid: false, error: `Validation error: ${error.message}` };
            }
        }
        return { valid: true };
    }
    /**
     * Evaluate validation criteria
     */
    async evaluateValidationCriteria(criteria, result) {
        switch (criteria.type) {
            case 'status':
                return this.evaluateStatusCriteria(criteria.condition, criteria.expectedValue, result);
            case 'metric':
                return await this.evaluateMetricCriteria(criteria.condition, criteria.expectedValue);
            case 'config':
                return await this.evaluateConfigCriteria(criteria.condition, criteria.expectedValue);
            case 'external':
                return await this.evaluateExternalCriteria(criteria.condition, criteria.expectedValue);
            default:
                return false;
        }
    }
    /**
     * Execute rollback
     */
    async executeRollback(execution, plan) {
        this.addExecutionLog(execution, 'warn', undefined, 'Initiating rollback procedure');
        try {
            await this.rollbackManager.executeRollback(plan, execution);
            this.addExecutionLog(execution, 'info', undefined, 'Rollback completed successfully');
        }
        catch (error) {
            this.addExecutionLog(execution, 'error', undefined, `Rollback failed: ${error.message}`);
        }
    }
    /**
     * Request approval for remediation
     */
    async requestApproval(execution, plan) {
        if (plan.approvalWorkflow.required) {
            const approvalResult = await this.approvalManager.requestApproval(plan, execution);
            execution.approvals.push(...approvalResult.approvals);
            if (!approvalResult.approved) {
                execution.status = 'cancelled';
                this.addExecutionLog(execution, 'warn', undefined, 'Remediation cancelled - approval denied');
            }
        }
    }
    /**
     * Check if plan is auto-approved
     */
    isAutoApproved(plan) {
        return plan.approvalWorkflow.autoApprovalConditions.some(condition => {
            // Evaluate auto-approval conditions
            switch (condition) {
                case 'severity <= medium':
                    return ['low', 'medium'].includes(plan.severity);
                case 'business_hours':
                    const now = new Date();
                    const hour = now.getHours();
                    return hour >= 9 && hour <= 17;
                case 'emergency_plan':
                    return plan.type === 'control_failure' && plan.severity === 'critical';
                default:
                    return false;
            }
        });
    }
    /**
     * Helper methods
     */
    selectPlanTemplate(type, severity) {
        // Select appropriate template based on type and severity
        return 'compliance-drift-standard'; // Default template
    }
    customizeRemediationSteps(templateSteps, params) {
        return templateSteps.map(step => ({
            ...step,
            parameters: { ...step.parameters, framework: params.framework, controls: params.controls }
        }));
    }
    addExecutionLog(execution, level, step, message, data) {
        execution.logs.push({
            timestamp: new Date(),
            level,
            step,
            message,
            data,
            source: 'remediation_orchestrator'
        });
    }
    evaluateStatusCriteria(condition, expectedValue, result) {
        // Mock evaluation - in production would parse condition and evaluate against result
        return true;
    }
    async evaluateMetricCriteria(condition, expectedValue) {
        // Mock evaluation - in production would fetch metrics and evaluate condition
        return true;
    }
    async evaluateConfigCriteria(condition, expectedValue) {
        // Mock evaluation - in production would check configuration state
        return true;
    }
    async evaluateExternalCriteria(condition, expectedValue) {
        // Mock evaluation - in production would call external validation services
        return true;
    }
    /**
     * Public API methods
     */
    getRemediationPlan(planId) {
        return this.remediationPlans.get(planId) || null;
    }
    getActiveExecutions() {
        return Array.from(this.activeExecutions.values());
    }
    getExecutionHistory(limit) {
        return limit ? this.executionHistory.slice(-limit) : this.executionHistory;
    }
    async cancelExecution(executionId) {
        const execution = this.activeExecutions.get(executionId);
        if (execution) {
            execution.status = 'cancelled';
            execution.endTime = new Date();
            this.addExecutionLog(execution, 'warn', undefined, 'Execution cancelled by user');
            this.executionHistory.push(execution);
            this.activeExecutions.delete(executionId);
            this.emit('remediation_execution_cancelled', { executionId });
            return true;
        }
        return false;
    }
}
exports.RemediationOrchestrator = RemediationOrchestrator;
/**
 * Action Executor for remediation actions
 */
class ActionExecutor {
    async executeAction(action, parameters) {
        // Mock action execution - in production would implement actual action execution
        switch (action.type) {
            case 'config_update':
                return { success: true, message: 'Configuration updated successfully' };
            case 'script_execution':
                return { success: true, output: 'Script executed successfully' };
            case 'api_request':
                return { success: true, response: 'API request completed' };
            case 'notification':
                return { success: true, sent: true };
            default:
                throw new Error(`Unknown action type: ${action.type}`);
        }
    }
}
/**
 * Approval Manager for remediation approvals
 */
class ApprovalManager {
    constructor(config) {
        this.config = config;
    }
    async requestApproval(plan, execution) {
        // Mock approval process - in production would integrate with approval systems
        const approvals = plan.approvalWorkflow.approvers.map(approver => ({
            approver,
            timestamp: new Date(),
            decision: 'approved',
            comments: 'Auto-approved for demonstration',
            conditions: []
        }));
        return { approved: true, approvals };
    }
}
/**
 * Rollback Manager for remediation rollbacks
 */
class RollbackManager {
    async executeRollback(plan, execution) {
        // Mock rollback execution - in production would execute actual rollback steps
        if (plan.rollbackPlan.steps.length > 0) {
            for (const step of plan.rollbackPlan.steps) {
                // Execute rollback step
            }
        }
    }
}
exports.default = RemediationOrchestrator;
//# sourceMappingURL=remediation-orchestrator.js.map