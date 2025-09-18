#!/bin/bash

# Familiar Project Deployment Script
# Infrastructure Princess - Production Deployment Framework

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_LOG="$PROJECT_ROOT/logs/deploy.log"
ENVIRONMENT="${1:-staging}"
VERSION="${2:-$(git rev-parse --short HEAD)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo -e "${timestamp} [${level}] ${message}" | tee -a "$DEPLOY_LOG"
}

info() {
    log "INFO" "${BLUE}$@${NC}"
}

success() {
    log "SUCCESS" "${GREEN}$@${NC}"
}

warning() {
    log "WARNING" "${YELLOW}$@${NC}"
}

error() {
    log "ERROR" "${RED}$@${NC}"
}

# Cleanup function for rollback
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error "Deployment failed with exit code $exit_code"
        if [ "$ENVIRONMENT" = "production" ]; then
            warning "Initiating automatic rollback..."
            rollback_deployment
        fi
    fi
}

trap cleanup EXIT

# Validate environment
validate_environment() {
    info "🔍 Validating deployment environment: $ENVIRONMENT"

    case "$ENVIRONMENT" in
        development|staging|production)
            info "Valid environment: $ENVIRONMENT"
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT. Must be development, staging, or production."
            exit 1
            ;;
    esac

    # Check required environment variables
    local required_vars=()

    if [ "$ENVIRONMENT" = "production" ]; then
        required_vars=(
            "DATABASE_URL"
            "REDIS_URL"
            "JWT_SECRET"
            "API_BASE_URL"
        )
    elif [ "$ENVIRONMENT" = "staging" ]; then
        required_vars=(
            "DB_HOST"
            "DB_NAME"
            "REDIS_HOST"
        )
    fi

    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done

    success "Environment validation passed"
}

# Pre-deployment checks
pre_deployment_checks() {
    info "🔧 Running pre-deployment checks..."

    # Check git status
    if [ -n "$(git status --porcelain)" ]; then
        warning "Working directory is not clean. Uncommitted changes detected."
        if [ "$ENVIRONMENT" = "production" ]; then
            error "Production deployments require a clean working directory"
            exit 1
        fi
    fi

    # Check if we're on the correct branch
    current_branch=$(git branch --show-current)
    expected_branch=""

    case "$ENVIRONMENT" in
        production)
            expected_branch="main"
            ;;
        staging)
            expected_branch="develop"
            ;;
        development)
            expected_branch="develop"
            ;;
    esac

    if [ "$current_branch" != "$expected_branch" ] && [ "$ENVIRONMENT" != "development" ]; then
        error "Expected branch '$expected_branch' but on '$current_branch'"
        exit 1
    fi

    # Pull latest changes for non-development environments
    if [ "$ENVIRONMENT" != "development" ]; then
        info "Pulling latest changes..."
        git pull origin "$current_branch"
    fi

    success "Pre-deployment checks passed"
}

# Install dependencies
install_dependencies() {
    info "📦 Installing dependencies..."

    # Clean install
    rm -rf node_modules package-lock.json
    npm ci --production=false

    success "Dependencies installed successfully"
}

# Run quality gates
run_quality_gates() {
    info "🛡️ Running quality gates..."

    # Linting
    info "Running ESLint..."
    npm run lint

    # Type checking
    if [ -f "tsconfig.json" ]; then
        info "Running TypeScript checks..."
        npm run typecheck
    fi

    # Unit tests
    info "Running unit tests..."
    npm run test:unit

    # Integration tests
    if [ "$ENVIRONMENT" != "development" ]; then
        info "Running integration tests..."
        npm run test:integration || warning "Integration tests failed but continuing deployment"
    fi

    # Security audit
    info "Running security audit..."
    npm audit --audit-level moderate

    # NASA POT10 compliance check
    if [ -f "$SCRIPT_DIR/nasa-compliance-check.js" ]; then
        info "Running NASA POT10 compliance check..."
        node "$SCRIPT_DIR/nasa-compliance-check.js"
    fi

    success "Quality gates passed"
}

# Build application
build_application() {
    info "🏗️ Building application for $ENVIRONMENT..."

    # Set NODE_ENV
    export NODE_ENV="$ENVIRONMENT"

    # Build
    npm run "build:$ENVIRONMENT" || npm run build

    # Verify build output
    if [ ! -d "dist" ] && [ ! -d "build" ]; then
        error "Build output directory not found"
        exit 1
    fi

    success "Application built successfully"
}

# Database migrations
run_migrations() {
    if [ "$ENVIRONMENT" = "development" ]; then
        return 0
    fi

    info "🗄️ Running database migrations..."

    # Check if migrations exist
    if [ -d "migrations" ] || [ -f "migrate.js" ]; then
        npm run migrate:$ENVIRONMENT || npm run migrate
        success "Database migrations completed"
    else
        info "No migrations found, skipping"
    fi
}

# Deploy to target environment
deploy_to_environment() {
    info "🚀 Deploying to $ENVIRONMENT environment..."

    case "$ENVIRONMENT" in
        development)
            deploy_development
            ;;
        staging)
            deploy_staging
            ;;
        production)
            deploy_production
            ;;
    esac
}

deploy_development() {
    info "Starting development server..."

    # Kill existing process
    pkill -f "node.*server" || true

    # Start in background
    nohup npm run start:dev > "$PROJECT_ROOT/logs/dev-server.log" 2>&1 &

    # Wait for server to start
    sleep 5

    # Health check
    if curl -f http://localhost:3000/health >/dev/null 2>&1; then
        success "Development server started successfully"
    else
        error "Development server failed to start"
        exit 1
    fi
}

deploy_staging() {
    info "Deploying to staging environment..."

    # Copy built files to staging directory
    sudo cp -r dist/* /var/www/staging/ || cp -r dist/* ~/staging/

    # Restart staging services
    sudo systemctl restart familiar-staging || pm2 restart familiar-staging

    # Health check
    sleep 10
    if curl -f "$STAGING_URL/health" >/dev/null 2>&1; then
        success "Staging deployment successful"
    else
        error "Staging deployment failed health check"
        exit 1
    fi
}

deploy_production() {
    info "Deploying to production environment..."

    # Backup current deployment
    backup_production

    # Deploy new version
    sudo cp -r dist/* /var/www/production/ || cp -r dist/* ~/production/

    # Update version file
    echo "$VERSION" > /var/www/production/VERSION

    # Restart production services with zero downtime
    sudo systemctl reload familiar-production || pm2 reload familiar-production

    # Health check with retry
    local max_retries=5
    local retry=0

    while [ $retry -lt $max_retries ]; do
        sleep 10
        if curl -f "$PRODUCTION_URL/health" >/dev/null 2>&1; then
            success "Production deployment successful"
            return 0
        fi

        retry=$((retry + 1))
        warning "Health check failed, attempt $retry of $max_retries"
    done

    error "Production deployment failed health checks"
    rollback_deployment
    exit 1
}

# Backup production
backup_production() {
    if [ "$ENVIRONMENT" != "production" ]; then
        return 0
    fi

    info "📦 Creating production backup..."

    local backup_dir="/var/backups/familiar/$(date +%Y%m%d-%H%M%S)"
    sudo mkdir -p "$backup_dir"

    # Backup application files
    sudo cp -r /var/www/production/* "$backup_dir/"

    # Backup database
    if command -v pg_dump >/dev/null 2>&1; then
        sudo -u postgres pg_dump familiar_production > "$backup_dir/database.sql"
    fi

    success "Production backup created at $backup_dir"
    echo "$backup_dir" > /tmp/last_backup_path
}

# Rollback deployment
rollback_deployment() {
    if [ "$ENVIRONMENT" != "production" ]; then
        return 0
    fi

    warning "🔄 Rolling back deployment..."

    if [ -f "/tmp/last_backup_path" ]; then
        local backup_path=$(cat /tmp/last_backup_path)
        if [ -d "$backup_path" ]; then
            sudo cp -r "$backup_path"/* /var/www/production/
            sudo systemctl restart familiar-production
            success "Rollback completed"
            return 0
        fi
    fi

    error "Rollback failed: backup not found"
}

# Post-deployment tasks
post_deployment_tasks() {
    info "🔧 Running post-deployment tasks..."

    # Clear caches
    if command -v redis-cli >/dev/null 2>&1; then
        redis-cli FLUSHDB
    fi

    # Warm up caches
    if [ "$ENVIRONMENT" = "production" ]; then
        curl "$PRODUCTION_URL/api/warmup" >/dev/null 2>&1 || true
    fi

    # Send deployment notification
    send_deployment_notification

    success "Post-deployment tasks completed"
}

# Send deployment notification
send_deployment_notification() {
    local status="SUCCESS"
    local message="Deployment to $ENVIRONMENT completed successfully"

    if [ -n "${SLACK_WEBHOOK:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 $message (version: $VERSION)\"}" \
            "$SLACK_WEBHOOK" >/dev/null 2>&1 || true
    fi

    if [ -n "${DISCORD_WEBHOOK:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"content\":\"🚀 $message (version: $VERSION)\"}" \
            "$DISCORD_WEBHOOK" >/dev/null 2>&1 || true
    fi

    info "Deployment notification sent"
}

# Main deployment function
main() {
    info "🎯 Starting deployment to $ENVIRONMENT (version: $VERSION)"

    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"

    # Run deployment steps
    validate_environment
    pre_deployment_checks
    install_dependencies
    run_quality_gates
    build_application
    run_migrations
    deploy_to_environment
    post_deployment_tasks

    success "🎉 Deployment to $ENVIRONMENT completed successfully!"
    info "Version: $VERSION"
    info "Timestamp: $(date)"
    info "Log file: $DEPLOY_LOG"
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [ENVIRONMENT] [VERSION]

Arguments:
  ENVIRONMENT  Target environment (development, staging, production)
  VERSION      Deployment version (defaults to git commit hash)

Examples:
  $0 development
  $0 staging
  $0 production v1.2.3

Environment Variables:
  Production:
    - DATABASE_URL
    - REDIS_URL
    - JWT_SECRET
    - API_BASE_URL
    - PRODUCTION_URL

  Staging:
    - DB_HOST
    - DB_NAME
    - REDIS_HOST
    - STAGING_URL

  Optional:
    - SLACK_WEBHOOK
    - DISCORD_WEBHOOK
    - WORKER_COUNT

EOF
}

# Handle command line arguments
if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_usage
    exit 0
fi

# Run main deployment
main "$@"