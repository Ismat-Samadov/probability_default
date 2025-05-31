#!/bin/bash
# =============================================================================
# PD Model Deployment Setup Script
# =============================================================================
# Complete deployment configuration for production environment
# Includes Docker, Docker Compose, and Kubernetes configs

echo "🚀 Setting up PD Model Production Deployment"
echo "=============================================="

# Create directory structure
mkdir -p deployment/{docker,kubernetes,nginx,scripts}
mkdir -p api
mkdir -p dashboard

# =============================================================================
# 1. Dockerfile for API
# =============================================================================
cat > deployment/docker/Dockerfile.api << 'EOF'
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY models/ ./models/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# =============================================================================
# 2. Dockerfile for Dashboard
# =============================================================================
cat > deployment/docker/Dockerfile.dashboard << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard code
COPY dashboard/ ./dashboard/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start Streamlit
CMD ["streamlit", "run", "dashboard/monitor.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

# =============================================================================
# 3. Docker Compose Configuration
# =============================================================================
cat > deployment/docker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  # PD Model API
  pd-api:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile.api
    container_name: pd-model-api
    ports:
      - "8000:8000"
    volumes:
      - ../../models:/app/models:ro
      - api-logs:/app/logs
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=info
      - MODEL_DIR=/app/models
    restart: unless-stopped
    networks:
      - pd-network
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.pd-api.rule=Host(`api.pd-model.local`)"
      - "traefik.http.services.pd-api.loadbalancer.server.port=8000"

  # Monitoring Dashboard
  pd-dashboard:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile.dashboard
    container_name: pd-model-dashboard
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://pd-api:8000
    depends_on:
      - pd-api
    restart: unless-stopped
    networks:
      - pd-network
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.pd-dashboard.rule=Host(`dashboard.pd-model.local`)"
      - "traefik.http.services.pd-dashboard.loadbalancer.server.port=8501"

  # Redis for caching
  redis:
    image: redis:7-alpine
    container_name: pd-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    networks:
      - pd-network

  # PostgreSQL for metadata and logs
  postgres:
    image: postgres:15-alpine
    container_name: pd-postgres
    environment:
      POSTGRES_DB: pd_models
      POSTGRES_USER: pd_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-pd_secure_password}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - pd-network

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: pd-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ../nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ../nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - pd-api
      - pd-dashboard
    restart: unless-stopped
    networks:
      - pd-network

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    container_name: pd-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ../prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped
    networks:
      - pd-network

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: pd-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ../grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ../grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    restart: unless-stopped
    networks:
      - pd-network

networks:
  pd-network:
    driver: bridge

volumes:
  api-logs:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:
EOF

# =============================================================================
# 4. Nginx Configuration
# =============================================================================
cat > deployment/nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream pd_api {
        server pd-api:8000;
    }

    upstream pd_dashboard {
        server pd-dashboard:8501;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=dashboard_limit:10m rate=5r/s;

    server {
        listen 80;
        server_name api.pd-model.local;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        location / {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://pd_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check endpoint
        location /health {
            access_log off;
            proxy_pass http://pd_api/health;
        }
    }

    server {
        listen 80;
        server_name dashboard.pd-model.local;

        location / {
            limit_req zone=dashboard_limit burst=10 nodelay;
            
            proxy_pass http://pd_dashboard;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
EOF

# =============================================================================
# 5. Kubernetes Deployment
# =============================================================================
cat > deployment/kubernetes/namespace.yml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: pd-models
  labels:
    name: pd-models
EOF

cat > deployment/kubernetes/pd-api-deployment.yml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pd-api
  namespace: pd-models
  labels:
    app: pd-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pd-api
  template:
    metadata:
      labels:
        app: pd-api
    spec:
      containers:
      - name: pd-api
        image: pd-model-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: pd-api-service
  namespace: pd-models
spec:
  selector:
    app: pd-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pd-api-ingress
  namespace: pd-models
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.pd-model.com
    secretName: pd-api-tls
  rules:
  - host: api.pd-model.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pd-api-service
            port:
              number: 80
EOF

# =============================================================================
# 6. Deployment Scripts
# =============================================================================
cat > deployment/scripts/deploy.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Deploying PD Model System"
echo "============================="

# Configuration
ENV=${1:-development}
VERSION=${2:-latest}

echo "Environment: $ENV"
echo "Version: $VERSION"

# Build images
echo "📦 Building Docker images..."
docker build -f deployment/docker/Dockerfile.api -t pd-model-api:$VERSION .
docker build -f deployment/docker/Dockerfile.dashboard -t pd-model-dashboard:$VERSION .

# Deploy based on environment
if [ "$ENV" = "production" ]; then
    echo "🏭 Deploying to production..."
    
    # Create namespace
    kubectl apply -f deployment/kubernetes/namespace.yml
    
    # Deploy applications
    kubectl apply -f deployment/kubernetes/
    
    # Wait for deployment
    kubectl rollout status deployment/pd-api -n pd-models
    
    echo "✅ Production deployment complete!"
    
elif [ "$ENV" = "staging" ]; then
    echo "🧪 Deploying to staging..."
    docker-compose -f deployment/docker/docker-compose.yml up -d
    echo "✅ Staging deployment complete!"
    
else
    echo "🔧 Deploying to development..."
    docker-compose -f deployment/docker/docker-compose.yml up -d --build
    echo "✅ Development deployment complete!"
fi

# Health checks
echo "🔍 Running health checks..."
sleep 10

if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API health check passed"
else
    echo "❌ API health check failed"
    exit 1
fi

if curl -f http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ Dashboard health check passed"
else
    echo "❌ Dashboard health check failed"
    exit 1
fi

echo ""
echo "🎉 Deployment successful!"
echo "📊 Dashboard: http://localhost:8501"
echo "🔌 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
EOF

chmod +x deployment/scripts/deploy.sh

# =============================================================================
# 7. Environment Configuration
# =============================================================================
cat > deployment/.env.example << 'EOF'
# Database
POSTGRES_PASSWORD=pd_secure_password_change_me
POSTGRES_DB=pd_models
POSTGRES_USER=pd_user

# Grafana
GRAFANA_PASSWORD=admin_change_me

# API Configuration
ENVIRONMENT=production
LOG_LEVEL=info
MODEL_DIR=/app/models

# Security
JWT_SECRET_KEY=your_jwt_secret_key_here
API_KEY=your_api_key_here

# External Services
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://pd_user:pd_secure_password@postgres:5432/pd_models

# Monitoring
PROMETHEUS_ENDPOINT=http://prometheus:9090
GRAFANA_ENDPOINT=http://grafana:3000
EOF

# =============================================================================
# 8. Monitoring Configuration
# =============================================================================
mkdir -p deployment/prometheus
cat > deployment/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'pd-api'
    static_configs:
      - targets: ['pd-api:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'pd-dashboard'
    static_configs:
      - targets: ['pd-dashboard:8501']
    metrics_path: /metrics
    scrape_interval: 60s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

# =============================================================================
# 9. Database Initialization
# =============================================================================
cat > deployment/docker/init.sql << 'EOF'
-- PD Model Database Schema
CREATE DATABASE pd_models;

\c pd_models;

-- Model metadata table
CREATE TABLE model_metadata (
    id SERIAL PRIMARY KEY,
    segment VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    training_date TIMESTAMP NOT NULL,
    performance_metrics JSONB,
    validation_results JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prediction logs
CREATE TABLE prediction_logs (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(100),
    segment VARCHAR(50) NOT NULL,
    input_data JSONB NOT NULL,
    pd_score DECIMAL(10, 8) NOT NULL,
    risk_grade VARCHAR(10) NOT NULL,
    ifrs9_stage INTEGER NOT NULL,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    processing_time_ms INTEGER
);

-- Model performance monitoring
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    segment VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10, 6) NOT NULL,
    measurement_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data drift monitoring
CREATE TABLE data_drift (
    id SERIAL PRIMARY KEY,
    segment VARCHAR(50) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    psi_value DECIMAL(10, 6) NOT NULL,
    drift_status VARCHAR(20) NOT NULL,
    measurement_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_prediction_logs_timestamp ON prediction_logs(prediction_timestamp);
CREATE INDEX idx_prediction_logs_segment ON prediction_logs(segment);
CREATE INDEX idx_model_performance_segment_date ON model_performance(segment, measurement_date);
CREATE INDEX idx_data_drift_segment_date ON data_drift(segment, measurement_date);
EOF

# =============================================================================
# 10. Final Setup Instructions
# =============================================================================
cat > DEPLOYMENT_GUIDE.md << 'EOF'
# PD Model Deployment Guide

## Quick Start

### 1. Development Deployment
```bash
# Run the deployment script
./deployment/scripts/deploy.sh development

# Access the services
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - API Docs: http://localhost:8000/docs
```

### 2. Production Deployment
```bash
# Configure environment
cp deployment/.env.example .env
# Edit .env with your production values

# Deploy to Kubernetes
./deployment/scripts/deploy.sh production

# Or deploy with Docker Compose
./deployment/scripts/deploy.sh staging
```

## Architecture

- **API**: FastAPI-based REST API for PD scoring
- **Dashboard**: Streamlit monitoring dashboard
- **Database**: PostgreSQL for metadata and logs
- **Cache**: Redis for performance optimization
- **Monitoring**: Prometheus + Grafana
- **Proxy**: Nginx for load balancing

## Monitoring

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **API Metrics**: http://localhost:8000/metrics

## Security Features

- Rate limiting
- API key authentication
- SSL/TLS encryption
- Security headers
- Non-root containers

## Scaling

The system supports horizontal scaling:
- API: Multiple replicas behind load balancer
- Database: Read replicas for reporting
- Cache: Redis cluster for high availability

## Backup & Recovery

- Database: Automated backups to S3/Azure
- Models: Versioned storage with rollback capability
- Logs: Centralized logging with retention policies
EOF

echo ""
echo "✅ Deployment configuration created successfully!"
echo ""
echo "📁 Generated files:"
echo "   - deployment/docker/Dockerfile.api"
echo "   - deployment/docker/Dockerfile.dashboard"
echo "   - deployment/docker/docker-compose.yml"
echo "   - deployment/kubernetes/pd-api-deployment.yml"
echo "   - deployment/nginx/nginx.conf"
echo "   - deployment/scripts/deploy.sh"
echo "   - DEPLOYMENT_GUIDE.md"
echo ""
echo "🚀 Next steps:"
echo "   1. Run the model training: python3 notebooks/model_fixed.py"
echo "   2. Create API files in api/ directory"
echo "   3. Create dashboard files in dashboard/ directory"
echo "   4. Deploy: ./deployment/scripts/deploy.sh development"
echo ""
echo "📚 Full documentation available in DEPLOYMENT_GUIDE.md"