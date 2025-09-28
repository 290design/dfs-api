# DFS API Service

High-performance FastAPI microservice for Daily Fantasy Sports optimization and data endpoints.

## Overview

This service replaces Django Lambda for performance-critical endpoints:
- **Optimizer**: 25-30s → 5-10s (60-80% improvement)
- **Zero cold starts**: Always-on containers
- **4 vCPUs**: Better than Lambda's 1.3 vCPUs

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload --port 8000

# Test optimizer
curl -X POST http://localhost:8000/api/v2/optimize \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

### Deploy to Fly.io

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Deploy
fly launch --copy-config --yes

# Check status
fly status

# View logs
fly logs
```

## API Endpoints

### Optimizer

**POST /api/v2/optimize**

Generate optimized DFS lineups using Linear Programming.

```json
{
  "players": [...],
  "options": {...},
  "game": {...},
  "exposure": {...},
  "num_solutions": 25
}
```

**GET /api/v2/health**

Health check endpoint.

## Architecture

- **FastAPI**: High-performance async web framework
- **PuLP**: Linear Programming optimization
- **Fly.io**: Global edge deployment
- **SQLAlchemy**: Database ORM (Phase 2)

## Performance

| Metric | Before (Lambda) | After (Fly.io) |
|--------|----------------|----------------|
| Optimizer (25 lineups) | 25-30s | 5-10s |
| Cold starts | 2-5s | <100ms |
| CPU | 1.3 vCPU | 4 vCPU |

## Migration from Django

See `FASTAPI_MIGRATION_PLAN.md` in the main dailyfantasyserver repo for complete migration instructions.