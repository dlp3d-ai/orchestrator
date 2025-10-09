
# Docker Deployment Guide

## Table of Contents

- [Overview](#overview)
- [Standalone Orchestrator Service](#standalone-orchestrator-service)
- [Building from Source](#building-from-source)

## Overview

This guide covers Docker deployment options for the Orchestrator service, including standalone deployment with environment variable configuration and custom image building from source.

## Standalone Orchestrator Service

To run only the Orchestrator service using Docker, you need a pre-configured MongoDB server running separately.

### Quick Start

```bash
# Pull and run the pre-built image
docker run -it \
  -p 18081:18081 \
  -e MONGODB_HOST=127.0.0.1 \
  -e MONGODB_PORT=27017 \
  -e MONGODB_MEMORY_USER=memory_user \
  -e MONGODB_MEMORY_PASSWORD=memory_password \
  -e MONGODB_MEMORY_DB=memory_database \
  -e MONGODB_WEB_USER=web_user \
  -e MONGODB_WEB_PASSWORD=web_password \
  -e MONGODB_WEB_DB=web_database \
  -e MONGODB_ADMIN_USERNAME=admin \
  -e MONGODB_ADMIN_PASSWORD=admin_password \
  -e BACKEND_URL=http://127.0.0.1:18080/api/v1/motion_keywords \
  -e A2F_WS_URL=ws://127.0.0.1:18083/api/v1/streaming_audio2face/ws \
  -e S2M_WS_URL=ws://127.0.0.1:18084/api/v3/streaming_speech2motion/ws \
  dockersenseyang/dlp3d_orchestrator:latest
```

### Prerequisites

- Make sure Docker is installed and running on your system
- **MongoDB server must be already running and accessible** with the provided connection parameters
- The orchestrator service will automatically create necessary databases in the existing MongoDB server
- **Audio2Face server must be already running and accessible**
- **Speech2Motion server must be already running and accessible**


The following environment variables are configured in the Docker image:

**MongoDB Configuration:**
- `MONGODB_HOST`: MongoDB server hostname
- `MONGODB_PORT`: MongoDB server port (optional, default: 27017)
- `MONGODB_MEMORY_USER`: Username for memory database access (optional, default: memory_user)
- `MONGODB_MEMORY_PASSWORD`: Password for memory database user (optional, default: memory_password)
- `MONGODB_MEMORY_DB`: Name of the memory database (optional, default: memory_database)
- `MONGODB_WEB_USER`: Username for web configuration database access (optional, default: web_user)
- `MONGODB_WEB_PASSWORD`: Password for web configuration database user (optional, default: web_password)
- `MONGODB_WEB_DB`: Name of the web configuration database (optional, default: web_database)
- `MONGODB_ADMIN_USERNAME`: MongoDB admin username for database setup (optional, default: admin)
- `MONGODB_ADMIN_PASSWORD`: MongoDB admin password for database setup (optional, default: empty)

**Service URLs:**
- `A2F_WS_URL`: Audio2Face WebSocket service URL for 3D facial animation
- `S2M_WS_URL`: Speech2Motion WebSocket service URL for 3D motion generation
- `BACKEND_URL`: Backend service URL for API calls (optional, default: empty)
- `PROXY_URL`: Network proxy URL for external service access (optional, default: empty)

**Note:**
- **Required variables**: `MONGODB_HOST`, `A2F_WS_URL`, and `S2M_WS_URL` must be provided for the Orchestrator to function properly
- **Optional variables**: All other environment variables use sensible defaults and do not affect normal operation if not specified
- **Enhanced functionality**: If you provide a correct `BACKEND_URL`, the Orchestrator can respond more accurately to user needs and perform appropriate motions
- **Network proxy**: If you provide a `PROXY_URL`, the Orchestrator will establish connections through the proxy server when accessing upstream services located outside mainland China
- The application will automatically create the required databases and users during startup if they don't exist


## Building from Source

If you prefer to build the image from source instead of using the pre-built image:

### Build Process

```cmd
# Build the Docker image
docker build -t orchestrator:local .
```

---

*For basic Docker usage, see the [main documentation](../README.md).*
