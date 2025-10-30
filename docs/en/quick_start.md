# Quick Start

This guide helps you run Orchestrator quickly using Docker. For production or a full backend stack, consider the complete Docker Compose approach.

## Using Docker

### Recommended: Complete Backend Services with Docker Compose

For the best experience, we recommend using Docker Compose to start the complete DLP3D backend services, which includes the Orchestrator along with all required dependencies (MongoDB, Audio2Face, Speech2Motion, etc.).

Please follow the [Complete DLP3D Backend Services](https://github.com/dlp3d-ai/web_backend?tab=readme-ov-file#complete-dlp3d-backend-services) documentation to set up and run the entire backend infrastructure.

> **Note:** The above link will redirect you to the [web_backend repository](https://github.com/dlp3d-ai/web_backend) for complete backend setup instructions.

### Standalone Orchestrator Service

If you only want to run the Orchestrator service with your own infrastructure (MongoDB, Audio2Face, Speech2Motion already running), use the following:

#### Quick Start

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
  dlp3d/orchestrator:latest
```

#### Prerequisites

- Make sure Docker is installed and running on your system
- **MongoDB server must be already running and accessible** with the provided connection parameters
- The orchestrator service will automatically create necessary databases in the existing MongoDB server
- **Audio2Face server must be already running and accessible**
- **Speech2Motion server must be already running and accessible**

#### Environment Variables

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

### Building from Source

If you prefer to build the image from source instead of using the pre-built image:

#### Build Process

```cmd
# Build the Docker image
docker build -t orchestrator:local .
```

## Environment Setup

The following sections describe how to set up a local development environment.

### Linux Environment Setup

#### Prerequisites

Before starting, ensure you have the following system requirements:
- Ubuntu 20.04 or compatible Linux distribution
- Internet connection for downloading packages

#### Step 1: Install Protocol Buffers Compiler

Download and install protoc for protocol buffer compilation:

```bash
# Create protoc directory
mkdir -p protoc
cd protoc

# Download protoc
curl -LjO https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protoc-31.1-linux-x86_64.zip

# Extract and set permissions
unzip protoc-31.1-linux-x86_64.zip
rm -f protoc-31.1-linux-x86_64.zip
chmod +x bin/protoc

# Verify installation
bin/protoc --version

# Go back to the root directory
cd ..
```

#### Step 2: Set Up Python

You need Python 3.10 or higher to run this project. This document provides one method using conda for Python installation as a reference.

**Install Python using Miniconda:**

```bash
# Download Miniconda installer
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda
bash Miniconda3-latest-Linux-x86_64.sh

# Clean up installer
rm -f Miniconda3-latest-Linux-x86_64.sh

# Configure conda channels
conda config --add channels conda-forge
conda tos accept

# Create orchestrator environment with Python 3.10
conda create -n orchestrator python=3.10 -y

# Activate the environment
conda activate orchestrator

```

#### Step 3: Install the Project

Install the orchestrator package:

```bash
# Ensure you're in the project root directory
cd /path/to/orchestrator

# Activate conda environment
conda activate orchestrator

# Install the package
pip install .
```

#### Step 4: Verify Installation

Test that everything is working correctly:

```bash
# Activate the environment
conda activate orchestrator

# Check if orchestrator.service can be imported
python -c "import orchestrator.service; print('orchestrator.service imported successfully')"

# Check if the main application runs
python main.py --help
```

#### Environment Activation

To work with the orchestrator backend project, always activate the conda environment first:

```bash
# Activate the environment
conda activate orchestrator

# Your terminal prompt should now show (orchestrator)
# You can now run Python scripts and use the orchestrator package
```

### Windows Environment Setup

#### Prerequisites

Before starting, ensure you have the following system requirements:
- Windows 10/11 or compatible Windows distribution
- Internet connection for downloading packages

#### Step 1: Install Protocol Buffers Compiler

Download and install protoc for protocol buffer compilation:

1. **Download protoc:**
   - Visit [Protocol Buffers v31.1 Release Page](https://github.com/protocolbuffers/protobuf/releases/tag/v31.1)
   - Download the Windows version: `protoc-31.1-win64.zip`

2. **Extract the files:**
   - Create a `protoc` folder in your project root directory
   - Extract the downloaded `protoc-31.1-win64.zip` file into the `protoc` folder
   - Ensure the executable file is located at: `protoc\bin\protoc.exe`

3. **Verify installation:**
   ```cmd
   # Open Command Prompt in your project directory
   protoc\bin\protoc.exe --version
   ```

#### Step 2: Set Up Python

You need Python 3.10 or higher to run this project. This document provides one method using conda for Python installation as a reference.

**Install Python using Miniconda:**

1. **Download and Install Miniconda:**
   - Visit [Miniconda Installation Guide](https://www.anaconda.com/docs/getting-started/miniconda/install)
   - Download the Windows installer from the Anaconda website
   - Follow the official installation instructions to install Miniconda
   - **Important**: During installation, make sure to check "Add Miniconda3 to my PATH environment variable" or add the Miniconda3/Scripts directory to the PATH environment variable manually to enable conda commands from any terminal

2. **Create and Activate Environment:**
   ```cmd
   # Create orchestrator environment with Python 3.10
   conda create -n orchestrator python=3.10 -y

   # Activate the environment
   conda activate orchestrator
   ```

#### Step 3: Install the Project

Install the orchestrator backend package:

```cmd
# Ensure you're in the project root directory
cd /path/to/orchestrator

# Activate conda environment
conda activate orchestrator

# Temporarily add protoc to PATH for this session
set PATH=%PATH%;%CD%\protoc\bin

# Install the package
pip install .
```

#### Step 4: Verify Installation

Test that everything is working correctly:

```cmd
# Activate the environment
conda activate orchestrator

# Check if orchestrator.service can be imported
python -c "import orchestrator.service; print('orchestrator.service imported successfully')"

# Check if the main application runs
python main.py --help
```

#### Environment Activation

To work with the orchestrator project, always activate the conda environment first:

```cmd
# Activate the environment
conda activate orchestrator

# Your terminal prompt should now show (orchestrator)
# You can now run Python scripts and use the orchestrator package
```

## Local Development

After completing the environment setup, you can start the service locally:

```bash
# Activate the conda environment
conda activate orchestrator

# Start the service
python main.py --config_path configs/local.py
```
