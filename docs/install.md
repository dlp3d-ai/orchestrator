# Installation Guide

This document provides step-by-step instructions for setting up the DLP3D Orchestrator development environment on different operating systems.

## Table of Contents

- [Linux Environment Setup](#linux-environment-setup)
  - [Prerequisites](#prerequisites)
  - [Step 1: Install Protocol Buffers Compiler](#step-1-install-protocol-buffers-compiler)
  - [Step 2: Set Up Python](#step-2-set-up-python)
  - [Step 3: Install the Project](#step-3-install-the-project)
  - [Step 4: Verify Installation](#step-4-verify-installation)
  - [Environment Activation](#environment-activation)
- [Windows Environment Setup](#windows-environment-setup)
  - [Prerequisites](#prerequisites-1)
  - [Step 1: Install Protocol Buffers Compiler](#step-1-install-protocol-buffers-compiler-1)
  - [Step 2: Set Up Python](#step-2-set-up-python-1)
  - [Step 3: Install the Project](#step-3-install-the-project-1)
  - [Step 4: Verify Installation](#step-4-verify-installation-1)
  - [Environment Activation](#environment-activation-1)

## Linux Environment Setup

### Prerequisites

Before starting, ensure you have the following system requirements:
- Ubuntu 20.04 or compatible Linux distribution
- Internet connection for downloading packages

### Step 1: Install Protocol Buffers Compiler

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

### Step 2: Set Up Python

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

### Step 3: Install the Project

Install the orchestrator package:

```bash
# Ensure you're in the project root directory
cd /path/to/orchestrator

# Activate conda environment
conda activate orchestrator

# Install the package
pip install .
```

### Step 4: Verify Installation

Test that everything is working correctly:

```bash
# Activate the environment
conda activate orchestrator

# Check if orchestrator.service can be imported
python -c "import orchestrator.service; print('orchestrator.service imported successfully')"

# Check if the main application runs
python main.py --help
```

### Environment Activation

To work with the orchestrator backend project, always activate the conda environment first:

```bash
# Activate the environment
conda activate orchestrator

# Your terminal prompt should now show (orchestrator)
# You can now run Python scripts and use the orchestrator package
```


## Windows Environment Setup

### Prerequisites

Before starting, ensure you have the following system requirements:
- Windows 10/11 or compatible Windows distribution
- Internet connection for downloading packages

### Step 1: Install Protocol Buffers Compiler

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

### Step 2: Set Up Python

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

### Step 3: Install the Project

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

### Step 4: Verify Installation

Test that everything is working correctly:

```cmd
# Activate the environment
conda activate orchestrator

# Check if orchestrator.service can be imported
python -c "import orchestrator.service; print('orchestrator.service imported successfully')"

# Check if the main application runs
python main.py --help
```

### Environment Activation

To work with the orchestrator project, always activate the conda environment first:

```cmd
# Activate the environment
conda activate orchestrator

# Your terminal prompt should now show (orchestrator)
# You can now run Python scripts and use the orchestrator package
```


