FROM ubuntu:20.04

# Install apt packages
RUN apt-get update && \
    apt-get install -y \
        ca-certificates curl git vim unzip \
        gcc g++ make \
    && \
    apt-get autoclean

# Set timezone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -yq tzdata && \
    dpkg-reconfigure -f noninteractive tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get autoclean

# Install uv and managed Python 3.10.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.10.12

# Download protoc
RUN mkdir -p /opt/protoc && cd /opt/protoc && \
    curl -LjO https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protoc-31.1-linux-x86_64.zip && \
    unzip protoc-31.1-linux-x86_64.zip && \
    rm -f protoc-31.1-linux-x86_64.zip && \
    chmod +x bin/protoc && \
    ln -s /opt/protoc/bin/protoc /usr/bin/protoc

# Create virtual environment
RUN uv venv --python 3.10.12 /opt/venv && \
    uv pip install --python /opt/venv/bin/python --upgrade pip setuptools wheel && \
    uv cache clean

# Update PATH to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# COPY orchestrator's requirements
COPY requirements.txt /opt/requirements.txt
RUN uv pip install --python /opt/venv/bin/python -r /opt/requirements.txt && \
    uv pip install --python /opt/venv/bin/python pytest && \
    uv cache clean

# COPY code
COPY . /workspace/orchestrator
# Install code
RUN cd /workspace/orchestrator && \
    uv pip install --python /opt/venv/bin/python . && \
    uv cache clean

# required environment variables
ENV MONGODB_HOST=
ENV MONGODB_PORT=27017
ENV MONGODB_MEMORY_DB=memory_database
ENV MONGODB_MEMORY_USER=memory_user
ENV MONGODB_MEMORY_PASSWORD=memory_password
ENV MONGODB_WEB_DB=web_database
ENV MONGODB_WEB_USER=web_user
ENV MONGODB_WEB_PASSWORD=web_password
ENV MONGODB_ADMIN_USERNAME=admin
ENV MONGODB_ADMIN_PASSWORD=
ENV A2F_WS_URL=
ENV S2M_WS_URL=
ENV ORCHESTRATOR_CONFIG_PATH=configs/docker.py
# optional environment variables
ENV ZOETROPE_ASR_WS_URL=
ENV ZOETROPE_TTS_WS_URL=
# PROXY_URL is optional, default value is None
# BACKEND_URL is optional, default value is None


# Set working directory
WORKDIR /workspace/orchestrator

# Set entrypoint
ENTRYPOINT ["/opt/venv/bin/python", "main.py", "--config_path", "configs/docker.py"]
