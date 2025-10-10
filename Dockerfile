FROM ubuntu:20.04

# Install apt packages
RUN apt-get update && \
    apt-get install -y \
        wget curl git vim unzip \
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

# Install Python 3.10 from source
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        wget \
        libbz2-dev && \
    cd /tmp && \
    wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    tar -xf Python-3.10.12.tgz && \
    cd Python-3.10.12 && \
    ./configure --enable-optimizations --prefix=/usr/local && \
    make -j $(nproc) && \
    make altinstall && \
    cd / && \
    rm -rf /tmp/Python-3.10.12* && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

# Create symlinks for python3.10
RUN ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip

# Download protoc
RUN mkdir -p /opt/protoc && cd /opt/protoc && \
    curl -LjO https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protoc-31.1-linux-x86_64.zip && \
    unzip protoc-31.1-linux-x86_64.zip && \
    rm -f protoc-31.1-linux-x86_64.zip && \
    chmod +x bin/protoc && \
    ln -s /opt/protoc/bin/protoc /usr/bin/protoc

# Create virtual environment
RUN /usr/local/bin/python3.10 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip cache purge

# Update PATH to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# COPY orchestrator's requirements
COPY requirements.txt /opt/requirements.txt
RUN /opt/venv/bin/pip install -r /opt/requirements.txt && \
    /opt/venv/bin/pip install pytest && \
    /opt/venv/bin/pip cache purge

# COPY code
COPY . /workspace/orchestrator
# Install code
RUN cd /workspace/orchestrator && \
    /opt/venv/bin/pip install . && \
    /opt/venv/bin/pip cache purge

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
