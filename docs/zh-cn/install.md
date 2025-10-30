# 安装

## 从源码构建 Docker

如需从源码自行构建镜像：

### 构建流程

```cmd
# 构建 Docker 镜像
docker build -t orchestrator:local .
```

## 环境设置

以下为本地开发环境的设置方法。

### Linux 环境设置

#### 前置要求

开始之前，请确保：
- Ubuntu 20.04 或兼容发行版
- 可访问互联网以下载依赖

#### 步骤 1：安装 Protocol Buffers 编译器

下载并安装 protoc：

```bash
# 创建 protoc 目录
mkdir -p protoc
cd protoc

# 下载 protoc
curl -LjO https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protoc-31.1-linux-x86_64.zip

# 解压并设置权限
unzip protoc-31.1-linux-x86_64.zip
rm -f protoc-31.1-linux-x86_64.zip
chmod +x bin/protoc

# 验证安装
bin/protoc --version

# 返回项目根目录
cd ..
```

#### 步骤 2：安装 Python

需要 Python 3.10 及以上。本节以 conda 为例：

使用 Miniconda 安装 Python：

```bash
# 下载 Miniconda 安装器
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装 Miniconda
bash Miniconda3-latest-Linux-x86_64.sh

# 清理安装包
rm -f Miniconda3-latest-Linux-x86_64.sh

# 配置 conda 源
conda config --add channels conda-forge
conda tos accept

# 创建 Python 3.10 的环境
conda create -n orchestrator python=3.10 -y

# 激活环境
conda activate orchestrator

```

#### 步骤 3：安装项目

安装 orchestrator 包：

```bash
# 确认当前在项目根目录
cd /path/to/orchestrator

# 激活 conda 环境
conda activate orchestrator

# 安装包
pip install .
```

#### 步骤 4：验证安装

验证一切正常：

```bash
# 激活环境
conda activate orchestrator

# 检查 orchestrator.service 可导入
python -c "import orchestrator.service; print('orchestrator.service imported successfully')"

# 检查主程序可运行
python main.py --help
```

#### 环境激活

开发 Orchestrator 后端前，请先激活 conda 环境：

```bash
# 激活环境
conda activate orchestrator

# 终端提示应出现 (orchestrator)
# 之后可运行 Python 脚本并使用 orchestrator 包
```

### Windows 环境设置

#### 前置要求

开始之前，请确保：
- Windows 10/11 或兼容版本
- 可访问互联网以下载依赖

#### 步骤 1：安装 Protocol Buffers 编译器

下载并安装 protoc：

1. 下载 protoc：
   - 访问 [Protocol Buffers v31.1 发布页](https://github.com/protocolbuffers/protobuf/releases/tag/v31.1)
   - 下载 Windows 版本：`protoc-31.1-win64.zip`

2. 解压文件：
   - 在项目根目录创建 `protoc` 文件夹
   - 将 `protoc-31.1-win64.zip` 解压至 `protoc` 文件夹
   - 确认可执行文件路径：`protoc\bin\protoc.exe`

3. 验证安装：
   ```cmd
   # 在项目目录打开命令行
   protoc\bin\protoc.exe --version
   ```

#### 步骤 2：安装 Python

需要 Python 3.10 及以上。本节以 conda 为例：

使用 Miniconda 安装 Python：

1. 下载并安装 Miniconda：
   - 访问 [Miniconda 安装指南](https://www.anaconda.com/docs/getting-started/miniconda/install)
   - 从官网下载安装包
   - 按官方说明安装
   - 重要：安装时勾选“Add Miniconda3 to my PATH environment variable”，或手动将 Miniconda3/Scripts 加入 PATH 以便任意终端可用 conda

2. 创建并激活环境：
   ```cmd
   # 创建 Python 3.10 的环境
   conda create -n orchestrator python=3.10 -y

   # 激活环境
   conda activate orchestrator
   ```

#### 步骤 3：安装项目

安装 orchestrator 后端包：

```cmd
# 确认当前在项目根目录
cd /path/to/orchestrator

# 激活 conda 环境
conda activate orchestrator

# 临时将 protoc 加入 PATH（当前会话）
set PATH=%PATH%;%CD%\protoc\bin

# 安装包
pip install .
```

#### 步骤 4：验证安装

验证一切正常：

```cmd
# 激活环境
conda activate orchestrator

# 检查 orchestrator.service 可导入
python -c "import orchestrator.service; print('orchestrator.service imported successfully')"

# 检查主程序可运行
python main.py --help
```

#### 环境激活

进行 Orchestrator 项目开发前，请先激活 conda 环境：

```cmd
# 激活环境
conda activate orchestrator

# 终端提示应出现 (orchestrator)
# 之后可运行 Python 脚本并使用 orchestrator 包
```

## 本地开发

完成环境设置后，可以在本地启动服务：

```bash
# 激活 conda 环境
conda activate orchestrator

# 启动服务
python main.py --config_path configs/local.py
```


