# 快速开始

本指南帮助你使用 Docker 迅速运行 Orchestrator。若用于生产或需要完整后端栈，推荐使用 Docker Compose 的完整方案。

## 使用 Docker

### 推荐：使用 Docker Compose 启动完整服务

为获得最佳体验，推荐使用 Docker Compose 启动完整的 DLP3D 服务，其中包含 Orchestrator 及所有必需依赖（MongoDB、Audio2Face、Speech2Motion 等）。

请参考文档 [DLP3D 快速启动](https://dlp3d.readthedocs.io/zh-cn/latest/getting_started/quick_start.html) 来搭建并运行整个服务。

### 独立 Orchestrator 服务

如果你只想在自有基础设施上运行 Orchestrator（MongoDB、Audio2Face、Speech2Motion 已经运行），可按以下方式：

#### 快速启动

```bash
# 拉取并运行预构建镜像
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

#### 先决条件

- 确保你的系统已安装并运行 Docker
- 需确保提供的连接参数下，已有可访问的 MongoDB 实例
- Orchestrator 服务会在已有 MongoDB 上自动创建所需的数据库
- 需确保 Audio2Face 服务已经运行并可访问
- 需确保 Speech2Motion 服务已经运行并可访问

#### 环境变量

Docker 镜像中可配置以下环境变量：

MongoDB 配置：
- `MONGODB_HOST`：MongoDB 主机名
- `MONGODB_PORT`：MongoDB 端口（可选，默认 27017）
- `MONGODB_MEMORY_USER`：Memory 数据库用户名（可选，默认 memory_user）
- `MONGODB_MEMORY_PASSWORD`：Memory 数据库密码（可选，默认 memory_password）
- `MONGODB_MEMORY_DB`：Memory 数据库名（可选，默认 memory_database）
- `MONGODB_WEB_USER`：Web 配置数据库用户名（可选，默认 web_user）
- `MONGODB_WEB_PASSWORD`：Web 配置数据库密码（可选，默认 web_password）
- `MONGODB_WEB_DB`：Web 配置数据库名（可选，默认 web_database）
- `MONGODB_ADMIN_USERNAME`：用于初始化的 MongoDB 管理员用户名（可选，默认 admin）
- `MONGODB_ADMIN_PASSWORD`：用于初始化的 MongoDB 管理员密码（可选，默认空）

服务地址：
- `A2F_WS_URL`：Audio2Face WebSocket 地址（用于 3D 面部动画）
- `S2M_WS_URL`：Speech2Motion WebSocket 地址（用于 3D 动作生成）
- `BACKEND_URL`：后端服务地址（可选，默认空）
- `PROXY_URL`：外网代理地址（可选，默认空）

说明：
- 必填变量：`MONGODB_HOST`、`A2F_WS_URL`、`S2M_WS_URL`
- 可选变量：其他均有合理默认值，未提供不影响基本功能
- 增强功能：提供正确的 `BACKEND_URL` 可使 Orchestrator 更贴合用户意图并驱动更合适的动作
- 网络代理：若提供 `PROXY_URL`，当访问境外上游服务时将通过代理建立连接
- 应用启动时如不存在所需数据库与用户，会自动创建
