# Orchestrator

> **English Documentation** | [中文文档](README_CN.md)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/Version-2.0.0-green.svg)](orchestrator/version.py)

## 目录
- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [AI 服务](#ai-服务)
- [文档](#文档)
- [许可证](#许可证)

## 项目简介
Orchestrator 是一个实时智能对话系统，用于构建个性化多模态 AI 交互流程，包括语音识别（ASR）、文本对话（LLM）、语音合成（TTS）、情感分析（Classification & Reaction）、记忆管理（Memory）、3D 动画生成（Audio2Face & Speech2Motion）。该系统通过模块化设计支持多种 AI 服务提供商，提供流式处理和完整的对话管理功能。

主要应用场景：个性化角色扮演、定制化虚拟伴侣、教育培训、智能客服、办公助手等。

## 核心特性

### 技术特性
- **多模态交互**：语音交互、文本对话、3D 动画生成
- **实时流式处理**：实时数据流处理，低延迟响应
- **多 AI 服务商支持**：集成 SenseNova、OpenAI、Anthropic、Gemini、xAI、DeepSeek、ElevenLabs、火山引擎等主流 AI 服务
- **智能记忆管理**：多级对话记忆、关系状态、情绪状态管理
- **情感智能分析**：实时分析角色的情绪变化、关系变化和触发动作
- **高度可扩展架构**：模块化设计，易于添加新的 AI 服务和定制功能

### 定制化能力
- **角色定制**：自定义角色人设、音色、情绪、动作
- **交互定制**：灵活配置对话模式、反应机制、记忆管理
- **服务组合**：支持多 AI 服务商组合使用，根据场景需求灵活选择

## 系统架构

### 项目结构
```
orchestrator/
├── proxy.py                   # 核心编排器，管理 DAG 工作流
├── service/                   # Web 服务层
│   ├── server.py              # FastAPI 服务器，提供 WebSocket 接口
│   ├── requests.py            # 请求数据模型
│   └── responses.py           # 响应数据模型
├── conversation/              # 对话管理模块
│   ├── conversation_adapter.py        # 文本对话适配器基类
│   ├── audio_conversation_adapter.py  # 音频对话适配器基类
│   ├── openai_conversation_client.py  # OpenAI 文本对话客户端
│   ├── openai_audio_client.py         # OpenAI 音频对话客户端
│   ├── anthropic_conversation_client.py # Anthropic 对话客户端
│   ├── gemini_conversation_client.py   # Gemini 对话客户端
│   ├── xai_conversation_client.py      # xAI 对话客户端
|   ├── deepseek_conversation_client.py # DeepSeek 对话客户端
│   └── sensenova_omni_conversation_client.py  # SenseNova 实时对话客户端
|  
├── generation/                # 生成管理模块
│   ├── speech_recognition/    # 语音识别 (ASR)
│   │   ├── asr_adapter.py     # ASR 适配器基类
│   │   ├── openai_realtime_asr_client.py # OpenAI 实时 ASR
│   │   ├── sensetime_asr_client.py      # 商汤 ASR
│   │   └── softsugar_asr_client.py      # Softsugar ASR
│   ├── text2speech/          # 语音合成 (TTS)
│   │   ├── tts_adapter.py     # TTS适配器基类
│   │   ├── elevenlabs_tts_client.py     # ElevenLabs TTS
│   │   ├── huoshan_tts_client.py        # 火山引擎 TTS
|   |   ├── sensenova_tts_client.py      # SenseNova TTS
│   │   ├── sensetime_tts_client.py      # 商汤 TTS
│   │   └── softsugar_tts_client.py      # Softsugar TTS
│   ├── speech2motion/        # 语音转动作
│   │   ├── speech2motion_adapter.py     # S2M 适配器基类
│   │   └── speech2motion_streaming_client.py # S2M 流式客户端
│   └── audio2face/           # 音频转面部表情
│       ├── audio2face_adapter.py        # A2F 适配器基类
│       └── audio2face_streaming_client.py # A2F 流式客户端
├── memory/                   # 记忆管理模块
│   ├── memory_adapter.py     # 记忆适配器基类
│   ├── memory_manager.py     # 记忆管理器
│   ├── memory_processor.py   # 记忆处理器
│   ├── task_manager.py       # 任务管理器
│   ├── xai_memory_client.py  # xAI 记忆客户端
│   └── sensenova_omni_memory_client.py # SenseNova 实时记忆客户端
├── classification/           # 分类模块
│   ├── classification_adapter.py # 分类适配器基类
|   ├── sensenova_omni_classification_client.py # SenseNova 实时分类客户端
│   ├── openai_classification_client.py # OpenAI 分类客户端
│   ├── gemini_classification_client.py # Gemini 分类客户端
│   └── xai_classification_client.py    # xAI 分类客户端
├── reaction/                # 反应模块
│   ├── reaction_adapter.py   # 反应适配器基类
|   ├── sensenova_omni_reaction_client.py # SenseNova 实时反应客户端
│   ├── openai_reaction_client.py # OpenAI 反应客户端
│   ├── gemini_reaction_client.py # Gemini 反应客户端
│   └── xai_reaction_client.py    # xAI 反应客户端
├── aggregator/              # 数据聚合器
│   ├── conversation_aggregator.py # 对话聚合器
│   ├── tts_reaction_aggregator.py # TTS 反应聚合器
│   ├── blendshapes_aggregator.py  # 面部表情聚合器
│   └── callback_aggregator.py     # 回调聚合器
├── io/                      # 数据存储接口
│   ├── config/              # 配置存储
│   │   ├── database_config_client.py # 数据库配置客户端
│   │   ├── dynamodb_config_client.py # DynamoDB 配置客户端
│   │   └── mongodb_config_client.py  # MongoDB 配置客户端
│   └── memory/              # 记忆存储
│       ├── database_memory_client.py # 数据库记忆客户端
│       ├── dynamodb_memory_client.py # DynamoDB 记忆客户端
│       └── mongodb_memory_client.py  # MongoDB 记忆客户端
├── data_structures/         # 数据结构定义
└── utils/                   # 工具模块
```

### 核心组件

#### 1. 对话管理模块 (Conversation)
- **功能**: 处理文本和音频对话，支持多种大语言模型
- **核心组件**:
  - `ConversationAdapter`: 文本对话适配器基类，处理流式文本对话
  - `AudioConversationAdapter`: 音频对话适配器基类，处理实时语音交互
  - 支持服务商: SenseNova、OpenAI、Anthropic、Gemini、xAI、DeepSeek 等
- **特性**: 支持流式输出、长上下文、多模态对话

#### 2. 语音合成模块 (TTS)
- **功能**: 将文本转换为自然语音，支持多种音色和情感表达
- **核心组件**:
  - `TextToSpeechAdapter`: TTS 适配器基类，处理流式音频生成
  - 支持服务商: ElevenLabs、火山引擎、商汤、Softsugar 等
- **特性**: 多音色、多情感、多语言支持、实时合成

#### 3. 语音识别模块 (ASR)
- **功能**: 实时语音识别，支持多语言和实时处理
- **核心组件**:
  - `ASRAdapter`: ASR 适配器基类，处理流式语音识别
  - 支持服务商: OpenAI、商汤、Softsugar 等
- **特性**: 多语言支持、流式识别

#### 4. 记忆管理模块 (Memory)
- **功能**: 多级对话记忆、情感状态、关系状态管理
- **核心组件**:
  - `MemoryAdapter`: 记忆适配器基类
  - `MemoryManager`: 记忆管理器，处理对话历史和上下文
  - `MemoryProcessor`: 记忆处理器，分析和管理记忆数据
- **特性**: 多级记忆存储、情感状态跟踪、关系状态管理

#### 5. 情感分析模块 (Classification & Reaction)
- **功能**: 实时情感分析、用户意图分类、反应生成
- **核心组件**:
  - `ClassificationAdapter`: 分类适配器，分析用户意图
  - `ReactionAdapter`: 反应适配器，分析角色情绪变化、关系变化和触发动作
- **特性**: 实时情感分析、意图分类、个性化反应生成

#### 6. 3D 动画生成模块
- **功能**: 语音到动作转换、音频到面部表情转换
- **核心组件**:
  - `Speech2MotionAdapter`: 语音转动作适配器
  - `Audio2FaceAdapter`: 音频转面部表情适配器
- **特性**: 实时动作生成、面部表情同步、3D动画输出

#### 7. 数据聚合器 (Aggregator)
- **功能**: 协调多个模块间的数据流，确保数据同步
- **核心组件**:
  - `ConversationAggregator`: 对话聚合器，协调对话流程
  - `TTSReactionAggregator`: TTS反应聚合器，同步语音和反应
  - `BlendshapesAggregator`: 面部表情聚合器
- **特性**: 数据流协调、实时同步、错误处理

#### 8. 核心编排器 (Proxy)
- **功能**: 管理 DAG 工作流，协调所有模块的交互
- **核心组件**:
  - `Proxy`: 主编排器，管理复杂的 AI 交互流程
  - 支持多种对话模式：音频对话、文本对话、混合模式
- **特性**: DAG 工作流管理、模块协调、流程控制

### DAG 工作流架构
系统采用有向无环图（DAG）架构来管理复杂的 AI 交互流程。每个对话请求都会创建一个 DAG 实例，包含多个处理节点和依赖关系。

**图例说明：**
- **实线箭头** (→): 表示在单个生成请求中两个节点之间的一次性完整数据传输
- **虚线箭头** (⇢): 表示在单个生成请求中两个节点之间的流式数据传输

**流程图：**

- **完整版音频对话流程** (`audio_chat_with_text_llm_v4`)
   ![完整版音频对话流程](_static/audio_chat_with_text_llm_v4.svg)

- **极速版音频对话流程** (`audio_chat_with_audio_llm_v4`)
   ![极速版音频对话流程](_static/audio_chat_with_audio_llm_v4.svg)

- **完整版文本对话流程** (`text_chat_with_text_llm_v4`)
   ![完整版文本对话流程](_static/text_chat_with_text_llm_v4.svg)

- **极速版文本对话流程** (`text_chat_with_audio_llm_v4`)
   ![极速版文本对话流程](_static/text_chat_with_audio_llm_v4.svg)

- **直接生成流程** (`direct_generation_v4`)
   ![直接生成流程](_static/direct_generation_v4.svg)

## 快速开始

### 使用 Docker

#### 推荐：使用 Docker Compose 启动完整的后端服务

为了获得最佳体验，我们推荐使用 Docker Compose 启动完整的 DLP3D 后端服务，包括 Orchestrator 以及所有必需的依赖项（MongoDB、Audio2Face、Speech2Motion 等）。

请按照 [Complete DLP3D Backend Services](https://github.com/dlp3d-ai/web_backend?tab=readme-ov-file#complete-dlp3d-backend-services) 文档来设置和运行整个后端基础设施。

> **注意：** 上述链接将跳转到 [web_backend 仓库](https://github.com/dlp3d-ai/web_backend) 获取完整的后端设置说明。

#### 独立 Orchestrator 服务

如果您需要独立运行 Orchestrator 服务或配置高级选项，请参考 [Docker 配置指南](docker.md) 获取详细的设置说明、环境变量和配置选项。

### 环境设置

对于本地开发和部署，请按照详细的安装指南操作：

📖 **[完整安装指南](install.md)**

安装指南提供了以下步骤说明：
- 设置 Python 3.10+ 环境
- 安装 Protocol Buffers 编译器
- 配置开发环境
- 安装项目依赖

### 本地开发

按照安装指南完成环境设置后，您可以在本地启动服务：

```bash
# 激活 conda 环境
conda activate orchestrator

# 启动服务
python main.py --config_path configs/local.py
```


## AI 服务

### LLM
| 服务商 | 适配器类 | 默认模型 |
|--------|----------|----------|
| OpenAI | `OpenAIConversationClient` | `gpt-4.1-2025-04-14` |
| Anthropic | `AnthropicConversationClient` | `claude-sonnet-4-5-20250929` |
| Google | `GeminiConversationClient` | `gemini-2.5-flash-lite` |
| DeepSeek | `DeepSeekConversationClient` | `deepseek-chat` |
| xAI | `XAIConversationClient` | `grok-3` |
| SenseNova | `SenseNovaOmniConversationClient` | `SenseNova Omni` |
| OpenAI | `OpenAIAudioClient` | `gpt-4o-mini-realtime-preview-2024-12-17` |

### ASR
| 服务商 | 适配器类 |
|--------|----------|
| Softsugar | `SoftSugarASRClient` |
| OpenAI | `OpenAIRealtimeASRClient` |
| SenseTime | `SensetimeASRClient` |

### TTS
| 服务商 | 适配器类 |
|--------|----------|
| 火山引擎 | `HuoshanTTSClient` |
| Softsugar | `SoftSugarTTSClient` |
| SenseNova | `SensenovaTTSClient` |
| ElevenLabs | `ElevenLabsTTSClient` |
| SenseTime | `SensetimeTTSClient` |

### Memory
| 服务商 | 适配器类 | 默认模型 |
|--------|----------|------|
| xAI | `XAIMemoryClient` | `Grok-3` |
| SenseNova | `SenseNovaOmniMemoryClient` | `SenseNova Omni` |

### Classification
| 服务商 | 适配器类 | 默认模型 |
|--------|----------|----------|
| OpenAI | `OpenAIClassificationClient` | `gpt-4.1-mini-2025-04-14` |
| xAI | `XAIClassificationClient` | `grok-3` |
| Gemini | `GeminiClassificationClient` | `gemini-2.5-flash-lite` |
| SenseNova | `SenseNovaOmniClassificationClient` | `SenseNova Omni` |

### Reaction
| 服务商 | 适配器类 | 默认模型 |
|--------|----------|----------|
| OpenAI | `OpenAIReactionClient` | `gpt-4.1-mini-2025-04-14` |
| xAI | `XAIReactionClient` | `grok-3` |
| Gemini | `GeminiReactionClient` | `gemini-2.5-flash-lite` |
| SenseNova | `SenseNovaOmniReactionClient` | `SenseNova Omni` |

## 文档

- **[完整文档](https://dlp3d.readthedocs.io/zh-cn/latest/_subrepos/orchestrator/overview.html)** - 详细文档
- **[API 文档](API_Documentation.md)** - WebSocket 和 HTTP 端点的完整 API 参考
- **[开发指南](Development_Guide.md)** - 添加新 AI 服务、测试和代码质量标准的指南

## 许可证

本项目采用 MIT 许可证。详情请参见 [LICENSE](../LICENSE) 文件。

MIT 许可证是一个宽松的自由软件许可证，允许您使用、复制、修改、合并、发布、分发、再许可和/或销售软件副本，限制很少。唯一的要求是在所有副本或软件的重要部分中必须包含原始版权声明和许可证文本。
