# Orchestrator

> **English Documentation** | [中文文档](docs/README_CN.md)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/Version-2.0.0-green.svg)](orchestrator/version.py)

## 目录
- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [AI 服务](#ai-服务)
- [开发指南](#开发指南)
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

- **完整音频对话流程** (`audio_chat_with_text_llm_v4`)
   ```
   Audio input -> ASR -> Classification -> Conversation -> TTS -> Reaction -> A2F/S2M -> Callback
   ```

- **快速音频对话流程** (`audio_chat_with_audio_llm_v4`)
   ```
   Audio input -> Audio Conversation -> A2F/S2M -> Callback
   ```

- **完整文本对话流程** (`text_chat_with_text_llm_v4`)
   ```
   Text input -> Classification -> Conversation -> TTS -> Reaction -> A2F/S2M -> Callback
   ```

- **快速文本对话流程** (`text_chat_with_audio_llm_v4`)
   ```
   Text input -> TTS -> Audio conversation -> A2F/S2M -> Callback
   ```

- **直接生成流程** (`direct_generation_v4`)
   ```
   Text input -> TTS -> Reaction -> A2F/S2M -> Callback
   ```

## 快速开始

### 使用 Docker

### 本地安装

## API 文档

### WebSocket 端点

#### 1. 音频对话（文本LLM）
- **端点**: `/api/v4/audio_chat_with_text_llm`
- **描述**: 音频流式对话，使用文本模态 LLM 进行对话处理
- **请求模型**: `AudioChatCompleteStartRequestV4`

#### 2. 音频对话（音频LLM）
- **端点**: `/api/v4/audio_chat_with_audio_llm`
- **描述**: 音频流式对话，使用音频模态 LLM 进行对话处理
- **请求模型**: `AudioChatExpressStartRequestV4`

#### 3. 文本对话（文本LLM）
- **端点**: `/api/v4/text_chat_with_text_llm`
- **描述**: 文本对话，使用文本模态 LLM 进行对话处理
- **请求模型**: `TextChatCompleteRequestV4`

#### 4. 文本对话（音频LLM）
- **端点**: `/api/v4/text_chat_with_audio_llm`
- **描述**: 文本对话，使用音频模态 LLM 进行对话处理
- **请求模型**: `TextChatExpressRequestV4`

#### 5. 直接生成动画
- **端点**: `/api/v4/text_generate`
- **描述**: 直接从文本生成动画，不进行对话
- **请求模型**: `DirectGenerationRequest`

### HTTP 端点

#### 1. 健康检查
- **端点**: `GET /health` 或 `GET /api/v1/health`
- **描述**: 检查服务健康状态
- **响应**: `{"status": "healthy"}`

#### 2. 日志相关

**查看日志**
- **端点**: `GET /tail_log/{n_lines}` 或 `GET /api/v1/tail_log/{n_lines}`
- **描述**: 获取日志文件的最后N行
- **参数**: `n_lines` - 要获取的行数
- **响应**: HTML格式的日志内容

**下载日志**
- **端点**: `GET /download_log_file`
- **描述**: 下载完整的日志文件
- **响应**: 二进制日志文件

#### 3. 适配器选择

**ASR 适配器**
- **端点**: `GET /api/v1/asr_adapter_choices`
- **描述**: 获取可用的 ASR（语音识别）适配器列表
- **响应**: `AdapterChoicesResponse`

**TTS 适配器**
- **端点**: `GET /api/v1/tts_adapter_choices`
- **描述**: 获取可用的 TTS（语音合成）适配器列表
- **响应**: `AdapterChoicesResponse`

**对话适配器**
- **端点**: `GET /api/v1/conversation_adapter_choices`
- **描述**: 获取可用的 LLM 适配器列表
- **响应**: `AdapterChoicesResponse`

**反应适配器**
- **端点**: `GET /api/v1/reaction_adapter_choices`
- **描述**: 获取可用的反应适配器列表
- **响应**: `AdapterChoicesResponse`

**分类适配器**
- **端点**: `GET /api/v1/classification_adapter_choices`
- **描述**: 获取可用的分类适配器列表
- **响应**: `AdapterChoicesResponse`

**记忆适配器**
- **端点**: `GET /api/v1/memory_adapter_choices`
- **描述**: 获取可用的记忆适配器列表
- **响应**: `AdapterChoicesResponse`

#### 4. 语音和设置

**音色管理**
- **端点**: `GET /api/v1/tts_voice_names/{tts_adapter_key}`
- **描述**: 获取指定 TTS 适配器的可用音色列表
- **参数**: `tts_adapter_key` - TTS 适配器标识
- **响应**: `VoiceNamesResponse`

**用户设置**
- **端点**: `GET /api/v4/get_voice_settings/{user_id}/{character_id}`
- **描述**: 获取指定用户和角色的语音设置
- **参数**:
  - `user_id` - 用户 ID
  - `character_id` - 角色 ID
- **响应**: `VoiceSettingsResponse`

**动作设置**
- **端点**: `GET /api/v4/get_motion_settings/{user_id}/{character_id}`
- **描述**: 获取指定用户和角色的动作设置
- **参数**:
  - `user_id` - 用户 ID
  - `character_id` - 角色 ID
- **响应**: `MotionSettingsResponse`

## AI 服务

### LLM
| 服务商 | 适配器类 | 默认模型 |
|--------|----------|----------|
| OpenAI | `OpenAIConversationClient` | `gpt-4.1-2025-04-14` |
| Anthropic | `AnthropicConversationClient` | `claude-sonnet-4-20250514` |
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
| SenseTime | `SenseNovaOmniMemoryClient` | `SenseNova Omni` |

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

## 开发指南

### 添加新的 AI 服务

以添加新的 TTS 服务为例，需要完成以下步骤：

#### 1. 创建新的客户端类

在 `orchestrator/generation/text2speech/` 目录下创建新的客户端文件，例如 `new_tts_client.py`：

```python
from .tts_adapter import TextToSpeechAdapter

class NewTTSClient(TextToSpeechAdapter):
    """新的 TTS 客户端实现"""

    AVAILABLE_FOR_STREAM = True  # 是否支持流式处理

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        # 初始化客户端特定参数

    async def _generate_tts(
        self,
        request_id: str,
        text: str,
        voice_name: str,
        voice_speed: float = 1.0,
        voice_style: Union[None, str] = None,
        language: str = "zh",
        start_time: float = 0.0,
    ) -> Dict[str, Any]:
        """实现TTS生成逻辑"""
        # 调用第三方TTS API
        # 返回包含audio、speech_text、speech_time、duration的字典
        pass

    async def get_voice_names(self, **kwargs: Any) -> Dict[str, Any]:
        """返回可用的声音列表"""
        return {"voice_id": "voice_name"}
```

#### 2. 更新构建器

在 `orchestrator/generation/text2speech/builder.py` 中注册新的客户端：

```python
from .new_tts_client import NewTTSClient

_TTS_ADAPTERS = dict(
    # 现有适配器...
    NewTTSClient=NewTTSClient,
)
```

#### 3. 更新配置文件

在 `configs/local.py`、`configs/docker.py` 等配置文件中添加新的 TTS 适配器配置：

```python
tts_adapters=dict(
    # 现有适配器...
    new_tts=dict(
        type="NewTTSClient",
        name="new_tts_client",
        # 客户端特定参数...
    ),
),
```

#### 4. 添加测试

在 `tests/adapters/test_tts_adapters.py` 中添加测试函数：

```python
@pytest.mark.asyncio
async def test_new_tts_client_stream():
    """测试新的 TTS 客户端流式功能"""
    # 检查环境变量
    api_key = os.environ.get("NEW_TTS_API_KEY")
    if not api_key:
        pytest.skip("NEW_TTS_API_KEY is not set")

    # 配置和测试逻辑
    tts_client_cfg = dict(
        type="NewTTSClient",
        name="new_tts_client",
    )
    # 测试实现...
```

### 测试
项目包含使用 pytest 和异步测试支持的全面测试。测试覆盖了适配器、聚合器和 IO 模块的各个组件。

#### 测试分类

项目包含以下测试模块：

- **适配器测试** (`tests/adapters/`): 测试各种服务适配器
  - `test_a2f_adapters.py` - Audio2Face 适配器测试
  - `test_asr_adapters.py` - 语音识别适配器测试
  - `test_audio_conversation_adapters.py` - 音频对话适配器测试
  - `test_classification_adapters.py` - 分类适配器测试
  - `test_conversation_adapter.py` - 对话适配器测试
  - `test_memory_adapters.py` - 内存适配器测试
  - `test_reaction_adapters.py` - 反应适配器测试
  - `test_s2m_adapters.py` - 语音到动作适配器测试
  - `test_tts_adapters.py` - 文本转语音适配器测试

- **聚合器测试** (`tests/aggregator/`): 测试数据聚合器
  - `test_blendshapes_aggregator.py` - 混合形状聚合器测试
  - `test_conversation_aggregator.py` - 对话聚合器测试
  - `test_tts_reaction_aggregator.py` - TTS 反应聚合器测试

- **IO 测试** (`tests/io/`): 测试输入输出模块
  - `test_config_client.py` - 配置客户端测试
  - `test_memory_client.py` - 内存客户端测试

#### 测试数据准备

下载所需的测试文件并将其组织到正确的目录结构中：

1. **创建测试输入目录：**
   ```bash
   mkdir -p input
   ```

2. **下载测试文件：**
   ```bash
   cd input
   # 下载不同采样率的测试音频文件
   curl -LjO https://github.com/LazyBusyYang/CatStream/releases/download/orchestrator_cicd_files/test_audio_16kHz.wav
   curl -LjO https://github.com/LazyBusyYang/CatStream/releases/download/orchestrator_cicd_files/test_audio_24kHz.wav
   # 创建默认测试音频文件
   cp test_audio_16kHz.wav test_audio.wav
   cd ..
   ```

#### 运行测试

1. **运行所有测试：**
   ```bash
   # 创建日志目录
   mkdir -p logs

   # 运行所有测试
   python -m pytest tests --log-cli-level=ERROR
   ```

2. **运行特定测试模块：**
   ```bash
   # 运行适配器测试
   python -m pytest tests/adapters/

   # 运行聚合器测试
   python -m pytest tests/aggregator/

   # 运行 IO 测试
   python -m pytest tests/io/
   ```

3. **运行特定测试文件：**
   ```bash
   # 运行对话适配器测试
   python -m pytest tests/adapters/test_conversation_adapter.py

   # 运行配置客户端测试
   python -m pytest tests/io/test_config_client.py
   ```

### 代码质量

项目通过以下方式保持高代码质量：

- **代码检查**：使用 Ruff 进行代码风格和质量检查
- **类型提示**：完整的类型注解支持
- **CI/CD**：自动化测试和部署管道

## 许可证

本项目采用 MIT 许可证。详情请参见 [LICENSE](LICENSE) 文件。

MIT 许可证是一个宽松的自由软件许可证，允许您使用、复制、修改、合并、发布、分发、再许可和/或销售软件副本，限制很少。唯一的要求是在所有副本或软件的重要部分中必须包含原始版权声明和许可证文本。
