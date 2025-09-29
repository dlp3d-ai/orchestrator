# Orchestrator

> **English Documentation** | [中文文档](docs/README_CN.md)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/Version-2.0.0-green.svg)](orchestrator/version.py)

## Table of Contents
- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [AI Services](#ai-services)
- [Development Guide](#development-guide)
- [License](#license)

## Project Overview
Orchestrator is a real-time intelligent conversation system for building personalized multimodal AI interaction workflows, including speech recognition (ASR), text conversation (LLM), text-to-speech (TTS), emotion analysis (Classification & Reaction), memory management (Memory), and 3D animation generation (Audio2Face & Speech2Motion). The system supports multiple AI service providers through modular design, providing streaming processing and complete conversation management capabilities.

Main application scenarios: personalized role-playing, customized virtual companions, education and training, intelligent customer service, office assistants, etc.

## Core Features

### Technical Features
- **Multimodal Interaction**: Voice interaction, text conversation, 3D animation generation
- **Real-time Streaming Processing**: Real-time data stream processing with low-latency response
- **Multi-AI Service Provider Support**: Integration with mainstream AI services including SenseNova, OpenAI, Anthropic, Gemini, xAI, DeepSeek, ElevenLabs, Volcano Engine, etc.
- **Intelligent Memory Management**: Multi-level conversation memory, relationship status, and emotional state management
- **Emotional Intelligence Analysis**: Real-time analysis of character emotional changes, relationship changes, and triggered actions
- **Highly Scalable Architecture**: Modular design, easy to add new AI services and custom features

### Customization Capabilities
- **Character Customization**: Custom character personalities, voices, emotions, and actions
- **Interaction Customization**: Flexible configuration of conversation modes, reaction mechanisms, and memory management
- **Service Combination**: Support for combining multiple AI service providers, flexible selection based on scenario requirements

## System Architecture

### Project Structure
```
orchestrator/
├── proxy.py                   # Core orchestrator, manages DAG workflows
├── service/                   # Web service layer
│   ├── server.py              # FastAPI server, provides WebSocket interface
│   ├── requests.py            # Request data models
│   └── responses.py           # Response data models
├── conversation/              # Conversation management module
│   ├── conversation_adapter.py        # Text conversation adapter base class
│   ├── audio_conversation_adapter.py  # Audio conversation adapter base class
│   ├── openai_conversation_client.py  # OpenAI text conversation client
│   ├── openai_audio_client.py         # OpenAI audio conversation client
│   ├── anthropic_conversation_client.py # Anthropic conversation client
│   ├── gemini_conversation_client.py   # Gemini conversation client
│   ├── xai_conversation_client.py      # xAI conversation client
│   ├── deepseek_conversation_client.py # DeepSeek conversation client
│   └── sensenova_omni_conversation_client.py  # SenseNova real-time conversation client
├── generation/                # Generation management module
│   ├── speech_recognition/    # Speech Recognition (ASR)
│   │   ├── asr_adapter.py     # ASR adapter base class
│   │   ├── openai_realtime_asr_client.py # OpenAI real-time ASR
│   │   ├── sensetime_asr_client.py      # SenseTime ASR
│   │   └── softsugar_asr_client.py      # Softsugar ASR
│   ├── text2speech/          # Text-to-Speech (TTS)
│   │   ├── tts_adapter.py     # TTS adapter base class
│   │   ├── elevenlabs_tts_client.py     # ElevenLabs TTS
│   │   ├── huoshan_tts_client.py        # Volcano Engine TTS
│   │   ├── sensenova_tts_client.py      # SenseNova TTS
│   │   ├── sensetime_tts_client.py      # SenseTime TTS
│   │   └── softsugar_tts_client.py      # Softsugar TTS
│   ├── speech2motion/        # Speech-to-Motion
│   │   ├── speech2motion_adapter.py     # S2M adapter base class
│   │   └── speech2motion_streaming_client.py # S2M streaming client
│   └── audio2face/           # Audio-to-Face
│       ├── audio2face_adapter.py        # A2F adapter base class
│       └── audio2face_streaming_client.py # A2F streaming client
├── memory/                   # Memory management module
│   ├── memory_adapter.py     # Memory adapter base class
│   ├── memory_manager.py     # Memory manager
│   ├── memory_processor.py   # Memory processor
│   ├── task_manager.py       # Task manager
│   ├── xai_memory_client.py  # xAI memory client
│   └── sensenova_omni_memory_client.py # SenseNova real-time memory client
├── classification/           # Classification module
│   ├── classification_adapter.py # Classification adapter base class
│   ├── sensenova_omni_classification_client.py # SenseNova real-time classification client
│   ├── openai_classification_client.py # OpenAI classification client
│   ├── gemini_classification_client.py # Gemini classification client
│   └── xai_classification_client.py    # xAI classification client
├── reaction/                # Reaction module
│   ├── reaction_adapter.py   # Reaction adapter base class
│   ├── sensenova_omni_reaction_client.py # SenseNova real-time reaction client
│   ├── openai_reaction_client.py # OpenAI reaction client
│   ├── gemini_reaction_client.py # Gemini reaction client
│   └── xai_reaction_client.py    # xAI reaction client
├── aggregator/              # Data aggregators
│   ├── conversation_aggregator.py # Conversation aggregator
│   ├── tts_reaction_aggregator.py # TTS reaction aggregator
│   ├── blendshapes_aggregator.py  # Facial expression aggregator
│   └── callback_aggregator.py     # Callback aggregator
├── io/                      # Data storage interfaces
│   ├── config/              # Configuration storage
│   │   ├── database_config_client.py # Database configuration client
│   │   ├── dynamodb_config_client.py # DynamoDB configuration client
│   │   └── mongodb_config_client.py  # MongoDB configuration client
│   └── memory/              # Memory storage
│       ├── database_memory_client.py # Database memory client
│       ├── dynamodb_memory_client.py # DynamoDB memory client
│       └── mongodb_memory_client.py  # MongoDB memory client
├── data_structures/         # Data structure definitions
└── utils/                   # Utility modules
```

### Core Components

#### 1. Conversation Management Module (Conversation)
- **Function**: Handles text and audio conversations, supports multiple large language models
- **Core Components**:
  - `ConversationAdapter`: Text conversation adapter base class, handles streaming text conversations
  - `AudioConversationAdapter`: Audio conversation adapter base class, handles real-time voice interactions
  - Supported providers: SenseNova, OpenAI, Anthropic, Gemini, xAI, DeepSeek, etc.
- **Features**: Streaming output support, long context, multimodal conversations

#### 2. Text-to-Speech Module (TTS)
- **Function**: Converts text to natural speech, supports multiple voices and emotional expressions
- **Core Components**:
  - `TextToSpeechAdapter`: TTS adapter base class, handles streaming audio generation
  - Supported providers: ElevenLabs, Volcano Engine, SenseTime, Softsugar, etc.
- **Features**: Multiple voices, multiple emotions, multi-language support, real-time synthesis

#### 3. Speech Recognition Module (ASR)
- **Function**: Real-time speech recognition, supports multiple languages and real-time processing
- **Core Components**:
  - `ASRAdapter`: ASR adapter base class, handles streaming speech recognition
  - Supported providers: OpenAI, SenseTime, Softsugar, etc.
- **Features**: Multi-language support, streaming recognition

#### 4. Memory Management Module (Memory)
- **Function**: Multi-level conversation memory, emotional state, relationship state management
- **Core Components**:
  - `MemoryAdapter`: Memory adapter base class
  - `MemoryManager`: Memory manager, handles conversation history and context
  - `MemoryProcessor`: Memory processor, analyzes and manages memory data
- **Features**: Multi-level memory storage, emotional state tracking, relationship state management

#### 5. Emotion Analysis Module (Classification & Reaction)
- **Function**: Real-time emotion analysis, user intent classification, reaction generation
- **Core Components**:
  - `ClassificationAdapter`: Classification adapter, analyzes user intent
  - `ReactionAdapter`: Reaction adapter, analyzes character emotional changes, relationship changes, and triggered actions
- **Features**: Real-time emotion analysis, intent classification, personalized reaction generation

#### 6. 3D Animation Generation Module
- **Function**: Speech-to-motion conversion, audio-to-facial expression conversion
- **Core Components**:
  - `Speech2MotionAdapter`: Speech-to-motion adapter
  - `Audio2FaceAdapter`: Audio-to-facial expression adapter
- **Features**: Real-time motion generation, facial expression synchronization, 3D animation output

#### 7. Data Aggregators (Aggregator)
- **Function**: Coordinates data flow between multiple modules, ensures data synchronization
- **Core Components**:
  - `ConversationAggregator`: Conversation aggregator, coordinates conversation flow
  - `TTSReactionAggregator`: TTS reaction aggregator, synchronizes voice and reactions
  - `BlendshapesAggregator`: Facial expression aggregator
- **Features**: Data flow coordination, real-time synchronization, error handling

#### 8. Core Orchestrator (Proxy)
- **Function**: Manages DAG workflows, coordinates interactions between all modules
- **Core Components**:
  - `Proxy`: Main orchestrator, manages complex AI interaction workflows
  - Supports multiple conversation modes: audio conversation, text conversation, mixed mode
- **Features**: DAG workflow management, module coordination, process control

### DAG Workflow Architecture
The system uses a Directed Acyclic Graph (DAG) architecture to manage complex AI interaction workflows. Each conversation request creates a DAG instance containing multiple processing nodes and dependencies.

- **Complete Audio Conversation Flow** (`audio_chat_with_text_llm_v4`)
   ```
   Audio input -> ASR -> Classification -> Conversation -> TTS -> Reaction -> A2F/S2M -> Callback
   ```

- **Fast Audio Conversation Flow** (`audio_chat_with_audio_llm_v4`)
   ```
   Audio input -> Audio Conversation -> A2F/S2M -> Callback
   ```

- **Complete Text Conversation Flow** (`text_chat_with_text_llm_v4`)
   ```
   Text input -> Classification -> Conversation -> TTS -> Reaction -> A2F/S2M -> Callback
   ```

- **Fast Text Conversation Flow** (`text_chat_with_audio_llm_v4`)
   ```
   Text input -> TTS -> Audio conversation -> A2F/S2M -> Callback
   ```

- **Direct Generation Flow** (`direct_generation_v4`)
   ```
   Text input -> TTS -> Reaction -> A2F/S2M -> Callback
   ```

## Quick Start

### Using Docker

### Local Installation

## API Documentation

### WebSocket Endpoints

#### 1. Audio Conversation (Text LLM)
- **Endpoint**: `/api/v4/audio_chat_with_text_llm`
- **Description**: Audio streaming conversation using text modal LLM for conversation processing
- **Request Model**: `AudioChatCompleteStartRequestV4`

#### 2. Audio Conversation (Audio LLM)
- **Endpoint**: `/api/v4/audio_chat_with_audio_llm`
- **Description**: Audio streaming conversation using audio modal LLM for conversation processing
- **Request Model**: `AudioChatExpressStartRequestV4`

#### 3. Text Conversation (Text LLM)
- **Endpoint**: `/api/v4/text_chat_with_text_llm`
- **Description**: Text conversation using text modal LLM for conversation processing
- **Request Model**: `TextChatCompleteRequestV4`

#### 4. Text Conversation (Audio LLM)
- **Endpoint**: `/api/v4/text_chat_with_audio_llm`
- **Description**: Text conversation using audio modal LLM for conversation processing
- **Request Model**: `TextChatExpressRequestV4`

#### 5. Direct Animation Generation
- **Endpoint**: `/api/v4/text_generate`
- **Description**: Direct animation generation from text without conversation
- **Request Model**: `DirectGenerationRequest`

### HTTP Endpoints

#### 1. Health Check
- **Endpoint**: `GET /health` or `GET /api/v1/health`
- **Description**: Check service health status
- **Response**: `{"status": "healthy"}`

#### 2. Logging

**View Logs**
- **Endpoint**: `GET /tail_log/{n_lines}` or `GET /api/v1/tail_log/{n_lines}`
- **Description**: Get the last N lines of the log file
- **Parameters**: `n_lines` - Number of lines to retrieve
- **Response**: HTML formatted log content

**Download Logs**
- **Endpoint**: `GET /download_log_file`
- **Description**: Download complete log file
- **Response**: Binary log file

#### 3. Adapter Selection

**ASR Adapter**
- **Endpoint**: `GET /api/v1/asr_adapter_choices`
- **Description**: Get available ASR (Speech Recognition) adapter list
- **Response**: `AdapterChoicesResponse`

**TTS Adapter**
- **Endpoint**: `GET /api/v1/tts_adapter_choices`
- **Description**: Get available TTS (Text-to-Speech) adapter list
- **Response**: `AdapterChoicesResponse`

**Conversation Adapter**
- **Endpoint**: `GET /api/v1/conversation_adapter_choices`
- **Description**: Get available LLM adapter list
- **Response**: `AdapterChoicesResponse`

**Reaction Adapter**
- **Endpoint**: `GET /api/v1/reaction_adapter_choices`
- **Description**: Get available reaction adapter list
- **Response**: `AdapterChoicesResponse`

**Classification Adapter**
- **Endpoint**: `GET /api/v1/classification_adapter_choices`
- **Description**: Get available classification adapter list
- **Response**: `AdapterChoicesResponse`

**Memory Adapter**
- **Endpoint**: `GET /api/v1/memory_adapter_choices`
- **Description**: Get available memory adapter list
- **Response**: `AdapterChoicesResponse`

#### 4. Voice and Settings

**Voice Management**
- **Endpoint**: `GET /api/v1/tts_voice_names/{tts_adapter_key}`
- **Description**: Get available voice list for specified TTS adapter
- **Parameters**: `tts_adapter_key` - TTS adapter identifier
- **Response**: `VoiceNamesResponse`

**User Settings**
- **Endpoint**: `GET /api/v4/get_voice_settings/{user_id}/{character_id}`
- **Description**: Get voice settings for specified user and character
- **Parameters**:
  - `user_id` - User ID
  - `character_id` - Character ID
- **Response**: `VoiceSettingsResponse`

**Motion Settings**
- **Endpoint**: `GET /api/v4/get_motion_settings/{user_id}/{character_id}`
- **Description**: Get motion settings for specified user and character
- **Parameters**:
  - `user_id` - User ID
  - `character_id` - Character ID
- **Response**: `MotionSettingsResponse`

## AI Services

### LLM
| Provider | Adapter Class | Default Model |
|----------|---------------|---------------|
| OpenAI | `OpenAIConversationClient` | `gpt-4.1-2025-04-14` |
| Anthropic | `AnthropicConversationClient` | `claude-sonnet-4-20250514` |
| Google | `GeminiConversationClient` | `gemini-2.5-flash-lite` |
| DeepSeek | `DeepSeekConversationClient` | `deepseek-chat` |
| xAI | `XAIConversationClient` | `grok-3` |
| SenseNova | `SenseNovaOmniConversationClient` | `SenseNova Omni` |
| OpenAI | `OpenAIAudioClient` | `gpt-4o-mini-realtime-preview-2024-12-17` |

### ASR
| Provider | Adapter Class |
|----------|---------------|
| Softsugar | `SoftSugarASRClient` |
| OpenAI | `OpenAIRealtimeASRClient` |
| SenseTime | `SensetimeASRClient` |

### TTS
| Provider | Adapter Class |
|----------|---------------|
| Volcano Engine | `HuoshanTTSClient` |
| Softsugar | `SoftSugarTTSClient` |
| SenseNova | `SensenovaTTSClient` |
| ElevenLabs | `ElevenLabsTTSClient` |
| SenseTime | `SensetimeTTSClient` |

### Memory
| Provider | Adapter Class | Default Model |
|----------|---------------|---------------|
| xAI | `XAIMemoryClient` | `Grok-3` |
| SenseNova | `SenseNovaOmniMemoryClient` | `SenseNova Omni` |

### Classification
| Provider | Adapter Class | Default Model |
|----------|---------------|---------------|
| OpenAI | `OpenAIClassificationClient` | `gpt-4.1-mini-2025-04-14` |
| xAI | `XAIClassificationClient` | `grok-3` |
| Gemini | `GeminiClassificationClient` | `gemini-2.5-flash-lite` |
| SenseNova | `SenseNovaOmniClassificationClient` | `SenseNova Omni` |

### Reaction
| Provider | Adapter Class | Default Model |
|----------|---------------|---------------|
| OpenAI | `OpenAIReactionClient` | `gpt-4.1-mini-2025-04-14` |
| xAI | `XAIReactionClient` | `grok-3` |
| Gemini | `GeminiReactionClient` | `gemini-2.5-flash-lite` |
| SenseNova | `SenseNovaOmniReactionClient` | `SenseNova Omni` |

## Development Guide

### Adding New AI Services

Taking adding a new TTS service as an example, you need to complete the following steps:

#### 1. Create New Client Class

Create a new client file in the `orchestrator/generation/text2speech/` directory, for example `new_tts_client.py`:

```python
from .tts_adapter import TextToSpeechAdapter

class NewTTSClient(TextToSpeechAdapter):
    """New TTS client implementation"""

    AVAILABLE_FOR_STREAM = True  # Whether streaming is supported

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        # Initialize client-specific parameters

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
        """Implement TTS generation logic"""
        # Call third-party TTS API
        # Return dictionary containing audio, speech_text, speech_time, duration
        pass

    async def get_voice_names(self, **kwargs: Any) -> Dict[str, Any]:
        """Return available voice list"""
        return {"voice_id": "voice_name"}
```

#### 2. Update Builder

Register the new client in `orchestrator/generation/text2speech/builder.py`:

```python
from .new_tts_client import NewTTSClient

_TTS_ADAPTERS = dict(
    # Existing adapters...
    NewTTSClient=NewTTSClient,
)
```

#### 3. Update Configuration Files

Add new TTS adapter configuration in `configs/local.py`, `configs/docker.py`, etc.:

```python
tts_adapters=dict(
    # Existing adapters...
    new_tts=dict(
        type="NewTTSClient",
        name="new_tts_client",
        # Client-specific parameters...
    ),
),
```

#### 4. Add Tests

Add test functions in `tests/adapters/test_tts_adapters.py`:

```python
@pytest.mark.asyncio
async def test_new_tts_client_stream():
    """Test new TTS client streaming functionality"""
    # Check environment variables
    api_key = os.environ.get("NEW_TTS_API_KEY")
    if not api_key:
        pytest.skip("NEW_TTS_API_KEY is not set")

    # Configuration and test logic
    tts_client_cfg = dict(
        type="NewTTSClient",
        name="new_tts_client",
    )
    # Test implementation...
```

### Testing
The project includes comprehensive testing using pytest with async test support. Tests cover various components of adapters, aggregators, and IO modules.

#### Test Categories

The project includes the following test modules:

- **Adapter Tests** (`tests/adapters/`): Test various service adapters
  - `test_a2f_adapters.py` - Audio2Face adapter tests
  - `test_asr_adapters.py` - Speech recognition adapter tests
  - `test_audio_conversation_adapters.py` - Audio conversation adapter tests
  - `test_classification_adapters.py` - Classification adapter tests
  - `test_conversation_adapter.py` - Conversation adapter tests
  - `test_memory_adapters.py` - Memory adapter tests
  - `test_reaction_adapters.py` - Reaction adapter tests
  - `test_s2m_adapters.py` - Speech-to-motion adapter tests
  - `test_tts_adapters.py` - Text-to-speech adapter tests

- **Aggregator Tests** (`tests/aggregator/`): Test data aggregators
  - `test_blendshapes_aggregator.py` - Blendshapes aggregator tests
  - `test_conversation_aggregator.py` - Conversation aggregator tests
  - `test_tts_reaction_aggregator.py` - TTS reaction aggregator tests

- **IO Tests** (`tests/io/`): Test input/output modules
  - `test_config_client.py` - Configuration client tests
  - `test_memory_client.py` - Memory client tests

#### Test Data Preparation

Download required test files and organize them into the correct directory structure:

1. **Create test input directory:**
   ```bash
   mkdir -p input
   ```

2. **Download test files:**
   ```bash
   cd input
   # Download test audio files with different sample rates
   curl -LjO https://github.com/LazyBusyYang/CatStream/releases/download/orchestrator_cicd_files/test_audio_16kHz.wav
   curl -LjO https://github.com/LazyBusyYang/CatStream/releases/download/orchestrator_cicd_files/test_audio_24kHz.wav
   # Create default test audio file
   cp test_audio_16kHz.wav test_audio.wav
   cd ..
   ```

#### Running Tests

1. **Run all tests:**
   ```bash
   # Create logs directory
   mkdir -p logs

   # Run all tests
   python -m pytest tests --log-cli-level=ERROR
   ```

2. **Run specific test modules:**
   ```bash
   # Run adapter tests
   python -m pytest tests/adapters/

   # Run aggregator tests
   python -m pytest tests/aggregator/

   # Run IO tests
   python -m pytest tests/io/
   ```

3. **Run specific test files:**
   ```bash
   # Run conversation adapter tests
   python -m pytest tests/adapters/test_conversation_adapter.py

   # Run configuration client tests
   python -m pytest tests/io/test_config_client.py
   ```

### Code Quality

The project maintains high code quality through:

- **Code Inspection**: Using Ruff for code style and quality checks
- **Type Hints**: Complete type annotation support
- **CI/CD**: Automated testing and deployment pipelines

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The MIT License is a permissive free software license that allows you to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software with very few restrictions. The only requirement is that the original copyright notice and license text must be included in all copies or substantial portions of the software.
