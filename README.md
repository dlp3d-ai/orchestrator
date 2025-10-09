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
- [AI Services](#ai-services)
- [Documentation](#documentation)
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

#### Recommended: Complete Backend Services with Docker Compose

For the best experience, we recommend using Docker Compose to start the complete DLP3D backend services, which includes the Orchestrator along with all required dependencies (MongoDB, Audio2Face, Speech2Motion, etc.).

Please follow the [Complete DLP3D Backend Services](https://github.com/dlp3d-ai/web_backend?tab=readme-ov-file#complete-dlp3d-backend-services) documentation to set up and run the entire backend infrastructure.

#### Standalone Orchestrator Service

If you need to run the Orchestrator service independently or configure advanced options, please refer to the [Docker Configuration Guide](docs/docker.md) for detailed setup instructions, environment variables, and configuration options.

### Environment Setup

For local development and deployment, please follow the detailed installation guide:

📖 **[Complete Installation Guide](docs/install.md)**

The installation guide provides step-by-step instructions for:
- Setting up Python 3.10+ environment
- Installing Protocol Buffers compiler
- Configuring the development environment
- Installing project dependencies

### Local Development

After completing the environment setup as described in the installation guide, you can start the service locally:

```bash
# Activate the conda environment
conda activate orchestrator

# Start the service
python main.py --config_path configs/local.py
```

## AI Services

### LLM
| Provider | Adapter Class | Default Model |
|----------|---------------|---------------|
| OpenAI | `OpenAIConversationClient` | `gpt-4.1-2025-04-14` |
| Anthropic | `AnthropicConversationClient` | `claude-sonnet-4-5-20250929` |
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

## Documentation

- **[API Documentation](docs/API_Documentation.md)** - Complete API reference for WebSocket and HTTP endpoints
- **[Development Guide](docs/Development_Guide.md)** - Guide for adding new AI services, testing, and code quality standards

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The MIT License is a permissive free software license that allows you to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software with very few restrictions. The only requirement is that the original copyright notice and license text must be included in all copies or substantial portions of the software.
