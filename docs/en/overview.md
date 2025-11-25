# Overview
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
│   ├── sensechat_conversation_client.py # SenseChat conversation client
│   ├── sensenova_conversation_client.py # SenseNova conversation client
│   └── sensenova_omni_conversation_client.py  # SenseNova real-time conversation client
├── generation/                # Generation management module
│   ├── speech_recognition/    # Speech Recognition (ASR)
│   │   ├── asr_adapter.py     # ASR adapter base class
│   │   ├── huoshan_asr_client.py # Volcano Engine ASR
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
|   ├── openai_memory_client.py  # OpenAI memory client
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

**Diagram Legend:**
- **Solid arrows** (→): One-time complete data transmission between nodes in a single generation request
- **Dashed arrows** (⇢): Streaming data transmission between nodes in a single generation request

**Workflow Diagrams:**

 - **Complete Audio Conversation Flow** (`audio_chat_with_text_llm_v4`)
   <div style="text-align: center;">
     <img src="_static/audio_chat_with_text_llm_v4.svg" style="width: 100%; max-width: 100%;">
     <p><em>Complete Audio Conversation Flow</em></p>
   </div>

 - **Express Audio Conversation Flow** (`audio_chat_with_audio_llm_v4`)
   <div style="text-align: center;">
     <img src="_static/audio_chat_with_audio_llm_v4.svg" style="width: 100%; max-width: 100%;">
     <p><em>Express Audio Conversation Flow</em></p>
   </div>

 - **Complete Text Conversation Flow** (`text_chat_with_text_llm_v4`)
   <div style="text-align: center;">
     <img src="_static/text_chat_with_text_llm_v4.svg" style="width: 100%; max-width: 100%;">
     <p><em>Complete Text Conversation Flow</em></p>
   </div>

 - **Express Text Conversation Flow** (`text_chat_with_audio_llm_v4`)
   <div style="text-align: center;">
     <img src="_static/text_chat_with_audio_llm_v4.svg" style="width: 100%; max-width: 100%;">
     <p><em>Express Text Conversation Flow</em></p>
   </div>

 - **Direct Generation Flow** (`direct_generation_v4`)
   <div style="text-align: center;">
     <img src="_static/direct_generation_v4.svg" style="width: 100%; max-width: 100%;">
     <p><em>Direct Generation Flow</em></p>
   </div>

## AI Services

### LLM
| Provider | Adapter Class | Default Model |
|----------|---------------|---------------|
| OpenAI | `OpenAIConversationClient` | `gpt-4.1-2025-04-14` |
| Anthropic | `AnthropicConversationClient` | `claude-sonnet-4-5-20250929` |
| Google | `GeminiConversationClient` | `gemini-2.5-flash-lite` |
| DeepSeek | `DeepSeekConversationClient` | `deepseek-chat` |
| xAI | `XAIConversationClient` | `grok-3` |
| SenseNova | `SenseChatConversationClient` | `SenseChat-5-1202` (Large Language Model) |
| SenseNova | `SenseNovaConversationClient` | `SenseNova-V6-5-Pro` (Multimodal Model) |
| SenseNova | `SenseNovaOmniConversationClient` | `SenseNova-V6-5-Omni` (Real-time Interactive Multimodal Model) |
| OpenAI | `OpenAIAudioClient` | `gpt-4o-mini-realtime-preview-2024-12-17` |

### ASR
| Provider | Adapter Class |
|----------|---------------|
| OpenAI | `OpenAIRealtimeASRClient` |
| Volcano Engine | `HuoshanASRClient` |
| SenseTime | `SensetimeASRClient` |
| Softsugar | `SoftSugarASRClient` |

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
| OpenAI | `OpenAIMemoryClient` | `gpt-4.1-mini-2025-04-14` |
| xAI | `XAIMemoryClient` | `Grok-3` |
| SenseNova | `SenseNovaOmniMemoryClient` | `SenseNova-V6-5-Omni` |

### Classification
| Provider | Adapter Class | Default Model |
|----------|---------------|---------------|
| OpenAI | `OpenAIClassificationClient` | `gpt-4.1-mini-2025-04-14` |
| xAI | `XAIClassificationClient` | `grok-3` |
| Gemini | `GeminiClassificationClient` | `gemini-2.5-flash-lite` |
| SenseNova | `SenseNovaOmniClassificationClient` | `SenseNova-V6-5-Omni` |

### Reaction
| Provider | Adapter Class | Default Model |
|----------|---------------|---------------|
| OpenAI | `OpenAIReactionClient` | `gpt-4.1-mini-2025-04-14` |
| xAI | `XAIReactionClient` | `grok-3` |
| Gemini | `GeminiReactionClient` | `gemini-2.5-flash-lite` |
| SenseNova | `SenseNovaOmniReactionClient` | `SenseNova-V6-5-Omni` |
