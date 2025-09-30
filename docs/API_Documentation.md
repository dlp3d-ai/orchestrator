# API Documentation

## WebSocket Endpoints

### 1. Audio Conversation (Text LLM)
- **Endpoint**: `/api/v4/audio_chat_with_text_llm`
- **Description**: Audio streaming conversation using text modal LLM for conversation processing
- **Request Model**: `AudioChatCompleteStartRequestV4`

### 2. Audio Conversation (Audio LLM)
- **Endpoint**: `/api/v4/audio_chat_with_audio_llm`
- **Description**: Audio streaming conversation using audio modal LLM for conversation processing
- **Request Model**: `AudioChatExpressStartRequestV4`

### 3. Text Conversation (Text LLM)
- **Endpoint**: `/api/v4/text_chat_with_text_llm`
- **Description**: Text conversation using text modal LLM for conversation processing
- **Request Model**: `TextChatCompleteRequestV4`

### 4. Text Conversation (Audio LLM)
- **Endpoint**: `/api/v4/text_chat_with_audio_llm`
- **Description**: Text conversation using audio modal LLM for conversation processing
- **Request Model**: `TextChatExpressRequestV4`

### 5. Direct Animation Generation
- **Endpoint**: `/api/v4/text_generate`
- **Description**: Direct animation generation from text without conversation
- **Request Model**: `DirectGenerationRequest`

## HTTP Endpoints

### 1. Health Check
- **Endpoint**: `GET /health` or `GET /api/v1/health`
- **Description**: Check service health status
- **Response**: `{"status": "healthy"}`

### 2. Logging

**View Logs**
- **Endpoint**: `GET /tail_log/{n_lines}` or `GET /api/v1/tail_log/{n_lines}`
- **Description**: Get the last N lines of the log file
- **Parameters**: `n_lines` - Number of lines to retrieve
- **Response**: HTML formatted log content

**Download Logs**
- **Endpoint**: `GET /download_log_file`
- **Description**: Download complete log file
- **Response**: Binary log file

### 3. Adapter Selection

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

### 4. Voice and Settings

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
