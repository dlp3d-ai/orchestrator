# API 文档

## WebSocket 端点

### 1. 音频对话（文本 LLM）
- 端点：`/api/v4/audio_chat_with_text_llm`
- 描述：使用文本模态 LLM 进行对话处理的音频流式对话
- 请求模型：`AudioChatCompleteStartRequestV4`

### 2. 音频对话（音频 LLM）
- 端点：`/api/v4/audio_chat_with_audio_llm`
- 描述：使用音频模态 LLM 进行对话处理的音频流式对话
- 请求模型：`AudioChatExpressStartRequestV4`

### 3. 文本对话（文本 LLM）
- 端点：`/api/v4/text_chat_with_text_llm`
- 描述：使用文本模态 LLM 进行对话处理的文本对话
- 请求模型：`TextChatCompleteRequestV4`

### 4. 文本对话（音频 LLM）
- 端点：`/api/v4/text_chat_with_audio_llm`
- 描述：使用音频模态 LLM 进行对话处理的文本对话
- 请求模型：`TextChatExpressRequestV4`

### 5. 直接动画生成
- 端点：`/api/v4/text_generate`
- 描述：从文本直接生成动画，不经过对话
- 请求模型：`DirectGenerationRequest`

## HTTP 端点

### 1. 健康检查
- 端点：`GET /health` 或 `GET /api/v4/health`
- 描述：检查服务健康状态
- 响应：`{"status": "healthy"}`

### 2. 日志

查看日志
- 端点：`GET /tail_log/{n_lines}` 或 `GET /api/v4/tail_log/{n_lines}`
- 描述：获取日志文件的最后 N 行
- 参数：`n_lines`——要获取的行数
- 响应：HTML 格式日志内容

下载日志
- 端点：`GET /download_log_file`
- 描述：下载完整日志文件
- 响应：二进制日志文件

### 3. 适配器选择

ASR 适配器
- 端点：`GET /api/v4/asr_adapter_choices`
- 描述：获取可用 ASR（语音识别）适配器列表
- 响应：`AdapterChoicesResponse`

TTS 适配器
- 端点：`GET /api/v4/tts_adapter_choices`
- 描述：获取可用 TTS（语音合成）适配器列表
- 响应：`AdapterChoicesResponse`

对话适配器
- 端点：`GET /api/v4/conversation_adapter_choices`
- 描述：获取可用 LLM 适配器列表
- 响应：`AdapterChoicesResponse`

反应适配器
- 端点：`GET /api/v4/reaction_adapter_choices`
- 描述：获取可用 Reaction 适配器列表
- 响应：`AdapterChoicesResponse`

分类适配器
- 端点：`GET /api/v4/classification_adapter_choices`
- 描述：获取可用 Classification 适配器列表
- 响应：`AdapterChoicesResponse`

记忆适配器
- 端点：`GET /api/v4/memory_adapter_choices`
- 描述：获取可用 Memory 适配器列表
- 响应：`AdapterChoicesResponse`

### 4. 音色与设置

音色管理
- 端点：`GET /api/v4/tts_voice_names/{tts_adapter_key}`
- 描述：获取指定 TTS 适配器的可用音色列表
- 参数：`tts_adapter_key`——TTS 适配器标识
- 响应：`VoiceNamesResponse`

用户音色设置
- 端点：`GET /api/v4/get_voice_settings/{user_id}/{character_id}`
- 描述：获取指定用户与角色的音色设置
- 参数：
  - `user_id`——用户 ID
  - `character_id`——角色 ID
- 响应：`VoiceSettingsResponse`

动作设置
- 端点：`GET /api/v4/get_motion_settings/{user_id}/{character_id}`
- 描述：获取指定用户与角色的动作设置
- 参数：
  - `user_id`——用户 ID
  - `character_id`——角色 ID
- 响应：`MotionSettingsResponse`


