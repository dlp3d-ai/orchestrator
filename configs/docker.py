import os

# CRITICAL = 50
# ERROR = 40
# WARNING = 30
# INFO = 20
# DEBUG = 10
# NOTSET = 0
__logger_cfg__ = dict(
    logger_name="root",
    aws_level=10,
    file_level=10,
    console_level=20,
    logger_path="logs/server.log",
)
server = dict(
    type="OrchestratorProxyServer",
    name="proxy_server",
    proxy=None,
    enable_cors=True,
    host="0.0.0.0",
    port=18081,
    logger_cfg=__logger_cfg__,
)
proxy = dict(
    type="Proxy",
    db_memory_cfg=dict(
        type="MongoDBMemoryClient",
        host=os.environ["MONGODB_HOST"],
        port=int(os.environ["MONGODB_PORT"]),
        username=os.environ["MONGODB_MEMORY_USER"],
        password=os.environ["MONGODB_MEMORY_PASSWORD"],
        database=os.environ["MONGODB_MEMORY_DB"],
        auth_database=os.environ["MONGODB_MEMORY_DB"],
    ),
    db_config_cfg=dict(
        type="MongoDBConfigClient",
        host=os.environ["MONGODB_HOST"],
        port=int(os.environ["MONGODB_PORT"]),
        username=os.environ["MONGODB_WEB_USER"],
        password=os.environ["MONGODB_WEB_PASSWORD"],
        database=os.environ["MONGODB_WEB_DB"],
        auth_database=os.environ["MONGODB_WEB_DB"],
    ),
    db_config_cache_sync_trigger="健康检查",
    memory_adapters=dict(
        openai_memory=dict(
            type="OpenAIMemoryClient",
            name="openai_memory_client",
            openai_model_name="gpt-4.1-mini-2025-04-14",
            proxy_url=os.environ.get("PROXY_URL", None),
        ),
        xai_memory=dict(
            type="XAIMemoryClient",
            name="xai_memory_client",
            xai_model_name="grok-3",
            proxy_url=os.environ.get("PROXY_URL", None),
        ),
        sensenova_omni_memory=dict(
            type="SenseNovaOmniMemoryClient",
            name="sensenova_omni_memory_client",
            wss_url="wss://api-gai.sensetime.com/agent-5o/duplex/ws2",
        ),
    ),
    conversation_adapters=dict(
        anthropic_agent=dict(
            type="AnthropicConversationClient",
            name="anthropic_agent_client",
            agent_prompts_file="configs/agent_prompts.yaml",
            anthropic_model_name="claude-sonnet-4-5-20250929",
            proxy_url=os.environ.get("PROXY_URL", None),
        ),
        openai_agent=dict(
            type="OpenAIConversationClient",
            agent_prompts_file="configs/agent_prompts.yaml",
            name="openai_agent_client",
            openai_model_name="gpt-4.1-2025-04-14",
            proxy_url=os.environ.get("PROXY_URL", None),
        ),
        gemini_agent=dict(
            type="GeminiConversationClient",
            name="gemini_agent_client",
            agent_prompts_file="configs/agent_prompts.yaml",
            gemini_model_name="gemini-2.5-flash-lite",
            proxy_url=os.environ.get("PROXY_URL", None),
        ),
        deepseek_agent=dict(
            type="DeepSeekConversationClient",
            name="deepseek_agent_client",
            agent_prompts_file="configs/agent_prompts.yaml",
            deepseek_model_name="deepseek-chat",
            proxy_url=os.environ.get("PROXY_URL", None),
        ),
        xai_agent=dict(
            type="XAIConversationClient",
            name="xai_agent_client",
            agent_prompts_file="configs/agent_prompts.yaml",
            xai_model_name="grok-3",
            proxy_url=os.environ.get("PROXY_URL", None),
        ),
        sensenova_omni_agent=dict(
            type="SenseNovaOmniConversationClient",
            name="sensenova_omni_agent_client",
            agent_prompts_file="configs/agent_prompts.yaml",
            wss_url="wss://api-gai.sensetime.com/agent-5o/duplex/ws2",
        ),
        openai_audio_agent=dict(
            type="OpenAIAudioClient",
            name="openai_audio_agent_client",
            agent_prompts_file="configs/agent_prompts.yaml",
            wss_url="wss://api.openai.com/v1/realtime",
            proxy_url=os.environ.get("PROXY_URL", None),
        ),
    ),
    classfication_adapters=dict(
        openai_classification=dict(
            type="OpenAIClassificationClient",
            name="openai_classification_client",
            motion_keywords=os.environ.get("BACKEND_URL", None),
            proxy_url=os.environ.get("PROXY_URL", None),
            openai_model_name="gpt-4.1-mini-2025-04-14",
        ),
        xai_classification=dict(
            type="XAIClassificationClient",
            name="xai_classification_client",
            motion_keywords=os.environ.get("BACKEND_URL", None),
            proxy_url=os.environ.get("PROXY_URL", None),
            xai_model_name="grok-3",
        ),
        gemini_classification=dict(
            type="GeminiClassificationClient",
            name="gemini_classification_client",
            motion_keywords=os.environ.get("BACKEND_URL", None),
            proxy_url=os.environ.get("PROXY_URL", None),
            gemini_model_name="gemini-2.5-flash-lite",
        ),
        sensenova_omni_classification=dict(
            type="SenseNovaOmniClassificationClient",
            name="sensenova_omni_classification_client",
            motion_keywords=os.environ.get("BACKEND_URL", None),
            wss_url="wss://api-gai.sensetime.com/agent-5o/duplex/ws2",
        ),
    ),
    reaction_adapters=dict(
        openai_reaction=dict(
            type="OpenAIReactionClient",
            name="openai_reaction_client",
            motion_keywords=os.environ.get("BACKEND_URL", None),
            proxy_url=os.environ.get("PROXY_URL", None),
            openai_model_name="gpt-4.1-mini-2025-04-14",
        ),
        xai_reaction=dict(
            type="XAIReactionClient",
            name="xai_reaction_client",
            motion_keywords=os.environ.get("BACKEND_URL", None),
            proxy_url=os.environ.get("PROXY_URL", None),
            xai_model_name="grok-3",
        ),
        gemini_reaction=dict(
            type="GeminiReactionClient",
            name="gemini_reaction_client",
            motion_keywords=os.environ.get("BACKEND_URL", None),
            proxy_url=os.environ.get("PROXY_URL", None),
            gemini_model_name="gemini-2.5-flash-lite",
        ),
        sensenova_omni_reaction=dict(
            type="SenseNovaOmniReactionClient",
            name="sensenova_omni_reaction_client",
            motion_keywords=os.environ.get("BACKEND_URL", None),
            wss_url="wss://api-gai.sensetime.com/agent-5o/duplex/ws2",
        ),
    ),
    a2f_cfg=dict(
        type="Audio2FaceStreamingClient",
        ws_url=os.environ["A2F_WS_URL"],
        timeout=10.0,
    ),
    asr_adapters=dict(
        softsugar=dict(
            type="SoftSugarASRClient",
            name="softsugar_asr_client",
            ws_url="ws://aigc.softsugar.com/api/voice/stream/v1",
            softsugar_api="https://aigc.softsugar.com",
            queue_size=5000,
        ),
        openai_realtime=dict(
            type="OpenAIRealtimeASRClient",
            name="openai_realtime_asr_client",
            wss_url="wss://api.openai.com/v1/realtime?intent=transcription",
            openai_model_name="gpt-4o-mini-transcribe",
            proxy_url=os.environ.get("PROXY_URL", None),
            queue_size=5000,
        ),
        huoshan=dict(
            type="HuoshanASRClient",
            name="huoshan_asr_client",
            wss_url="wss://openspeech.bytedance.com/api/v2/asr",
            cluster_id="volcengine_streaming_common",
            default_language="zh-CN",
            request_timeout=5,
            commit_timeout=10,
            queue_size=5000,
        ),
    ),
    s2m_cfg=dict(
        type="Speech2MotionStreamingClient",
        ws_url=os.environ["S2M_WS_URL"],
        timeout=20.0,
    ),
    tts_adapters=dict(
        huoshan=dict(
            type="HuoshanTTSClient",
            name="huoshan_tts_client",
            tts_ws_url="wss://openspeech.bytedance.com/api/v1/tts/ws_binary",
            cluster="volcano_tts",
        ),
        huoshan_icl=dict(
            type="HuoshanTTSClient",
            name="huoshan_icl_client",
            tts_ws_url="wss://openspeech.bytedance.com/api/v1/tts/ws_binary",
            cluster="volcano_icl",
        ),
        softsugar=dict(
            type="SoftSugarTTSClient",
            name="softsugar_tts_client",
            tts_ws_url="ws://aigc.softsugar.com/api/voice/stream/v3",
            softsugar_token_url="https://aigc.softsugar.com/api/uc/v1/access/api/token",
        ),
        sensenova=dict(
            type="SensenovaTTSClient",
            name="sensenova_tts_client",
            tts_ws_url="wss://api-gai-internal.sensetime.com/agent-5o/signpost/v0/prod-agent-5o-tts-sovits-cosyvoice2-public/ws/stream",
        ),
        elevenlabs=dict(
            type="ElevenLabsTTSClient",
            name="elevenlabs_tts_client",
            elevenlabs_model_name="eleven_flash_v2_5",
        ),
    ),
    conversation_aggregator_cfg=dict(type="ConversationAggregator"),
    tts_reaction_aggregator_cfg=dict(type="TTSReactionAggregator"),
    blendshapes_aggregator_cfg=dict(
        type="BlendshapesAggregator",
        motion_first_blendshape_names="configs/eyelid_bs_names.json",
        add_blendshape_names="configs/eyebrow_mouthcorner_bs_names.json",
    ),
    callback_aggregator_cfg=dict(
        type="CallbackAggregator",
        verbose=True,
    ),
    process_timeout=120.0,
    sleep_time=1.0,
    logger_cfg=__logger_cfg__,
)
