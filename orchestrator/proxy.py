import asyncio
import io
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple, Union

from .aggregator.builder import (
    BlendshapesAggregator,
    CallbackAggregator,
    ConversationAggregator,
    TTSReactionAggregator,
    build_aggregator,
)
from .classification.builder import ClassificationAdapter, build_classification_adapter
from .conversation.audio_conversation_adapter import AudioConversationAdapter
from .conversation.builder import build_conversation_adapter
from .conversation.conversation_adapter import ConversationAdapter
from .data_structures import orchestrator_v4_pb2 as orchestrator_pb2
from .data_structures.audio_chunk import AudioChunkBody, AudioChunkEnd, AudioChunkStart
from .data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from .data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from .generation.audio2face.builder import build_audio2face_adapter
from .generation.speech2motion.builder import build_speech2motion_adapter
from .generation.speech_recognition.asr_adapter import AutomaticSpeechRecognitionAdapter
from .generation.speech_recognition.builder import build_asr_adapter
from .generation.text2speech.builder import build_tts_adapter
from .generation.text2speech.tts_adapter import TextToSpeechAdapter
from .io.config.builder import build_db_config_client
from .io.config.dynamodb_redis_config_client import DynamoDBRedisConfigClient
from .io.memory.builder import build_db_memory_client
from .memory.builder import build_memory_adapter
from .memory.memory_adapter import INITIAL_EMOTION_STATE, INITIAL_RELATIONSHIP_STATE
from .memory.memory_manager import MemoryManager
from .reaction.builder import ReactionAdapter, build_reaction_adapter
from .utils.executor_registry import ExecutorRegistry
from .utils.super import Super


class AdapterNotFoundError(Exception):
    """Exception raised when a required adapter is not found.

    This exception is raised when the system attempts to use an adapter (such
    as TTS, ASR, conversation, etc.) that is not available in the configured
    adapters dictionary.
    """

    pass


class RunningDAGNotFoundError(Exception):
    """Exception raised when a running DAG is not found.

    This exception is raised when attempting to interact with a DAG (Directed
    Acyclic Graph) that is not currently running or has already
    completed/failed.
    """

    pass


class Proxy(Super):
    """Main orchestrator proxy class for managing AI conversation workflows.

    The Proxy class serves as the central orchestrator that manages various AI
    adapters and coordinates complex workflows involving speech recognition,
    text-to-speech, conversation, memory management, and 3D animation
    generation. It handles multiple types of conversation modes including audio
    chat with text LLM, audio chat with audio LLM, and text chat scenarios.
    """

    ExecutorRegistry.register_class("Proxy")

    def __init__(
        self,
        a2f_cfg: Dict[str, Any],
        asr_adapters: Dict[str, Dict[str, Any]],
        s2m_cfg: Dict[str, Any],
        tts_adapters: Dict[str, Dict[str, Any]],
        memory_adapters: Dict[str, Dict[str, Any]],
        conversation_adapters: Dict[str, Dict[str, Any]],
        classfication_adapters: Dict[str, Dict[str, Any]],
        reaction_adapters: Dict[str, Dict[str, Any]],
        conversation_aggregator_cfg: Dict[str, Any],
        tts_reaction_aggregator_cfg: Dict[str, Any],
        blendshapes_aggregator_cfg: Dict[str, Any],
        callback_aggregator_cfg: Dict[str, Any],
        db_memory_cfg: Dict[str, Any],
        db_config_cfg: Dict[str, Any],
        db_config_cache_sync_trigger: Union[None, str] = None,
        process_timeout: float = 120.0,
        sleep_time: float = 1.0,
        logger_cfg: Union[None, Dict[str, Any]] = None,
        max_workers: int = 4,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        style_file: str = "configs/voice_style.json",
    ) -> None:
        """Initialize the orchestrator proxy with all required adapters and
        configurations.

        Args:
            a2f_cfg (Dict[str, Any]):
                Configuration for the audio2face adapter.
            asr_adapters (Dict[str, Dict[str, Any]]):
                Dictionary of ASR adapter configurations.
                Key is the adapter name, value is its configuration.
            s2m_cfg (Dict[str, Any]):
                Configuration for the speech2motion adapter.
            tts_adapters (Dict[str, Dict[str, Any]]):
                Dictionary of TTS adapter configurations.
                Key is the adapter name, value is its configuration.
            memory_adapters (Dict[str, Dict[str, Any]]):
                Dictionary of memory adapter configurations.
                Key is the adapter name, value is its configuration.
            conversation_adapters (Dict[str, Dict[str, Any]]):
                Dictionary of conversation adapter configurations.
                Key is the adapter name, value is its configuration.
            classfication_adapters (Dict[str, Dict[str, Any]]):
                Dictionary of classification adapter configurations.
                Key is the adapter name, value is its configuration.
            reaction_adapters (Dict[str, Dict[str, Any]]):
                Dictionary of reaction adapter configurations.
                Key is the adapter name, value is its configuration.
            conversation_aggregator_cfg (Dict[str, Any]):
                Configuration for the conversation aggregator.
            tts_reaction_aggregator_cfg (Dict[str, Any]):
                Configuration for the TTS reaction aggregator.
            blendshapes_aggregator_cfg (Dict[str, Any]):
                Configuration for the blendshapes aggregator.
            callback_aggregator_cfg (Dict[str, Any]):
                Configuration for the callback aggregator.
            db_memory_cfg (Dict[str, Any]):
                Configuration for the character's memory database client.
            db_config_cfg (Dict[str, Any]):
                Configuration for the character/user config database client.
            db_config_cache_sync_trigger (Union[None, str], optional):
                Trigger string for manual database cache synchronization.
                If None, manual sync is disabled. Defaults to None.
            process_timeout (float, optional):
                Maximum time in seconds for process execution.
                Defaults to 120.0.
            sleep_time (float, optional):
                Sleep interval in seconds between status checks.
                Defaults to 1.0.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary.
                If None, default logger configuration is used.
                Defaults to None.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 4.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor.
                If None, a new thread pool executor will be created based on
                max_workers. Defaults to None.
            style_file (str, optional):
                Path to the voice style configuration file.
                Defaults to "configs/voice_style.json".
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.sleep_time = sleep_time
        self.process_timeout = process_timeout
        self.executor = (
            thread_pool_executor if thread_pool_executor is not None else ThreadPoolExecutor(max_workers=max_workers)
        )
        self.executor_external = True if thread_pool_executor is not None else False

        self.a2f_cfg = a2f_cfg.copy()
        self.a2f_cfg["logger_cfg"] = logger_cfg
        if ExecutorRegistry.validate_class(self.a2f_cfg["type"]):
            self.a2f_cfg["thread_pool_executor"] = self.executor
        self.a2f_adapter = build_audio2face_adapter(self.a2f_cfg)

        self.asr_adapters: Dict[str, AutomaticSpeechRecognitionAdapter] = dict()
        self.asr_cfgs = dict()
        for asr_key, asr_cfg in asr_adapters.items():
            self.asr_cfgs[asr_key] = asr_cfg.copy()
            self.asr_cfgs[asr_key]["logger_cfg"] = logger_cfg
            if ExecutorRegistry.validate_class(self.asr_cfgs[asr_key]["type"]):
                self.asr_cfgs[asr_key]["thread_pool_executor"] = self.executor
            self.asr_adapters[asr_key] = build_asr_adapter(self.asr_cfgs[asr_key])

        self.db_memory_cfg = db_memory_cfg.copy()
        self.db_memory_cfg["logger_cfg"] = logger_cfg
        if ExecutorRegistry.validate_class(self.db_memory_cfg["type"]):
            self.db_memory_cfg["thread_pool_executor"] = self.executor
        self.db_memory_client = build_db_memory_client(self.db_memory_cfg)

        self.memory_adapters: Dict[str, Any] = dict()
        self.memory_cfgs = dict()
        for memory_key, memory_cfg in memory_adapters.items():
            self.memory_cfgs[memory_key] = memory_cfg.copy()
            self.memory_cfgs[memory_key]["logger_cfg"] = logger_cfg
            self.memory_cfgs[memory_key]["db_client"] = self.db_memory_client
            if ExecutorRegistry.validate_class(self.memory_cfgs[memory_key]["type"]):
                self.memory_cfgs[memory_key]["thread_pool_executor"] = self.executor
            self.memory_adapters[memory_key] = build_memory_adapter(self.memory_cfgs[memory_key])

            # Create corresponding MemoryManager for each memory adapter
            MemoryManager(
                db_client=self.db_memory_client,
                memory_adapter=self.memory_adapters[memory_key],
                conversation_char_threshold=memory_cfg.get("conversation_char_threshold", 10000),
                conversation_char_target=memory_cfg.get("conversation_char_target", 8000),
                short_term_length_threshold=memory_cfg.get("short_term_length_threshold", 20),
                short_term_target_size=memory_cfg.get("short_term_target_size", 10),
                medium_term_length_threshold=memory_cfg.get("medium_term_length_threshold", 10),
                sleep_time=memory_cfg.get("sleep_time", 1.0),
                logger_cfg=logger_cfg,
            )

        self.conversation_adapters: Dict[str, Union[ConversationAdapter, AudioConversationAdapter]] = dict()
        self.conversation_cfgs = dict()
        for conversation_key, conversation_cfg in conversation_adapters.items():
            self.conversation_cfgs[conversation_key] = conversation_cfg.copy()
            self.conversation_cfgs[conversation_key]["logger_cfg"] = logger_cfg
            if ExecutorRegistry.validate_class(self.conversation_cfgs[conversation_key]["type"]):
                self.conversation_cfgs[conversation_key]["thread_pool_executor"] = self.executor
            self.conversation_adapters[conversation_key] = build_conversation_adapter(
                self.conversation_cfgs[conversation_key]
            )

        self.s2m_cfg = s2m_cfg.copy()
        self.s2m_cfg["logger_cfg"] = logger_cfg
        if ExecutorRegistry.validate_class(self.s2m_cfg["type"]):
            self.s2m_cfg["thread_pool_executor"] = self.executor
        self.s2m_adapter = build_speech2motion_adapter(self.s2m_cfg)

        self.tts_adapters: Dict[str, TextToSpeechAdapter] = dict()
        self.tts_cfgs = dict()
        for tts_key, tts_cfg in tts_adapters.items():
            self.tts_cfgs[tts_key] = tts_cfg.copy()
            self.tts_cfgs[tts_key]["logger_cfg"] = logger_cfg
            if ExecutorRegistry.validate_class(self.tts_cfgs[tts_key]["type"]):
                self.tts_cfgs[tts_key]["thread_pool_executor"] = self.executor
            self.tts_adapters[tts_key] = build_tts_adapter(self.tts_cfgs[tts_key])

        self.classfication_adapters: Dict[str, ClassificationAdapter] = dict()
        self.classfication_cfgs = dict()
        for classfication_key, classfication_cfg in classfication_adapters.items():
            self.classfication_cfgs[classfication_key] = classfication_cfg.copy()
            self.classfication_cfgs[classfication_key]["logger_cfg"] = logger_cfg
            if ExecutorRegistry.validate_class(self.classfication_cfgs[classfication_key]["type"]):
                self.classfication_cfgs[classfication_key]["thread_pool_executor"] = self.executor
            self.classfication_adapters[classfication_key] = build_classification_adapter(
                self.classfication_cfgs[classfication_key]
            )

        self.reaction_adapters: Dict[str, ReactionAdapter] = dict()
        self.reaction_cfgs = dict()
        for reaction_key, reaction_cfg in reaction_adapters.items():
            self.reaction_cfgs[reaction_key] = reaction_cfg.copy()
            self.reaction_cfgs[reaction_key]["logger_cfg"] = logger_cfg
            if ExecutorRegistry.validate_class(self.reaction_cfgs[reaction_key]["type"]):
                self.reaction_cfgs[reaction_key]["thread_pool_executor"] = self.executor
            self.reaction_adapters[reaction_key] = build_reaction_adapter(self.reaction_cfgs[reaction_key])

        self.conversation_aggregator_cfg = conversation_aggregator_cfg.copy()
        self.conversation_aggregator_cfg["logger_cfg"] = logger_cfg
        if ExecutorRegistry.validate_class(self.conversation_aggregator_cfg["type"]):
            self.conversation_aggregator_cfg["thread_pool_executor"] = self.executor
        self.conversation_aggregator: ConversationAggregator = build_aggregator(
            self.conversation_aggregator_cfg,
        )

        self.tts_reaction_aggregator_cfg = tts_reaction_aggregator_cfg.copy()
        self.tts_reaction_aggregator_cfg["logger_cfg"] = logger_cfg
        if ExecutorRegistry.validate_class(self.tts_reaction_aggregator_cfg["type"]):
            self.tts_reaction_aggregator_cfg["thread_pool_executor"] = self.executor
        self.tts_reaction_aggregator: TTSReactionAggregator = build_aggregator(self.tts_reaction_aggregator_cfg)

        self.blendshapes_aggregator_cfg = blendshapes_aggregator_cfg.copy()
        self.blendshapes_aggregator_cfg["logger_cfg"] = logger_cfg
        if ExecutorRegistry.validate_class(self.blendshapes_aggregator_cfg["type"]):
            self.blendshapes_aggregator_cfg["thread_pool_executor"] = self.executor
        self.blendshapes_aggregator: BlendshapesAggregator = build_aggregator(self.blendshapes_aggregator_cfg)

        self.callback_aggregator_cfg = callback_aggregator_cfg.copy()
        self.callback_aggregator_cfg["logger_cfg"] = logger_cfg
        if ExecutorRegistry.validate_class(self.callback_aggregator_cfg["type"]):
            self.callback_aggregator_cfg["thread_pool_executor"] = self.executor
        self.callback_aggregator: CallbackAggregator = build_aggregator(self.callback_aggregator_cfg)

        self.db_config_cfg = db_config_cfg.copy()
        self.db_config_cfg["logger_cfg"] = logger_cfg
        if ExecutorRegistry.validate_class(self.db_config_cfg["type"]):
            self.db_config_cfg["thread_pool_executor"] = self.executor
        self.db_config_client = build_db_config_client(self.db_config_cfg)
        self.db_config_cache_sync_trigger = db_config_cache_sync_trigger

        self.running_dags: Dict[str, DirectedAcyclicGraph] = dict()

        with open(style_file, "r", encoding="utf-8") as f:
            self.style_dict = json.load(f)

    def __del__(self) -> None:
        """Destructor, cleanup thread pool executor."""
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    async def start_streamable_instances(self) -> None:
        """Start all streamable adapter instances.

        Initializes and starts all adapters that support streaming operations,
        including ASR, memory, conversation, TTS, classification, and reaction
        adapters, as well as all aggregators.
        """
        streamable_instances = [
            self.a2f_adapter,
            *self.asr_adapters.values(),
            *self.memory_adapters.values(),
            *self.conversation_adapters.values(),
            self.s2m_adapter,
            *self.tts_adapters.values(),
            *self.classfication_adapters.values(),
            *self.reaction_adapters.values(),
            self.conversation_aggregator,
            self.tts_reaction_aggregator,
            self.blendshapes_aggregator,
            self.callback_aggregator,
        ]
        for streamable_instance in streamable_instances:
            if hasattr(streamable_instance, "AVAILABLE_FOR_STREAM") and streamable_instance.AVAILABLE_FOR_STREAM:
                asyncio.create_task(streamable_instance.run())

    async def maintain_loop(self) -> None:
        """Main maintenance loop for managing running DAGs.

        Continuously monitors the status of running DAGs, removes completed or
        failed DAGs, and handles timeout scenarios. Also starts all streamable
        instances before entering the monitoring loop.
        """
        await self.start_streamable_instances()
        while True:
            cur_time = time.time()
            keys = list(self.running_dags.keys())
            for dag_idx in keys:
                dag = self.running_dags[dag_idx]
                if dag.status != DAGStatus.RUNNING:
                    self.running_dags.pop(dag_idx)
                else:
                    dag_start_time = dag.conf["start_time"]
                    time_diff = cur_time - dag_start_time
                    if time_diff > self.process_timeout:
                        msg = f"DAG {dag.name} has been running for {time_diff} seconds, timeout."
                        self.logger.warning(msg)
                        dag.set_status(DAGStatus.FAILED)
                        self.running_dags.pop(dag_idx)
            await asyncio.sleep(self.sleep_time)

    async def direct_generation_v4(
        self,
        request: Dict[str, Any],
        callback_instances: List[Any],
        callback_bytes_fn: Callable,
    ) -> Tuple[DirectedAcyclicGraph, str]:
        """Generate 3DAC animation from speech text with streaming output.

        Creates a DAG for direct text-to-animation generation without conversation
        processing. The animation is streamed back through the callback function.

        Args:
            request (Dict[str, Any]):
                User request containing speech text and configuration parameters.
            callback_instances (List[Any]):
                List of callback instances to prevent garbage collection.
            callback_bytes_fn (Callable):
                Callback function for streaming bytes output.

        Returns:
            Tuple[DirectedAcyclicGraph, str]:
                Tuple containing the created DAG and unique request ID.
        """
        request_id = str(uuid.uuid4())
        user_id = request["user_id"]
        conf = dict(
            start_time=0.0,
            # for tts
            chunk_n_char_lowerbound=request.get("chunk_n_char_lowerbound", 10),
            chunk_n_char_lowerbound_en=request.get("chunk_n_char_lowerbound_en", 25),
            # for s2m
            max_front_extension_duration=request["max_front_extension_duration"],
            max_rear_extension_duration=request["max_rear_extension_duration"],
            idle_long_extendable=False,
            # for callback aggregator
            # 3 types: audio, face, motion
            chunk_type_required=3,
            callback_instances=callback_instances,
            callback_bytes_fn=callback_bytes_fn,
            # for both reaction and s2m
            first_body_fast_response=request["first_body_fast_response"],
            # for asr, tts, conversation, reaction
            language=request["language"] if request["language"] else "zh",
            # for app
            app_name=request["app_name"],
        )
        character_id = request.get("character_id", None)
        if character_id is not None:
            db_config_start_time = time.time()
            if isinstance(self.db_config_client, DynamoDBRedisConfigClient):
                if (
                    self.db_config_cache_sync_trigger is not None
                    and request["speech_text"] == self.db_config_cache_sync_trigger
                ):
                    read_cache = False
                else:
                    read_cache = True
                kwargs = dict(user_id=user_id, character_id=character_id, read_cache=read_cache)
            else:
                kwargs = dict(
                    user_id=user_id,
                    character_id=character_id,
                )
            character_settings, user_settings = await asyncio.gather(
                self.db_config_client.get_character_settings(**kwargs),
                self.db_config_client.get_user_settings(user_id),
            )
            db_config_end_time = time.time()
            self.logger.debug(
                "Database reading completed, "
                + f"config takes {db_config_end_time - db_config_start_time: .3f} seconds."
            )
            # user settings
            conf["user_settings"] = user_settings
            # for tts
            tts_adapter_key = character_settings["tts_adapter"]
            conf["voice_name"] = character_settings["voice"]
            conf["voice_speed"] = character_settings["voice_speed"]
            # for s2m
            conf["avatar"] = character_settings["avatar"]
        else:
            character_id = user_id
            self.logger.warning(
                f"character_id is not provided, use user_id {user_id} as character_id, and use settings from request."
            )
            # for tts
            tts_adapter_key = request["tts_adapter"]
            conf["voice_name"] = request["voice"]
            conf["voice_speed"] = request["voice_speed"]
            # for s2m
            conf["avatar"] = request["avatar"]
        # for s2m, conversation, reaction
        conf["character_id"] = character_id
        if request["speech_text"].startswith("[label_expression:"):
            label_expression = request["speech_text"].split("[label_expression:")[1].split("]")[0]
            if len(label_expression) > 0:
                conf["label_expression"] = label_expression
                request["speech_text"] = request["speech_text"].split("]", 1)[1]
        graph = DirectedAcyclicGraph(
            name="direct_generation_v4",
            conf=conf,
            logger_cfg=self.logger_cfg,
        )
        if tts_adapter_key not in self.tts_adapters:
            msg = f"TTS adapter {tts_adapter_key} not found, please choose among {list(self.tts_adapters.keys())}."
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        tts_adapter = self.tts_adapters[tts_adapter_key]
        tts_node = DAGNode(
            name="tts_node",
            payload=tts_adapter,
        )
        graph.add_node(tts_node)
        a2f_node = DAGNode(
            name="a2f_node",
            payload=self.a2f_adapter,
        )
        graph.add_node(a2f_node)
        s2m_node = DAGNode(
            name="s2m_node",
            payload=self.s2m_adapter,
        )
        graph.add_node(s2m_node)
        blendshapes_node = DAGNode(
            name="blendshapes_node",
            payload=self.blendshapes_aggregator,
        )
        graph.add_node(blendshapes_node)
        callback_node = DAGNode(
            name="callback_node",
            payload=self.callback_aggregator,
        )
        graph.add_node(callback_node)
        graph.add_edge(tts_node.name, a2f_node.name)
        graph.add_edge(tts_node.name, s2m_node.name)
        graph.add_edge(tts_node.name, callback_node.name)
        graph.add_edge(a2f_node.name, blendshapes_node.name)
        graph.add_edge(s2m_node.name, blendshapes_node.name)
        graph.add_edge(blendshapes_node.name, callback_node.name)
        # graph.check_cycle() has been confirmed
        start_time = time.time()
        graph.conf["start_time"] = start_time
        graph.set_status(DAGStatus.RUNNING)
        pb_response = orchestrator_pb2.OrchestratorV4Response()
        pb_response.class_name = "RequestIDResponse"
        pb_response.request_id = request_id
        pb_response_bytes = pb_response.SerializeToString()
        await callback_bytes_fn(pb_response_bytes)
        text_chunk_start = TextChunkStart(
            request_id=request_id,
            node_name=tts_node.name,
            dag=graph,
        )
        text_chunk_body = TextChunkBody(
            request_id=request_id,
            text_segment=request["speech_text"],
        )
        text_chunk_end = TextChunkEnd(
            request_id=request_id,
        )
        start_time = time.time()
        await tts_adapter.feed_stream(text_chunk_start)
        await tts_adapter.feed_stream(text_chunk_body)
        await tts_adapter.feed_stream(text_chunk_end)
        msg = f"DAG {graph.name}, request id {request_id}, character_id {character_id}, has been triggered."
        self.logger.info(msg)
        return graph, request_id

    async def start_audio_chat_with_text_llm_v4(
        self,
        request: Dict[str, Any],
        callback_instances: List[Any],
        callback_bytes_fn: Callable,
    ) -> Tuple[DirectedAcyclicGraph, str]:
        """Start streaming audio chat with text-based LLM.

        Initiates a conversation workflow where user uploads PCM audio data,
        which is processed through ASR, conversation, TTS, and animation
        generation, with streaming output.

        Args:
            request (Dict[str, Any]):
                User request containing audio parameters and configuration.
                Should be a dictionary from AudioChatCompleteStartRequestV4.
            callback_instances (List[Any]):
                List of callback instances to prevent garbage collection.
            callback_bytes_fn (Callable):
                Callback function for streaming bytes output.

        Returns:
            Tuple[DirectedAcyclicGraph, str]:
                Tuple containing the created DAG and unique request ID.
        """
        request_id = str(uuid.uuid4())
        user_id = request["user_id"]
        n_channels = request["n_channels"]
        sample_width = request["sample_width"]
        frame_rate = request["frame_rate"]
        conf = dict(
            start_time=0.0,
            # for tts
            chunk_n_char_lowerbound=request.get("chunk_n_char_lowerbound", 10),
            chunk_n_char_lowerbound_en=request.get("chunk_n_char_lowerbound_en", 25),
            # for s2m
            max_front_extension_duration=request["max_front_extension_duration"],
            max_rear_extension_duration=request["max_rear_extension_duration"],
            idle_long_extendable=False,
            # for callback aggregator
            # 4 types: audio, face, motion, classification
            chunk_type_required=4,
            callback_instances=callback_instances,
            callback_bytes_fn=callback_bytes_fn,
            # for both reaction and s2m
            first_body_fast_response=request["first_body_fast_response"],
            # for asr, tts, conversation, reaction
            language=request["language"] if request["language"] else "zh",
            # for app
            app_name=request["app_name"],
        )
        character_id = request.get("character_id", None)
        if character_id is not None:
            db_config_start_time = time.time()
            if isinstance(self.db_config_client, DynamoDBRedisConfigClient):
                kwargs = dict(user_id=user_id, character_id=character_id, read_cache=True)
            else:
                kwargs = dict(
                    user_id=user_id,
                    character_id=character_id,
                )
            character_settings, user_settings = await asyncio.gather(
                self.db_config_client.get_character_settings(**kwargs),
                self.db_config_client.get_user_settings(user_id),
            )
            db_config_end_time = time.time()
            profile_memory, cascade_memories, relationship, emotion = await asyncio.gather(
                self.db_memory_client.get_profile_memory(character_id),
                self.db_memory_client.get_cascade_memories(character_id),
                self.db_memory_client.get_relationship(character_id),
                self.db_memory_client.get_emotion(character_id),
            )
            if relationship is None:
                relationship = (INITIAL_RELATIONSHIP_STATE["stage"], INITIAL_RELATIONSHIP_STATE["value"])
            if emotion is None:
                emotion = INITIAL_EMOTION_STATE
            db_memory_end_time = time.time()
            self.logger.debug(
                "Database reading completed, "
                + f"config takes {db_config_end_time - db_config_start_time: .3f} seconds, "
                + f"memory takes {db_memory_end_time - db_config_end_time: .3f} seconds."
            )
            # user settings
            conf["user_settings"] = user_settings
            # for memory
            memory_adapter_key = character_settings["memory_adapter"]
            conf["memory_adapter"] = self.memory_adapters[memory_adapter_key]
            conf["memory_db_client"] = self.db_memory_client
            conf["memory_model_override"] = character_settings["memory_model_override"]
            # for asr
            asr_adapter_key = character_settings["asr_adapter"]
            # for tts
            tts_adapter_key = character_settings["tts_adapter"]
            conf["voice_name"] = character_settings["voice"]
            conf["voice_speed"] = character_settings["voice_speed"]
            # for classification
            classification_adapter_key = character_settings["classification_adapter"]
            conf["classification_model_override"] = character_settings["classification_model_override"]
            # for conversation
            conversation_adapter_key = character_settings["conversation_adapter"]
            conf["conversation_model_override"] = character_settings["conversation_model_override"]
            conf["user_prompt"] = character_settings["prompt"]
            conf["profile_memory"] = profile_memory
            conf["cascade_memories"] = cascade_memories
            conf["relationship"] = relationship
            # for reaction
            reaction_adapter_key = character_settings["reaction_adapter"]
            conf["reaction_model_override"] = character_settings["reaction_model_override"]
            conf["emotion"] = emotion
            conf["acquaintance_threshold"] = character_settings["acquaintance_threshold"]
            conf["friend_threshold"] = character_settings["friend_threshold"]
            conf["situationship_threshold"] = character_settings["situationship_threshold"]
            conf["lover_threshold"] = character_settings["lover_threshold"]
            conf["neutral_threshold"] = character_settings["neutral_threshold"]
            conf["happiness_threshold"] = character_settings["happiness_threshold"]
            conf["sadness_threshold"] = character_settings["sadness_threshold"]
            conf["fear_threshold"] = character_settings["fear_threshold"]
            conf["anger_threshold"] = character_settings["anger_threshold"]
            conf["disgust_threshold"] = character_settings["disgust_threshold"]
            conf["surprise_threshold"] = character_settings["surprise_threshold"]
            conf["shyness_threshold"] = character_settings["shyness_threshold"]
            # for s2m
            conf["avatar"] = character_settings["avatar"]
        else:
            character_id = user_id
            self.logger.warning(
                f"character_id is not provided, use user_id {user_id} as character_id, and use settings from request."
            )
            # for memory
            memory_adapter_key = request["memory_adapter"]
            conf["memory_adapter"] = self.memory_adapters[memory_adapter_key]
            conf["memory_db_client"] = self.db_memory_client
            # for asr
            asr_adapter_key = request["asr_adapter"]
            # for tts
            tts_adapter_key = request["tts_adapter"]
            conf["voice_name"] = request["voice"]
            conf["voice_speed"] = request["voice_speed"]
            # for classification
            classification_adapter_key = request["classification_adapter"]
            # for conversation
            conversation_adapter_key = request["conversation_adapter"]
            # for reaction
            reaction_adapter_key = request["reaction_adapter"]
            # for s2m
            conf["avatar"] = request["avatar"]
        conf["character_id"] = character_id
        conf["style_list"] = self.style_dict.get(conf["voice_name"], self.style_dict["others"])
        graph = DirectedAcyclicGraph(
            name="audio_chat_with_text_llm_v4",
            conf=conf,
            logger_cfg=self.logger_cfg,
        )
        if asr_adapter_key not in self.asr_adapters:
            msg = f"ASR adapter {asr_adapter_key} not found, please choose among {list(self.asr_adapters.keys())}."
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        asr_adapter = self.asr_adapters[asr_adapter_key]
        asr_node = DAGNode(
            name="asr_node",
            payload=asr_adapter,
        )
        graph.add_node(asr_node)
        if classification_adapter_key not in self.classfication_adapters:
            msg = (
                f"Classification adapter {classification_adapter_key} not found,"
                + f"please choose among {list(self.classfication_adapters.keys())}."
            )
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        classification_adapter = self.classfication_adapters[classification_adapter_key]
        classification_node = DAGNode(
            name="classification_node",
            payload=classification_adapter,
        )
        graph.add_node(classification_node)
        if conversation_adapter_key not in self.conversation_adapters:
            msg = (
                f"Conversation adapter {conversation_adapter_key} not found,"
                + f"please choose among {list(self.conversation_adapters.keys())}."
            )
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        conversation_adapter = self.conversation_adapters[conversation_adapter_key]
        conversation_node = DAGNode(
            name="conversation_node",
            payload=conversation_adapter,
        )
        graph.add_node(conversation_node)
        # use conversation adapter as reject adapter
        reject_adapter_key = conversation_adapter_key
        if reject_adapter_key not in self.conversation_adapters:
            msg = (
                f"Reject adapter {reject_adapter_key} not found,"
                + f"please choose among {list(self.conversation_adapters.keys())}."
            )
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        reject_adapter = self.conversation_adapters[reject_adapter_key]
        reject_node = DAGNode(
            name="reject_node",
            payload=reject_adapter,
        )
        graph.add_node(reject_node)
        if reaction_adapter_key not in self.reaction_adapters:
            msg = (
                f"Reaction adapter {reaction_adapter_key} not found,"
                + f"please choose among {list(self.reaction_adapters.keys())}."
            )
            self.logger.error(msg)
            raise AdapterNotFoundError(msg)
        reaction_adapter = self.reaction_adapters[reaction_adapter_key]
        reaction_node = DAGNode(
            name="reaction_node",
            payload=reaction_adapter,
        )
        graph.add_node(reaction_node)
        if tts_adapter_key not in self.tts_adapters:
            msg = f"TTS adapter {tts_adapter_key} not found, please choose among {list(self.tts_adapters.keys())}."
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        tts_adapter = self.tts_adapters[tts_adapter_key]
        tts_node = DAGNode(
            name="tts_node",
            payload=tts_adapter,
        )
        graph.add_node(tts_node)
        a2f_node = DAGNode(
            name="a2f_node",
            payload=self.a2f_adapter,
        )
        graph.add_node(a2f_node)
        s2m_node = DAGNode(
            name="s2m_node",
            payload=self.s2m_adapter,
        )
        graph.add_node(s2m_node)
        conversation_aggregator_node = DAGNode(
            name="conversation_aggregator_node",
            payload=self.conversation_aggregator,
        )
        graph.add_node(conversation_aggregator_node)
        tts_reaction_aggregator_node = DAGNode(
            name="tts_reaction_aggregator_node",
            payload=self.tts_reaction_aggregator,
        )
        graph.add_node(tts_reaction_aggregator_node)
        blendshapes_node = DAGNode(
            name="blendshapes_node",
            payload=self.blendshapes_aggregator,
        )
        graph.add_node(blendshapes_node)
        callback_node = DAGNode(
            name="callback_node",
            payload=self.callback_aggregator,
        )
        graph.add_node(callback_node)
        graph.add_edge(asr_node.name, classification_node.name)
        graph.add_edge(asr_node.name, conversation_node.name)
        graph.add_edge(classification_node.name, reject_node.name)
        graph.add_edge(reject_node.name, conversation_aggregator_node.name)
        graph.add_edge(conversation_node.name, conversation_aggregator_node.name)
        graph.add_edge(classification_node.name, conversation_aggregator_node.name)
        graph.add_edge(conversation_aggregator_node.name, tts_node.name)
        graph.add_edge(conversation_aggregator_node.name, reaction_node.name)
        graph.add_edge(tts_node.name, tts_reaction_aggregator_node.name)
        graph.add_edge(reaction_node.name, tts_reaction_aggregator_node.name)
        graph.add_edge(tts_reaction_aggregator_node.name, a2f_node.name)
        graph.add_edge(tts_reaction_aggregator_node.name, s2m_node.name)
        graph.add_edge(classification_node.name, callback_node.name)
        graph.add_edge(tts_node.name, callback_node.name)
        graph.add_edge(a2f_node.name, blendshapes_node.name)
        graph.add_edge(s2m_node.name, blendshapes_node.name)
        graph.add_edge(blendshapes_node.name, callback_node.name)
        # graph.check_cycle() has been confirmed
        start_time = time.time()
        graph.conf["start_time"] = start_time
        graph.set_status(DAGStatus.RUNNING)
        pb_response = orchestrator_pb2.OrchestratorV4Response()
        pb_response.class_name = "RequestIDResponse"
        pb_response.request_id = request_id
        pb_response_bytes = pb_response.SerializeToString()
        await callback_bytes_fn(pb_response_bytes)
        msg = f"DAG {graph.name}, request id {request_id}, character_id {character_id}, has been triggered."
        self.logger.info(msg)
        self.running_dags[request_id] = graph
        audio_chunk_start = AudioChunkStart(
            request_id=request_id,
            audio_type="pcm",
            node_name=asr_node.name,
            dag=graph,
            n_channels=n_channels,
            sample_width=sample_width,
            frame_rate=frame_rate,
        )
        await asr_node.payload.feed_stream(audio_chunk_start)
        return graph, request_id

    async def feed_audio_chat_with_text_llm_v4(
        self,
        request_id: str,
        pcm_bytes: bytes,
        seq_number: int,
    ) -> None:
        """Feed PCM audio data to the running audio chat DAG.

        Args:
            request_id (str):
                Unique identifier for the running DAG.
            pcm_bytes (bytes):
                PCM audio data bytes to process.
            seq_number (int):
                Sequence number for ordering audio chunks.
        """
        if request_id not in self.running_dags:
            msg = f"Request id {request_id} not found, please check the request id."
            self.logger.error(msg)
            raise RunningDAGNotFoundError(msg)
        graph = self.running_dags[request_id]
        asr_node = graph.get_node("asr_node")
        pcm_io = io.BytesIO(pcm_bytes)
        pcm_io.seek(0)
        audio_chunk_body = AudioChunkBody(
            request_id=request_id,
            # ASR does not care about duration attribute, get it directly from pcm bytes
            duration=0.0,
            audio_io=pcm_io,
            seq_number=seq_number,
        )
        await asr_node.payload.feed_stream(audio_chunk_body)

    async def stop_audio_chat_with_text_llm_v4(
        self,
        request_id: str,
    ) -> None:
        """Stop the running audio chat with text LLM DAG.

        Args:
            request_id (str):
                Unique identifier for the DAG to stop.
        """
        if request_id not in self.running_dags:
            msg = f"Request id {request_id} not found, please check the request id."
            self.logger.error(msg)
            raise RunningDAGNotFoundError(msg)
        graph = self.running_dags[request_id]
        asr_node = graph.get_node("asr_node")
        await asr_node.payload.feed_stream(AudioChunkEnd(request_id=request_id))

    async def start_audio_chat_with_audio_llm_v4(
        self,
        request: Dict[str, Any],
        callback_instances: List[Any],
        callback_bytes_fn: Callable,
    ) -> Tuple[DirectedAcyclicGraph, str]:
        """Start streaming audio chat with audio-based LLM.

        Initiates a conversation workflow where user uploads PCM audio data,
        which is processed through an audio conversation adapter that handles
        both audio input and output, with streaming animation and audio response.

        Args:
            request (Dict[str, Any]):
                User request containing audio parameters and configuration.
                Should include keys like audio_conversation_adapter, n_channels,
                sample_width, frame_rate, voice_name, avatar, etc.
            callback_instances (List[Any]):
                List of callback instances to prevent garbage collection.
            callback_bytes_fn (Callable):
                Callback function for streaming bytes output.

        Returns:
            Tuple[DirectedAcyclicGraph, str]:
                Tuple containing the created DAG and unique request ID.
        """
        request_id = str(uuid.uuid4())
        user_id = request["user_id"]
        n_channels = request["n_channels"]
        sample_width = request["sample_width"]
        frame_rate = request["frame_rate"]
        conf = dict(
            start_time=0.0,
            # for s2m
            max_front_extension_duration=request["max_front_extension_duration"],
            max_rear_extension_duration=request["max_rear_extension_duration"],
            idle_long_extendable=True,
            # for callback aggregator
            # 3 types: audio (from conversation), face, motion
            chunk_type_required=3,
            callback_instances=callback_instances,
            callback_bytes_fn=callback_bytes_fn,
            # for conversation
            language=request["language"] if request["language"] else "zh",
            # for app
            app_name=request["app_name"],
        )
        character_id = request.get("character_id", None)
        if character_id is not None:
            db_config_start_time = time.time()
            if isinstance(self.db_config_client, DynamoDBRedisConfigClient):
                kwargs = dict(user_id=user_id, character_id=character_id, read_cache=True)
            else:
                kwargs = dict(
                    user_id=user_id,
                    character_id=character_id,
                )
            character_settings, user_settings = await asyncio.gather(
                self.db_config_client.get_character_settings(**kwargs),
                self.db_config_client.get_user_settings(user_id),
            )
            db_config_end_time = time.time()
            profile_memory, cascade_memories, relationship, emotion = await asyncio.gather(
                self.db_memory_client.get_profile_memory(character_id),
                self.db_memory_client.get_cascade_memories(character_id),
                self.db_memory_client.get_relationship(character_id),
                self.db_memory_client.get_emotion(character_id),
            )
            if relationship is None:
                relationship = (INITIAL_RELATIONSHIP_STATE["stage"], INITIAL_RELATIONSHIP_STATE["value"])
            if emotion is None:
                emotion = INITIAL_EMOTION_STATE
            db_memory_end_time = time.time()
            self.logger.debug(
                "Database reading completed, "
                + f"config takes {db_config_end_time - db_config_start_time: .3f} seconds, "
                + f"memory takes {db_memory_end_time - db_config_end_time: .3f} seconds."
            )
            # user settings
            conf["user_settings"] = user_settings
            # for memory
            memory_adapter_key = character_settings["memory_adapter"]
            conf["memory_adapter"] = self.memory_adapters[memory_adapter_key]
            conf["memory_db_client"] = self.db_memory_client
            conf["memory_model_override"] = character_settings["memory_model_override"]
            # for conversation
            conf["conversation_voice_name"] = character_settings["voice"]
            audio_conversation_adapter_key = character_settings["conversation_adapter"]
            conf["conversation_model_override"] = character_settings["conversation_model_override"]
            conf["user_prompt"] = character_settings["prompt"]
            conf["profile_memory"] = profile_memory
            conf["cascade_memories"] = cascade_memories
            conf["relationship"] = relationship
            conf["emotion"] = emotion
            # for s2m
            conf["avatar"] = character_settings["avatar"]
        else:
            character_id = user_id
            self.logger.warning(
                f"character_id is not provided, use user_id {user_id} as character_id, and use settings from request."
            )
            # for memory
            memory_adapter_key = request["memory_adapter"]
            conf["memory_adapter"] = self.memory_adapters[memory_adapter_key]
            conf["memory_db_client"] = self.db_memory_client
            # for conversation
            conf["conversation_voice_name"] = request["voice"]
            audio_conversation_adapter_key = request["conversation_adapter"]
            # for s2m
            conf["avatar"] = request["avatar"]
        conf["character_id"] = character_id
        graph = DirectedAcyclicGraph(
            name="audio_chat_with_audio_llm_v4",
            conf=conf,
            logger_cfg=self.logger_cfg,
        )
        if audio_conversation_adapter_key not in self.conversation_adapters:
            msg = (
                f"Conversation adapter {audio_conversation_adapter_key} not found,"
                + f"please choose among {list(self.conversation_adapters.keys())}."
            )
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        audio_conversation_adapter = self.conversation_adapters[audio_conversation_adapter_key]
        audio_conversation_node = DAGNode(
            name="audio_conversation_node",
            payload=audio_conversation_adapter,
        )
        graph.add_node(audio_conversation_node)
        a2f_node = DAGNode(
            name="a2f_node",
            payload=self.a2f_adapter,
        )
        graph.add_node(a2f_node)
        s2m_node = DAGNode(
            name="s2m_node",
            payload=self.s2m_adapter,
        )
        graph.add_node(s2m_node)
        blendshapes_node = DAGNode(
            name="blendshapes_node",
            payload=self.blendshapes_aggregator,
        )
        graph.add_node(blendshapes_node)
        callback_node = DAGNode(
            name="callback_node",
            payload=self.callback_aggregator,
        )
        graph.add_node(callback_node)
        graph.add_edge(audio_conversation_node.name, a2f_node.name)
        graph.add_edge(audio_conversation_node.name, s2m_node.name)
        graph.add_edge(audio_conversation_node.name, callback_node.name)
        graph.add_edge(a2f_node.name, blendshapes_node.name)
        graph.add_edge(s2m_node.name, blendshapes_node.name)
        graph.add_edge(blendshapes_node.name, callback_node.name)
        # graph.check_cycle() has been confirmed
        start_time = time.time()
        graph.conf["start_time"] = start_time
        graph.set_status(DAGStatus.RUNNING)
        pb_response = orchestrator_pb2.OrchestratorV4Response()
        pb_response.class_name = "RequestIDResponse"
        pb_response.request_id = request_id
        pb_response_bytes = pb_response.SerializeToString()
        await callback_bytes_fn(pb_response_bytes)
        msg = f"DAG {graph.name}, request id {request_id}, has been triggered."
        self.logger.info(msg)
        self.running_dags[request_id] = graph
        audio_chunk_start = AudioChunkStart(
            request_id=request_id,
            audio_type="pcm",
            node_name=audio_conversation_node.name,
            dag=graph,
            n_channels=n_channels,
            sample_width=sample_width,
            frame_rate=frame_rate,
        )
        await audio_conversation_node.payload.feed_stream(audio_chunk_start)
        return graph, request_id

    async def feed_audio_chat_with_audio_llm_v4(
        self,
        request_id: str,
        pcm_bytes: bytes,
        seq_number: int,
    ) -> None:
        """Feed PCM audio data to the running audio chat with audio LLM DAG.

        Args:
            request_id (str):
                Unique identifier for the running DAG.
            pcm_bytes (bytes):
                PCM audio data bytes to process.
            seq_number (int):
                Sequence number for ordering audio chunks.
        """
        if request_id not in self.running_dags:
            msg = f"Request id {request_id} not found, please check the request id."
            self.logger.error(msg)
            raise RunningDAGNotFoundError(msg)
        graph = self.running_dags[request_id]
        audio_conversation_node = graph.get_node("audio_conversation_node")
        pcm_io = io.BytesIO(pcm_bytes)
        pcm_io.seek(0)
        audio_chunk_body = AudioChunkBody(
            request_id=request_id,
            # Audio chat does not care about duration attribute, get it directly from pcm bytes
            duration=0.0,
            audio_io=pcm_io,
            seq_number=seq_number,
        )
        await audio_conversation_node.payload.feed_stream(audio_chunk_body)

    async def stop_audio_chat_with_audio_llm_v4(
        self,
        request_id: str,
    ) -> None:
        """Stop the running audio chat with audio LLM DAG.

        Args:
            request_id (str):
                Unique identifier for the DAG to stop.
        """
        if request_id not in self.running_dags:
            msg = f"Request id {request_id} not found, please check the request id."
            self.logger.error(msg)
            raise RunningDAGNotFoundError(msg)
        graph = self.running_dags[request_id]
        audio_conversation_node = graph.get_node("audio_conversation_node")
        await audio_conversation_node.payload.feed_stream(AudioChunkEnd(request_id=request_id))

    async def text_chat_with_text_llm_v4(
        self,
        request: Dict[str, Any],
        callback_instances: List[Any],
        callback_bytes_fn: Callable,
    ) -> Tuple[DirectedAcyclicGraph, str]:
        """Start streaming text chat with text-based LLM.

        Initiates a conversation workflow where user provides speech text,
        which is processed through classification, conversation, TTS, and
        animation generation, with streaming output.

        Args:
            request (Dict[str, Any]):
                User request containing speech text and configuration.
                Should be a dictionary from AudioChatCompleteStartRequestV4.
            callback_instances (List[Any]):
                List of callback instances to prevent garbage collection.
            callback_bytes_fn (Callable):
                Callback function for streaming bytes output.

        Returns:
            Tuple[DirectedAcyclicGraph, str]:
                Tuple containing the created DAG and unique request ID.
        """
        request_id = str(uuid.uuid4())
        user_id = request["user_id"]
        conf = dict(
            start_time=0.0,
            # for tts
            chunk_n_char_lowerbound=request.get("chunk_n_char_lowerbound", 10),
            chunk_n_char_lowerbound_en=request.get("chunk_n_char_lowerbound_en", 25),
            # for s2m
            max_front_extension_duration=request["max_front_extension_duration"],
            max_rear_extension_duration=request["max_rear_extension_duration"],
            idle_long_extendable=False,
            # for callback aggregator
            # 4 types: audio, face, motion, classification
            chunk_type_required=4,
            callback_instances=callback_instances,
            callback_bytes_fn=callback_bytes_fn,
            # for both reaction and s2m
            first_body_fast_response=request["first_body_fast_response"],
            # for asr, tts, conversation, reaction
            language=request["language"] if request["language"] else "zh",
            # for app
            app_name=request["app_name"],
        )
        character_id = request.get("character_id", None)
        if character_id is not None:
            db_config_start_time = time.time()
            if isinstance(self.db_config_client, DynamoDBRedisConfigClient):
                if (
                    self.db_config_cache_sync_trigger is not None
                    and request["speech_text"] == self.db_config_cache_sync_trigger
                ):
                    read_cache = False
                else:
                    read_cache = True
                kwargs = dict(user_id=user_id, character_id=character_id, read_cache=read_cache)
            else:
                kwargs = dict(
                    user_id=user_id,
                    character_id=character_id,
                )
            character_settings, user_settings = await asyncio.gather(
                self.db_config_client.get_character_settings(**kwargs),
                self.db_config_client.get_user_settings(user_id),
            )
            db_config_end_time = time.time()
            profile_memory, cascade_memories, relationship, emotion = await asyncio.gather(
                self.db_memory_client.get_profile_memory(character_id),
                self.db_memory_client.get_cascade_memories(character_id),
                self.db_memory_client.get_relationship(character_id),
                self.db_memory_client.get_emotion(character_id),
            )
            if relationship is None:
                relationship = (
                    INITIAL_RELATIONSHIP_STATE["stage"],
                    INITIAL_RELATIONSHIP_STATE["value"],
                )
            if emotion is None:
                emotion = INITIAL_EMOTION_STATE
            db_memory_end_time = time.time()
            self.logger.debug(
                "Database reading completed, "
                + f"config takes {db_config_end_time - db_config_start_time: .3f} seconds, "
                + f"memory takes {db_memory_end_time - db_config_end_time: .3f} seconds."
            )
            # user settings
            conf["user_settings"] = user_settings
            # for memory
            memory_adapter_key = character_settings["memory_adapter"]
            conf["memory_adapter"] = self.memory_adapters[memory_adapter_key]
            conf["memory_db_client"] = self.db_memory_client
            conf["memory_model_override"] = character_settings["memory_model_override"]
            # for tts
            tts_adapter_key = character_settings["tts_adapter"]
            conf["voice_name"] = character_settings["voice"]
            conf["voice_speed"] = character_settings["voice_speed"]
            # for classification
            classification_adapter_key = character_settings["classification_adapter"]
            conf["classification_model_override"] = character_settings["classification_model_override"]
            # for conversation
            conversation_adapter_key = character_settings["conversation_adapter"]
            conf["conversation_model_override"] = character_settings["conversation_model_override"]
            conf["user_prompt"] = character_settings["prompt"]
            conf["profile_memory"] = profile_memory
            conf["cascade_memories"] = cascade_memories
            conf["relationship"] = relationship
            # for reaction
            reaction_adapter_key = character_settings["reaction_adapter"]
            conf["reaction_model_override"] = character_settings["reaction_model_override"]
            conf["emotion"] = emotion
            conf["acquaintance_threshold"] = character_settings["acquaintance_threshold"]
            conf["friend_threshold"] = character_settings["friend_threshold"]
            conf["situationship_threshold"] = character_settings["situationship_threshold"]
            conf["lover_threshold"] = character_settings["lover_threshold"]
            conf["neutral_threshold"] = character_settings["neutral_threshold"]
            conf["happiness_threshold"] = character_settings["happiness_threshold"]
            conf["sadness_threshold"] = character_settings["sadness_threshold"]
            conf["fear_threshold"] = character_settings["fear_threshold"]
            conf["anger_threshold"] = character_settings["anger_threshold"]
            conf["disgust_threshold"] = character_settings["disgust_threshold"]
            conf["surprise_threshold"] = character_settings["surprise_threshold"]
            conf["shyness_threshold"] = character_settings["shyness_threshold"]
            # for s2m
            conf["avatar"] = character_settings["avatar"]
        else:
            character_id = user_id
            self.logger.warning(
                f"character_id is not provided, use user_id {user_id} as character_id, and use settings from request."
            )
            # for memory
            memory_adapter_key = request["memory_adapter"]
            conf["memory_adapter"] = self.memory_adapters[memory_adapter_key]
            conf["memory_db_client"] = self.db_memory_client
            # for tts
            tts_adapter_key = request["tts_adapter"]
            conf["voice_name"] = request["voice"]
            conf["voice_speed"] = request["voice_speed"]
            # for classification
            classification_adapter_key = request["classification_adapter"]
            # for conversation
            conversation_adapter_key = request["conversation_adapter"]
            # for reaction
            reaction_adapter_key = request["reaction_adapter"]
            # for s2m
            conf["avatar"] = request["avatar"]
        conf["character_id"] = character_id
        conf["style_list"] = self.style_dict.get(conf["voice_name"], self.style_dict["others"])
        graph = DirectedAcyclicGraph(
            name="text_chat_with_text_llm_v4",
            conf=conf,
            logger_cfg=self.logger_cfg,
        )
        if classification_adapter_key not in self.classfication_adapters:
            msg = (
                f"Classification adapter {classification_adapter_key} not found,"
                + f"please choose among {list(self.classfication_adapters.keys())}."
            )
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        classification_adapter = self.classfication_adapters[classification_adapter_key]
        classification_node = DAGNode(
            name="classification_node",
            payload=classification_adapter,
        )
        graph.add_node(classification_node)
        if conversation_adapter_key not in self.conversation_adapters:
            msg = (
                f"Conversation adapter {conversation_adapter_key} not found,"
                + f"please choose among {list(self.conversation_adapters.keys())}."
            )
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        conversation_adapter = self.conversation_adapters[conversation_adapter_key]
        conversation_node = DAGNode(
            name="conversation_node",
            payload=conversation_adapter,
        )
        graph.add_node(conversation_node)
        reject_adapter_key = conversation_adapter_key
        if reject_adapter_key not in self.conversation_adapters:
            msg = (
                f"Reject adapter {reject_adapter_key} not found,"
                + f"please choose among {list(self.conversation_adapters.keys())}."
            )
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        reject_adapter = self.conversation_adapters[reject_adapter_key]
        reject_node = DAGNode(
            name="reject_node",
            payload=reject_adapter,
        )
        graph.add_node(reject_node)
        if reaction_adapter_key not in self.reaction_adapters:
            msg = (
                f"Reaction adapter {reaction_adapter_key} not found,"
                + f"please choose among {list(self.reaction_adapters.keys())}."
            )
            self.logger.error(msg)
            raise AdapterNotFoundError(msg)
        reaction_adapter = self.reaction_adapters[reaction_adapter_key]
        reaction_node = DAGNode(
            name="reaction_node",
            payload=reaction_adapter,
        )
        graph.add_node(reaction_node)
        if tts_adapter_key not in self.tts_adapters:
            msg = f"TTS adapter {tts_adapter_key} not found, please choose among {list(self.tts_adapters.keys())}."
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        tts_adapter = self.tts_adapters[tts_adapter_key]
        tts_node = DAGNode(
            name="tts_node",
            payload=tts_adapter,
        )
        graph.add_node(tts_node)
        a2f_node = DAGNode(
            name="a2f_node",
            payload=self.a2f_adapter,
        )
        graph.add_node(a2f_node)
        s2m_node = DAGNode(
            name="s2m_node",
            payload=self.s2m_adapter,
        )
        graph.add_node(s2m_node)
        conversation_aggregator_node = DAGNode(
            name="conversation_aggregator_node",
            payload=self.conversation_aggregator,
        )
        graph.add_node(conversation_aggregator_node)
        tts_reaction_aggregator_node = DAGNode(
            name="tts_reaction_aggregator_node",
            payload=self.tts_reaction_aggregator,
        )
        graph.add_node(tts_reaction_aggregator_node)
        blendshapes_node = DAGNode(
            name="blendshapes_node",
            payload=self.blendshapes_aggregator,
        )
        graph.add_node(blendshapes_node)
        callback_node = DAGNode(
            name="callback_node",
            payload=self.callback_aggregator,
        )
        graph.add_node(callback_node)
        graph.add_edge(classification_node.name, reject_node.name)
        graph.add_edge(reject_node.name, conversation_aggregator_node.name)
        graph.add_edge(conversation_node.name, conversation_aggregator_node.name)
        graph.add_edge(classification_node.name, conversation_aggregator_node.name)
        graph.add_edge(conversation_aggregator_node.name, tts_node.name)
        graph.add_edge(conversation_aggregator_node.name, reaction_node.name)
        graph.add_edge(tts_node.name, tts_reaction_aggregator_node.name)
        graph.add_edge(reaction_node.name, tts_reaction_aggregator_node.name)
        graph.add_edge(tts_reaction_aggregator_node.name, a2f_node.name)
        graph.add_edge(tts_reaction_aggregator_node.name, s2m_node.name)
        graph.add_edge(classification_node.name, callback_node.name)
        graph.add_edge(tts_node.name, callback_node.name)
        graph.add_edge(a2f_node.name, blendshapes_node.name)
        graph.add_edge(s2m_node.name, blendshapes_node.name)
        graph.add_edge(blendshapes_node.name, callback_node.name)
        # graph.check_cycle() has been confirmed
        start_time = time.time()
        graph.conf["start_time"] = start_time
        graph.set_status(DAGStatus.RUNNING)
        pb_response = orchestrator_pb2.OrchestratorV4Response()
        pb_response.class_name = "RequestIDResponse"
        pb_response.request_id = request_id
        pb_response_bytes = pb_response.SerializeToString()
        await callback_bytes_fn(pb_response_bytes)
        msg = f"DAG {graph.name}, request id {request_id}, character_id {character_id}, has been triggered."
        self.logger.info(msg)
        self.running_dags[request_id] = graph
        for node in (classification_node, conversation_node):
            text_chunk_start = TextChunkStart(
                request_id=request_id,
                node_name=node.name,
                dag=graph,
            )
            await node.payload.feed_stream(text_chunk_start)
            text_chunk_body = TextChunkBody(
                request_id=request_id,
                text_segment=request["speech_text"],
            )
            await node.payload.feed_stream(text_chunk_body)
            text_chunk_end = TextChunkEnd(
                request_id=request_id,
            )
            await node.payload.feed_stream(text_chunk_end)
        return graph, request_id

    async def text_chat_with_audio_llm_v4(
        self,
        request: Dict[str, Any],
        callback_instances: List[Any],
        callback_bytes_fn: Callable,
    ) -> Tuple[DirectedAcyclicGraph, str]:
        """Start streaming text chat with audio-based LLM.

        Initiates a conversation workflow where user provides speech text,
        which is processed through TTS, audio conversation adapter, and
        animation generation, with streaming output.

        Args:
            request (Dict[str, Any]):
                User request containing speech text and configuration.
                Should be a dictionary from AudioChatCompleteStartRequestV4.
            callback_instances (List[Any]):
                List of callback instances to prevent garbage collection.
            callback_bytes_fn (Callable):
                Callback function for streaming bytes output.

        Returns:
            Tuple[DirectedAcyclicGraph, str]:
                Tuple containing the created DAG and unique request ID.
        """
        request_id = str(uuid.uuid4())
        user_id = request["user_id"]
        conf = dict(
            start_time=0.0,
            # for tts
            chunk_n_char_lowerbound=request.get("chunk_n_char_lowerbound", 10),
            # for s2m
            max_front_extension_duration=request["max_front_extension_duration"],
            max_rear_extension_duration=request["max_rear_extension_duration"],
            idle_long_extendable=True,
            # for callback aggregator
            # 4 types: audio, face, motion
            chunk_type_required=3,
            callback_instances=callback_instances,
            callback_bytes_fn=callback_bytes_fn,
            # for s2m
            first_body_fast_response=request["first_body_fast_response"],
            # for tts, conversation
            language=request["language"] if request["language"] else "zh",
            # for app
            app_name=request["app_name"],
        )
        character_id = request.get("character_id", None)
        if character_id is not None:
            db_config_start_time = time.time()
            if isinstance(self.db_config_client, DynamoDBRedisConfigClient):
                if (
                    self.db_config_cache_sync_trigger is not None
                    and request["speech_text"] == self.db_config_cache_sync_trigger
                ):
                    read_cache = False
                else:
                    read_cache = True
                kwargs = dict(user_id=user_id, character_id=character_id, read_cache=read_cache)
            else:
                kwargs = dict(
                    user_id=user_id,
                    character_id=character_id,
                )
            character_settings, user_settings = await asyncio.gather(
                self.db_config_client.get_character_settings(**kwargs),
                self.db_config_client.get_user_settings(user_id),
            )
            db_config_end_time = time.time()
            profile_memory, cascade_memories, relationship, emotion = await asyncio.gather(
                self.db_memory_client.get_profile_memory(character_id),
                self.db_memory_client.get_cascade_memories(character_id),
                self.db_memory_client.get_relationship(character_id),
                self.db_memory_client.get_emotion(character_id),
            )
            if relationship is None:
                relationship = (
                    INITIAL_RELATIONSHIP_STATE["stage"],
                    INITIAL_RELATIONSHIP_STATE["value"],
                )
            if emotion is None:
                emotion = INITIAL_EMOTION_STATE
            db_memory_end_time = time.time()
            self.logger.debug(
                "Database reading completed, "
                + f"config takes {db_config_end_time - db_config_start_time: .3f} seconds, "
                + f"memory takes {db_memory_end_time - db_config_end_time: .3f} seconds."
            )
            # user settings
            conf["user_settings"] = user_settings
            # for memory
            memory_adapter_key = character_settings["memory_adapter"]
            conf["memory_adapter"] = self.memory_adapters[memory_adapter_key]
            conf["memory_db_client"] = self.db_memory_client
            conf["memory_model_override"] = character_settings["memory_model_override"]
            # for tts
            tts_adapter_key = character_settings["tts_adapter"]
            conf["conversation_voice_name"] = character_settings["voice"]
            # for conversation
            conversation_adapter_key = character_settings["conversation_adapter"]
            conf["conversation_model_override"] = character_settings["conversation_model_override"]
            conf["user_prompt"] = character_settings["prompt"]
            conf["profile_memory"] = profile_memory
            conf["cascade_memories"] = cascade_memories
            conf["relationship"] = relationship
            conf["emotion"] = emotion
            # for s2m
            conf["avatar"] = character_settings["avatar"]
        else:
            character_id = user_id
            self.logger.warning(
                f"character_id is not provided, use user_id {user_id} as character_id, and use settings from request."
            )
            # for memory
            memory_adapter_key = request["memory_adapter"]
            conf["memory_adapter"] = self.memory_adapters[memory_adapter_key]
            conf["memory_db_client"] = self.db_memory_client
            # for tts
            tts_adapter_key = request["tts_adapter"]
            conf["conversation_voice_name"] = request["voice"]
            # for conversation
            conversation_adapter_key = request["conversation_adapter"]
            # for s2m
            conf["avatar"] = request["avatar"]
        conf["character_id"] = character_id
        graph = DirectedAcyclicGraph(
            name="text_chat_with_audio_llm_v4",
            conf=conf,
            logger_cfg=self.logger_cfg,
        )
        if conversation_adapter_key not in self.conversation_adapters:
            msg = (
                f"Conversation adapter {conversation_adapter_key} not found,"
                + f"please choose among {list(self.conversation_adapters.keys())}."
            )
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        conversation_adapter = self.conversation_adapters[conversation_adapter_key]
        audio_conversation_node = DAGNode(
            name="audio_conversation_node",
            payload=conversation_adapter,
        )
        graph.add_node(audio_conversation_node)
        if tts_adapter_key not in self.tts_adapters:
            msg = f"TTS adapter {tts_adapter_key} not found, please choose among {list(self.tts_adapters.keys())}."
            self.logger.error(msg)
            pb_response = orchestrator_pb2.OrchestratorV4Response()
            pb_response.class_name = "FailedResponse"
            pb_response.message = msg
            pb_response_bytes = pb_response.SerializeToString()
            await callback_bytes_fn(pb_response_bytes)
            raise AdapterNotFoundError(msg)
        tts_adapter = self.tts_adapters[tts_adapter_key]
        tts_node = DAGNode(
            name="tts_node",
            payload=tts_adapter,
        )
        graph.add_node(tts_node)
        a2f_node = DAGNode(
            name="a2f_node",
            payload=self.a2f_adapter,
        )
        graph.add_node(a2f_node)
        s2m_node = DAGNode(
            name="s2m_node",
            payload=self.s2m_adapter,
        )
        graph.add_node(s2m_node)
        blendshapes_node = DAGNode(
            name="blendshapes_node",
            payload=self.blendshapes_aggregator,
        )
        graph.add_node(blendshapes_node)
        callback_node = DAGNode(
            name="callback_node",
            payload=self.callback_aggregator,
        )
        graph.add_node(callback_node)
        graph.add_edge(tts_node.name, audio_conversation_node.name)
        graph.add_edge(audio_conversation_node.name, a2f_node.name)
        graph.add_edge(audio_conversation_node.name, s2m_node.name)
        graph.add_edge(audio_conversation_node.name, callback_node.name)
        graph.add_edge(a2f_node.name, blendshapes_node.name)
        graph.add_edge(s2m_node.name, blendshapes_node.name)
        graph.add_edge(blendshapes_node.name, callback_node.name)
        # graph.check_cycle() has been confirmed
        start_time = time.time()
        graph.conf["start_time"] = start_time
        graph.set_status(DAGStatus.RUNNING)
        pb_response = orchestrator_pb2.OrchestratorV4Response()
        pb_response.class_name = "RequestIDResponse"
        pb_response.request_id = request_id
        pb_response_bytes = pb_response.SerializeToString()
        await callback_bytes_fn(pb_response_bytes)
        msg = f"DAG {graph.name}, request id {request_id}, character_id {character_id}, has been triggered."
        self.logger.info(msg)
        self.running_dags[request_id] = graph
        text_chunk_start = TextChunkStart(
            request_id=request_id,
            node_name=tts_node.name,
            dag=graph,
        )
        await tts_node.payload.feed_stream(text_chunk_start)
        text_chunk_body = TextChunkBody(
            request_id=request_id,
            text_segment=request["speech_text"],
        )
        await tts_node.payload.feed_stream(text_chunk_body)
        text_chunk_end = TextChunkEnd(
            request_id=request_id,
        )
        await tts_node.payload.feed_stream(text_chunk_end)
        return graph, request_id

    async def get_voice_settings(self, user_id: str, character_id: str) -> Dict[str, Any]:
        """Get voice settings for a specific character.

        Args:
            user_id (str):
                Unique identifier for the user.
            character_id (str):
                Unique identifier for the character.

        Returns:
            Dict[str, Any]:
                Dictionary containing voice configuration settings.
        """
        return await self.db_config_client.get_voice_settings(user_id, character_id)

    async def get_motion_settings(self, user_id: str, character_id: str) -> Dict[str, Any]:
        """Get motion settings for a specific character.

        Args:
            user_id (str):
                Unique identifier for the user.
            character_id (str):
                Unique identifier for the character.

        Returns:
            Dict[str, Any]:
                Dictionary containing motion configuration settings.
        """
        return await self.db_config_client.get_motion_settings(user_id, character_id)

    async def get_asr_adapter_choices(self) -> List[str]:
        """Get available ASR adapter choices.

        Returns:
            List[str]:
                List of available ASR adapter names.
        """
        return list(self.asr_adapters.keys())

    async def get_tts_adapter_choices(self) -> List[str]:
        """Get available TTS adapter choices.

        Returns:
            List[str]:
                List of available TTS adapter names.
        """
        return list(self.tts_adapters.keys())

    async def get_tts_voice_names(self, tts_adapter_key: str) -> Dict[str, Any]:
        """Get available voice names for a specific TTS adapter.

        Args:
            tts_adapter_key (str):
                Key identifying the TTS adapter.

        Returns:
            Dict[str, Any]:
                Dictionary containing available voice names and their properties.
        """
        if tts_adapter_key not in self.tts_adapters:
            msg = f"TTS adapter {tts_adapter_key} not found,please choose among {list(self.tts_adapters.keys())}."
            self.logger.error(msg)
            raise AdapterNotFoundError(msg)
        tts_adapter = self.tts_adapters[tts_adapter_key]
        return await tts_adapter.get_voice_names()

    async def get_conversation_adapter_choices(self) -> List[str]:
        """Get available conversation adapter choices.

        Returns:
            List[str]:
                List of available conversation adapter names.
        """
        return list(self.conversation_adapters.keys())

    async def get_reaction_adapter_choices(self) -> List[str]:
        """Get available reaction adapter choices.

        Returns:
            List[str]:
                List of available reaction adapter names.
        """
        return list(self.reaction_adapters.keys())

    async def get_classification_adapter_choices(self) -> List[str]:
        """Get available classification adapter choices.

        Returns:
            List[str]:
                List of available classification adapter names.
        """
        return list(self.classfication_adapters.keys())

    async def get_memory_adapter_choices(self) -> List[str]:
        """Get available memory adapter choices.

        Returns:
            List[str]:
                List of available memory adapter names.
        """
        return list(self.memory_adapters.keys())
