import asyncio
import json
import os
from asyncio import QueueFull
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..data_structures import orchestrator_v4_pb2 as orchestrator_pb2
from ..data_structures.process_flow import DAGStatus
from ..proxy import Proxy
from ..utils.executor_registry import ExecutorRegistry
from .base_fast_api_service import BaseFastAPIService
from .exceptions import OPENAPI_RESPONSE_404, OPENAPI_RESPONSE_500
from .requests import (
    AudioChatCompleteStartRequestV4,
    AudioChatExpressStartRequestV4,
    DirectGenerationRequest,
    TextChatCompleteRequestV4,
    TextChatExpressRequestV4,
)
from .responses import (
    AdapterChoicesResponse,
    EmotionResponse,
    MotionSettingsResponse,
    RelationshipResponse,
    VoiceNamesResponse,
    VoiceSettingsResponse,
)


class OrchestratorProxyServer(BaseFastAPIService):
    """Orchestrator proxy server for handling real-time streaming requests.

    This server provides WebSocket endpoints for various types of real-time
    interactions including audio chat with text/audio LLMs, text chat, and
    direct animation generation. It manages streaming data through protobuf
    messages and coordinates with the underlying orchestrator proxy.
    """

    ExecutorRegistry.register_class("OrchestratorProxyServer")

    def __init__(
        self,
        name: str,
        proxy: Proxy,
        enable_cors: bool = False,
        host: str = "0.0.0.0",
        port: int = 80,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        startup_event_listener: Union[None, list] = None,
        shutdown_event_listener: Union[None, list] = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ) -> None:
        """Initialize the orchestrator proxy server.

        Args:
            name (str):
                The name of the server for identification.
            proxy (Proxy):
                The orchestrator proxy instance for handling requests.
            enable_cors (bool, optional):
                Whether to enable Cross-Origin Resource Sharing (CORS).
                Defaults to False.
            host (str, optional):
                Host address to bind the server to.
                Defaults to "0.0.0.0".
            port (int, optional):
                Port number to bind the server to.
                Defaults to 80.
            max_workers (int, optional):
                Maximum number of worker threads for the thread pool executor.
                Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                External thread pool executor to use. If None, creates a new one.
                Defaults to None.
            startup_event_listener (Union[None, list], optional):
                List of event listeners to execute on server startup.
                Defaults to None.
            shutdown_event_listener (Union[None, list], optional):
                List of event listeners to execute on server shutdown.
                Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary.
                Defaults to None.
        """
        BaseFastAPIService.__init__(
            self,
            name=name,
            enable_cors=enable_cors,
            host=host,
            port=port,
            startup_event_listener=startup_event_listener,
            shutdown_event_listener=shutdown_event_listener,
            logger_cfg=logger_cfg,
        )
        self.proxy = proxy
        # for tailing the log file
        log_path = None
        for logger_handler in self.logger.handlers:
            if hasattr(logger_handler, "baseFilename"):
                log_path = logger_handler.baseFilename
                break
        self.log_path = log_path
        self.templates = Jinja2Templates(directory="templates")
        self.executor = (
            thread_pool_executor if thread_pool_executor is not None else ThreadPoolExecutor(max_workers=max_workers)
        )
        self.executor_external = True if thread_pool_executor is not None else False

    def __del__(self) -> None:
        """Destructor to cleanup thread pool executor.

        Shuts down the thread pool executor if it was created internally.
        External executors are not managed by this class.
        """
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    async def audio_chat_with_text_llm_v4(
        self,
        websocket: WebSocket,
    ) -> None:
        """Handle streaming audio chat with text-based LLM.

        This method is called when users engage in streaming PCM audio
        conversation with text-modal LLM. The client should first send
        an AudioChatCompleteStartRequestV4, then send PCM audio data
        multiple times, and finally send an AudioChatCompleteStopRequestV4.

        Args:
            websocket (WebSocket):
                WebSocket connection for real-time communication.

        Raises:
            WebSocketDisconnect:
                When the WebSocket connection is closed or invalid data is received.
        """
        loop = asyncio.get_running_loop()
        await websocket.accept()
        pb_bytes = await websocket.receive_bytes()
        pb_request = orchestrator_pb2.OrchestratorV4Request()
        await loop.run_in_executor(self.executor, pb_request.ParseFromString, pb_bytes)
        if pb_request.class_name != "AudioChatCompleteStartRequestV4":
            msg = f"Received non-AudioChatCompleteStartRequestV4 format data from websocket, data type: {pb_request.class_name}."
            self.logger.error(msg)
            raise WebSocketDisconnect(code=1000)
        if pb_request.app_name not in ("babylon", "python_backend"):
            app_name = "python_backend"
        else:
            app_name = pb_request.app_name
        request_instance = AudioChatCompleteStartRequestV4(
            n_channels=pb_request.n_channels,
            sample_width=pb_request.sample_width,
            frame_rate=pb_request.frame_rate,
            language=pb_request.language,
            first_body_fast_response=pb_request.first_body_fast_response,
            app_name=app_name,
            max_front_extension_duration=pb_request.max_front_extension_duration,
            max_rear_extension_duration=pb_request.max_rear_extension_duration,
            # with dynamodb
            user_id=pb_request.user_id if len(pb_request.user_id) > 0 else None,
            character_id=pb_request.character_id if len(pb_request.character_id) > 0 else None,
            # without dynamodb
            asr_adapter=pb_request.asr_adapter if len(pb_request.asr_adapter) > 0 else None,
            classification_adapter=pb_request.classification_adapter
            if len(pb_request.classification_adapter) > 0
            else None,
            conversation_adapter=pb_request.conversation_adapter if len(pb_request.conversation_adapter) > 0 else None,
            reaction_adapter=pb_request.reaction_adapter if len(pb_request.reaction_adapter) > 0 else None,
            tts_adapter=pb_request.tts_adapter if len(pb_request.tts_adapter) > 0 else None,
            voice_name=pb_request.voice_name if len(pb_request.voice_name) > 0 else None,
            voice_speed=pb_request.voice_speed if pb_request.voice_speed > 0 else None,
            avatar=pb_request.avatar if len(pb_request.avatar) > 0 else None,
        )
        request_dict = await loop.run_in_executor(self.executor, request_instance.model_dump)
        dag, request_id = await self.proxy.start_audio_chat_with_text_llm_v4(
            request_dict,
            callback_instances=[websocket],
            callback_bytes_fn=websocket.send_bytes,
        )
        client_stream_closed = False
        try:
            seq_number = 0
            self.logger.info(f"Listening for connection with request ID: {request_id}.")
            while True:
                data = await websocket.receive()
                if "bytes" in data:
                    pb_request = orchestrator_pb2.OrchestratorV4Request()
                    await loop.run_in_executor(self.executor, pb_request.ParseFromString, data["bytes"])
                    if pb_request.class_name == "AudioChunkBody":
                        pcm_bytes = pb_request.data
                        try:
                            await self.proxy.feed_audio_chat_with_text_llm_v4(
                                request_id=request_id,
                                pcm_bytes=pcm_bytes,
                                seq_number=seq_number,
                            )
                            seq_number += 1
                        except QueueFull:
                            msg = f"Audio queue is full for request_id {request_id}, please try again later."
                            self.logger.error(msg)
                            pb_response = orchestrator_pb2.OrchestratorV4Response()
                            pb_response.class_name = "FailedResponse"
                            pb_response.message = msg
                            pb_response_bytes = await loop.run_in_executor(self.executor, pb_response.SerializeToString)
                            await websocket.send_bytes(pb_response_bytes)
                    elif pb_request.class_name == "AudioChatCompleteStopRequestV4":
                        await self.proxy.stop_audio_chat_with_text_llm_v4(request_id)
                        client_stream_closed = True
                # websocket signal or text(json)
                else:
                    if data["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect
                    else:
                        msg = f"Received unknown non-bytes format data from websocket, data content: {data}, "
                        +f"request ID: {request_id}. Ending streaming audio conversation."
                        self.logger.warning(msg)
                        await self.proxy.stop_audio_chat_with_text_llm_v4(request_id)
        except WebSocketDisconnect:
            if not client_stream_closed:
                msg = f"Connection with request ID {request_id} was unexpectedly disconnected by user."
                self.logger.warning(msg)
                dag.set_status(DAGStatus.FAILED)
                client_stream_closed = True
            else:
                msg = f"Connection with request ID {request_id} was disconnected by user."
                self.logger.info(msg)
            return

    async def audio_chat_with_audio_llm_v4(
        self,
        websocket: WebSocket,
    ) -> None:
        """Handle streaming audio chat with audio-based LLM.

        This method is called when users engage in streaming PCM audio
        conversation with audio-native LLM. The client should first send
        an AudioChatExpressStartRequestV4, then send PCM audio data
        multiple times, and finally send an AudioChatExpressStopRequestV4.

        Args:
            websocket (WebSocket):
                WebSocket connection for real-time communication.

        Raises:
            WebSocketDisconnect:
                When the WebSocket connection is closed or invalid data is received.
        """
        loop = asyncio.get_running_loop()
        await websocket.accept()
        pb_bytes = await websocket.receive_bytes()
        pb_request = orchestrator_pb2.OrchestratorV4Request()
        await loop.run_in_executor(self.executor, pb_request.ParseFromString, pb_bytes)
        if pb_request.class_name != "AudioChatExpressStartRequestV4":
            msg = f"Received non-AudioChatExpressStartRequestV4 format data from websocket, data type: {pb_request.class_name}."
            self.logger.error(msg)
            raise WebSocketDisconnect(code=1000)
        if pb_request.app_name not in ("babylon", "python_backend"):
            app_name = "python_backend"
        else:
            app_name = pb_request.app_name
        request_instance = AudioChatExpressStartRequestV4(
            n_channels=pb_request.n_channels,
            sample_width=pb_request.sample_width,
            frame_rate=pb_request.frame_rate,
            language=pb_request.language,
            app_name=app_name,
            max_front_extension_duration=pb_request.max_front_extension_duration,
            max_rear_extension_duration=pb_request.max_rear_extension_duration,
            # with dynamodb
            user_id=pb_request.user_id if len(pb_request.user_id) > 0 else None,
            character_id=pb_request.character_id if len(pb_request.character_id) > 0 else None,
            # without dynamodb
            conversation_adapter=pb_request.conversation_adapter if len(pb_request.conversation_adapter) > 0 else None,
            voice_name=pb_request.voice_name if len(pb_request.voice_name) > 0 else None,
            avatar=pb_request.avatar if len(pb_request.avatar) > 0 else None,
        )
        request_dict = await loop.run_in_executor(self.executor, request_instance.model_dump)
        dag, request_id = await self.proxy.start_audio_chat_with_audio_llm_v4(
            request_dict,
            callback_instances=[websocket],
            callback_bytes_fn=websocket.send_bytes,
        )
        client_stream_closed = False
        try:
            seq_number = 0
            self.logger.info(f"Listening for connection with request ID: {request_id}.")
            while True:
                data = await websocket.receive()
                # PCM bytes data
                if "bytes" in data:
                    pb_request = orchestrator_pb2.OrchestratorV4Request()
                    await loop.run_in_executor(self.executor, pb_request.ParseFromString, data["bytes"])
                    if pb_request.class_name == "AudioChunkBody":
                        pcm_bytes = pb_request.data
                        await self.proxy.feed_audio_chat_with_audio_llm_v4(
                            request_id=request_id,
                            pcm_bytes=pcm_bytes,
                            seq_number=seq_number,
                        )
                        seq_number += 1
                    elif pb_request.class_name == "AudioChatExpressStopRequestV4":
                        await self.proxy.stop_audio_chat_with_audio_llm_v4(request_id)
                        client_stream_closed = True
                # websocket signal or text(json)
                else:
                    if data["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect
                    else:
                        msg = f"Received unknown non-bytes format data from websocket, data content: {data}, "
                        +f"request ID: {request_id}. Ending streaming audio conversation."
                        self.logger.warning(msg)
                        await self.proxy.stop_audio_chat_with_audio_llm_v4(request_id)
        except WebSocketDisconnect:
            if not client_stream_closed:
                msg = f"Connection with request ID {request_id} was unexpectedly disconnected by user."
                self.logger.warning(msg)
                dag.set_status(DAGStatus.FAILED)
                client_stream_closed = True
            else:
                msg = f"Connection with request ID {request_id} was disconnected by user."
                self.logger.info(msg)
            return

    async def text_generate_v4(
        self,
        websocket: WebSocket,
    ) -> None:
        """Handle direct animation generation from text.

        This method is called when users generate animations from dialogue text.
        The client should first send a DirectGenerationRequest, then
        continuously receive results without sending additional data.

        Args:
            websocket (WebSocket):
                WebSocket connection for real-time communication.

        Raises:
            WebSocketDisconnect:
                When the WebSocket connection is closed or invalid data is received.
        """
        loop = asyncio.get_running_loop()
        await websocket.accept()
        pb_bytes = await websocket.receive_bytes()
        pb_request = orchestrator_pb2.OrchestratorV4Request()
        await loop.run_in_executor(self.executor, pb_request.ParseFromString, pb_bytes)
        if pb_request.class_name != "DirectGenerationRequest":
            msg = (
                f"Received non-DirectGenerationRequest format data from websocket, data type: {pb_request.class_name}."
            )
            self.logger.error(msg)
            raise WebSocketDisconnect(code=1000)
        if pb_request.app_name not in ("babylon", "python_backend"):
            app_name = "python_backend"
        else:
            app_name = pb_request.app_name
        request_instance = DirectGenerationRequest(
            # for text2speech
            speech_text=pb_request.speech_text,
            language=pb_request.language,
            first_body_fast_response=pb_request.first_body_fast_response,
            app_name=app_name,
            max_front_extension_duration=pb_request.max_front_extension_duration,
            max_rear_extension_duration=pb_request.max_rear_extension_duration,
            # with dynamodb
            user_id=pb_request.user_id if len(pb_request.user_id) > 0 else None,
            character_id=pb_request.character_id if len(pb_request.character_id) > 0 else None,
            # without dynamodb
            tts_adapter=pb_request.tts_adapter if len(pb_request.tts_adapter) > 0 else None,
            voice_name=pb_request.voice_name if len(pb_request.voice_name) > 0 else None,
            voice_speed=pb_request.voice_speed if pb_request.voice_speed > 0 else None,
            avatar=pb_request.avatar if len(pb_request.avatar) > 0 else None,
        )
        request_dict = await loop.run_in_executor(self.executor, request_instance.model_dump)
        graph, request_id = await self.proxy.direct_generation_v4(
            request=request_dict,
            callback_instances=[websocket],
            callback_bytes_fn=websocket.send_bytes,
        )
        try:
            while True:
                data = await websocket.receive_text()
                type_str = ""
                code_str = ""
                try:
                    data_dict = await loop.run_in_executor(self.executor, json.loads, data)
                    type_str = data_dict.get("type", "")
                    code_str = data_dict.get("code", "")
                except json.JSONDecodeError:
                    pass
                if type_str == "websocket.disconnect" and code_str == "1000":
                    raise WebSocketDisconnect
                msg = (
                    "text_generate_v4 service did not expect to receive data from websocket after stream establishment, "
                    + f"data type: {type(data)}, data content: {data}, "
                    + f"request ID: {request_id}"
                )
                self.logger.warning(msg)
                graph.set_status(DAGStatus.FAILED)
        except WebSocketDisconnect:
            msg = f"Connection with request ID {request_id} was disconnected by user."
            if graph.status == DAGStatus.RUNNING:
                msg += " Due to connection being disconnected by user, setting running task to failed status."
                graph.set_status(DAGStatus.FAILED)
                self.logger.warning(msg)
            else:
                self.logger.info(msg)
            return

    async def text_chat_with_text_llm_v4(
        self,
        websocket: WebSocket,
    ) -> None:
        """Handle text chat with text-based LLM.

        This method is called when users engage in text conversation
        with text-modal LLM. The client should send a TextChatCompleteRequestV4
        at the beginning.

        Args:
            websocket (WebSocket):
                WebSocket connection for real-time communication.

        Raises:
            WebSocketDisconnect:
                When the WebSocket connection is closed or invalid data is received.
        """
        loop = asyncio.get_running_loop()
        await websocket.accept()
        pb_bytes = await websocket.receive_bytes()
        pb_request = orchestrator_pb2.OrchestratorV4Request()
        await loop.run_in_executor(self.executor, pb_request.ParseFromString, pb_bytes)
        if pb_request.class_name != "TextChatCompleteRequestV4":
            msg = f"Received non-TextChatCompleteStartRequestV4 format data from websocket, data type: {pb_request.class_name}."
            self.logger.error(msg)
            raise WebSocketDisconnect(code=1000)
        if pb_request.app_name not in ("babylon", "python_backend"):
            app_name = "python_backend"
        else:
            app_name = pb_request.app_name
        request_instance = TextChatCompleteRequestV4(
            speech_text=pb_request.speech_text,
            language=pb_request.language,
            first_body_fast_response=pb_request.first_body_fast_response,
            app_name=app_name,
            max_front_extension_duration=pb_request.max_front_extension_duration,
            max_rear_extension_duration=pb_request.max_rear_extension_duration,
            # with dynamodb
            user_id=pb_request.user_id if len(pb_request.user_id) > 0 else None,
            character_id=pb_request.character_id if len(pb_request.character_id) > 0 else None,
            # without dynamodb
            classification_adapter=pb_request.classification_adapter
            if len(pb_request.classification_adapter) > 0
            else None,
            conversation_adapter=pb_request.conversation_adapter if len(pb_request.conversation_adapter) > 0 else None,
            reaction_adapter=pb_request.reaction_adapter if len(pb_request.reaction_adapter) > 0 else None,
            tts_adapter=pb_request.tts_adapter if len(pb_request.tts_adapter) > 0 else None,
            voice_name=pb_request.voice_name if len(pb_request.voice_name) > 0 else None,
            voice_speed=pb_request.voice_speed if pb_request.voice_speed > 0 else None,
            avatar=pb_request.avatar if len(pb_request.avatar) > 0 else None,
        )
        request_dict = await loop.run_in_executor(self.executor, request_instance.model_dump)
        graph, request_id = await self.proxy.text_chat_with_text_llm_v4(
            request_dict,
            callback_instances=[websocket],
            callback_bytes_fn=websocket.send_bytes,
        )
        try:
            while True:
                data = await websocket.receive_text()
                type_str = ""
                code_str = ""
                try:
                    data_dict = await loop.run_in_executor(self.executor, json.loads, data)
                    type_str = data_dict.get("type", "")
                    code_str = data_dict.get("code", "")
                except json.JSONDecodeError:
                    pass
                if type_str == "websocket.disconnect" and code_str == "1000":
                    raise WebSocketDisconnect
                msg = (
                    "text_chat_with_text_llm_v4 service did not expect to receive data from websocket after stream establishment, "
                    + f"data type: {type(data)}, data content: {data}, "
                    + f"request ID: {request_id}"
                )
                self.logger.warning(msg)
                graph.set_status(DAGStatus.FAILED)
        except WebSocketDisconnect:
            msg = f"Connection with request ID {request_id} was disconnected by user."
            if graph.status == DAGStatus.RUNNING:
                msg += " Due to connection being disconnected by user, setting running task to failed status."
                graph.set_status(DAGStatus.FAILED)
                self.logger.warning(msg)
            else:
                self.logger.info(msg)
            return

    async def text_chat_with_audio_llm_v4(
        self,
        websocket: WebSocket,
    ) -> None:
        """Handle text chat with audio-based LLM.

        This method is called when users engage in text conversation
        with audio-modal LLM. The client should send a TextChatExpressRequestV4
        at the beginning.

        Args:
            websocket (WebSocket):
                WebSocket connection for real-time communication.

        Raises:
            WebSocketDisconnect:
                When the WebSocket connection is closed or invalid data is received.
        """
        loop = asyncio.get_running_loop()
        await websocket.accept()
        pb_bytes = await websocket.receive_bytes()
        pb_request = orchestrator_pb2.OrchestratorV4Request()
        await loop.run_in_executor(self.executor, pb_request.ParseFromString, pb_bytes)
        if pb_request.class_name != "TextChatExpressRequestV4":
            msg = (
                f"Received non-TextChatExpressRequestV4 format data from websocket, data type: {pb_request.class_name}."
            )
            self.logger.error(msg)
            raise WebSocketDisconnect(code=1000)
        if pb_request.app_name not in ("babylon", "python_backend"):
            app_name = "python_backend"
        else:
            app_name = pb_request.app_name
        request_instance = TextChatExpressRequestV4(
            speech_text=pb_request.speech_text,
            language=pb_request.language,
            first_body_fast_response=pb_request.first_body_fast_response,
            app_name=app_name,
            max_front_extension_duration=pb_request.max_front_extension_duration,
            max_rear_extension_duration=pb_request.max_rear_extension_duration,
            # with dynamodb
            user_id=pb_request.user_id if len(pb_request.user_id) > 0 else None,
            character_id=pb_request.character_id if len(pb_request.character_id) > 0 else None,
            # without dynamodb
            conversation_adapter=pb_request.conversation_adapter if len(pb_request.conversation_adapter) > 0 else None,
            tts_adapter=pb_request.tts_adapter if len(pb_request.tts_adapter) > 0 else None,
            voice_name=pb_request.voice_name if len(pb_request.voice_name) > 0 else None,
            avatar=pb_request.avatar if len(pb_request.avatar) > 0 else None,
        )
        request_dict = await loop.run_in_executor(self.executor, request_instance.model_dump)
        graph, request_id = await self.proxy.text_chat_with_audio_llm_v4(
            request_dict,
            callback_instances=[websocket],
            callback_bytes_fn=websocket.send_bytes,
        )
        try:
            while True:
                data = await websocket.receive_text()
                type_str = ""
                code_str = ""
                try:
                    data_dict = await loop.run_in_executor(self.executor, json.loads, data)
                    type_str = data_dict.get("type", "")
                    code_str = data_dict.get("code", "")
                except json.JSONDecodeError:
                    pass
                if type_str == "websocket.disconnect" and code_str == "1000":
                    raise WebSocketDisconnect
                msg = (
                    "text_chat_with_audio_llm_v4 service did not expect to receive data from websocket after stream establishment, "
                    + f"data type: {type(data)}, data content: {data}, "
                    + f"request ID: {request_id}"
                )
                self.logger.warning(msg)
                graph.set_status(DAGStatus.FAILED)
        except WebSocketDisconnect:
            msg = f"Connection with request ID {request_id} was disconnected by user."
            if graph.status == DAGStatus.RUNNING:
                msg += " Due to connection being disconnected by user, setting running task to failed status."
                graph.set_status(DAGStatus.FAILED)
                self.logger.warning(msg)
            else:
                self.logger.info(msg)
            return

    async def get_voice_settings(self, user_id: str, character_id: str) -> VoiceSettingsResponse:
        """Get the voice settings for a specific character.

        Args:
            user_id (str):
                Unique identifier for the user.
            character_id (str):
                Unique identifier for the character.

        Returns:
            VoiceSettingsResponse:
                Response containing the voice settings for the character.
        """
        settings_dict = await self.proxy.get_voice_settings(user_id, character_id)
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(self.executor, VoiceSettingsResponse.model_validate, settings_dict)
        return resp

    async def get_motion_settings(self, user_id: str, character_id: str) -> MotionSettingsResponse:
        """Get the motion settings for a specific character.

        Args:
            user_id (str):
                Unique identifier for the user.
            character_id (str):
                Unique identifier for the character.

        Returns:
            MotionSettingsResponse:
                Response containing the motion settings for the character.
        """
        settings_dict = await self.proxy.get_motion_settings(user_id, character_id)
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(self.executor, MotionSettingsResponse.model_validate, settings_dict)
        return resp

    async def get_relationship(self, character_id: str) -> RelationshipResponse:
        """Get the relationship status for a specific character.

        Args:
            character_id (str):
                Unique identifier for the character.

        Returns:
            RelationshipResponse:
                Response containing the relationship status and score.

        Raises:
            HTTPException:
                When the relationship is not found for the specified character_id.
        """
        relationship = await self.proxy.get_relationship(character_id)
        if relationship is None:
            raise HTTPException(status_code=404, detail=f"Relationship not found for character_id: {character_id}")
        resp = RelationshipResponse(relationship=relationship[0], score=relationship[1])
        return resp

    async def get_emotion(self, user_id: str, character_id: str) -> EmotionResponse:
        """Get the emotion status for a specific character.

        Args:
            user_id (str):
                Unique identifier for the user.
            character_id (str):
                Unique identifier for the character.

        Returns:
            EmotionResponse:
                Response containing the emotion status.

        Raises:
            HTTPException:
                When the emotion is not found for the specified character_id.
        """
        emotions = await self.proxy.get_emotion(user_id, character_id)
        resp = EmotionResponse(emotions=emotions)
        return resp

    async def asr_adapter_choices(self) -> AdapterChoicesResponse:
        """Get available ASR (Automatic Speech Recognition) adapter choices.

        Returns:
            AdapterChoicesResponse:
                Response containing the list of available ASR adapters.
        """
        choices = await self.proxy.get_asr_adapter_choices()
        return AdapterChoicesResponse(choices=choices)

    async def tts_adapter_choices(self) -> AdapterChoicesResponse:
        """Get available TTS (Text-to-Speech) adapter choices.

        Returns:
            AdapterChoicesResponse:
                Response containing the list of available TTS adapters.
        """
        choices = await self.proxy.get_tts_adapter_choices()
        return AdapterChoicesResponse(choices=choices)

    async def tts_voice_names(self, tts_adapter_key: str) -> VoiceNamesResponse:
        """Get available voice names for a specific TTS adapter.

        Args:
            tts_adapter_key (str):
                The key identifier of the TTS adapter.

        Returns:
            VoiceNamesResponse:
                Response containing the list of available voice names.

        Raises:
            AdapterNotFoundError:
                When the specified TTS adapter is not found.
        """
        voice_names = await self.proxy.get_tts_voice_names(tts_adapter_key)
        return VoiceNamesResponse(voice_names=voice_names)

    async def conversation_adapter_choices(self) -> AdapterChoicesResponse:
        """Get available conversation adapter choices.

        Returns:
            AdapterChoicesResponse:
                Response containing the list of available conversation adapters.
        """
        choices = await self.proxy.get_conversation_adapter_choices()
        return AdapterChoicesResponse(choices=choices)

    async def reaction_adapter_choices(self) -> AdapterChoicesResponse:
        """Get available reaction adapter choices.

        Returns:
            AdapterChoicesResponse:
                Response containing the list of available reaction adapters.
        """
        choices = await self.proxy.get_reaction_adapter_choices()
        return AdapterChoicesResponse(choices=choices)

    async def classification_adapter_choices(self) -> AdapterChoicesResponse:
        """Get available classification adapter choices.

        Returns:
            AdapterChoicesResponse:
                Response containing the list of available classification adapters.
        """
        choices = await self.proxy.get_classification_adapter_choices()
        return AdapterChoicesResponse(choices=choices)

    async def memory_adapter_choices(self) -> AdapterChoicesResponse:
        """Get available memory adapter choices.

        Returns:
            AdapterChoicesResponse:
                Response containing the list of available memory adapters.
        """
        choices = await self.proxy.get_memory_adapter_choices()
        return AdapterChoicesResponse(choices=choices)

    async def tail_log(self, request: Request, n_lines: int) -> str:
        """Tail the log file and return the last N lines.

        Args:
            request (Request):
                FastAPI request object for template rendering.
            n_lines (int):
                Maximum number of lines to tail from the log file.

        Returns:
            str:
                HTML template response containing the log content.

        Raises:
            HTTPException:
                When no log file is found or accessible.
        """
        if self.log_path is None:
            msg = "No log file found."
            self.logger.error(msg)
            raise HTTPException(status_code=503, detail=msg)
        # read the last n_lines lines from the log file
        loop = asyncio.get_running_loop()
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = await loop.run_in_executor(self.executor, f.readlines)
            n_lines = min(int(n_lines), len(lines))
            log_content = "".join(lines[-n_lines:])
        # Render template and return
        return self.templates.TemplateResponse("log_template.html", {"request": request, "log_content": log_content})

    async def download_log_file(self) -> Response:
        """Download the log file as a binary attachment.

        Returns:
            Response:
                Response object with the log file content as binary data.

        Raises:
            HTTPException:
                When no log file is found or accessible.
        """
        if self.log_path is None:
            msg = "No log file found."
            self.logger.error(msg)
            raise HTTPException(status_code=503, detail=msg)
        loop = asyncio.get_running_loop()
        with open(self.log_path, "rb") as f:
            content = await loop.run_in_executor(self.executor, f.read)
            resp = Response(content=content, media_type="application/octet-stream")
            base_name = os.path.basename(self.log_path)
            resp.headers["Content-Disposition"] = f"attachment; filename={base_name}"
            return resp

    async def metrics(self) -> Response:
        """Get the metrics for the orchestrator proxy server.

        Returns:
            Response:
                Response object with the metrics content.
        """
        if self.proxy.prometheus_registry:
            data = generate_latest(self.proxy.prometheus_registry)
            return Response(content=data, media_type=CONTENT_TYPE_LATEST)
        else:
            msg = "Prometheus registry is not enabled."
            self.logger.error(msg)
            raise HTTPException(status_code=503, detail=msg)

    def root(self) -> RedirectResponse:
        """Redirect to the API documentation.

        Returns:
            RedirectResponse:
                Redirect response object pointing to the API docs.
        """
        return RedirectResponse(url="/docs")

    def _add_api_routes(self, router: APIRouter) -> None:
        """Add API routes to the router.

        Configures all REST API endpoints and WebSocket routes for the
        orchestrator proxy server, including health checks, adapter choices,
        settings retrieval, and real-time streaming endpoints.

        Args:
            router (APIRouter):
                FastAPI router to add routes to.
        """
        router.add_api_route(
            "/",
            self.root,
            methods=["GET"],
        )
        router.add_api_route(
            "/tail_log/{n_lines}",
            self.tail_log,
            methods=["GET"],
            status_code=200,
            response_model=str,
        )
        router.add_api_route(
            "/api/v4/tail_log/{n_lines}",
            self.tail_log,
            methods=["GET"],
            status_code=200,
            response_model=str,
        )
        router.add_api_route(
            "/download_log_file",
            self.download_log_file,
            methods=["GET"],
            status_code=200,
            response_class=Response,
        )
        if self.proxy.prometheus_registry:
            router.add_api_route(
                "/metrics",
                self.metrics,
                methods=["GET"],
                status_code=200,
                response_class=Response,
            )
        router.add_api_route(
            "/health",
            self.health,
            methods=["GET"],
            status_code=200,
            responses={
                200: {},
                **OPENAPI_RESPONSE_500,
            },
        )
        router.add_api_route(
            "/api/v4/health",
            self.health,
            methods=["GET"],
            status_code=200,
            responses={
                200: {},
                **OPENAPI_RESPONSE_500,
            },
        )
        router.add_api_route(
            "/api/v4/asr_adapter_choices",
            self.asr_adapter_choices,
            methods=["GET"],
            status_code=200,
            response_model=AdapterChoicesResponse,
        )
        router.add_api_route(
            "/api/v4/tts_adapter_choices",
            self.tts_adapter_choices,
            methods=["GET"],
            status_code=200,
            response_model=AdapterChoicesResponse,
        )
        router.add_api_route(
            "/api/v4/tts_voice_names/{tts_adapter_key}",
            self.tts_voice_names,
            methods=["GET"],
            status_code=200,
            response_model=VoiceNamesResponse,
            responses={
                200: {
                    "model": VoiceNamesResponse,
                },
                **OPENAPI_RESPONSE_404,
            },
        )
        router.add_api_route(
            "/api/v4/conversation_adapter_choices",
            self.conversation_adapter_choices,
            methods=["GET"],
            status_code=200,
            response_model=AdapterChoicesResponse,
        )
        router.add_api_route(
            "/api/v4/reaction_adapter_choices",
            self.reaction_adapter_choices,
            methods=["GET"],
            status_code=200,
            response_model=AdapterChoicesResponse,
        )
        router.add_api_route(
            "/api/v4/classification_adapter_choices",
            self.classification_adapter_choices,
            methods=["GET"],
            status_code=200,
            response_model=AdapterChoicesResponse,
        )
        router.add_api_route(
            "/api/v4/memory_adapter_choices",
            self.memory_adapter_choices,
            methods=["GET"],
            status_code=200,
            response_model=AdapterChoicesResponse,
        )
        router.add_api_route(
            "/api/v4/get_voice_settings/{user_id}/{character_id}",
            self.get_voice_settings,
            methods=["GET"],
            status_code=200,
            response_model=VoiceSettingsResponse,
        )
        router.add_api_route(
            "/api/v4/get_motion_settings/{user_id}/{character_id}",
            self.get_motion_settings,
            methods=["GET"],
            status_code=200,
            response_model=MotionSettingsResponse,
        )
        router.add_api_route(
            "/api/v4/get_relationship/{character_id}",
            self.get_relationship,
            methods=["GET"],
            status_code=200,
            response_model=RelationshipResponse,
        )
        router.add_api_route(
            "/api/v4/get_emotion/{user_id}/{character_id}",
            self.get_emotion,
            methods=["GET"],
            status_code=200,
            response_model=EmotionResponse,
        )
        # WebSocket
        router.add_api_websocket_route(
            "/api/v4/audio_chat_with_text_llm",
            endpoint=self.audio_chat_with_text_llm_v4,
        )
        router.add_api_websocket_route(
            "/api/v4/audio_chat_with_audio_llm",
            endpoint=self.audio_chat_with_audio_llm_v4,
        )
        router.add_api_websocket_route(
            "/api/v4/text_generate",
            endpoint=self.text_generate_v4,
        )
        router.add_api_websocket_route(
            "/api/v4/text_chat_with_text_llm",
            endpoint=self.text_chat_with_text_llm_v4,
        )
        router.add_api_websocket_route(
            "/api/v4/text_chat_with_audio_llm",
            endpoint=self.text_chat_with_audio_llm_v4,
        )
