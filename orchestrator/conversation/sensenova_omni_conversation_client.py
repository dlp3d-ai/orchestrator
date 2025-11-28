import asyncio
import json
import re
import ssl
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

import jwt
import websockets
from prometheus_client import Histogram

from ..data_structures.conversation import ConversationChunkBody, RejectChunkBody
from ..utils.exception import MissingAPIKeyException
from ..utils.executor_registry import ExecutorRegistry
from .conversation_adapter import ConversationAdapter


class SenseNovaOmniConversationClient(ConversationAdapter):
    """SenseNova Omni conversation client for streaming chat and reject
    operations.

    This client provides streaming conversation capabilities using SenseNova
    Omni API, with support for bracket content filtering and downstream task
    processing.
    """

    AVAILABLE_FOR_STREAM = True
    AVAILABLE_FOR_REJECT = True
    # Class-level SSL context cache
    _ssl_context_cache: Optional[ssl.SSLContext] = None
    ExecutorRegistry.register_class("SenseNovaOmniConversationClient")

    def __init__(
        self,
        name: str,
        agent_prompts_file: str,
        wss_url: str = "wss://api-gai.sensetime.com/agent-5o/duplex/ws2",
        proxy_url: Union[None, str] = None,
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        latency_histogram: Histogram | None = None,
        input_token_number_histogram: Histogram | None = None,
        output_token_number_histogram: Histogram | None = None,
        token_number_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the OpenAI conversation client.

        Args:
            name (str):
                The name of the conversation adapter.
            agent_prompts_file (str):
                The path to the agent prompts file.
            wss_url (str, optional):
                The WebSocket URL for the conversation.
                Defaults to "wss://api-gai.sensetime.com/agent-5o/duplex/ws2".
            proxy_url (Union[None, str], optional):
                The proxy URL for the conversation.
                Defaults to None.
            request_timeout (float, optional):
                The timeout for requests in seconds.
                Defaults to 20.0.
            queue_size (int, optional):
                The size of the input buffer queue.
                Defaults to 100.
            sleep_time (float, optional):
                The sleep time between operations in seconds.
                Defaults to 0.01.
            clean_interval (float, optional):
                The interval to clean expired requests in seconds.
                Defaults to 10.0.
            expire_time (float, optional):
                The time to expire requests in seconds.
                Defaults to 120.0.
            max_workers (int, optional):
                The maximum number of worker threads.
                Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                External thread pool executor to use.
                Defaults to None.
            latency_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording request latency distribution
                in seconds. If provided, latency metrics will be collected for monitoring
                purposes. Defaults to None.
            input_token_number_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording input token count distribution
                per request. If provided, input token usage metrics will be collected for
                monitoring purposes. Defaults to None.
            output_token_number_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording output token count distribution
                per request. If provided, output token usage metrics will be collected for
                monitoring purposes. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
        """
        ConversationAdapter.__init__(
            self,
            name=name,
            agent_prompts_file=agent_prompts_file,
            proxy_url=proxy_url,
            request_timeout=request_timeout,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            latency_histogram=latency_histogram,
            input_token_number_histogram=input_token_number_histogram,
            output_token_number_histogram=output_token_number_histogram,
            logger_cfg=logger_cfg,
        )
        self.wss_url = wss_url

        self.executor = (
            thread_pool_executor if thread_pool_executor is not None else ThreadPoolExecutor(max_workers=max_workers)
        )
        self.executor_external = True if thread_pool_executor is not None else False

    def __del__(self) -> None:
        """Destructor, cleanup thread pool executor."""
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    async def _init_llm_client(self, request_id: str) -> None:
        """Initialize the LLM client for the given request.

        Args:
            request_id (str):
                The unique identifier for the request.
        """
        pass

    def _gen_token(self, task_space: dict) -> str:
        """Generate a JWT token for the SenseNova Omni API.

        Args:
            task_space (dict):
                The task space for the request.

        Returns:
            str:
                JWT token string for API authentication.
        """
        api_keys = task_space.get("api_keys", {})
        iss = api_keys.get("sensenovaomni_ak", "")
        secret = api_keys.get("sensenovaomni_sk", "")
        if not iss or not secret:
            msg = "SenseNova Omni API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        payload = {"iss": iss, "exp": int(time.time()) + 3600}  # 1 hour expiration
        return jwt.encode(payload, secret, algorithm="HS256")

    async def _ws_create_session(self, ws) -> str:
        """Create a WebSocket session with the SenseNova Omni API.

        Args:
            ws:
                WebSocket connection object.

        Returns:
            str:
                Session ID returned from the API.
        """
        create_session_msg = {"type": "CreateSession", "request_id": uuid.uuid4().hex}
        await ws.send(json.dumps(create_session_msg))
        self.logger.debug("SenseNova sent CreateSession request")

        response = await ws.recv()
        json_res = json.loads(response)
        session_id = json_res.get("session_id")
        self.logger.debug(f"SenseNova created session, ID: {session_id}")
        return session_id

    async def _ws_request_agora_channel_info(self, ws):
        """Request Agora channel information via WebSocket.

        Args:
            ws:
                WebSocket connection object.

        Returns:
            tuple:
                Tuple containing (app_id, channel_id, server_uid).
        """
        request = {"type": "RequestAgoraChannelInfo", "request_id": uuid.uuid4().hex}
        await ws.send(json.dumps(request))
        self.logger.debug("SenseNova sent RequestAgoraChannelInfo request")

        response = await ws.recv()
        json_res = json.loads(response)

        app_id = json_res.get("appid")
        channel_id = json_res.get("channel_id")
        server_uid = json_res.get("server_uid")

        self.logger.debug(f"SenseNova received Agora channel info: AppID={app_id}, ChannelID={channel_id}")
        return app_id, channel_id, server_uid

    async def _ws_request_agora_token(self, ws):
        """Request Agora token via WebSocket.

        Args:
            ws:
                WebSocket connection object.

        Returns:
            tuple:
                Tuple containing (token, user_id).
        """
        request = {"type": "RequestAgoraToken", "duration": 600, "request_id": uuid.uuid4().hex}
        await ws.send(json.dumps(request))
        self.logger.debug("SenseNova sent RequestAgoraToken request")

        response = await ws.recv()
        json_res = json.loads(response)

        token = json_res.get("token")
        user_id = json_res.get("client_uid")

        self.logger.debug(f"SenseNova received Agora token: UserID={user_id}")
        return token, user_id

    async def _ws_message_listener(
        self,
        ws,
        request_id: str,
        start_time: float,
        is_reject: bool = False,
    ) -> str:
        """WebSocket message listener for receiving responses.

        Args:
            ws:
                WebSocket connection object.
            request_id (str):
                The unique identifier for the request.
            start_time (float):
                The start time of the request.
            is_reject (bool):
                Whether the request is a reject request.
                Defaults to False.

        Returns:
            str:
                Assistant output text from the API response.
        """
        assistant_output = ""
        first_body_trunk = True
        style = "正常"

        if is_reject:
            task_space = self.input_buffer[request_id]["reject_task"]
        else:
            task_space = self.input_buffer[request_id]["chat_task"]

        user_id = task_space["user_id"]
        dag = task_space["dag"]
        dag_start_time = task_space["dag_start_time"]
        node_name = task_space["node_name"]
        dag_node = dag.get_node(node_name)
        downstream_nodes = dag_node.downstreams

        try:
            while True:
                try:
                    response = await ws.recv()
                    try:
                        json_res = json.loads(response)
                        message_type = json_res.get("type")
                        self.logger.debug(f"{message_type}: {json_res}")
                        if message_type == "ResponseTextSegment":
                            text_seg = json_res.get("text").strip()
                            style_match = re.search(r"<style>(.*?)</style>", text_seg)
                            if style_match:
                                style = style_match.group(1)
                                text_seg = text_seg.replace(style_match.group(0), "")
                            if len(text_seg) > 0:
                                assistant_output += text_seg
                                coroutines = list()
                                for node in downstream_nodes:
                                    payload = node.payload
                                    if is_reject:
                                        body_trunk = RejectChunkBody(
                                            request_id=request_id,
                                            text_segment=text_seg,
                                        )
                                    else:
                                        body_trunk = ConversationChunkBody(
                                            request_id=request_id,
                                            text_segment=text_seg,
                                            style=style,
                                        )
                                    coroutines.append(payload.feed_stream(body_trunk))
                                    await asyncio.gather(*coroutines)
                                if first_body_trunk:
                                    first_body_trunk = False
                                    if dag_start_time is not None:
                                        time_diff = time.time() - dag_start_time
                                        self.logger.debug(
                                            f"request {request_id} LLM delay from DAG start: {time_diff:.2f} seconds"
                                        )
                                    latency = time.time() - start_time
                                    self.logger.debug(
                                        f"request {request_id} first chunk latency: {latency:.2f} seconds"
                                    )
                                    if self.latency_histogram:
                                        self.latency_histogram.labels(adapter=self.name, user_id=user_id).observe(
                                            latency
                                        )

                        if message_type == "ResponseEndTextStream":
                            await ws.close()
                            break
                    except json.JSONDecodeError:
                        self.logger.warning(f"SenseNova unable to parse JSON message: {response}")
                except asyncio.TimeoutError:
                    continue  # Continue loop, check status
            if self.output_token_number_histogram:
                self.output_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(
                    len(assistant_output)
                )
            return assistant_output
        except websockets.exceptions.ConnectionClosedError:
            self.logger.debug("SenseNova WebSocket connection closed")
            return ""
        except Exception as e:
            self.logger.error(f"SenseNova WebSocket message listener error: {e}")
            return ""

    async def _llm_stream_chat(
        self,
        message: str,
        conversation_context: str,
        conversation_history: list[Any],
        language: str,
        request_id: str,
    ) -> str:
        """Stream LLM chat conversation.

        Args:
            message (str):
                The current message to process.
            conversation_context (str):
                The context of the conversation.
            conversation_history (list[Any]):
                The history of previous conversation messages.
            language (str):
                The language of the conversation.
            request_id (str):
                The unique identifier for the request.

        Returns:
            str:
                The complete response from the LLM.
        """
        try:
            start_time = time.time()

            # Prepare downstream tasks
            task_space = self.input_buffer[request_id]["chat_task"]
            dag = task_space["dag"]
            style_list = task_space["style_list"]
            node_name = task_space["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            downstream_instances = dict()
            for node in downstream_nodes:
                next_node_name = node.name
                payload = node.payload
                downstream_instances[next_node_name] = payload

            # Start conversation
            loop = asyncio.get_running_loop()
            jwt_token = await loop.run_in_executor(self.executor, self._gen_token, task_space)
            wss_url_with_token = f"{self.wss_url}?jwt={jwt_token}"
            if self.__class__._ssl_context_cache is None:
                ssl_context = await loop.run_in_executor(self.executor, ssl.create_default_context)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                self.__class__._ssl_context_cache = ssl_context
            else:
                ssl_context = self.__class__._ssl_context_cache
            ws = await websockets.connect(wss_url_with_token, ssl=ssl_context)

            session_id = await self._ws_create_session(ws)

            app_id, channel_id, server_uid = await self._ws_request_agora_channel_info(ws)

            token, user_id = await self._ws_request_agora_token(ws)
            start_serving_msg = {"type": "StartServing"}
            await ws.send(json.dumps(start_serving_msg))

            system_chat = self.agent_prompts["system_chat"].format(style_list=style_list)
            conversation_prompt = task_space["user_prompt"] + "\n" + system_chat
            prompt_msg = {
                "type": "SetSystemPrompt",
                "system_prompt": conversation_prompt
                + "\n"
                + f"conversation_context: {conversation_context}"
                + "\n"
                + f"conversation_history: {conversation_history}",
            }
            prompt_msg = await loop.run_in_executor(self.executor, _json_dumps_not_ensure_ascii, prompt_msg)
            await ws.send(prompt_msg)

            send_message = {
                "type": "PostMultimodalGenerate",
                "request_id": uuid.uuid4().hex,
                "text": message,
            }
            send_message = await loop.run_in_executor(self.executor, _json_dumps_not_ensure_ascii, send_message)
            await ws.send(send_message)

            chat_rsp = await self._ws_message_listener(ws, request_id, start_time)
            if self.input_token_number_histogram:
                input_token_number = len(prompt_msg.replace(" ", "").replace("\n", "").replace("\t", ""))
                self.input_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(input_token_number)
            return chat_rsp
        except Exception as e:
            msg = f"Error in streaming chat: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            return ""

    async def _llm_stream_reject(self, message: str, language: str, request_id: str) -> str:
        """Stream LLM reject processing.

        Args:
            message (str):
                The message to reject.
            language (str):
                The language of the message.
            request_id (str):
                The unique identifier for the request.

        Returns:
            str:
                The reject response from the LLM.
        """
        try:
            start_time = time.time()

            # Prepare downstream tasks
            task_space = self.input_buffer[request_id]["reject_task"]
            dag = task_space["dag"]
            node_name = task_space["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            downstream_instances = dict()
            for node in downstream_nodes:
                next_node_name = node.name
                payload = node.payload
                downstream_instances[next_node_name] = payload

            # Start reject
            loop = asyncio.get_running_loop()
            jwt_token = await loop.run_in_executor(self.executor, self._gen_token, task_space)
            wss_url_with_token = f"{self.wss_url}?jwt={jwt_token}"
            if self.__class__._ssl_context_cache is None:
                ssl_context = await loop.run_in_executor(self.executor, ssl.create_default_context)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                self.__class__._ssl_context_cache = ssl_context
            else:
                ssl_context = self.__class__._ssl_context_cache
            ws = await websockets.connect(wss_url_with_token, ssl=ssl_context)

            session_id = await self._ws_create_session(ws)

            app_id, channel_id, server_uid = await self._ws_request_agora_channel_info(ws)

            token, user_id = await self._ws_request_agora_token(ws)
            start_serving_msg = {"type": "StartServing"}
            await ws.send(json.dumps(start_serving_msg))
            reject_prompt = self.agent_prompts["system_reject"]

            prompt_msg = {
                "type": "SetSystemPrompt",
                "system_prompt": reject_prompt,
            }
            prompt_msg = await loop.run_in_executor(self.executor, _json_dumps_not_ensure_ascii, prompt_msg)
            await ws.send(prompt_msg)

            send_message = {
                "type": "PostMultimodalGenerate",
                "request_id": uuid.uuid4().hex,
                "text": message,
            }
            send_message = await loop.run_in_executor(self.executor, _json_dumps_not_ensure_ascii, send_message)
            await ws.send(send_message)

            reject_rsp = await self._ws_message_listener(ws, request_id, start_time, is_reject=True)
            return reject_rsp
        except Exception as e:
            msg = f"Error in streaming reject: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            return ""


def _json_dumps_not_ensure_ascii(obj: Any, **kwargs: Any) -> str:
    """JSON dumps without ensure_ascii."""
    kwargs = kwargs.copy()
    if "ensure_ascii" in kwargs:
        kwargs.pop("ensure_ascii")
    return json.dumps(obj, ensure_ascii=False, **kwargs)
