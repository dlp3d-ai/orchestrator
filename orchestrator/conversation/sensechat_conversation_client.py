import asyncio
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union

import httpx
import jwt
from prometheus_client import Histogram

from ..data_structures.conversation import ConversationChunkBody, RejectChunkBody
from ..utils.exception import MissingAPIKeyException
from ..utils.executor_registry import ExecutorRegistry
from .conversation_adapter import BracketFilter, ConversationAdapter


class SenseChatConversationClient(ConversationAdapter):
    """SenseChat conversation client for streaming chat and reject operations.

    This client provides streaming conversation capabilities using SenseChat
    API, with support for bracket content filtering and downstream task
    processing.
    """

    AVAILABLE_FOR_STREAM = True
    AVAILABLE_FOR_REJECT = True
    ExecutorRegistry.register_class("SenseChatConversationClient")

    def __init__(
        self,
        name: str,
        agent_prompts_file: str,
        sensechat_model_name: str = "SenseChat-5-1202",
        sensechat_url: str = "https://api.sensenova.cn/v1/llm/chat-completions",
        proxy_url: Union[None, str] = None,
        request_timeout: float = 600.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        latency_histogram: Histogram | None = None,
        input_token_number_histogram: Histogram | None = None,
        output_token_number_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
        enable_bracket_filter: bool = True,
        bracket_pairs: list[tuple[str, str]] = [("*", "*"), ("(", ")"), ("[", "]"), ("{", "}"), ("「", "」"), ("（", "）")],
    ):
        """Initialize the SenseChat conversation client.

        Args:
            name (str):
                The name of the conversation adapter.
            agent_prompts_file (str):
                The path to the agent prompts file.
            sensechat_model_name (str, optional):
                The name of the SenseChat model to use.
                Defaults to "SenseChat-5-1202".
            sensechat_url (str, optional):
                The SenseChat API URL.
                Defaults to "https://api.sensenova.cn/v1/llm/chat-completions".
            proxy_url (Union[None, str], optional):
                The proxy URL for the conversation.
                Defaults to None.
            request_timeout (float, optional):
                The timeout for requests in seconds.
                Defaults to 600.0.
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
            enable_bracket_filter (bool, optional):
                Whether to enable bracket content filtering. Defaults to True.
            bracket_pairs (list[tuple[str, str]], optional):
                Bracket pairs to filter. Defaults to [('*', '*'), ('(', ')'), ('[', ']'), ('{', '}'), ('「', '」'), ('（', '）')].
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
        self.sensechat_model_name = sensechat_model_name
        self.sensechat_url = sensechat_url

        if self.proxy_url is not None:
            self.http_client = httpx.AsyncClient(proxy=self.proxy_url, timeout=self.request_timeout)
        else:
            self.http_client = httpx.AsyncClient(timeout=self.request_timeout)

        self.enable_bracket_filter = enable_bracket_filter
        self.bracket_pairs = bracket_pairs

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
        """Generate a JWT token for the SenseChat API.

        Args:
            task_space (dict):
                The task space for the request.

        Returns:
            str:
                JWT token string for API authentication.
        """
        api_keys = task_space.get("api_keys", {})
        ak = api_keys.get("sensechat_ak", "")
        sk = api_keys.get("sensechat_sk", "")
        if not ak or not sk:
            msg = "SenseChat API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        headers = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": ak,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5,
        }
        return jwt.encode(payload, sk, headers=headers)

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

            task_space = self.input_buffer[request_id]["chat_task"]
            dag = task_space["dag"]
            dag_start_time = task_space["dag_start_time"]
            style_list = task_space["style_list"]
            node_name = task_space["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            downstream_instances = dict()
            for node in downstream_nodes:
                next_node_name = node.name
                payload = node.payload
                downstream_instances[next_node_name] = payload

            bracket_filter = None
            if self.enable_bracket_filter:
                bracket_filter = BracketFilter(self.bracket_pairs)

            chat_rsp = ""
            first_body_trunk = True
            model_name_override = task_space.get("conversation_model_override")
            system_chat = self.agent_prompts["system_chat"].format(style_list=style_list)
            conversation_prompt = task_space["user_prompt"] + "\n" + system_chat

            loop = asyncio.get_running_loop()
            jwt_token = await loop.run_in_executor(self.executor, self._gen_token, task_space)

            user_id = task_space["user_id"]
            model_name = model_name_override if model_name_override else self.sensechat_model_name

            messages = [
                {
                    "role": "system",
                    "content": conversation_prompt.replace(
                        "{style_list}", str(style_list) if style_list is not None else ""
                    ),
                },
                {"role": "user", "content": conversation_context},
                *conversation_history,
                {"role": "user", "content": message},
            ]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {jwt_token}",
            }

            data = {
                "max_new_tokens": 4096,
                "messages": messages,
                "model": model_name,
                "stream": True,
            }

            input_token_number = 0
            output_token_number = 0

            async with self.http_client.stream("POST", self.sensechat_url, headers=headers, json=data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line == "data:[DONE]":
                        continue

                    if line.startswith("data:"):
                        line = line[5:].strip()

                    if line == "[DONE]" or not line:
                        break

                    try:
                        chunk_data = json.loads(line)
                        data_obj = chunk_data.get("data", {})
                        choices = data_obj.get("choices", [])

                        if choices and len(choices) > 0:
                            choice = choices[0]
                            delta = choice.get("delta", "")
                            finish_reason = choice.get("finish_reason", "")

                            if finish_reason == "stop":
                                break

                            if isinstance(delta, str) and delta:
                                text_seg = delta

                                if bracket_filter is not None:
                                    text_seg = await loop.run_in_executor(
                                        self.executor, bracket_filter.filter_text_segment, text_seg
                                    )

                                if len(text_seg) > 0:
                                    chat_rsp += text_seg
                                    style = await loop.run_in_executor(self.executor, self.extract_style_tag, chat_rsp)
                                    coroutines = list()
                                    for next_node_name, payload in downstream_instances.items():
                                        body_trunk = ConversationChunkBody(
                                            request_id=request_id,
                                            text_segment=text_seg,
                                            style=style,
                                        )
                                        coroutines.append(payload.feed_stream(body_trunk))
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
                                                self.latency_histogram.labels(
                                                    adapter=self.name, user_id=user_id
                                                ).observe(latency)
                                    await asyncio.gather(*coroutines)

                        usage_data = data_obj.get("usage")
                        if usage_data:
                            input_token_number = usage_data.get("prompt_tokens", 0)
                            output_token_number = usage_data.get("completion_tokens", 0)
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        self.logger.warning(f"SenseChat error processing chunk: {e}")
                        continue

            if self.input_token_number_histogram:
                self.input_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(input_token_number)
            if self.output_token_number_histogram:
                self.output_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(
                    output_token_number
                )
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

            task_space = self.input_buffer[request_id]["reject_task"]
            dag = task_space["dag"]
            dag_start_time = task_space["dag_start_time"]
            node_name = task_space["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            downstream_instances = dict()
            for node in downstream_nodes:
                next_node_name = node.name
                payload = node.payload
                downstream_instances[next_node_name] = payload

            bracket_filter = None
            if self.enable_bracket_filter:
                bracket_filter = BracketFilter(self.bracket_pairs)

            reject_rsp = ""
            first_body_trunk = True
            model_name_override = task_space.get("conversation_model_override")
            reject_prompt = self.agent_prompts["system_reject"]

            loop = asyncio.get_running_loop()
            jwt_token = await loop.run_in_executor(self.executor, self._gen_token, task_space)

            user_id = task_space["user_id"]
            model_name = model_name_override if model_name_override else self.sensechat_model_name

            messages = [
                {"role": "system", "content": reject_prompt},
                {"role": "user", "content": message},
            ]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {jwt_token}",
            }

            data = {
                "max_new_tokens": 1000,
                "messages": messages,
                "model": model_name,
                "stream": True,
            }

            input_token_number = 0
            output_token_number = 0

            async with self.http_client.stream("POST", self.sensechat_url, headers=headers, json=data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    line_stripped = line.strip()

                    if line_stripped.startswith("data:"):
                        line_stripped = line_stripped[5:].strip()

                    if line_stripped == "[DONE]" or line_stripped == "":
                        break

                    try:
                        chunk_data = json.loads(line_stripped)

                        choices = None
                        if "data" in chunk_data and isinstance(chunk_data["data"], dict):
                            choices = chunk_data["data"].get("choices", [])
                        elif "choices" in chunk_data:
                            choices = chunk_data["choices"]

                        if choices and len(choices) > 0:
                            choice = choices[0]
                            delta = choice.get("delta")
                            message = choice.get("message", {})
                            finish_reason = choice.get("finish_reason", "")

                            if finish_reason == "stop":
                                break

                            text_seg = None
                            if isinstance(delta, str) and delta and delta != "None":
                                text_seg = delta
                            elif isinstance(delta, dict) and "content" in delta and delta.get("content") is not None:
                                text_seg = delta.get("content")
                            elif (
                                isinstance(message, dict)
                                and "content" in message
                                and message.get("content") is not None
                            ):
                                text_seg = message.get("content")

                            if text_seg:
                                if bracket_filter is not None:
                                    text_seg = await loop.run_in_executor(
                                        self.executor, bracket_filter.filter_text_segment, text_seg
                                    )

                                if len(text_seg) > 0:
                                    reject_rsp += text_seg
                                    coroutines = list()
                                    for next_node_name, payload in downstream_instances.items():
                                        body_trunk = RejectChunkBody(
                                            request_id=request_id,
                                            text_segment=text_seg,
                                        )
                                        coroutines.append(payload.feed_stream(body_trunk))
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
                                                self.latency_histogram.labels(
                                                    adapter=self.name, user_id=user_id
                                                ).observe(latency)
                                    await asyncio.gather(*coroutines)

                        usage_data = None
                        if "data" in chunk_data and isinstance(chunk_data["data"], dict):
                            usage_data = chunk_data["data"].get("usage")
                        elif "usage" in chunk_data:
                            usage_data = chunk_data["usage"]

                        if usage_data:
                            if "prompt_tokens" in usage_data:
                                input_token_number = usage_data["prompt_tokens"]
                            if "completion_tokens" in usage_data:
                                output_token_number = usage_data["completion_tokens"]
                    except json.JSONDecodeError as e:
                        if line_stripped and not line_stripped.startswith("[DONE]"):
                            self.logger.debug(
                                f"SenseChat unable to parse JSON message (may be incomplete): {line_stripped[:200]}, error: {e}"
                            )
                        continue
                    except Exception as e:
                        self.logger.warning(f"SenseChat error processing chunk: {e}, line: {line_stripped[:200]}")
                        continue

            if self.input_token_number_histogram:
                self.input_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(input_token_number)
            if self.output_token_number_histogram:
                self.output_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(
                    output_token_number
                )
            return reject_rsp
        except Exception as e:
            msg = f"Error in streaming reject: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            return ""
