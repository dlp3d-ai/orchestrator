import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union

import anthropic
import httpx

from ..data_structures.conversation import ConversationChunkBody, RejectChunkBody
from ..utils.executor_registry import ExecutorRegistry
from .conversation_adapter import BracketFilter, ConversationAdapter


class AnthropicConversationClient(ConversationAdapter):
    """Anthropic conversation client for streaming chat and reject operations.

    This client provides streaming conversation capabilities using Anthropic's
    Claude models, with support for bracket content filtering and downstream
    task processing.
    """

    AVAILABLE_FOR_STREAM = True
    AVAILABLE_FOR_REJECT = True
    ExecutorRegistry.register_class("AnthropicConversationClient")

    def __init__(
        self,
        name: str,
        agent_prompts_file: str,
        anthropic_model_name: str = "claude-sonnet-4-20250514",
        proxy_url: Union[None, str] = None,
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
        enable_bracket_filter: bool = True,
        bracket_pairs: list[tuple[str, str]] = [("*", "*"), ("(", ")"), ("[", "]"), ("{", "}"), ("「", "」"), ("（", "）")],
    ):
        """Initialize the Anthropic conversation client.

        Args:
            name (str):
                The name of the conversation adapter.
            agent_prompts_file (str):
                The path to the agent prompts file.
            anthropic_model_name (str, optional):
                The name of the Anthropic model to use.
                Defaults to "claude-sonnet-4-20250514".
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
            logger_cfg=logger_cfg,
        )
        self.anthropic_model_name = anthropic_model_name

        if self.proxy_url is not None:
            self.http_client = httpx.AsyncClient(proxy=self.proxy_url)
        else:
            self.http_client = None

        # Initialize bracket filter
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
        anthropic_api_key = self.input_buffer[request_id]["chat_task"].get("api_keys", {}).get("anthropic_api_key", "")
        if not anthropic_api_key:
            msg = "Anthropic API key is not found in the API keys."
            self.logger.error(msg)
            raise ValueError(msg)

        self.input_buffer[request_id]["chat_task"]["llm_client"] = anthropic.AsyncAnthropic(
            api_key=anthropic_api_key,
            http_client=self.http_client,
        )
        self.input_buffer[request_id]["reject_task"]["llm_client"] = anthropic.AsyncAnthropic(
            api_key=anthropic_api_key,
            http_client=self.http_client,
        )

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
            dag_start_time = task_space["dag_start_time"]
            node_name = task_space["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            downstream_instances = dict()
            for node in downstream_nodes:
                next_node_name = node.name
                payload = node.payload
                downstream_instances[next_node_name] = payload

            # Initialize bracket filter
            bracket_filter = None
            if self.enable_bracket_filter:
                bracket_filter = BracketFilter(self.bracket_pairs)

            # Start conversation
            chat_rsp = ""
            first_body_trunk = True
            model_name_override = task_space["conversation_model_override"]
            system_chat = self.agent_prompts["system_chat"].format(style_list=style_list)
            conversation_prompt = task_space["user_prompt"] + "\n" + system_chat

            llm_client = task_space.get("llm_client", None)
            while llm_client is None:
                await asyncio.sleep(self.sleep_time)
                llm_client = task_space.get("llm_client", None)

            async with llm_client.messages.stream(
                model=model_name_override if model_name_override else self.anthropic_model_name,
                max_tokens=1000,
                temperature=1,
                system=conversation_prompt,
                messages=[
                    {"role": "user", "content": conversation_context},
                    *conversation_history,
                    {"role": "user", "content": message},
                ],
            ) as stream:
                loop = asyncio.get_event_loop()
                async for chunk in stream.text_stream:
                    if chunk is not None and len(chunk.strip()) > 0:
                        text_seg = chunk

                        # Apply bracket filter
                        if bracket_filter is not None and text_seg:
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
                            asyncio.gather(*coroutines)
            return chat_rsp
        except Exception as e:
            msg = f"Error in streaming chat: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            return ""

    async def _llm_stream_reject(self, message: str, language: str, request_id: str) -> str:
        """Stream reject task processing.

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
            dag_start_time = task_space["dag_start_time"]
            node_name = task_space["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            downstream_instances = dict()
            for node in downstream_nodes:
                next_node_name = node.name
                payload = node.payload
                downstream_instances[next_node_name] = payload

            # Initialize bracket filter
            bracket_filter = None
            if self.enable_bracket_filter:
                bracket_filter = BracketFilter(self.bracket_pairs)

            # Process reject
            reject_rsp = ""
            first_body_trunk = True
            model_name_override = task_space["conversation_model_override"]
            reject_prompt = self.agent_prompts["system_reject"]

            llm_client = task_space.get("llm_client", None)
            while llm_client is None:
                await asyncio.sleep(self.sleep_time)
                llm_client = task_space.get("llm_client", None)

            async with llm_client.messages.stream(
                model=model_name_override if model_name_override else self.anthropic_model_name,
                max_tokens=1000,
                temperature=1,
                system=reject_prompt,
                messages=[{"role": "user", "content": message}],
            ) as stream:
                loop = asyncio.get_event_loop()
                async for chunk in stream.text_stream:
                    if chunk is not None and len(chunk.strip()) > 0:
                        text_seg = chunk

                        # Apply bracket filter
                        if bracket_filter is not None and text_seg:
                            text_seg = await loop.run_in_executor(
                                self.executor, bracket_filter.filter_text_segment, text_seg
                            )

                        if len(text_seg) > 0:
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
                            asyncio.gather(*coroutines)
                            reject_rsp += text_seg
            return reject_rsp
        except Exception as e:
            msg = f"Error in streaming reject: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            return ""
