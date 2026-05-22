import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union

from prometheus_client import Histogram

from ..data_structures.conversation import ConversationChunkBody, RejectChunkBody
from ..llm.openai_chat import create_client, stream
from ..llm.sensenova import (
    SENSENOVA_DEFAULT_BASE_URL,
    SENSENOVA_DEFAULT_MODEL,
    SENSENOVA_EXTRA_BODY,
    build_sensenova_config,
)
from ..utils.executor_registry import ExecutorRegistry
from .conversation_adapter import BracketFilter, ConversationAdapter


class SenseNovaConversationClient(ConversationAdapter):
    """SenseNova conversation client for streaming chat and reject operations."""

    AVAILABLE_FOR_STREAM = True
    AVAILABLE_FOR_REJECT = True
    ExecutorRegistry.register_class("SenseNovaConversationClient")

    def __init__(
        self,
        name: str,
        agent_prompts_file: str,
        sensenova_model_name: str = SENSENOVA_DEFAULT_MODEL,
        sensenova_url: str = SENSENOVA_DEFAULT_BASE_URL,
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
        self.sensenova_model_name = sensenova_model_name
        self.sensenova_url = sensenova_url
        self.llm_provider_config = build_sensenova_config(
            model_name=sensenova_model_name,
            base_url=sensenova_url,
            timeout=request_timeout,
            proxy_url=proxy_url,
        )

        self.enable_bracket_filter = enable_bracket_filter
        self.bracket_pairs = bracket_pairs
        self.executor = (
            thread_pool_executor if thread_pool_executor is not None else ThreadPoolExecutor(max_workers=max_workers)
        )
        self.executor_external = True if thread_pool_executor is not None else False

    def __del__(self) -> None:
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    async def _init_llm_client(self, request_id: str) -> None:
        llm_client = create_client(
            self.input_buffer[request_id]["chat_task"].get("api_keys", {}),
            self.llm_provider_config,
        )
        self.input_buffer[request_id]["chat_task"]["llm_client"] = llm_client
        self.input_buffer[request_id]["reject_task"]["llm_client"] = llm_client

    async def _llm_stream_chat(
        self,
        message: str,
        conversation_context: str,
        conversation_history: list[Any],
        language: str,
        request_id: str,
    ) -> str:
        try:
            start_time = time.time()
            task_space = self.input_buffer[request_id]["chat_task"]
            dag = task_space["dag"]
            dag_start_time = task_space["dag_start_time"]
            style_list = task_space["style_list"]
            node_name = task_space["node_name"]
            downstream_instances = {
                node.name: node.payload for node in dag.get_node(node_name).downstreams
            }

            bracket_filter = BracketFilter(self.bracket_pairs) if self.enable_bracket_filter else None
            chat_rsp = ""
            first_body_trunk = True
            model_name_override = task_space.get("conversation_model_override")
            system_chat = self.agent_prompts["system_chat"].format(style_list=style_list)
            conversation_prompt = task_space["user_prompt"] + "\n" + system_chat
            llm_client = task_space.get("llm_client", None)
            while llm_client is None:
                await asyncio.sleep(self.sleep_time)
                llm_client = task_space.get("llm_client", None)

            user_id = task_space["user_id"]
            chat_rsp_stream = stream(
                client=llm_client,
                api_keys=None,
                config=self.llm_provider_config,
                model_override=model_name_override if model_name_override else self.sensenova_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": conversation_prompt.replace(
                            "{style_list}", str(style_list) if style_list is not None else ""
                        ),
                    },
                    {"role": "user", "content": conversation_context},
                    *conversation_history,
                    {"role": "user", "content": message},
                ],
                temperature=1,
                max_tokens=4096,
                stream_options={"include_usage": True},
                extra_body=SENSENOVA_EXTRA_BODY,
            )
            input_token_number = 0
            output_token_number = 0
            loop = asyncio.get_event_loop()
            async for chunk in chat_rsp_stream:
                if chunk.text_delta:
                    text_seg = chunk.text_delta
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
                                self.logger.debug(f"request {request_id} first chunk latency: {latency:.2f} seconds")
                                if self.latency_histogram:
                                    self.latency_histogram.labels(adapter=self.name, user_id=user_id).observe(latency)
                        asyncio.gather(*coroutines)
                if chunk.usage:
                    input_token_number += chunk.usage.prompt_tokens
                    output_token_number += chunk.usage.completion_tokens

            if self.input_token_number_histogram:
                self.input_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(input_token_number)
            if self.output_token_number_histogram:
                self.output_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(
                    output_token_number
                )
            return chat_rsp
        except Exception as e:
            msg = f"Error in streaming chat: {e} for request {request_id}"
            msg += f"\n{traceback.format_exc()}"
            self.logger.error(msg)
            return ""

    async def _llm_stream_reject(self, message: str, language: str, request_id: str) -> str:
        try:
            start_time = time.time()
            task_space = self.input_buffer[request_id]["reject_task"]
            dag = task_space["dag"]
            dag_start_time = task_space["dag_start_time"]
            node_name = task_space["node_name"]
            downstream_instances = {
                node.name: node.payload for node in dag.get_node(node_name).downstreams
            }

            bracket_filter = BracketFilter(self.bracket_pairs) if self.enable_bracket_filter else None
            reject_rsp = ""
            first_body_trunk = True
            model_name_override = task_space.get("conversation_model_override")
            reject_prompt = self.agent_prompts["system_reject"]
            llm_client = task_space.get("llm_client", None)
            while llm_client is None:
                await asyncio.sleep(self.sleep_time)
                llm_client = task_space.get("llm_client", None)

            reject_rsp_stream = stream(
                client=llm_client,
                api_keys=None,
                config=self.llm_provider_config,
                model_override=model_name_override if model_name_override else self.sensenova_model_name,
                messages=[
                    {"role": "system", "content": reject_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=1,
                max_tokens=1000,
                stream_options={"include_usage": True},
                extra_body=SENSENOVA_EXTRA_BODY,
            )
            input_token_number = 0
            output_token_number = 0
            user_id = task_space["user_id"]
            loop = asyncio.get_event_loop()
            async for chunk in reject_rsp_stream:
                if chunk.text_delta:
                    text_seg = chunk.text_delta
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
                                self.logger.debug(f"request {request_id} first chunk latency: {latency:.2f} seconds")
                                if self.latency_histogram:
                                    self.latency_histogram.labels(adapter=self.name, user_id=user_id).observe(latency)
                        asyncio.gather(*coroutines)
                if chunk.usage:
                    input_token_number += chunk.usage.prompt_tokens
                    output_token_number += chunk.usage.completion_tokens

            if self.input_token_number_histogram:
                self.input_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(input_token_number)
            if self.output_token_number_histogram:
                self.output_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(
                    output_token_number
                )
            return reject_rsp
        except Exception as e:
            msg = f"Error in streaming reject: {e} for request {request_id}"
            msg += f"\n{traceback.format_exc()}"
            self.logger.error(msg)
            return ""
