import asyncio
import re
import time
import traceback
from abc import abstractmethod
from typing import Any, Dict, Union, cast

import yaml

from ..data_structures.classification import (
    ClassificationChunkBody,
    ClassificationChunkEnd,
    ClassificationChunkStart,
    ClassificationType,
)
from ..data_structures.conversation import (
    ConversationChunkBody,
    ConversationChunkEnd,
    ConversationChunkStart,
    RejectChunkEnd,
    RejectChunkStart,
)
from ..data_structures.process_flow import DAGStatus
from ..data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from ..utils.exception import MissingAPIKeyException, failure_callback
from ..utils.log import setup_logger
from ..utils.streamable import ChunkWithoutStartError, Streamable


class ConversationAdapter(Streamable):
    """Base conversation adapter for large language model integration.

    This abstract base class provides the foundation for conversation adapters,
    handling text streaming, classification, and memory management.
    """

    AVAILABLE_FOR_STREAM = False
    AVAILABLE_FOR_REJECT = False

    def __init__(
        self,
        name: str,
        agent_prompts_file: str,
        proxy_url: Union[None, str] = None,
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the conversation adapter.

        Args:
            name (str):
                The name of the conversation adapter.
            agent_prompts_file (str):
                The path to the agent prompts file.
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
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
        """
        Streamable.__init__(
            self,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            logger_cfg=logger_cfg,
        )
        self.name = name
        self.logger_cfg["logger_name"] = name
        self.logger = setup_logger(**self.logger_cfg)

        with open(agent_prompts_file, "r", encoding="utf-8") as file:
            self.agent_prompts = yaml.safe_load(file)

        self.proxy_url = proxy_url
        self.request_timeout = request_timeout

    async def _handle_start(
        self,
        chunk: Union[TextChunkStart, ClassificationChunkStart],
        cur_time: float,
    ) -> None:
        """Handle the start chunk.

        Args:
            chunk (Union[TextChunkStart, ClassificationChunkStart]):
                The start chunk to process.
            cur_time (float):
                The current timestamp.
        """
        request_id = chunk.request_id
        conf = chunk.dag.conf
        dag_start_time = conf.get("start_time", None)
        character_id = conf.get("character_id", None)
        conversation_model_override = conf.get("conversation_model_override", "")
        api_keys = conf.get("user_settings", {})
        language = conf.get("language", "zh")
        style_list = conf.get("style_list", None)
        user_prompt = conf.get("user_prompt", None)
        profile_memory = conf.get("profile_memory", None)
        cascade_memories = conf.get("cascade_memories", None)
        relationship = conf.get("relationship", None)
        emotion = conf.get("emotion", None)
        memory_adapter = conf.get("memory_adapter", None)
        memory_db_client = conf.get("memory_db_client", None)
        memory_model_override = conf.get("memory_model_override", "")
        self.input_buffer[request_id] = {
            "last_update_time": cur_time,
            "chat_task": dict(),
            "reject_task": dict(),
        }
        chunk_class_str = chunk.__class__.__name__
        callback_bytes_fn = chunk.dag.conf.get("callback_bytes_fn", None)
        for task_type in {"chat_task", "reject_task"}:
            self.input_buffer[request_id][task_type] = {
                "dag_start_time": dag_start_time,
                "start_time": cur_time,
                "language": language,
                "style_list": style_list,
                "character_id": character_id,
                "llm_client": None,
                "conversation_model_override": conversation_model_override,
                "api_keys": api_keys,
                "user_prompt": user_prompt,
                "profile_memory": profile_memory,
                "cascade_memories": cascade_memories,
                "relationship": relationship,
                "emotion": emotion,
                "memory_adapter": memory_adapter,
                "memory_db_client": memory_db_client,
                "memory_model_override": memory_model_override,
                "dag": chunk.dag,
                "node_name": chunk.node_name,
                "start_chunk_classes": set(),
                "text_segments": "",
                "reject_segments": "",
                "classification_result": "",
                "message": "",
                "callback_bytes_fn": callback_bytes_fn,
            }
            self.input_buffer[request_id][task_type]["start_chunk_classes"].add(chunk_class_str)
        task = asyncio.create_task(self._init_llm_client(request_id))
        task.add_done_callback(lambda t: self._handle_init_task_exception(t, request_id))

    def _handle_init_task_exception(self, task: asyncio.Task, request_id: str) -> None:
        """Handle exceptions from the initialization task.

        Args:
            task (asyncio.Task): The completed task.
            request_id (str): The request ID associated with the task.
        """
        if task.exception() is not None:
            exception = task.exception()
            if isinstance(exception, MissingAPIKeyException):
                msg = f"Missing API key during LLM client initialization: {exception}"
                self.logger.error(msg)
                # Create an async task to handle the failure callback
                asyncio.create_task(self._send_failure_callback(msg, request_id))
            else:
                msg = f"Unexpected error during LLM client initialization: {exception}"
                self.logger.error(msg)
                # Create an async task to handle the failure callback for other exceptions too
                asyncio.create_task(self._send_failure_callback(f"Unexpected error: {exception}", request_id))

    async def _send_failure_callback(self, msg: str, request_id: str) -> None:
        """Send failure callback asynchronously.

        Args:
            msg (str): The error message to send.
            request_id (str): The request ID to get the callback function.
        """
        try:
            if request_id in self.input_buffer:
                callback_bytes_fn = None
                for task_type in {"chat_task", "reject_task"}:
                    callback_bytes_fn = self.input_buffer[request_id][task_type].get("callback_bytes_fn")
                    if callback_bytes_fn:
                        await failure_callback(msg, callback_bytes_fn)
                        break
            else:
                self.logger.warning(f"Request {request_id} not found in input buffer")
        except Exception as e:
            self.logger.error(f"Failed to send failure callback for request {request_id}: {e}")

    async def _handle_body(
        self,
        chunk: Union[TextChunkBody, ClassificationChunkBody],
        cur_time: float,
    ) -> None:
        """Handle the body chunk.

        Args:
            chunk (Union[TextChunkBody, ClassificationChunkBody]):
                The body chunk to process.
            cur_time (float):
                The current timestamp.

        Raises:
            ChunkWithoutStartError: If no corresponding start chunk is found.
            ValueError: If unexpected chunk type is received.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        body_chunk_class_str = chunk.__class__.__name__
        start_chunk_class_str = body_chunk_class_str.replace("ChunkBody", "ChunkStart")
        if (
            start_chunk_class_str not in self.input_buffer[request_id]["chat_task"]["start_chunk_classes"]
            and start_chunk_class_str not in self.input_buffer[request_id]["reject_task"]["start_chunk_classes"]
        ):
            msg = f"Start chunk {start_chunk_class_str} for request {request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        if body_chunk_class_str == "TextChunkBody":
            text_chunk = cast(TextChunkBody, chunk)
            self.input_buffer[request_id]["chat_task"]["text_segments"] += text_chunk.text_segment
            self.input_buffer[request_id]["reject_task"]["text_segments"] += text_chunk.text_segment
        elif body_chunk_class_str == "ClassificationChunkBody":
            classification_chunk = cast(ClassificationChunkBody, chunk)
            self.input_buffer[request_id]["chat_task"][
                "classification_result"
            ] = classification_chunk.classification_result
            self.input_buffer[request_id]["chat_task"]["message"] = classification_chunk.message
            self.input_buffer[request_id]["reject_task"][
                "classification_result"
            ] = classification_chunk.classification_result
            self.input_buffer[request_id]["reject_task"]["message"] = classification_chunk.message
        else:
            msg = f"Unexpected body chunk class: {body_chunk_class_str}"
            self.logger.error(msg)
            raise ValueError(msg)

    async def _handle_end(
        self,
        chunk: Union[TextChunkEnd, ClassificationChunkEnd],
        cur_time: float,
    ) -> None:
        """Handle the end chunk.

        Args:
            chunk (Union[TextChunkEnd, ClassificationChunkEnd]):
                The end chunk to process.
            cur_time (float):
                The current timestamp.

        Raises:
            ChunkWithoutStartError: If no corresponding start chunk is found.
            ValueError: If unexpected chunk type is received.
        """
        request_id = chunk.request_id
        end_chunk_class_str = chunk.__class__.__name__
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        start_chunk_class_str = end_chunk_class_str.replace("ChunkEnd", "ChunkStart")
        if (
            start_chunk_class_str not in self.input_buffer[request_id]["chat_task"]["start_chunk_classes"]
            and start_chunk_class_str not in self.input_buffer[request_id]["reject_task"]["start_chunk_classes"]
        ):
            msg = f"Start chunk {start_chunk_class_str} for request {request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        chat_dag = self.input_buffer[request_id]["chat_task"]["dag"]
        reject_dag = self.input_buffer[request_id]["reject_task"]["dag"]
        if chat_dag.status != DAGStatus.RUNNING or reject_dag.status != DAGStatus.RUNNING:
            return
        if end_chunk_class_str == "TextChunkEnd":
            self.logger.debug(f"Received text chunk end for request {request_id}")
            asyncio.create_task(self._stream_chat_task(request_id))
        elif end_chunk_class_str == "ClassificationChunkEnd":
            if self.input_buffer[request_id]["reject_task"]["classification_result"] == ClassificationType.REJECT:
                self.logger.debug(f"Received classification chunk end reject for request {request_id}")
                asyncio.create_task(
                    self._stream_reject_task(self.input_buffer[request_id]["reject_task"]["message"], request_id)
                )
        else:
            msg = f"Unexpected end chunk class: {end_chunk_class_str}"
            self.logger.error(msg)
            raise ValueError(msg)

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    async def _llm_stream_reject(
        self,
        message: str,
        language: str,
        request_id: str,
    ) -> str:
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
        raise NotImplementedError

    @abstractmethod
    async def _init_llm_client(self, request_id: str) -> None:
        """Initialize the LLM client for the given request.

        Args:
            request_id (str):
                The unique identifier for the request.
        """
        raise NotImplementedError

    async def handle_blank_response(
        self,
        request_id: str,
        task_space: dict,
    ) -> str:
        """Handle the blank response.

        Args:
            request_id (str):
                The unique identifier for the request.
            task_space (dict):
                The task space.

        Returns:
            str:
                The handled response.
        """
        chat_rsp = "抱歉没有听清。"

        dag = task_space["dag"]
        node_name = task_space["node_name"]
        dag_node = dag.get_node(node_name)
        downstream_nodes = dag_node.downstreams
        downstream_instances = dict()
        for node in downstream_nodes:
            next_node_name = node.name
            payload = node.payload
            downstream_instances[next_node_name] = payload

        coroutines = list()
        for next_node_name, payload in downstream_instances.items():
            body_trunk = TextChunkBody(
                request_id=request_id,
                text_segment=chat_rsp,
            )
            coroutines.append(payload.feed_stream(body_trunk))
        await asyncio.gather(*coroutines)
        return chat_rsp

    def extract_style_tag(self, text_seg: str) -> Union[None, str]:
        """Extract the style tag from the text segment.

        Args:
            text_seg (str):
                The text segment to extract style from.

        Returns:
            Union[None, str]:
                The extracted style tag or None if not found.
        """
        style_match = re.search(r"<style>(.*?)</style>", text_seg)
        if style_match:
            return style_match.group(1)
        return None

    async def _stream_chat_task(self, request_id: str) -> None:
        """Stream chat task processing.

        Args:
            request_id (str):
                The unique identifier for the request.
        """
        try:
            start_time = time.time()
            task_space = self.input_buffer[request_id]["chat_task"]

            language = task_space["language"]
            character_id = task_space["character_id"]
            profile_memory = task_space["profile_memory"]
            cascade_memories = task_space["cascade_memories"]
            relationship = task_space["relationship"]
            emotion = task_space["emotion"]
            memory_adapter = task_space["memory_adapter"]
            api_keys = task_space["api_keys"]
            memory_model_override = task_space["memory_model_override"]

            # user prompt
            conversation_context = await memory_adapter.build_chat_context(profile_memory, cascade_memories)
            conversation_history = await memory_adapter.build_chat_history(cascade_memories)
            relationship_stage = relationship[0]
            if not task_space["text_segments"]:
                message = "用户的音频有问题，没有说话。"
                self.logger.warning(f"Request {request_id} has no chat message")
            else:
                message = await memory_adapter.build_user_message(
                    task_space["text_segments"], start_time, relationship_stage
                )

            # memory manager
            callback_bytes_fn = task_space.get("callback_bytes_fn")
            asyncio.create_task(
                memory_adapter.handle_conversation(
                    character_id,
                    task_space["text_segments"],
                    profile_memory,
                    cascade_memories,
                    relationship,
                    api_keys,
                    memory_model_override,
                    callback_bytes_fn,
                )
            )

            # append user history
            memory_db_client = task_space["memory_db_client"]
            asyncio.create_task(
                memory_db_client.append_chat_history(
                    character_id=character_id,
                    unix_timestamp=start_time,
                    role="user",
                    content=task_space["text_segments"],
                    relationship=relationship_stage,
                )
            )
            # Prepare downstream start chunk
            dag = task_space["dag"]
            node_name = task_space["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            if len(downstream_nodes) == 0:
                self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
            downstream_instances = dict()
            coroutines = list()
            for node in downstream_nodes:
                next_node_name = node.name
                payload = node.payload
                downstream_instances[next_node_name] = payload
                start_trunk = ConversationChunkStart(
                    request_id=request_id,
                    node_name=next_node_name,
                    dag=dag,
                    client_name=self.name,
                    user_input=message,
                )
                coroutines.append(payload.feed_stream(start_trunk))
            asyncio.gather(*coroutines)

            # Prepare downstream body chunk
            if not task_space["text_segments"]:
                chat_rsp = "你好像没有说话哦。"
                coroutines = list()
                for next_node_name, payload in downstream_instances.items():
                    body_trunk = ConversationChunkBody(
                        request_id=request_id,
                        text_segment=chat_rsp,
                    )
                    coroutines.append(payload.feed_stream(body_trunk))
                asyncio.gather(*coroutines)

            else:
                # stream chat
                chat_rsp = await self._llm_stream_chat(
                    message, conversation_context, conversation_history, language, request_id
                )

            # handle blank response
            if not chat_rsp.strip():
                self.logger.warning(f"Request {request_id} has no chat response")
                chat_rsp = await self.handle_blank_response(request_id, task_space)

            # Prepare downstream end chunk
            coroutines = list()
            aggregator_downstream = False
            for next_node_name, payload in downstream_instances.items():
                payload_class_str = payload.__class__.__name__
                if payload_class_str == "ConversationAggregator":
                    aggregator_downstream = True
                end_trunk = ConversationChunkEnd(
                    request_id=request_id,
                )
                coroutines.append(payload.feed_stream(end_trunk))
            asyncio.gather(*coroutines)
            end_time = time.time()
            if not aggregator_downstream:
                memory_db_client = task_space["memory_db_client"]
                await memory_db_client.append_chat_history(
                    character_id=character_id,
                    unix_timestamp=end_time,
                    role="assistant",
                    content=chat_rsp,
                    **emotion,
                )
            self.logger.debug(f"Chat response: {chat_rsp}")
            msg = f"Streaming chat with the LLM API took {end_time - start_time} seconds"
            msg = msg + f" for request {request_id}"
            self.logger.debug(msg)
            self._cleanup_task_buffer(request_id, "chat_task")

        except Exception as e:
            msg = f"Error in streaming chat: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            dag = task_space["dag"]
            dag.set_status(DAGStatus.FAILED)
            return

    async def _stream_reject_task(self, message: str, request_id: str) -> None:
        """Stream reject task processing.

        Args:
            message (str):
                The message to reject.
            request_id (str):
                The unique identifier for the request.
        """
        try:
            start_time = time.time()
            task_space = self.input_buffer[request_id]["reject_task"]
            language = task_space["language"]
            cascade_memories = task_space["cascade_memories"]
            relationship = task_space["relationship"]

            # user prompt
            memory_adapter = task_space["memory_adapter"]
            relationship_stage = relationship[0]
            user_message = await memory_adapter.build_user_message(message, start_time, relationship_stage)

            # Prepare downstream start chunk
            dag = task_space["dag"]
            node_name = task_space["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            if len(downstream_nodes) == 0:
                self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
            downstream_instances = dict()
            coroutines = list()
            for node in downstream_nodes:
                next_node_name = node.name
                payload = node.payload
                downstream_instances[next_node_name] = payload
                start_trunk = RejectChunkStart(
                    request_id=request_id,
                    node_name=next_node_name,
                    dag=dag,
                )
                coroutines.append(payload.feed_stream(start_trunk))
            asyncio.gather(*coroutines)

            # Prepare downstream body chunk
            reject_rsp = await self._llm_stream_reject(user_message, language, request_id)

            # handle blank response
            if not reject_rsp.strip():
                self.logger.warning(f"Request {request_id} has no reject response")
                reject_rsp = await self.handle_blank_response(request_id, task_space)

            # Prepare downstream end chunk
            coroutines = list()
            for next_node_name, payload in downstream_instances.items():
                end_trunk = RejectChunkEnd(
                    request_id=request_id,
                )
                coroutines.append(payload.feed_stream(end_trunk))
            asyncio.gather(*coroutines)
            self.logger.debug(f"Reject response: {reject_rsp}")
            end_time = time.time()
            msg = f"Streaming reject with the LLM API took {end_time - start_time} seconds"
            msg = msg + f" for request {request_id}"
            self.logger.debug(msg)
            self._cleanup_task_buffer(request_id, "reject_task")
        except Exception as e:
            msg = f"Error in streaming reject: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            dag = task_space["dag"]
            dag.set_status(DAGStatus.FAILED)
            return

    def _cleanup_task_buffer(self, request_id: str, task_type: str) -> None:
        """Clean up the specified task type from the task buffer.

        If only last_update_time remains in the buffer, remove the entire
        request_id entry.

        Args:
            request_id (str):
                The unique identifier for the request.
            task_type (str):
                The task type to remove ("chat_task" or "reject_task").
        """
        if request_id not in self.input_buffer:
            return

        # Remove the specified task type
        if task_type in self.input_buffer[request_id]:
            self.input_buffer[request_id].pop(task_type)

        # If only last_update_time remains, remove the entire request_id
        if len(self.input_buffer[request_id]) == 1:
            self.input_buffer.pop(request_id)


class BracketFilter:
    """Filter for removing bracket content from streaming text.

    This filter can remove content enclosed in various bracket pairs from
    streaming text segments.
    """

    def __init__(self, bracket_pairs: list[tuple[str, str]]):
        """Initialize the bracket filter.

        Args:
            bracket_pairs (list[tuple[str, str]]):
                List of bracket pairs to filter, e.g. [('*', '*'), ('(', ')'), ('[', ']')].
        """
        self.bracket_pairs = bracket_pairs
        self.open_brackets = {}  # Record the opening state of each bracket
        self.buffer = ""  # Cache content that may contain brackets

        # Initialize the opening state
        for open_bracket, close_bracket in bracket_pairs:
            self.open_brackets[open_bracket] = {"close": close_bracket, "active": False}

    def filter_text_segment(self, text_segment: str) -> str:
        """Filter bracket content from text segment.

        Args:
            text_segment (str):
                The input text segment.

        Returns:
            str:
                The filtered text segment.
        """
        if not text_segment:
            return ""

        result = ""
        i = 0

        while i < len(text_segment):
            char = text_segment[i]

            # Check if it is an opening bracket
            found_open = False
            for open_bracket, bracket_info in self.open_brackets.items():
                if text_segment[i:].startswith(open_bracket) and not bracket_info["active"]:
                    # Found an opening bracket
                    bracket_info["active"] = True
                    i += len(open_bracket) - 1
                    found_open = True
                    break

            if found_open:
                i += 1
                continue

            # Check if it is a closing bracket
            found_close = False
            for open_bracket, bracket_info in self.open_brackets.items():
                if bracket_info["active"] and text_segment[i:].startswith(bracket_info["close"]):
                    # Found a closing bracket
                    bracket_info["active"] = False
                    i += len(bracket_info["close"]) - 1
                    found_close = True
                    break

            if found_close:
                i += 1
                continue

            # Check if the current character is in any bracket
            in_brackets = any(info["active"] for info in self.open_brackets.values())

            if not in_brackets:
                result += char

            i += 1

        return result
