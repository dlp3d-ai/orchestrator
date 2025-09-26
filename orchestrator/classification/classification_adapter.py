import asyncio
import time
import traceback
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import httpx

from ..data_structures.classification import (
    ClassificationChunkBody,
    ClassificationChunkEnd,
    ClassificationChunkStart,
    ClassificationType,
)
from ..data_structures.process_flow import DAGStatus
from ..data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from ..utils.log import setup_logger
from ..utils.streamable import ChunkWithoutStartError, Streamable
from .prompts import CLASSIFICATION_PROMPT_EN, CLASSIFICATION_PROMPT_ZH, RESPONSE_FORMAT, TAG_PROMPT


class ClassificationAdapter(Streamable):
    """Base class for text classification adapters using Large Language Models.

    This adapter provides a streaming interface for classifying user text input
    into different response types (accept, reject, leave) based on motion
    keywords and conversation context. It supports both Chinese and English
    language processing.
    """

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None] = None,
        proxy_url: Union[None, str] = None,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the classification adapter.

        Args:
            name (str):
                The name of the classification adapter.
            motion_keywords (Union[str, list[str], None]):
                The motion keywords.
                Defaults to None.
            proxy_url (Union[None, str], optional):
                The proxy URL for the classification API.
                Defaults to None.
            queue_size (int, optional):
                The size of the processing queue.
                Defaults to 100.
            sleep_time (float, optional):
                Sleep time between processing cycles.
                Defaults to 0.01.
            clean_interval (float, optional):
                The interval to clean expired requests.
                Defaults to 10.0.
            expire_time (float, optional):
                The time to expire requests.
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
        self.proxy_url = proxy_url

        if isinstance(motion_keywords, str):
            response = httpx.get(motion_keywords, verify=False)
            if response.status_code == 200:
                motion_kws = response.json()["motion_keywords"]
        elif isinstance(motion_keywords, list):
            motion_kws = motion_keywords
        elif motion_keywords is None:
            msg = "Motion keywords is None, using empty list."
            self.logger.warning(msg)
            motion_kws = []
        else:
            msg = f"Invalid motion keywords type: {type(motion_keywords)}"
            self.logger.error(msg)
            raise TypeError(msg)
        self.motion_kws: List[str] = motion_kws
        self.logger.info(f"Loaded {len(self.motion_kws)} motion keywords")

    @abstractmethod
    async def _init_llm_client(self, request_id: str) -> None:
        """Initialize the LLM client.

        Args:
            request_id (str):
                The request id.
        """
        raise NotImplementedError

    @abstractmethod
    async def classify(
        self,
        request_id: str,
        prompt: str,
        text: str,
        response_format: Optional[Dict[str, Any]] = None,
        tag_prompt: Optional[str] = None,
    ) -> ClassificationType:
        """Classify the required response type according to user's text input,
        based on LLM.

        Args:
            request_id (str):
                The request id.
            prompt (str):
                The prompt to classify the text.
            text (str):
                The text to classify.
            response_format (Optional[Dict[str, Any]], optional):
                The response format.
                Defaults to None.
            tag_prompt (Optional[str], optional):
                The tag prompt.
                Defaults to None.

        Returns:
            ClassificationType:
                The classification result.
        """
        raise NotImplementedError

    async def _handle_start(self, chunk: TextChunkStart, cur_time: float) -> None:
        """Handle the start chunk for text classification.

        Args:
            chunk (TextChunkStart):
                The start chunk containing request information.
            cur_time (float):
                Current timestamp.
        """
        request_id = chunk.request_id
        language = chunk.dag.conf.get("language", "zh")
        classification_model_override = chunk.dag.conf.get("classification_model_override", "")
        api_keys = chunk.dag.conf.get("user_settings", {})
        self.input_buffer[request_id] = {
            "start_time": cur_time,
            "last_update_time": cur_time,
            "dag": chunk.dag,
            "node_name": chunk.node_name,
            "language": language,
            "llm_client": None,
            "classification_model_override": classification_model_override,
            "api_keys": api_keys,
            "text_segments": "",
        }
        asyncio.create_task(self._init_llm_client(request_id))

    async def _handle_body(self, chunk: TextChunkBody, cur_time: float) -> None:
        """Handle the body chunk for text classification.

        Args:
            chunk (TextChunkBody):
                The body chunk containing text segments.
            cur_time (float):
                Current timestamp.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        self.input_buffer[request_id]["text_segments"] += chunk.text_segment

    async def _handle_end(self, chunk: TextChunkEnd, cur_time: float) -> None:
        """Handle the end chunk for text classification.

        Args:
            chunk (TextChunkEnd):
                The end chunk indicating completion of text input.
            cur_time (float):
                Current timestamp.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        dag = self.input_buffer[request_id]["dag"]
        if dag.status == DAGStatus.RUNNING:
            asyncio.create_task(self._classify_task(request_id))

    async def _classify_task(self, request_id: str) -> None:
        """Classify the required response type according to user's text input.

        This method processes the accumulated text segments and classifies them
        using the configured LLM. It handles both Chinese and English text
        and sends the classification result to downstream nodes.

        Args:
            request_id (str):
                The unique identifier for the classification request.
        """
        try:
            process_start_time = time.time()
            text = self.input_buffer[request_id]["text_segments"]
            if not text.strip():
                # if text_segments is empty, then the user is not speaking, return ACCEPT
                result = ClassificationType.ACCEPT
            else:
                try:
                    language = self.input_buffer[request_id]["language"]
                    if language == "zh":
                        prompt = CLASSIFICATION_PROMPT_ZH
                    else:
                        prompt = CLASSIFICATION_PROMPT_EN
                    prompt = prompt.format(motions=self.motion_kws)
                    result = await self.classify(request_id, prompt, text, RESPONSE_FORMAT, TAG_PROMPT)
                except Exception:
                    self.logger.error(
                        f"Error in classifying request {request_id}, text: {text}, using ACCEPT as default."
                    )
                    result = ClassificationType.ACCEPT
            process_end_time = time.time()
            process_time = process_end_time - process_start_time
            self.logger.debug(f"Request {request_id} processed in {process_time} seconds, result: {result}")
            # sending to downstream nodes
            dag = self.input_buffer[request_id]["dag"]
            node_name = self.input_buffer[request_id]["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            if len(downstream_nodes) == 0:
                self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
            for node in downstream_nodes:
                next_node_name = node.name
                payload = node.payload
                start_chunk = ClassificationChunkStart(request_id=request_id, node_name=next_node_name, dag=dag)
                await payload.feed_stream(start_chunk)
                body_chunk = ClassificationChunkBody(request_id=request_id, message=text, classification_result=result)
                await payload.feed_stream(body_chunk)
                end_chunk = ClassificationChunkEnd(request_id=request_id)
                await payload.feed_stream(end_chunk)
            self.input_buffer.pop(request_id)
        except Exception as e:
            msg = f"Error in streaming classification: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.FAILED)
            return
