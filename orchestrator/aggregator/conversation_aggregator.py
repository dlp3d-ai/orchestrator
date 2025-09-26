import asyncio
from typing import Union, cast

from ..data_structures.classification import (
    ClassificationChunkBody,
    ClassificationChunkEnd,
    ClassificationChunkStart,
    ClassificationType,
)
from ..data_structures.conversation import (
    ClassifiedTextChunkBody,
    ClassifiedTextChunkEnd,
    ClassifiedTextChunkStart,
    ConversationChunkBody,
    ConversationChunkEnd,
    ConversationChunkStart,
    RejectChunkBody,
    RejectChunkEnd,
    RejectChunkStart,
)
from ..data_structures.process_flow import DAGStatus
from ..utils.streamable import ChunkWithoutStartError, Streamable


class ConversationAggregator(Streamable):
    """ConversationAggregator receives chunks from positive/negative
    conversation nodes and decides which conversation to send downstream based
    on classification results.

    This aggregator handles three types of input streams:
    1. ClassificationChunk - determines whether to accept, reject, or leave the conversation
    2. ConversationChunk - contains the actual conversation content
    3. RejectChunk - contains rejection messages

    The aggregator waits for classification results and then forwards the appropriate
    conversation content (either accepted conversation or rejection message) to downstream nodes.
    It also manages chat history storage in the memory database.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the ConversationAggregator.

        Args:
            *args: Variable length argument list passed to parent Streamable class.
            **kwargs: Arbitrary keyword arguments passed to parent Streamable class.
        """
        super().__init__(*args, **kwargs)

    async def _handle_start(
        self,
        chunk: Union[ConversationChunkStart, RejectChunkStart, ClassificationChunkStart],
        cur_time: float,
    ) -> None:
        """Handle the start chunk from conversation, reject, or classification
        streams.

        Args:
            chunk (Union[ConversationChunkStart, RejectChunkStart, ClassificationChunkStart]):
                The start chunk containing request metadata and configuration.
            cur_time (float): Current timestamp for tracking request timing.
        """
        request_id = chunk.request_id
        conf = chunk.dag.conf
        character_id = conf["character_id"]
        cascade_memories = conf["cascade_memories"]
        emotion = conf["emotion"]
        dag_start_time = conf["start_time"]
        memory_adapter = conf["memory_adapter"]
        memory_db_client = conf["memory_db_client"]

        if request_id not in self.input_buffer:
            self.input_buffer[request_id] = dict(
                start_chunk_classes=set(),
                end_chunk_classes=set(),
                last_update_time=cur_time,
                dag_start_time=dag_start_time,
                dag=chunk.dag,
                node_name=chunk.node_name,
                character_id=character_id,
                cascade_memories=cascade_memories,
                emotion=emotion,
                memory_adapter=memory_adapter,
                memory_db_client=memory_db_client,
                client_name=None,
                user_input=None,
                downstream_warned=False,
                stream_start_initiated=False,
                stream_start_sent=False,
                classification_result=None,
                full_chat_segments="",
                full_reject_segments="",
                chat_segments="",
                reject_segments="",
                classification_ended=False,
                conversation_ended=False,
                reject_ended=False,
                pending_text_segments_tasks=0,
                style=None,
            )
        chunk_class_str = chunk.__class__.__name__
        self.input_buffer[request_id]["start_chunk_classes"].add(chunk_class_str)
        self.logger.debug(f"Received start chunk {chunk_class_str} for request {request_id}")

        if chunk_class_str == "ConversationChunkStart":
            conversation_chunk = cast(ConversationChunkStart, chunk)
            self.input_buffer[request_id]["client_name"] = conversation_chunk.client_name
            self.input_buffer[request_id]["user_input"] = conversation_chunk.user_input

        asyncio.create_task(self._send_stream_start_task(request_id))

    async def _handle_body(
        self,
        chunk: Union[ConversationChunkBody, RejectChunkBody, ClassificationChunkBody],
        cur_time: float,
    ) -> None:
        """Handle the body chunk from conversation, reject, or classification
        streams.

        Args:
            chunk (Union[ConversationChunkBody, RejectChunkBody, ClassificationChunkBody]):
                The body chunk containing content data or classification results.
            cur_time (float): Current timestamp for tracking request timing.
        """
        request_id = chunk.request_id
        chunk_class_str = chunk.__class__.__name__

        if request_id not in self.input_buffer:
            msg = (
                f"Request {request_id} not found in input buffer, "
                + f"but received a body message of class {chunk_class_str}."
            )
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)

        start_chunk_class_str = chunk_class_str.replace("ChunkBody", "ChunkStart")
        if start_chunk_class_str not in self.input_buffer[request_id]["start_chunk_classes"]:
            msg = (
                f"Start chunk {start_chunk_class_str} for request {request_id} not found in input buffer, "
                + "but received a body message."
            )
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)

        dag = self.input_buffer[request_id]["dag"]
        self.input_buffer[request_id]["last_update_time"] = cur_time

        if chunk_class_str == "ClassificationChunkBody":
            classification_chunk = cast(ClassificationChunkBody, chunk)
            self.input_buffer[request_id]["classification_result"] = classification_chunk.classification_result

            asyncio.create_task(self._send_stream_start_task(request_id))

            if classification_chunk.classification_result == ClassificationType.REJECT:
                msg = f"Received reject signal from classification, request id: {request_id}"
                self.logger.info(msg)
                if self.input_buffer[request_id]["reject_segments"]:
                    while not self.input_buffer[request_id]["stream_start_sent"]:
                        await asyncio.sleep(self.sleep_time)
                    self.input_buffer[request_id]["pending_text_segments_tasks"] += 1
                    asyncio.create_task(
                        self._send_text_segments(request_id, self.input_buffer[request_id]["reject_segments"])
                    )
                    self.input_buffer[request_id]["reject_segments"] = ""
            elif classification_chunk.classification_result == ClassificationType.LEAVE:
                msg = f"Received leave signal from classification, request id: {request_id}"
                self.logger.info(msg)
            elif classification_chunk.classification_result == ClassificationType.ACCEPT:
                msg = f"Received accept signal from classification, request id: {request_id}"
                self.logger.info(msg)
                if self.input_buffer[request_id]["chat_segments"]:
                    while not self.input_buffer[request_id]["stream_start_sent"]:
                        await asyncio.sleep(self.sleep_time)
                    self.input_buffer[request_id]["pending_text_segments_tasks"] += 1
                    asyncio.create_task(
                        self._send_text_segments(request_id, self.input_buffer[request_id]["chat_segments"])
                    )
                    self.input_buffer[request_id]["chat_segments"] = ""
            else:
                dag.set_status(DAGStatus.FAILED)
                msg = (
                    f"Received unknown classification result={classification_chunk.classification_result} "
                    + f"from classification, request id: {request_id}"
                )
                self.logger.error(msg)
                self.input_buffer.pop(request_id, None)
                raise ValueError(msg)

        elif chunk_class_str == "ConversationChunkBody":
            conversation_chunk = cast(ConversationChunkBody, chunk)
            self.input_buffer[request_id]["style"] = conversation_chunk.style
            msg = f"Received text chunk body, request id: {request_id}, text segment: {conversation_chunk.text_segment}, style: {conversation_chunk.style}"
            self.logger.debug(msg)

            self.input_buffer[request_id]["full_chat_segments"] += conversation_chunk.text_segment

            if self.input_buffer[request_id]["classification_result"] == ClassificationType.ACCEPT:
                if conversation_chunk.text_segment:
                    while not self.input_buffer[request_id]["stream_start_sent"]:
                        await asyncio.sleep(self.sleep_time)
                    self.input_buffer[request_id]["pending_text_segments_tasks"] += 1
                    asyncio.create_task(self._send_text_segments(request_id, conversation_chunk.text_segment))
            else:
                self.input_buffer[request_id]["chat_segments"] += conversation_chunk.text_segment

        elif chunk_class_str == "RejectChunkBody":
            reject_chunk = cast(RejectChunkBody, chunk)
            msg = f"Received reject chunk body, request id: {request_id}, reject segment: {reject_chunk.text_segment}"
            self.logger.debug(msg)

            self.input_buffer[request_id]["full_reject_segments"] += reject_chunk.text_segment

            if self.input_buffer[request_id]["classification_result"] == ClassificationType.REJECT:
                if reject_chunk.text_segment:
                    while not self.input_buffer[request_id]["stream_start_sent"]:
                        await asyncio.sleep(self.sleep_time)
                    self.input_buffer[request_id]["pending_text_segments_tasks"] += 1
                    asyncio.create_task(self._send_text_segments(request_id, reject_chunk.text_segment))
            else:
                self.input_buffer[request_id]["reject_segments"] += reject_chunk.text_segment

        else:
            dag.set_status(DAGStatus.FAILED)
            msg = f"Received unknown body message of class {chunk_class_str}, request id: {request_id}"
            self.logger.error(msg)
            raise ValueError(msg)

    async def _handle_end(
        self,
        chunk: Union[ConversationChunkEnd, RejectChunkEnd, ClassificationChunkEnd],
        cur_time: float,
    ) -> None:
        """Handle the end chunk from conversation, reject, or classification
        streams.

        Args:
            chunk (Union[ConversationChunkEnd, RejectChunkEnd, ClassificationChunkEnd]):
                The end chunk indicating completion of a stream.
            cur_time (float): Current timestamp for tracking request timing.
        """
        request_id = chunk.request_id
        chunk_class_str = chunk.__class__.__name__
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        start_chunk_class_str = chunk_class_str.replace("ChunkEnd", "ChunkStart")
        if start_chunk_class_str not in self.input_buffer[request_id]["start_chunk_classes"]:
            msg = (
                f"Start chunk {start_chunk_class_str} for request {request_id} not found in input buffer, "
                + "but received an end message."
            )
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)

        start_time = self.input_buffer[request_id]["dag_start_time"]
        time_diff = cur_time - float(start_time)
        msg = f"Aggregator received end chunk {chunk_class_str} for request {request_id} in {time_diff:.3f} seconds"
        self.logger.debug(msg)

        dag = self.input_buffer[request_id]["dag"]

        if chunk_class_str == "ClassificationChunkEnd":
            self.input_buffer[request_id]["classification_ended"] = True
        elif chunk_class_str == "ConversationChunkEnd":
            self.input_buffer[request_id]["conversation_ended"] = True
        elif chunk_class_str == "RejectChunkEnd":
            self.input_buffer[request_id]["reject_ended"] = True
        else:
            dag.set_status(DAGStatus.FAILED)
            msg = f"Received unknown end message of class {chunk_class_str}, request id: {request_id}"
            self.logger.error(msg)
            raise ValueError(msg)

        classification_result = self.input_buffer[request_id]["classification_result"]

        if classification_result == ClassificationType.REJECT:
            required_ends = {"classification_ended", "conversation_ended", "reject_ended"}
        else:
            required_ends = {"classification_ended", "conversation_ended"}

        all_ended = all(self.input_buffer[request_id][end_key] for end_key in required_ends)

        if all_ended:
            while not self.input_buffer[request_id]["stream_start_sent"]:
                await asyncio.sleep(self.sleep_time)

            while self.input_buffer[request_id]["pending_text_segments_tasks"] > 0:
                await asyncio.sleep(self.sleep_time)

            dag = self.input_buffer[request_id]["dag"]
            dag_node = dag.get_node(self.input_buffer[request_id]["node_name"])
            for node in dag_node.downstreams:
                payload = node.payload
                end_chunk = ClassifiedTextChunkEnd(
                    request_id=request_id,
                )
                asyncio.create_task(payload.feed_stream(end_chunk))

            if classification_result == ClassificationType.REJECT:
                content = self.input_buffer[request_id]["full_reject_segments"]
            else:
                content = self.input_buffer[request_id]["full_chat_segments"]

            character_id = self.input_buffer[request_id]["character_id"]
            emotion = self.input_buffer[request_id]["emotion"]

            memory_db_client = self.input_buffer[request_id]["memory_db_client"]
            asyncio.create_task(
                memory_db_client.append_chat_history(
                    character_id=character_id,
                    unix_timestamp=cur_time,
                    role="assistant",
                    content=content,
                    **emotion,
                )
            )
            self.input_buffer.pop(request_id)

    async def _send_text_segments(self, request_id: str, text_segment: str) -> None:
        """Send text segments to downstream nodes.

        Args:
            request_id (str): The request identifier.
            text_segment (str): The text content to send downstream.
        """
        dag = self.input_buffer[request_id]["dag"]
        dag_node = dag.get_node(self.input_buffer[request_id]["node_name"])
        coroutines = list()
        for node in dag_node.downstreams:
            payload = node.payload
            body_chunk = ClassifiedTextChunkBody(
                request_id=request_id,
                text_segment=text_segment,
                style=self.input_buffer[request_id]["style"],
            )
            coroutines.append(payload.feed_stream(body_chunk))
        await asyncio.gather(*coroutines)
        self.input_buffer[request_id]["pending_text_segments_tasks"] -= 1

    async def _send_stream_start_task(self, request_id: str) -> None:
        """Send stream start chunk to downstream nodes when all required data
        is available.

        Args:
            request_id (str): The request identifier.
        """
        if self.input_buffer[request_id]["stream_start_initiated"]:
            return

        if (
            self.input_buffer[request_id]["classification_result"] is not None
            and self.input_buffer[request_id]["client_name"] is not None
            and self.input_buffer[request_id]["user_input"] is not None
        ):
            self.input_buffer[request_id]["stream_start_initiated"] = True

            dag = self.input_buffer[request_id]["dag"]
            node_name = self.input_buffer[request_id]["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            downstream_warned = self.input_buffer[request_id]["downstream_warned"]
            if len(downstream_nodes) == 0 and not downstream_warned:
                self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
                self.input_buffer[request_id]["downstream_warned"] = True
                return
            coroutines = list()
            for node in downstream_nodes:
                next_node_name = node.name
                payload = node.payload
                start_chunk = ClassifiedTextChunkStart(
                    request_id=request_id,
                    node_name=next_node_name,
                    dag=dag,
                    classification_result=self.input_buffer[request_id]["classification_result"],
                    client_name=self.input_buffer[request_id]["client_name"],
                    user_input=self.input_buffer[request_id]["user_input"],
                )
                coroutines.append(payload.feed_stream(start_chunk))
            await asyncio.gather(*coroutines)
            self.input_buffer[request_id]["stream_start_sent"] = True
