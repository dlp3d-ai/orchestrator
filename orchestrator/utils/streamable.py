import asyncio
import time
import traceback
from abc import abstractmethod
from asyncio import Queue, QueueEmpty, QueueFull
from typing import Any, Dict, Union

from ..utils.super import Super


class ChunkWithoutStartError(Exception):
    """Exception for chunk without start."""

    pass


class Streamable(Super):
    AVAILABLE_FOR_STREAM = True
    """Super class for streamable objects."""

    def __init__(
        self,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the streamable object.

        Args:
            queue_size (int, optional):
                The size of the queue.
                Defaults to 100.
            sleep_time (float, optional):
                The sleep time.
                Defaults to 0.01.
            clean_interval (float, optional):
                The interval to clean the expired requests.
                Defaults to 10.0.
            expire_time (float, optional):
                The time to expire the request.
                Defaults to 120.0.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
        """
        super().__init__(logger_cfg=logger_cfg)

        self.queue = Queue(maxsize=queue_size)
        self.sleep_time = sleep_time

        # keys of input_buffer: request_id
        self.input_buffer: Dict[str, Dict[str, Any]] = dict()
        self.running: bool = False
        self.last_clean_time: float = 0.0
        self.clean_interval: float = clean_interval
        self.expire_time: float = expire_time

    @abstractmethod
    async def _handle_start(self, chunk: Any, cur_time: float) -> None:
        """Handle the start chunk."""
        pass

    @abstractmethod
    async def _handle_body(self, chunk: Any, cur_time: float) -> None:
        """Handle the body chunk."""
        pass

    @abstractmethod
    async def _handle_end(self, chunk: Any, cur_time: float) -> None:
        """Handle the end chunk."""
        pass

    async def interrupt(self) -> None:
        """Interrupt the streamable object."""
        self.running = False

    async def feed_stream(self, chunk: Any) -> None:
        """Feed the chunk to the conversation adapter.

        Args:
            chunk (Any):
                The chunk instance,
                which is one of the following:
                - ChunkStart
                - ChunkBody
                - ChunkEnd
        """
        if not self.__class__.AVAILABLE_FOR_STREAM:
            msg = f"Class {self.__class__.__name__} is not available for stream"
            self.logger.error(msg)
            raise RuntimeError(msg)
        try:
            self.queue.put_nowait(chunk)
        except QueueFull as e:
            msg = "The queue is full"
            self.logger.error(msg)
            raise e

    async def _one_loop(self, cur_time: float) -> bool:
        """One loop of the streamable object.

        Args:
            cur_time (float):
                The current time.

        Returns:
            bool:
                Whether to continue the loop.
                For class Streamable, it's always True.
        """
        try:
            input_chunk = self.queue.get_nowait()
        except QueueEmpty:
            # Clean expired requests
            if cur_time - self.last_clean_time > self.clean_interval:
                keys = list(self.input_buffer.keys())
                keys_to_remove = []
                for key in keys:
                    last_update_time = self.input_buffer[key]["last_update_time"]
                    if cur_time - last_update_time > self.expire_time:
                        keys_to_remove.append(key)
                if len(keys_to_remove) > 0:
                    msg = f"Cleaned {len(keys_to_remove)} expired requests: {keys_to_remove}"
                    self.logger.warning(msg)
                    for key in keys_to_remove:
                        self.input_buffer.pop(key)
                self.last_clean_time = cur_time
            else:
                await asyncio.sleep(self.sleep_time)
            return True
        if isinstance(input_chunk, dict):
            if "chunk_type" not in input_chunk:
                msg = f"Missing chunk type in input: {input_chunk}"
                self.logger.error(msg)
                return True
            type_str = input_chunk["chunk_type"]
        else:
            type_str = input_chunk.chunk_type
        if type_str == "start":
            await self._handle_start(input_chunk, cur_time)
            return True
        elif type_str == "body":
            await self._handle_body(input_chunk, cur_time)
            return True
        elif type_str == "end":
            await self._handle_end(input_chunk, cur_time)
            return True
        else:
            msg = f"Received an unknown chunk type: {type_str}"
            self.logger.error(msg)
            return True

    async def run(self) -> None:
        """Run the stream-able text profile instance in a loop, fetch text
        chunks from queue, and log them."""
        await self._run_precheck()
        self.running = True
        while self.running:
            cur_time = time.time()
            try:
                continue_loop = await self._one_loop(cur_time)
                if not continue_loop:
                    self.running = False
                    break
            except Exception as e:
                traceback_str = traceback.format_exc()
                msg = f"Error in streamable loop: {e}"
                msg += f"\n{traceback_str}"
                self.logger.error(msg)

    async def _run_precheck(self) -> None:
        """Run the precheck for the conversation adapter."""
        if not self.__class__.AVAILABLE_FOR_STREAM:
            msg = "This conversation adapter is not available for stream"
            self.logger.error(msg)
            raise ValueError(msg)
        if self.running:
            msg = "This conversation adapter is already running"
            self.logger.error(msg)
            raise RuntimeError(msg)
