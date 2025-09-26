import asyncio
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from ..io.memory.database_memory_client import DatabaseMemoryClient
from ..utils.super import Super
from .memory_adapter import BaseMemoryAdapter
from .memory_processor import MemoryProcessor
from .task_manager import TaskManager, TaskStatus, TaskType


class MemoryManager(Super):
    """Memory manager that integrates task management and memory processing.

    This class coordinates memory operations by managing tasks and processing
    them through the memory processor in an asynchronous manner.
    """

    def __init__(
        self,
        db_client: DatabaseMemoryClient,
        memory_adapter: BaseMemoryAdapter,
        conversation_char_threshold: int = 10000,
        conversation_char_target: int = 8000,
        short_term_length_threshold: int = 20,
        short_term_target_size: int = 10,
        medium_term_length_threshold: int = 10,
        sleep_time: float = 1.0,
        logger_cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the memory manager.

        Args:
            db_client (DatabaseMemoryClient):
                Database client for memory operations.
            memory_adapter (BaseMemoryAdapter):
                Memory adapter for LLM calls.
            conversation_char_threshold (int, optional):
                Character threshold for conversation compression. Defaults to 10000.
            conversation_char_target (int, optional):
                Target character count for conversation compression. Defaults to 8000.
            short_term_length_threshold (int, optional):
                Length threshold for short-term memory compression. Defaults to 20.
            short_term_target_size (int, optional):
                Target size for short-term memory compression. Defaults to 10.
            medium_term_length_threshold (int, optional):
                Length threshold for medium-term memory compression. Defaults to 10.
            sleep_time (float, optional):
                Sleep time between task processing cycles. Defaults to 1.0.
            logger_cfg (Optional[Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.db_client = db_client
        self.task_manager = TaskManager(logger_cfg=logger_cfg)
        self.memory_processor = MemoryProcessor(
            db_client=db_client,
            task_manager=self.task_manager,
            memory_adapter=memory_adapter,
            conversation_char_threshold=conversation_char_threshold,
            conversation_char_target=conversation_char_target,
            short_term_length_threshold=short_term_length_threshold,
            short_term_target_size=short_term_target_size,
            medium_term_length_threshold=medium_term_length_threshold,
            logger_cfg=logger_cfg,
        )

        # Set memory manager to adapter, complete dependency injection
        memory_adapter.set_memory_manager(self)

        # Start task processing loop
        self._task_processing_task = None
        # Delay start task processing loop, wait for event loop to be available
        self._task_processing_started = False
        self.sleep_time = sleep_time

    def _start_task_processing(self):
        """Start the task processing loop.

        This method ensures the task processing loop is started only once and
        handles cases where no event loop is available.
        """
        if not self._task_processing_started:
            try:
                if self._task_processing_task is None or self._task_processing_task.done():
                    self._task_processing_task = asyncio.create_task(self._task_processing_loop())
                    self._task_processing_started = True
            except RuntimeError:
                # If no event loop is running, delay start
                self.logger.warning("No running event loop, task processing will be started later")
                pass

    async def _task_processing_loop(self):
        """Task processing loop that asynchronously handles all pending tasks.

        This method continuously processes pending tasks, cleans up expired and
        completed tasks, and handles failed task retries.
        """
        while True:
            try:
                # Get all pending tasks
                pending_tasks = await self.task_manager.get_pending_tasks()

                # Process all pending tasks concurrently
                if pending_tasks:
                    processing_tasks = []
                    for task in pending_tasks:
                        processing_task = asyncio.create_task(self._process_task(task))
                        processing_tasks.append(processing_task)

                    # Wait for all tasks to complete
                    if processing_tasks:
                        await asyncio.gather(*processing_tasks, return_exceptions=True)

                # Maintenance operations: clean up expired tasks
                await self.task_manager.clean_expired_tasks()

                # Maintenance operations: clean up completed tasks
                await self._clean_completed_tasks()

                # Maintenance operations: handle retry of failed tasks
                await self._handle_failed_tasks_retry()

                # Brief sleep to avoid excessive CPU usage
                await asyncio.sleep(self.sleep_time)

            except Exception as e:
                traceback_str = traceback.format_exc()
                self.logger.error(f"Error in task processing loop: {str(e)}" + "\n" + traceback_str)
                await asyncio.sleep(self.sleep_time * 10)  # Wait longer when error occurs

    async def _process_task(self, task):
        """Process a single task.

        Args:
            task:
                Memory task to process.
        """
        try:
            if task.task_type == TaskType.TURN_LEVEL_SUMMARY:
                await self.memory_processor.process_turn_level_summary(task)
            elif task.task_type == TaskType.CREATE_MEDIUM_TERM_MEMORY:
                await self.memory_processor.process_create_medium_term_memory(task)
            elif task.task_type == TaskType.PROFILE_MEMORY_UPDATE:
                await self.memory_processor.process_profile_memory_update(task)
            elif task.task_type == TaskType.MEDIUM_TERM_COMPRESSION:
                await self.memory_processor.process_medium_term_compression(task)
            elif task.task_type == TaskType.SHORT_TERM_COMPRESSION:
                await self.memory_processor.process_short_term_compression(task)
            elif task.task_type == TaskType.SHORT_TERM_CHAR_COMPRESSION:
                await self.memory_processor.process_short_term_char_compression(task)
            else:
                task.fail(f"Unknown task type: {task.task_type}")
        except Exception as e:
            traceback_str = traceback.format_exc()
            task.fail(f"Task processing error: {str(e)}" + "\n" + traceback_str)

    async def handle_user_entry(
        self,
        character_id: str,
        profile_memory: Optional[Dict[str, Any]] = None,
        cascade_memories: Optional[Dict[str, Any]] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        memory_model_override: Optional[str] = None,
        unix_timestamp: Optional[float] = None,
    ) -> None:
        """Handle user entry behavior.

        Args:
            character_id (str):
                Character identifier.
            profile_memory (Optional[Dict[str, Any]], optional):
                User profile memory data. Defaults to None.
            cascade_memories (Optional[Dict[str, Any]], optional):
                Cascade memories data. Defaults to None.
            api_keys (Optional[Dict[str, Any]], optional):
                API keys for the LLM. Defaults to None.
            memory_model_override (Optional[str], optional):
                Memory model override. Defaults to None.
            unix_timestamp (Optional[float], optional):
                Unix timestamp for the entry. Defaults to None.
        """
        # Ensure task processing loop is started
        self._start_task_processing()

        # 1. Create medium-term memory space for this round
        if unix_timestamp is None:
            unix_timestamp = time.time()
        await self.task_manager.create_task(
            task_type=TaskType.CREATE_MEDIUM_TERM_MEMORY,
            character_id=character_id,
            params={"unix_timestamp": unix_timestamp},
        )

        # 2. Trigger round conversation summary task
        await self.task_manager.create_task(
            task_type=TaskType.TURN_LEVEL_SUMMARY,
            character_id=character_id,
            params={
                "cascade_memories": cascade_memories,
                "api_keys": api_keys,
                "model_override": memory_model_override,
            },
        )

        # 3. Trigger user profile update task
        await self.task_manager.create_task(
            task_type=TaskType.PROFILE_MEMORY_UPDATE,
            character_id=character_id,
            params={
                "profile_memory": profile_memory,
                "cascade_memories": cascade_memories,
                "api_keys": api_keys,
                "model_override": memory_model_override,
            },
        )

        # 4. Check if medium-term memory needs compression
        if await self.memory_processor.need_medium_term_compression(cascade_memories):
            await self.task_manager.create_task(
                task_type=TaskType.MEDIUM_TERM_COMPRESSION,
                character_id=character_id,
                params={"cascade_memories": cascade_memories},
            )

    async def handle_normal_conversation(
        self,
        character_id: str,
        user_input: str,
        profile_memory: Optional[Dict[str, Any]] = None,
        cascade_memories: Optional[Dict[str, Any]] = None,
        relationship: Optional[Tuple[str, int]] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        memory_model_override: Optional[str] = None,
    ) -> None:
        """Handle normal conversation.

        Args:
            character_id (str):
                Character identifier.
            user_input (str):
                User input message.
            profile_memory (Optional[Dict[str, Any]], optional):
                User profile memory data. Defaults to None.
            cascade_memories (Optional[Dict[str, Any]], optional):
                Cascade memories data. Defaults to None.
            relationship (Optional[Tuple[str, int]], optional):
                Current relationship stage and value. Defaults to None.
            api_keys (Optional[Dict[str, Any]], optional):
                API keys for the LLM. Defaults to None.
            memory_model_override (Optional[str], optional):
                Memory model override. Defaults to None.
        """
        # Ensure task processing loop is started
        self._start_task_processing()

        # 1. Check if short-term memory count needs compression
        if await self.memory_processor.need_short_term_compression(cascade_memories):
            await self.task_manager.create_task(
                task_type=TaskType.SHORT_TERM_COMPRESSION,
                character_id=character_id,
                params={
                    "cascade_memories": cascade_memories,
                    "api_keys": api_keys,
                    "model_override": memory_model_override,
                },
            )

        # 2. Check if multi-level memory character count needs compression
        if await self.memory_processor.need_short_term_char_compression(
            cascade_memories, profile_memory, user_input, relationship
        ):
            await self.task_manager.create_task(
                task_type=TaskType.SHORT_TERM_CHAR_COMPRESSION,
                character_id=character_id,
                params={
                    "cascade_memories": cascade_memories,
                    "user_input": user_input,
                    "profile_memory": profile_memory,
                    "relationship": relationship,
                    "api_keys": api_keys,
                    "model_override": memory_model_override,
                },
            )

    async def _clean_completed_tasks(self) -> None:
        """Clean up completed tasks.

        This method removes all tasks that have been marked as completed from
        the task manager.
        """
        try:
            # Get all tasks
            all_tasks = list(self.task_manager.tasks.values())
            for task in all_tasks:
                if task.status.value == "completed":
                    await self.task_manager.remove_task(task.task_id)
        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.error(f"Error cleaning completed tasks: {str(e)}" + "\n" + traceback_str)

    async def _handle_failed_tasks_retry(self) -> None:
        """Handle retry logic for failed tasks.

        This method attempts to retry failed tasks that haven't exceeded their
        maximum retry count, and removes tasks that have failed after reaching
        the maximum retry limit.
        """
        try:
            # Get all tasks
            all_tasks = list(self.task_manager.tasks.values())
            for task in all_tasks:
                if task.status == TaskStatus.FAILED:
                    if task.can_retry():
                        # Retry task
                        task.retry()
                        self.logger.info(f"Retrying failed task {task.task_id}, retry count: {task.retry_count}")
                        # Reprocess task
                        await self._process_task(task)
                    elif task.retry_count >= task.max_retries:
                        # Retry count reached limit, log and delete task
                        self.logger.error(
                            f"Task {task.task_id} failed after {task.retry_count} retries: {task.error_message}"
                        )
                        await self.task_manager.remove_task(task.task_id)
        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.error(f"Error handling failed tasks retry: {str(e)}" + "\n" + traceback_str)

    def is_user_entry(self, user_input: str) -> bool:
        """Check if the input represents user entry behavior.

        Args:
            user_input (str):
                User input message to check.

        Returns:
            bool:
                True if the input is a user entry signal, False otherwise.
        """
        # Collection of entry signals in various languages
        entry_signals = {
            "用户已进入对话",
            "The user entered the chat",
        }

        return user_input.strip() in entry_signals
