import asyncio
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..utils.super import Super


class TaskStatus(Enum):
    """Task status enumeration.

    Defines the possible states a memory task can be in.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(Enum):
    """Task type enumeration.

    Defines the different types of memory tasks that can be created.
    """

    CREATE_MEDIUM_TERM_MEMORY = "create_medium_term_memory"
    TURN_LEVEL_SUMMARY = "turn_level_summary"
    PROFILE_MEMORY_UPDATE = "profile_memory_update"
    MEDIUM_TERM_COMPRESSION = "medium_term_compression"
    SHORT_TERM_COMPRESSION = "short_term_compression"
    SHORT_TERM_CHAR_COMPRESSION = "short_term_char_compression"


class TaskCategory(Enum):
    """Task category enumeration, classified by database tables.

    Categorizes tasks based on which database table they write to.
    """

    MEDIUM_TERM_MEMORY = "medium_term_memory"  # Medium-term memory table
    LONG_TERM_MEMORY = "long_term_memory"  # Long-term memory table
    PROFILE_MEMORY = "profile_memory"  # Profile memory table
    MEDIUM_TERM_MEMORY_CREATION = "medium_term_memory_creation"  # Medium-term memory creation table


# Mapping from task type to task category
TASK_TYPE_TO_CATEGORY = {
    TaskType.CREATE_MEDIUM_TERM_MEMORY: TaskCategory.MEDIUM_TERM_MEMORY_CREATION,
    TaskType.TURN_LEVEL_SUMMARY: TaskCategory.MEDIUM_TERM_MEMORY,
    TaskType.PROFILE_MEMORY_UPDATE: TaskCategory.PROFILE_MEMORY,
    TaskType.MEDIUM_TERM_COMPRESSION: TaskCategory.LONG_TERM_MEMORY,
    TaskType.SHORT_TERM_COMPRESSION: TaskCategory.MEDIUM_TERM_MEMORY,
    TaskType.SHORT_TERM_CHAR_COMPRESSION: TaskCategory.MEDIUM_TERM_MEMORY,
}


class MemoryTask:
    """Memory task class.

    Represents a single memory-related task with its state, parameters, and
    execution tracking.
    """

    def __init__(
        self, task_id: str, task_type: TaskType, character_id: str, params: Dict[str, Any], max_retries: int = 3
    ):
        """Initialize a memory task.

        Args:
            task_id (str):
                Unique identifier for the task.
            task_type (TaskType):
                Type of the memory task.
            character_id (str):
                Character identifier associated with the task.
            params (Dict[str, Any]):
                Task parameters and data.
            max_retries (int, optional):
                Maximum number of retry attempts. Defaults to 3.
        """
        self.task_id = task_id
        self.task_type = task_type
        self.character_id = character_id
        self.params = params
        self.status = TaskStatus.PENDING
        self.retry_count = 0
        self.max_retries = max_retries
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error_message: Optional[str] = None
        self.result: Optional[Any] = None

    @property
    def category(self) -> TaskCategory:
        """Get the task category.

        Returns:
            TaskCategory:
                The category this task belongs to.
        """
        return TASK_TYPE_TO_CATEGORY[self.task_type]

    def start(self):
        """Start the task.

        Marks the task as running and records the start time.
        """
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result: Any = None):
        """Complete the task.

        Args:
            result (Any, optional):
                Task result data. Defaults to None.
        """
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result

    def fail(self, error_message: str):
        """Mark the task as failed.

        Args:
            error_message (str):
                Error message describing the failure.
        """
        self.status = TaskStatus.FAILED
        self.completed_at = time.time()
        self.error_message = error_message

    def can_retry(self) -> bool:
        """Check if the task can be retried.

        Returns:
            bool:
                True if the task can be retried, False otherwise.
        """
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries

    def retry(self):
        """Retry the task.

        Returns:
            bool:
                True if retry was successful, False otherwise.
        """
        if self.can_retry():
            self.retry_count += 1
            self.status = TaskStatus.PENDING
            self.started_at = None
            self.completed_at = None
            self.error_message = None
            self.result = None
            return True
        return False


class TaskManager(Super):
    """Task manager for handling memory tasks.

    This class manages the lifecycle of memory tasks, including creation,
    execution tracking, and cleanup.
    """

    def __init__(
        self,
        logger_cfg: Optional[Dict[str, Any]] = None,
        clean_interval: float = 60.0,
        expire_time: float = 3600.0,
        sleep_time: float = 1.0,
    ):
        """Initialize the task manager.

        Args:
            logger_cfg (Optional[Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
            clean_interval (float, optional):
                Interval for cleaning expired tasks in seconds. Defaults to 60.0.
            expire_time (float, optional):
                Time after which completed tasks expire in seconds. Defaults to 3600.0.
            sleep_time (float, optional):
                Sleep time between operations in seconds. Defaults to 1.0.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.tasks: Dict[str, MemoryTask] = {}
        self.clean_interval = clean_interval
        self.expire_time = expire_time
        self.sleep_time = sleep_time
        self._last_clean_time = 0
        self._lock = asyncio.Lock()

    async def has_running_task_in_category(self, character_id: str, task_category: TaskCategory) -> bool:
        """Check if a character has a running task in the specified category.

        Args:
            character_id (str):
                Character identifier to check.
            task_category (TaskCategory):
                Task category to check for.

        Returns:
            bool:
                True if there is a running task in the category, False otherwise.
        """
        for task in self.tasks.values():
            if (
                task.character_id == character_id
                and task.category == task_category
                and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
            ):
                return True
        return False

    async def create_task(
        self, task_type: TaskType, character_id: str, params: Dict[str, Any], max_retries: int = 3
    ) -> Optional[str]:
        """Create a new task, returns None if there is already a running task
        in the same category.

        Args:
            task_type (TaskType):
                Type of task to create.
            character_id (str):
                Character identifier for the task.
            params (Dict[str, Any]):
                Task parameters and data.
            max_retries (int, optional):
                Maximum number of retry attempts. Defaults to 3.

        Returns:
            Optional[str]:
                Task ID if created successfully, None if skipped due to existing task.
        """
        async with self._lock:
            # Get task category
            task_category = TASK_TYPE_TO_CATEGORY[task_type]

            # Check if there are running tasks of the same category, skip if yes
            if await self.has_running_task_in_category(character_id, task_category):
                self.logger.info(
                    f"Skipping task creation for character {character_id}, "
                    f"task type {task_type.value}. "
                    f"There's already a running task in category {task_category.value}."
                )
                return None

            # Create new task
            task_id = str(uuid.uuid4())
            task = MemoryTask(
                task_id=task_id, task_type=task_type, character_id=character_id, params=params, max_retries=max_retries
            )
            self.tasks[task_id] = task
            self.logger.info(f"Created task {task_id} of type {task_type.value} for character {character_id}")
            return task_id

    async def get_task(self, task_id: str) -> Optional[MemoryTask]:
        """Get a task by its ID.

        Args:
            task_id (str):
                Task identifier.

        Returns:
            Optional[MemoryTask]:
                The task if found, None otherwise.
        """
        return self.tasks.get(task_id)

    async def get_tasks_by_character(self, character_id: str) -> List[MemoryTask]:
        """Get all tasks for a specific character.

        Args:
            character_id (str):
                Character identifier.

        Returns:
            List[MemoryTask]:
                List of tasks for the character.
        """
        return [task for task in self.tasks.values() if task.character_id == character_id]

    async def get_pending_tasks(self) -> List[MemoryTask]:
        """Get all pending tasks.

        Returns:
            List[MemoryTask]:
                List of tasks with pending status.
        """
        return [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]

    async def get_failed_tasks(self) -> List[MemoryTask]:
        """Get all failed tasks.

        Returns:
            List[MemoryTask]:
                List of tasks with failed status.
        """
        return [task for task in self.tasks.values() if task.status == TaskStatus.FAILED]

    async def remove_task(self, task_id: str) -> bool:
        """Remove a task.

        Args:
            task_id (str):
                Task identifier to remove.

        Returns:
            bool:
                True if task was removed, False if not found.
        """
        async with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                self.logger.info(f"Removed task {task_id}")
                return True
            return False

    async def clean_expired_tasks(self):
        """Clean up expired tasks.

        Removes tasks that have been completed or failed and have exceeded the
        expiration time.
        """
        current_time = time.time()
        if current_time - self._last_clean_time < self.clean_interval:
            return

        async with self._lock:
            expired_tasks = []
            for task_id, task in self.tasks.items():
                if (
                    task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                    and task.completed_at
                    and current_time - task.completed_at > self.expire_time
                ):
                    expired_tasks.append(task_id)

            for task_id in expired_tasks:
                del self.tasks[task_id]

            if expired_tasks:
                self.logger.info(f"Cleaned {len(expired_tasks)} expired tasks")

            self._last_clean_time = current_time

    async def get_task_status_summary(self) -> Dict[str, int]:
        """Get task status summary.

        Returns:
            Dict[str, int]:
                Dictionary mapping task status to count.
        """
        summary = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            summary[task.status.value] += 1
        return summary
