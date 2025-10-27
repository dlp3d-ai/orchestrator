from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple, Union

import pytz

from ...utils.log import logging, setup_logger
from ...utils.super import Super


class NoMatchingMediumTermMemoryError(Exception):
    """Exception raised when no matching medium term memory is found."""

    pass


class DatabaseMemoryClient(Super, ABC):
    """Abstract base class for database memory clients.

    Provides asynchronous interaction with databases for managing character
    memory data. Supports querying and managing character long-term memory,
    medium-term memory, chat history, relationship, and emotion data. This is
    an abstract base class that requires subclasses to implement specific
    database operations.
    """

    logger: logging.Logger = setup_logger(logger_name="DatabaseMemoryClient")

    def __init__(
        self,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the database memory client.

        Args:
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        Super.__init__(self, logger_cfg)
        self.__class__.logger = self.logger

    async def get_cascade_memories(
        self,
        character_id: str,
    ) -> Dict[str, Union[dict, list]]:
        """Get cascade memory information.

        Retrieves complete information of long-term memory, medium-term memory,
        and short-term memory for the specified character in chronological order.

        Args:
            character_id (str):
                Character ID.

        Returns:
            Dict[str, Union[dict, list]]:
                Dictionary containing cascade memory information with the following fields:
                - long_term_memory (dict): Long-term memory information
                - medium_term_memories (list): Medium-term memory records list, may contain
                  medium-term memories with empty content and last_short_term_timestamp.
                - short_term_memories (list): Short-term memory records list
        """
        ret_dict = dict()
        long_term_memory = await self.get_long_term_memory(character_id)
        if long_term_memory is not None:
            last_medium_term_timestamp = long_term_memory.get("last_medium_term_timestamp", None)
            ret_dict["long_term_memory"] = long_term_memory
        else:
            long_term_memory = None
            last_medium_term_timestamp = None
        medium_term_memories = await self.get_medium_term_memories_after(character_id, last_medium_term_timestamp)
        last_short_term_timestamp = None
        if len(medium_term_memories) > 0:
            ret_dict["medium_term_memories"] = medium_term_memories
            for medium_term_memory in medium_term_memories:
                cur_timestamp = medium_term_memory.get("last_short_term_timestamp", None)
                if cur_timestamp is not None:
                    if last_short_term_timestamp is None:
                        last_short_term_timestamp = cur_timestamp
                    else:
                        last_short_term_timestamp = max(last_short_term_timestamp, cur_timestamp)
        short_term_memories = await self.get_chat_histories_after(character_id, last_short_term_timestamp)
        if len(short_term_memories) > 0:
            ret_dict["short_term_memories"] = short_term_memories
        return ret_dict

    @abstractmethod
    async def append_chat_history(
        self,
        character_id: str,
        unix_timestamp: float,
        role: Literal["user", "assistant"],
        content: str,
        relationship: Union[str, None] = None,
        happiness: Union[int, None] = None,
        sadness: Union[int, None] = None,
        fear: Union[int, None] = None,
        anger: Union[int, None] = None,
        disgust: Union[int, None] = None,
        surprise: Union[int, None] = None,
        shyness: Union[int, None] = None,
        timezone: Union[str, None] = None,
    ) -> None:
        """Append chat history record.

        Inserts chat record into the database chat history table. Different fields
        are inserted based on role type (user or assistant).

        Args:
            character_id (str):
                Character ID.
            unix_timestamp (float):
                Unix timestamp for generating specified timezone timestamp string.
            role (Literal["user", "assistant"]):
                Role type, user or assistant.
            content (str):
                Chat content.
            relationship (Union[str, None], optional):
                Relationship status, only valid for user role. Defaults to None.
            happiness (Union[int, None], optional):
                Happiness value, only valid for assistant role. Defaults to None.
            sadness (Union[int, None], optional):
                Sadness value, only valid for assistant role. Defaults to None.
            fear (Union[int, None], optional):
                Fear value, only valid for assistant role. Defaults to None.
            anger (Union[int, None], optional):
                Anger value, only valid for assistant role. Defaults to None.
            disgust (Union[int, None], optional):
                Disgust value, only valid for assistant role. Defaults to None.
            surprise (Union[int, None], optional):
                Surprise value, only valid for assistant role. Defaults to None.
            shyness (Union[int, None], optional):
                Shyness value, only valid for assistant role. Defaults to None.
            timezone (Union[str, None], optional):
                Timezone name. If None, defaults to "Asia/Shanghai". Defaults to None.

        Raises:
            ValueError:
                When role type does not match the provided parameters.
        """
        pass

    @abstractmethod
    async def append_medium_term_memory(
        self,
        character_id: str,
        start_timestamp: str,
    ) -> None:
        """Append medium term memory record.

        Creates a new empty record in the medium term memory table with only
        primary key information.

        Args:
            character_id (str):
                Character ID.
            start_timestamp (str):
                Start timestamp in format "YYYY-MM-DD HH:MM:SS,mmm".

        Raises:
            ValueError:
                When record already exists.
        """
        pass

    @abstractmethod
    async def update_medium_term_memory(
        self,
        character_id: str,
        start_timestamp: str,
        content: str,
        last_short_term_timestamp: str,
    ) -> None:
        """Update medium term memory record.

        Updates the content field and the latest short-term memory timestamp
        in the medium term memory table.

        Args:
            character_id (str):
                Character ID.
            start_timestamp (str):
                Start timestamp in format "YYYY-MM-DD HH:MM:SS,mmm".
            content (str):
                Memory content.
            last_short_term_timestamp (str):
                Latest short-term memory timestamp in format "YYYY-MM-DD HH:MM:SS,mmm".

        Raises:
            KeyError:
                When record does not exist.
        """
        pass

    @abstractmethod
    async def set_long_term_memory(
        self,
        character_id: str,
        content: str,
        last_short_term_timestamp: str,
        last_medium_term_timestamp: str,
    ) -> None:
        """Set long term memory record.

        Inserts or updates record in the long term memory table, overwriting
        regardless of whether the record already exists.

        Args:
            character_id (str):
                Character ID.
            content (str):
                Long-term memory content.
            last_short_term_timestamp (str):
                Latest short-term memory timestamp in format "YYYY-MM-DD HH:MM:SS,mmm".
            last_medium_term_timestamp (str):
                Latest medium-term memory timestamp in format "YYYY-MM-DD HH:MM:SS,mmm".
        """
        pass

    @abstractmethod
    async def set_profile_memory(
        self,
        character_id: str,
        content: str,
        last_short_term_timestamp: str,
    ) -> None:
        """Set profile memory record.

        Inserts or updates record in the profile memory table, overwriting
        regardless of whether the record already exists.

        Args:
            character_id (str):
                Character ID.
            content (str):
                Profile memory content.
            last_short_term_timestamp (str):
                Latest short-term memory timestamp included in profile in format "YYYY-MM-DD HH:MM:SS,mmm".
        """
        pass

    @abstractmethod
    async def remove_medium_term_memory_not_after(
        self,
        character_id: str,
        not_after_timestamp: Union[str, None] = None,
    ) -> None:
        """Remove medium term memory records matching specified conditions.

        Deletes all medium term memory records matching the conditions,
        supports pagination for handling large amounts of data.

        Args:
            character_id (str):
                Character ID.
            not_after_timestamp (Union[str, None], optional):
                Timestamp, deletes records where start_timestamp <= not_after_timestamp.
                If None, deletes all records for the character_id. Defaults to None.
        """
        pass

    @abstractmethod
    async def remove_chat_history_memory(
        self,
        character_id: str,
    ) -> None:
        """Remove chat history records.

        Deletes all chat history records for the specified character.

        Args:
            character_id (str):
                Character ID.
        """
        pass

    @abstractmethod
    async def remove_long_term_memory(
        self,
        character_id: str,
    ) -> None:
        """Remove long term memory record.

        Deletes the long term memory record for the specified character.

        Args:
            character_id (str):
                Character ID.
        """
        pass

    @abstractmethod
    async def remove_profile_memory(
        self,
        character_id: str,
    ) -> None:
        """Remove profile memory record.

        Deletes the profile memory record for the specified character.

        Args:
            character_id (str):
                Character ID.
        """
        pass

    @abstractmethod
    async def get_n_rows_medium_term_memory(
        self,
        character_id: str,
    ) -> int:
        """Get number of medium term memory records for specified character.

        Args:
            character_id (str):
                Character ID.

        Returns:
            int:
                Number of records for the character in the medium term memory table.
        """
        pass

    @abstractmethod
    async def get_long_term_memory(
        self,
        character_id: str,
    ) -> Union[Dict[str, str], None]:
        """Get long term memory record.

        Retrieves long term memory information for the specified character
        from the long term memory table.

        Args:
            character_id (str):
                Character ID.

        Returns:
            Union[Dict[str, str], None]:
                Dictionary containing long term memory information with the following fields:
                - character_id (str): Character ID
                - content (str): Memory content
                - last_short_term_timestamp (str): Latest short-term memory timestamp included
                - last_medium_term_timestamp (str): Latest medium-term memory timestamp included
                Returns None if no record is found.
        """
        pass

    @abstractmethod
    async def get_profile_memory(
        self,
        character_id: str,
    ) -> Union[Dict[str, str], None]:
        """Get profile memory record.

        Retrieves profile memory information for the specified character
        from the profile memory table.

        Args:
            character_id (str):
                Character ID.

        Returns:
            Union[Dict[str, str], None]:
                Dictionary containing profile memory information with the following fields:
                - character_id (str): Character ID
                - content (str): Memory content
                - last_short_term_timestamp (str): Latest short-term memory timestamp included
                Returns None if no record is found.
        """
        pass

    @abstractmethod
    async def get_medium_term_memories_after(
        self,
        character_id: str,
        after_timestamp: Union[str, None] = None,
    ) -> List[Dict[str, str]]:
        """Get medium term memory records after specified timestamp.

        Args:
            character_id (str):
                Character ID.
            after_timestamp (Union[str, None], optional):
                Timestamp, retrieves records where start_timestamp > after_timestamp.
                If None, retrieves all medium term memory records for the character. Defaults to None.

        Returns:
            List[Dict[str, str]]:
                List of medium term memory records, each containing character_id,
                start_timestamp, content, last_short_term_timestamp fields.
                Retrieved results may contain medium term memories with empty content
                and last_short_term_timestamp.
        """
        pass

    @abstractmethod
    async def get_chat_histories_after(
        self,
        character_id: str,
        after_timestamp: Union[str, None] = None,
    ) -> List[Dict[str, Union[str, int]]]:
        """Get chat history records after specified timestamp.

        Args:
            character_id (str):
                Character ID.
            after_timestamp (Union[str, None], optional):
                Timestamp, retrieves records where timestamp > after_timestamp.
                If None, retrieves all chat history records for the character. Defaults to None.

        Returns:
            List[Dict[str, Union[str, int]]]:
                List of chat history records.
        """
        pass

    @abstractmethod
    async def set_relationship(
        self,
        character_id: str,
        relationship: Literal["Stranger", "Acquaintance", "Friend", "Situationship", "Lover"],
        score: int,
    ) -> None:
        """Set character relationship.

        Sets the relationship status and score for the specified character
        in the relationship table.

        Args:
            character_id (str):
                Character ID.
            relationship (Literal["Stranger", "Acquaintance", "Friend", "Situationship", "Lover"]):
                Relationship type.
            score (int):
                Relationship score.
        """
        pass

    @abstractmethod
    async def remove_relationship(
        self,
        character_id: str,
    ) -> None:
        """Remove character relationship.

        Deletes the relationship record for the specified character.

        Args:
            character_id (str):
                Character ID.
        """
        pass

    @abstractmethod
    async def get_relationship(
        self,
        character_id: str,
    ) -> Union[None, Tuple[str, int]]:
        """Get character relationship.

        Retrieves relationship information for the specified character
        from the relationship table.

        Args:
            character_id (str):
                Character ID.

        Returns:
            Union[None, Tuple[str, int]]:
                If relationship record is found, returns tuple of (relationship, score);
                if not found, returns None.
        """
        pass

    @abstractmethod
    async def set_emotion(
        self,
        character_id: str,
        happiness: int,
        sadness: int,
        fear: int,
        anger: int,
        disgust: int,
        surprise: int,
        shyness: int,
    ) -> None:
        """Set character emotion.

        Sets the emotion state for the specified character in the emotion table.

        Args:
            character_id (str):
                Character ID.
            happiness (int):
                Happiness value.
            sadness (int):
                Sadness value.
            fear (int):
                Fear value.
            anger (int):
                Anger value.
            disgust (int):
                Disgust value.
            surprise (int):
                Surprise value.
            shyness (int):
                Shyness value.
        """
        pass

    @abstractmethod
    async def remove_emotion(
        self,
        character_id: str,
    ) -> None:
        """Remove character emotion.

        Deletes the emotion record for the specified character.

        Args:
            character_id (str):
                Character ID.
        """
        pass

    @abstractmethod
    async def get_emotion(
        self,
        character_id: str,
    ) -> Union[None, Dict[str, int]]:
        """Get character emotion.

        Retrieves emotion information for the specified character
        from the emotion table.

        Args:
            character_id (str):
                Character ID.

        Returns:
            Union[None, Dict[str, int]]:
                If emotion record is found, returns dictionary containing emotion values;
                if not found, returns None.
        """
        pass

    @classmethod
    def convert_unix_timestamp_to_str(
        cls,
        unix_timestamp: float,
        timezone: Union[str, None] = None,
    ) -> str:
        """Convert Unix timestamp to string.

        Args:
            unix_timestamp (float):
                Unix timestamp.
            timezone (Union[str, None], optional):
                Timezone name. If None, defaults to "Asia/Shanghai". Defaults to None.

        Returns:
            str:
                Timestamp string in specified timezone format "YYYY-MM-DD HH:MM:SS,mmm".
        """
        if timezone is None:
            timezone = "Asia/Shanghai"
        elif timezone not in pytz.all_timezones:
            cls.logger.warning(
                f"Timezone name {timezone} not found in pytz.all_timezones" + ", using default timezone Asia/Shanghai"
            )
            timezone = "Asia/Shanghai"
        target_tz = pytz.timezone(timezone)
        time_str = datetime.fromtimestamp(unix_timestamp, target_tz).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        return time_str
