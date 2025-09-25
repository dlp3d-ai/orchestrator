import traceback
from typing import Any, Dict, List, Literal, Tuple, Union

from pymongo import ASCENDING, AsyncMongoClient, MongoClient
from pymongo.errors import PyMongoError

from .database_memory_client import DatabaseMemoryClient, NoMatchingMediumTermMemoryError


class MongoDBMemoryClient(DatabaseMemoryClient):
    """MongoDB memory client class.

    Inherits from DatabaseMemoryClient and implements interaction functionality
    with MongoDB database. Used for managing character-related long-term
    memory, medium-term memory, chat history, relationship, and emotion data.
    Provides complete CRUD operations supporting data creation, retrieval,
    update, and deletion, with data validation and exception handling
    mechanisms.
    """

    def __init__(
        self,
        host: str,
        port: int,
        username: Union[str, None] = None,
        password: Union[str, None] = None,
        database: str = "character",
        auth_database: str = "admin",
        chat_history_collection_name: str = "CharacterChatHistory",
        medium_term_memory_collection_name: str = "CharacterMediumTermMemory",
        long_term_memory_collection_name: str = "CharacterLongTermMemory",
        profile_memory_collection_name: str = "CharacterProfileMemory",
        relationship_collection_name: str = "CharacterRelationship",
        emotion_collection_name: str = "CharacterEmotion",
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the MongoDB memory client.

        Args:
            host (str):
                MongoDB server host address.
            port (int):
                MongoDB server port number.
            username (Union[str, None], optional):
                MongoDB username, None if authentication is not required. Defaults to None.
            password (Union[str, None], optional):
                MongoDB password, None if authentication is not required. Defaults to None.
            database (str, optional):
                MongoDB database name. Defaults to "character".
            chat_history_collection_name (str, optional):
                Name of the chat history collection. Defaults to "CharacterChatHistory".
            medium_term_memory_collection_name (str, optional):
                Name of the medium term memory collection. Defaults to "CharacterMediumTermMemory".
            long_term_memory_collection_name (str, optional):
                Name of the long term memory collection. Defaults to "CharacterLongTermMemory".
            profile_memory_collection_name (str, optional):
                Name of the profile memory collection. Defaults to "CharacterProfileMemory".
            relationship_collection_name (str, optional):
                Name of the relationship collection. Defaults to "CharacterRelationship".
            emotion_collection_name (str, optional):
                Name of the emotion collection. Defaults to "CharacterEmotion".
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        DatabaseMemoryClient.__init__(self, logger_cfg)
        self.host = host
        self.port = port
        self.database_name = database
        self.auth_database = auth_database
        self.username = username
        self.password = password
        # test auth with MongoClient
        with MongoClient(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            authSource=self.auth_database,
        ) as client:
            client.admin.command("ping")

        self.chat_history_collection_name = chat_history_collection_name
        self.medium_term_memory_collection_name = medium_term_memory_collection_name
        self.long_term_memory_collection_name = long_term_memory_collection_name
        self.profile_memory_collection_name = profile_memory_collection_name
        self.relationship_collection_name = relationship_collection_name
        self.emotion_collection_name = emotion_collection_name

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
    ) -> None:
        """Append chat history record.

        Saves chat record to MongoDB, supporting conversation records for both user and assistant.
        For user role, relationship information is required; for assistant role, emotion
        information is required. Includes data validation to ensure data integrity.

        Args:
            character_id (str):
                Character ID for identifying the character that the chat record belongs to.
            unix_timestamp (float):
                Unix timestamp representing when the chat occurred.
            role (Literal["user", "assistant"]):
                Role type, can only be "user" or "assistant".
            content (str):
                Chat content.
            relationship (Union[str, None], optional):
                Relationship information, only valid for user role. Defaults to None.
            happiness (Union[int, None], optional):
                Happiness level, only valid for assistant role. Defaults to None.
            sadness (Union[int, None], optional):
                Sadness level, only valid for assistant role. Defaults to None.
            fear (Union[int, None], optional):
                Fear level, only valid for assistant role. Defaults to None.
            anger (Union[int, None], optional):
                Anger level, only valid for assistant role. Defaults to None.
            disgust (Union[int, None], optional):
                Disgust level, only valid for assistant role. Defaults to None.
            surprise (Union[int, None], optional):
                Surprise level, only valid for assistant role. Defaults to None.
            shyness (Union[int, None], optional):
                Shyness level, only valid for assistant role. Defaults to None.

        Raises:
            ValueError:
                When data validation fails, e.g., user role missing relationship information
                or assistant role missing emotion information.
            PyMongoError:
                When MongoDB operation fails.
        """
        timestamp_str = self.__class__.convert_unix_timestamp_to_str(unix_timestamp)
        emotion_none_count = sum(1 for x in [happiness, sadness, fear, anger, disgust, surprise, shyness] if x is None)
        try:
            if role == "user":
                if relationship is None:
                    msg = "relationship is required for user role"
                    self.logger.error(msg)
                    raise ValueError(msg)
                if emotion_none_count != 7:
                    msg = (
                        "Expecting emotion values are None for user role, "
                        + f"but got {emotion_none_count} None values: "
                        + f"happiness={happiness}, sadness={sadness}, fear={fear}, "
                        + f"anger={anger}, disgust={disgust}, surprise={surprise}, "
                        + f"shyness={shyness}. Writing None values to MongoDB."
                    )
                    self.logger.warning(msg)
                doc = {
                    "character_id": character_id,
                    "chat_timestamp": timestamp_str,
                    "role": role,
                    "relationship": relationship,
                    "content": content,
                }
            else:
                if relationship is not None:
                    msg = "relationship is not allowed for assistant role"
                    self.logger.error(msg)
                    raise ValueError(msg)
                if emotion_none_count > 0:
                    msg = (
                        "Expecting valid emotion values for assistant role, "
                        + f"but got {emotion_none_count} None values: "
                        + f"happiness={happiness}, sadness={sadness}, fear={fear}, "
                        + f"anger={anger}, disgust={disgust}, surprise={surprise}, "
                        + f"shyness={shyness}."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                doc = {
                    "character_id": character_id,
                    "chat_timestamp": timestamp_str,
                    "role": role,
                    "content": content,
                    "happiness": happiness,
                    "sadness": sadness,
                    "fear": fear,
                    "anger": anger,
                    "disgust": disgust,
                    "surprise": surprise,
                    "shyness": shyness,
                }
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.chat_history_collection_name]
                await col.insert_one(doc)
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to append chat history to MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def append_medium_term_memory(
        self,
        character_id: str,
        start_timestamp: str,
    ) -> None:
        """Append medium term memory record.

        Creates a new medium term memory record in MongoDB, containing character ID
        and start timestamp.

        Args:
            character_id (str):
                Character ID for identifying the character that the medium term memory belongs to.
            start_timestamp (str):
                Start timestamp representing the beginning time of the medium term memory.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        doc = {
            "character_id": character_id,
            "start_timestamp": start_timestamp,
        }
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.medium_term_memory_collection_name]
                await col.insert_one(doc)
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to append medium term memory to MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def update_medium_term_memory(
        self,
        character_id: str,
        start_timestamp: str,
        content: str,
        last_short_term_timestamp: str,
    ) -> None:
        """Update medium term memory record.

        Updates the medium term memory record for the specified character and start timestamp,
        including content and last short-term memory timestamp.

        Args:
            character_id (str):
                Character ID for identifying the medium term memory record to update.
            start_timestamp (str):
                Start timestamp for locating the record to update.
            content (str):
                Content of the medium term memory.
            last_short_term_timestamp (str):
                Timestamp of the last short-term memory.

        Raises:
            NoMatchingMediumTermMemoryError:
                When no matching medium term memory record is found.
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.medium_term_memory_collection_name]
                filter_doc = {"character_id": character_id, "start_timestamp": start_timestamp}
                existed = await col.find_one(filter_doc, {"_id": 1})
                if existed is None:
                    msg = (
                        "Medium term memory not found for "
                        + f"character_id={character_id} and start_timestamp={start_timestamp}, "
                        + "please create it first before updating."
                    )
                    self.logger.error(msg)
                    raise NoMatchingMediumTermMemoryError(msg)
                await col.update_one(
                    filter_doc,
                    {
                        "$set": {
                            "content": content,
                            "last_short_term_timestamp": last_short_term_timestamp,
                        }
                    },
                )
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to update medium term memory to MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def set_long_term_memory(
        self,
        character_id: str,
        content: str,
        last_short_term_timestamp: str,
        last_medium_term_timestamp: str,
    ) -> None:
        """Set long term memory record.

        Sets or updates long term memory record in MongoDB using upsert operation
        to ensure the record exists.

        Args:
            character_id (str):
                Character ID for identifying the character that the long term memory belongs to.
            content (str):
                Content of the long term memory.
            last_short_term_timestamp (str):
                Timestamp of the last short-term memory.
            last_medium_term_timestamp (str):
                Timestamp of the last medium-term memory.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.long_term_memory_collection_name]
                doc = {
                    "character_id": character_id,
                    "content": content,
                    "last_short_term_timestamp": last_short_term_timestamp,
                    "last_medium_term_timestamp": last_medium_term_timestamp,
                }
                await col.update_one(
                    {"character_id": character_id},
                    {"$set": doc},
                    upsert=True,
                )
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to update long term memory to MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def set_profile_memory(
        self,
        character_id: str,
        content: str,
        last_short_term_timestamp: str,
    ) -> None:
        """Set profile memory record.

        Sets or updates profile memory record in MongoDB using upsert operation
        to ensure the record exists.

        Args:
            character_id (str):
                Character ID for identifying the character that the profile memory belongs to.
            content (str):
                Content of the profile memory.
            last_short_term_timestamp (str):
                Timestamp of the last short-term memory.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.profile_memory_collection_name]
                doc = {
                    "character_id": character_id,
                    "content": content,
                    "last_short_term_timestamp": last_short_term_timestamp,
                }
                await col.update_one(
                    {"character_id": character_id},
                    {"$set": doc},
                    upsert=True,
                )
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to update profile memory to MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def remove_medium_term_memory_not_after(
        self,
        character_id: str,
        not_after_timestamp: Union[str, None] = None,
    ) -> None:
        """Remove medium term memory records.

        Deletes medium term memory records for the specified character, with optional
        time range specification. If no timestamp is specified, deletes all records.

        Args:
            character_id (str):
                Character ID for identifying the medium term memory records to delete.
            not_after_timestamp (Union[str, None], optional):
                Timestamp, deletes records before or at this timestamp (inclusive).
                If None, deletes all records. Defaults to None.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.medium_term_memory_collection_name]
                if not_after_timestamp is None:
                    self.logger.info(
                        f"Deleting all rows for character_id={character_id} in medium term memory collection."
                    )
                    res = await col.delete_many({"character_id": character_id})
                else:
                    res = await col.delete_many(
                        {
                            "character_id": character_id,
                            "start_timestamp": {"$lte": not_after_timestamp},
                        }
                    )
            if res.deleted_count > 0:
                self.logger.debug(
                    f"Deleting {res.deleted_count} rows for character_id={character_id} in medium term memory collection."
                )
            else:
                self.logger.warning(
                    f"No rows to delete for character_id={character_id} in medium term memory collection."
                )
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to remove medium term memory from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def remove_chat_history_memory(
        self,
        character_id: str,
    ) -> None:
        """Remove chat history records.

        Deletes all chat history records for the specified character.

        Args:
            character_id (str):
                Character ID for identifying the chat history records to delete.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.chat_history_collection_name]
                res = await col.delete_many({"character_id": character_id})
            if res.deleted_count > 0:
                self.logger.info(
                    f"Deleting {res.deleted_count} rows for character_id={character_id} in chat history collection."
                )
            else:
                self.logger.warning(f"No rows to delete for character_id={character_id} in chat history collection.")
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to remove chat history from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def remove_long_term_memory(
        self,
        character_id: str,
    ) -> None:
        """Remove long term memory record.

        Deletes long term memory record for the specified character. If multiple records
        are found, issues a warning and deletes all records.

        Args:
            character_id (str):
                Character ID for identifying the long term memory record to delete.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.long_term_memory_collection_name]
                # First query the number of documents matching conditions
                count = await col.count_documents({"character_id": character_id})
                if count > 1:
                    self.logger.warning(
                        f"Found {count} rows for character_id={character_id} in long term memory collection, "
                        "deleting all of them."
                    )
                # Delete all documents matching conditions
                res = await col.delete_many({"character_id": character_id})
            if res.deleted_count > 0:
                self.logger.info(
                    f"Deleted {res.deleted_count} row(s) for character_id={character_id} in long term memory collection."
                )
            else:
                self.logger.warning(
                    f"No rows to delete for character_id={character_id} in long term memory collection."
                )
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to remove long term memory from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def remove_profile_memory(
        self,
        character_id: str,
    ) -> None:
        """Remove profile memory record.

        Deletes profile memory record for the specified character. If multiple records
        are found, issues a warning and deletes all records.

        Args:
            character_id (str):
                Character ID for identifying the profile memory record to delete.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.profile_memory_collection_name]
                # First query the number of documents matching conditions
                count = await col.count_documents({"character_id": character_id})
                if count > 1:
                    self.logger.warning(
                        f"Found {count} rows for character_id={character_id} in profile memory collection, "
                        "deleting all of them."
                    )
                # Delete all documents matching conditions
                res = await col.delete_many({"character_id": character_id})
            if res.deleted_count > 0:
                self.logger.info(
                    f"Deleted {res.deleted_count} row(s) for character_id={character_id} in profile memory collection."
                )
            else:
                self.logger.warning(f"No rows to delete for character_id={character_id} in profile memory collection.")
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to remove profile memory from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def get_n_rows_medium_term_memory(
        self,
        character_id: str,
    ) -> int:
        """Get number of medium term memory records.

        Queries the total number of medium term memory records for the specified character.

        Args:
            character_id (str):
                Character ID for querying the number of medium term memory records.

        Returns:
            int:
                Number of medium term memory records.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.medium_term_memory_collection_name]
                count = await col.count_documents({"character_id": character_id})
            return int(count)
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to get number of rows in medium term memory collection from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def get_long_term_memory(
        self,
        character_id: str,
    ) -> Union[Dict[str, str], None]:
        """Get long term memory record.

        Queries long term memory record for the specified character.

        Args:
            character_id (str):
                Character ID for querying the long term memory record.

        Returns:
            Union[Dict[str, str], None]:
                Dictionary containing long term memory information with the following fields:
                - character_id (str): Character ID
                - content (str): Long term memory content
                - last_short_term_timestamp (str): Last short-term memory timestamp
                - last_medium_term_timestamp (str): Last medium-term memory timestamp
                Returns None if no record is found.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.long_term_memory_collection_name]
                doc = await col.find_one({"character_id": character_id}, {"_id": 0})
            if doc is None:
                return None
            return doc
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to get long term memory from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def get_profile_memory(
        self,
        character_id: str,
    ) -> Union[Dict[str, str], None]:
        """Get profile memory record.

        Queries profile memory record for the specified character.

        Args:
            character_id (str):
                Character ID for querying the profile memory record.

        Returns:
            Union[Dict[str, str], None]:
                Dictionary containing profile memory information with the following fields:
                - character_id (str): Character ID
                - content (str): Profile memory content
                - last_short_term_timestamp (str): Last short-term memory timestamp
                Returns None if no record is found.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.profile_memory_collection_name]
                doc = await col.find_one({"character_id": character_id}, {"_id": 0})
            if doc is None:
                msg = f"Profile memory not found for character_id={character_id}"
                self.logger.info(msg)
                return None
            return doc
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to get profile memory from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def get_medium_term_memories_after(
        self,
        character_id: str,
        after_timestamp: Union[str, None] = None,
    ) -> List[Dict[str, str]]:
        """Get medium term memory records list.

        Queries medium term memory records for the specified character with optional
        time range specification. If no timestamp is specified, retrieves all records.

        Args:
            character_id (str):
                Character ID for querying medium term memory records.
            after_timestamp (Union[str, None], optional):
                Timestamp, retrieves records after this timestamp. If None, retrieves all records.
                Defaults to None.

        Returns:
            List[Dict[str, str]]:
                List of medium term memory records, each containing the following fields:
                - character_id (str): Character ID
                - start_timestamp (str): Start timestamp
                - content (str): Memory content
                - last_short_term_timestamp (str): Last short-term memory timestamp

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.medium_term_memory_collection_name]
                if after_timestamp is None:
                    cursor = col.find(
                        {"character_id": character_id},
                        {
                            "_id": 0,
                            "character_id": 1,
                            "start_timestamp": 1,
                            "content": 1,
                            "last_short_term_timestamp": 1,
                        },
                    ).sort("start_timestamp", ASCENDING)
                    results = await cursor.to_list(length=None)
                else:
                    cursor = col.find(
                        {
                            "character_id": character_id,
                            "start_timestamp": {"$gt": after_timestamp},
                        },
                        {
                            "_id": 0,
                            "character_id": 1,
                            "start_timestamp": 1,
                            "content": 1,
                            "last_short_term_timestamp": 1,
                        },
                    ).sort("start_timestamp", ASCENDING)
                    results = await cursor.to_list(length=None)
            return results
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to get medium term memories after {after_timestamp} from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def get_chat_histories_after(
        self,
        character_id: str,
        after_timestamp: Union[str, None] = None,
    ) -> List[Dict[str, Union[str, int]]]:
        """Get chat history records list.

        Queries chat history records for the specified character with optional
        time range specification. If no timestamp is specified, retrieves all records.

        Args:
            character_id (str):
                Character ID for querying chat history records.
            after_timestamp (Union[str, None], optional):
                Timestamp, retrieves records after this timestamp. If None, retrieves all records.
                Defaults to None.

        Returns:
            List[Dict[str, Union[str, int]]]:
                List of chat history records, each containing the following fields:
                - chat_timestamp (str): Chat timestamp
                - role (str): Role type ("user" or "assistant")
                - content (str): Chat content
                - relationship (str, optional): Relationship information (user role only)
                - happiness (int, optional): Happiness level (assistant role only)
                - sadness (int, optional): Sadness level (assistant role only)
                - fear (int, optional): Fear level (assistant role only)
                - anger (int, optional): Anger level (assistant role only)
                - disgust (int, optional): Disgust level (assistant role only)
                - surprise (int, optional): Surprise level (assistant role only)
                - shyness (int, optional): Shyness level (assistant role only)

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.chat_history_collection_name]
                if after_timestamp is None:
                    cursor = col.find(
                        {"character_id": character_id},
                        {
                            "_id": 0,
                            "chat_timestamp": 1,
                            "role": 1,
                            "content": 1,
                            "relationship": 1,
                            "happiness": 1,
                            "sadness": 1,
                            "fear": 1,
                            "anger": 1,
                            "disgust": 1,
                            "surprise": 1,
                            "shyness": 1,
                        },
                    ).sort("chat_timestamp", ASCENDING)
                    results = await cursor.to_list(length=None)
                else:
                    cursor = col.find(
                        {
                            "character_id": character_id,
                            "chat_timestamp": {"$gt": after_timestamp},
                        },
                        {
                            "_id": 0,
                            "chat_timestamp": 1,
                            "role": 1,
                            "content": 1,
                            "relationship": 1,
                            "happiness": 1,
                            "sadness": 1,
                            "fear": 1,
                            "anger": 1,
                            "disgust": 1,
                            "surprise": 1,
                            "shyness": 1,
                        },
                    ).sort("chat_timestamp", ASCENDING)
                    results = await cursor.to_list(length=None)
            ret_list: List[Dict[str, Union[str, int]]] = list()
            for item in results:
                chat_history: Dict[str, Union[str, int]] = dict(
                    chat_timestamp=item["chat_timestamp"],
                    role=item["role"],
                    content=item["content"],
                )
                if "relationship" in item and item["relationship"] is not None:
                    chat_history["relationship"] = item["relationship"]
                for emotion_key in (
                    "happiness",
                    "sadness",
                    "fear",
                    "anger",
                    "disgust",
                    "surprise",
                    "shyness",
                ):
                    if emotion_key in item and item[emotion_key] is not None:
                        chat_history[emotion_key] = int(item[emotion_key])
                ret_list.append(chat_history)
            return ret_list
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to get chat histories after {after_timestamp} from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def set_relationship(
        self,
        character_id: str,
        relationship: Literal["Stranger", "Acquaintance", "Friend", "Situationship", "Lover"],
        score: int,
    ) -> None:
        """Set relationship record.

        Sets or updates relationship record in MongoDB using upsert operation
        to ensure the record exists.

        Args:
            character_id (str):
                Character ID for identifying the character that the relationship record belongs to.
            relationship (Literal["Stranger", "Acquaintance", "Friend", "Situationship", "Lover"]):
                Relationship type, must be one of the predefined relationship types.
            score (int):
                Relationship score.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.relationship_collection_name]
                await col.update_one(
                    {"character_id": character_id},
                    {"$set": {"character_id": character_id, "relationship": relationship, "score": int(score)}},
                    upsert=True,
                )
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to set relationship for character_id={character_id} to {relationship} with score {score} in MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def remove_relationship(
        self,
        character_id: str,
    ) -> None:
        """Remove relationship record.

        Deletes relationship record for the specified character.

        Args:
            character_id (str):
                Character ID for identifying the relationship record to delete.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.relationship_collection_name]
                await col.delete_one({"character_id": character_id})
            self.logger.info(f"Deleting relationship for character_id={character_id} in relationship collection.")
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to remove relationship for character_id={character_id} from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def get_relationship(
        self,
        character_id: str,
    ) -> Union[None, Tuple[str, int]]:
        """Get relationship record.

        Queries relationship record for the specified character.

        Args:
            character_id (str):
                Character ID for querying the relationship record.

        Returns:
            Union[None, Tuple[str, int]]:
                If relationship record is found, returns tuple of (relationship_type, relationship_score);
                if no record is found, returns None.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.relationship_collection_name]
                doc = await col.find_one({"character_id": character_id}, {"_id": 0, "relationship": 1, "score": 1})
            if doc is None:
                return None
            return doc["relationship"], int(doc["score"])  # type: ignore[return-value]
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to get relationship for character_id={character_id} from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

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
        """Set emotion record.

        Sets or updates emotion record in MongoDB using upsert operation
        to ensure the record exists.

        Args:
            character_id (str):
                Character ID for identifying the character that the emotion record belongs to.
            happiness (int):
                Happiness level.
            sadness (int):
                Sadness level.
            fear (int):
                Fear level.
            anger (int):
                Anger level.
            disgust (int):
                Disgust level.
            surprise (int):
                Surprise level.
            shyness (int):
                Shyness level.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.emotion_collection_name]
                await col.update_one(
                    {"character_id": character_id},
                    {
                        "$set": {
                            "character_id": character_id,
                            "happiness": int(happiness),
                            "sadness": int(sadness),
                            "fear": int(fear),
                            "anger": int(anger),
                            "disgust": int(disgust),
                            "surprise": int(surprise),
                            "shyness": int(shyness),
                        }
                    },
                    upsert=True,
                )
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to set emotion for character_id={character_id} from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def remove_emotion(
        self,
        character_id: str,
    ) -> None:
        """Remove emotion record.

        Deletes emotion record for the specified character.

        Args:
            character_id (str):
                Character ID for identifying the emotion record to delete.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.emotion_collection_name]
                await col.delete_one({"character_id": character_id})
            self.logger.info(f"Deleting emotion for character_id={character_id} in emotion collection.")
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to remove emotion for character_id={character_id} from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def get_emotion(
        self,
        character_id: str,
    ) -> Union[None, Dict[str, int]]:
        """Get emotion record.

        Queries emotion record for the specified character.

        Args:
            character_id (str):
                Character ID for querying the emotion record.

        Returns:
            Union[None, Dict[str, int]]:
                If emotion record is found, returns dictionary containing emotion values with the following fields:
                - happiness (int): Happiness level
                - sadness (int): Sadness level
                - fear (int): Fear level
                - anger (int): Anger level
                - disgust (int): Disgust level
                - surprise (int): Surprise level
                - shyness (int): Shyness level
                If no record is found, returns None.

        Raises:
            PyMongoError:
                When MongoDB operation fails.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database,
            ) as client:
                db = client[self.database_name]
                col = db[self.emotion_collection_name]
                doc = await col.find_one(
                    {"character_id": character_id},
                    {
                        "_id": 0,
                        "happiness": 1,
                        "sadness": 1,
                        "fear": 1,
                        "anger": 1,
                        "disgust": 1,
                        "surprise": 1,
                        "shyness": 1,
                    },
                )
            if doc is None:
                return None
            ret_dict: Dict[str, int] = dict()
            for key in ("happiness", "sadness", "fear", "anger", "disgust", "surprise", "shyness"):
                value = doc.get(key, None)
                if value is not None:
                    ret_dict[key] = int(value)
            return ret_dict
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to get emotion for character_id={character_id} from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e
