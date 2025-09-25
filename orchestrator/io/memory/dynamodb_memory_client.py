import asyncio
import traceback
from typing import Any, Dict, List, Literal, Tuple, Union

import aioboto3
import aioboto3.session
from boto3.dynamodb.conditions import Key as DynamoDBKey
from botocore.exceptions import ClientError

from .database_memory_client import DatabaseMemoryClient, NoMatchingMediumTermMemoryError


class DynamoDBMemoryClient(DatabaseMemoryClient):
    """DynamoDB memory client class.

    Provides asynchronous interaction with AWS DynamoDB for managing character
    memory data. Supports querying and managing character long-term memory,
    medium-term memory, chat history, relationship, and emotion data from
    DynamoDB tables.
    """

    def __init__(
        self,
        region_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        chat_history_table_name: str = "CharacterChatHistory",
        medium_term_memory_table_name: str = "CharacterMediumTermMemory",
        long_term_memory_table_name: str = "CharacterLongTermMemory",
        profile_memory_table_name: str = "CharacterProfileMemory",
        relationship_table_name: str = "CharacterRelationship",
        emotion_table_name: str = "CharacterEmotion",
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the DynamoDB memory client.

        Args:
            region_name (str):
                AWS region name, e.g., 'us-east-1'.
            aws_access_key_id (str):
                AWS access key ID.
            aws_secret_access_key (str):
                AWS secret access key.
            chat_history_table_name (str, optional):
                Name of the chat history table. Defaults to "CharacterChatHistory".
            medium_term_memory_table_name (str, optional):
                Name of the medium term memory table. Defaults to "CharacterMediumTermMemory".
            long_term_memory_table_name (str, optional):
                Name of the long term memory table. Defaults to "CharacterLongTermMemory".
            profile_memory_table_name (str, optional):
                Name of the profile memory table storing assistant's memory of user profile.
                Defaults to "CharacterProfileMemory".
            relationship_table_name (str, optional):
                Name of the relationship table. Defaults to "CharacterRelationship".
            emotion_table_name (str, optional):
                Name of the emotion table. Defaults to "CharacterEmotion".
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        DatabaseMemoryClient.__init__(self, logger_cfg)
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.chat_history_table_name = chat_history_table_name
        self.medium_term_memory_table_name = medium_term_memory_table_name
        self.long_term_memory_table_name = long_term_memory_table_name
        self.profile_memory_table_name = profile_memory_table_name
        self.relationship_table_name = relationship_table_name
        self.emotion_table_name = emotion_table_name
        self.session = aioboto3.Session(
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

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

        Inserts chat record into the DynamoDB chat history table. Different fields
        are inserted based on role type (user or assistant).

        Args:
            character_id (str):
                Character ID.
            unix_timestamp (float):
                Unix timestamp for generating Beijing timezone timestamp string.
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

        Raises:
            ValueError:
                When role type does not match the provided parameters.
        """
        timestamp_str = self.__class__.convert_unix_timestamp_to_str(unix_timestamp)
        emotion_none_count = sum(1 for x in [happiness, sadness, fear, anger, disgust, surprise, shyness] if x is None)
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
                    + f"shyness={shyness}. Writing None values to DynamoDB."
                )
                self.logger.warning(msg)
            item = {
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
            item = {
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
        async with self.session.resource("dynamodb") as dynamo_resource:
            try:
                table = await dynamo_resource.Table(self.chat_history_table_name)
                await table.put_item(Item=item)
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to append chat history to DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        item = {
            "character_id": character_id,
            "start_timestamp": start_timestamp,
        }
        async with self.session.resource("dynamodb") as dynamo_resource:
            try:
                table = await dynamo_resource.Table(self.medium_term_memory_table_name)
                key = {"character_id": character_id, "start_timestamp": start_timestamp}
                response = await table.get_item(Key=key)
                if "Item" in response:
                    msg = (
                        "Medium term memory already exists for "
                        + f"character_id={character_id} and start_timestamp={start_timestamp}, "
                        + "please update it instead."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                await table.put_item(Item=item)
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to append medium term memory to DynamoDB: {traceback_str}"
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
        async with self.session.resource("dynamodb") as dynamo_resource:
            try:
                table = await dynamo_resource.Table(self.medium_term_memory_table_name)
                key = {"character_id": character_id, "start_timestamp": start_timestamp}
                response = await table.get_item(Key=key)
                if "Item" not in response:
                    msg = (
                        "Medium term memory not found for "
                        + f"character_id={character_id} and start_timestamp={start_timestamp}, "
                        + "please create it first before updating."
                    )
                    self.logger.error(msg)
                    raise NoMatchingMediumTermMemoryError(msg)
                await table.update_item(
                    Key={"character_id": character_id, "start_timestamp": start_timestamp},
                    UpdateExpression="SET content = :content, last_short_term_timestamp = :last_short_term_timestamp",
                    ExpressionAttributeValues={
                        ":content": content,
                        ":last_short_term_timestamp": last_short_term_timestamp,
                    },
                )
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to update medium term memory to DynamoDB: {traceback_str}"
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
        async with self.session.resource("dynamodb") as dynamo_resource:
            try:
                table = await dynamo_resource.Table(self.long_term_memory_table_name)
                item = {
                    "character_id": character_id,
                    "content": content,
                    "last_short_term_timestamp": last_short_term_timestamp,
                    "last_medium_term_timestamp": last_medium_term_timestamp,
                }
                # no matter whether the item already exists, always put_item
                await table.put_item(Item=item)
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to update long term memory to DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            try:
                table = await dynamo_resource.Table(self.profile_memory_table_name)
                item = {
                    "character_id": character_id,
                    "content": content,
                    "last_short_term_timestamp": last_short_term_timestamp,
                }
                # no matter whether the item already exists, always put_item
                await table.put_item(Item=item)
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to update profile memory to DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.medium_term_memory_table_name)
            try:
                # Query all records matching conditions
                if not_after_timestamp is None:
                    msg = f"Deleting all rows for character_id={character_id} in medium term memory table."
                    self.logger.info(msg)
                    response = await table.query(
                        KeyConditionExpression=DynamoDBKey("character_id").eq(character_id),
                        ProjectionExpression="character_id, start_timestamp",
                    )
                else:
                    # Delete records where start_timestamp <= not_after_timestamp
                    response = await table.query(
                        KeyConditionExpression=DynamoDBKey("character_id").eq(character_id)
                        & DynamoDBKey("start_timestamp").lte(not_after_timestamp),
                        ProjectionExpression="character_id, start_timestamp",
                    )
                # Delete all queried records
                delete_coroutines = list()
                for item in response.get("Items", []):
                    delete_coroutines.append(
                        table.delete_item(
                            Key={"character_id": item["character_id"], "start_timestamp": item["start_timestamp"]}
                        )
                    )

                # Handle pagination results
                while "LastEvaluatedKey" in response:
                    if not_after_timestamp is None:
                        response = await table.query(
                            KeyConditionExpression=DynamoDBKey("character_id").eq(character_id),
                            ProjectionExpression="character_id, start_timestamp",
                            ExclusiveStartKey=response["LastEvaluatedKey"],
                        )
                    else:
                        response = await table.query(
                            KeyConditionExpression=DynamoDBKey("character_id").eq(character_id)
                            & DynamoDBKey("start_timestamp").lte(not_after_timestamp),
                            ProjectionExpression="character_id, start_timestamp",
                            ExclusiveStartKey=response["LastEvaluatedKey"],
                        )

                    for item in response.get("Items", []):
                        delete_coroutines.append(
                            table.delete_item(
                                Key={"character_id": item["character_id"], "start_timestamp": item["start_timestamp"]}
                            )
                        )
                if len(delete_coroutines) > 0:
                    self.logger.debug(
                        f"Deleting {len(delete_coroutines)} rows for character_id={character_id} in medium term memory table."
                    )
                    await asyncio.gather(*delete_coroutines)
                else:
                    self.logger.warning(
                        f"No rows to delete for character_id={character_id} in medium term memory table."
                    )
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to remove medium term memory from DynamoDB: {traceback_str}"
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
                Character ID.
        """
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.chat_history_table_name)
            try:
                # Query all chat history records for this character
                response = await table.query(
                    KeyConditionExpression=DynamoDBKey("character_id").eq(character_id),
                    ProjectionExpression="character_id, chat_timestamp",
                )

                # Delete all queried records
                delete_coroutines = list()
                for item in response.get("Items", []):
                    delete_coroutines.append(
                        table.delete_item(
                            Key={"character_id": item["character_id"], "chat_timestamp": item["chat_timestamp"]}
                        )
                    )

                # Handle pagination results
                while "LastEvaluatedKey" in response:
                    response = await table.query(
                        KeyConditionExpression=DynamoDBKey("character_id").eq(character_id),
                        ProjectionExpression="character_id, chat_timestamp",
                        ExclusiveStartKey=response["LastEvaluatedKey"],
                    )

                    for item in response.get("Items", []):
                        delete_coroutines.append(
                            table.delete_item(
                                Key={"character_id": item["character_id"], "chat_timestamp": item["chat_timestamp"]}
                            )
                        )

                if len(delete_coroutines) > 0:
                    self.logger.info(
                        f"Deleting {len(delete_coroutines)} rows for character_id={character_id} in chat history table."
                    )
                    await asyncio.gather(*delete_coroutines)
                else:
                    self.logger.warning(f"No rows to delete for character_id={character_id} in chat history table.")
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to remove chat history from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.long_term_memory_table_name)
            try:
                await table.delete_item(
                    Key={
                        "character_id": character_id,
                    }
                )
                self.logger.info(f"Deleting the row for character_id={character_id} in long term memory table.")
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to remove long term memory from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.profile_memory_table_name)
            try:
                await table.delete_item(
                    Key={
                        "character_id": character_id,
                    }
                )
                self.logger.info(f"Deleting the row for character_id={character_id} in profile memory table.")
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to remove profile memory from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.medium_term_memory_table_name)

            try:
                response = await table.query(
                    KeyConditionExpression=DynamoDBKey("character_id").eq(character_id), Select="COUNT"
                )
                count = response.get("Count", 0)
                return count
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to get number of rows in medium term memory table from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.long_term_memory_table_name)
            try:
                response = await table.get_item(
                    Key={
                        "character_id": character_id,
                    }
                )
                if "Item" not in response:
                    return None
                return response["Item"]
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to get long term memory from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.profile_memory_table_name)
            try:
                response = await table.get_item(
                    Key={
                        "character_id": character_id,
                    }
                )
                if "Item" not in response:
                    msg = f"Profile memory not found for character_id={character_id}"
                    self.logger.info(msg)
                    return None
                return response["Item"]
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to get profile memory from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.medium_term_memory_table_name)
            try:
                if after_timestamp is None:
                    response = await table.query(
                        KeyConditionExpression=DynamoDBKey("character_id").eq(character_id),
                    )
                else:
                    response = await table.query(
                        KeyConditionExpression=DynamoDBKey("character_id").eq(character_id)
                        & DynamoDBKey("start_timestamp").gt(after_timestamp),
                    )
                return response.get("Items", [])
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to get medium term memories after {after_timestamp} from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.chat_history_table_name)
            try:
                if after_timestamp is None:
                    response = await table.query(
                        KeyConditionExpression=DynamoDBKey("character_id").eq(character_id),
                    )
                else:
                    response = await table.query(
                        KeyConditionExpression=DynamoDBKey("character_id").eq(character_id)
                        & DynamoDBKey("chat_timestamp").gt(after_timestamp),
                    )
                ret_list = list()
                for item in response.get("Items", []):
                    chat_history = dict(
                        chat_timestamp=item["chat_timestamp"],
                        role=item["role"],
                        content=item["content"],
                    )
                    if "relationship" in item:
                        chat_history["relationship"] = item["relationship"]
                    for emotion_key in ("happiness", "sadness", "fear", "anger", "disgust", "surprise", "shyness"):
                        if emotion_key in item:
                            chat_history[emotion_key] = int(item[emotion_key])
                    ret_list.append(chat_history)
                return ret_list
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to get chat histories after {after_timestamp} from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            try:
                table = await dynamo_resource.Table(self.relationship_table_name)
                item = {"character_id": character_id, "relationship": relationship, "score": score}
                await table.put_item(Item=item)
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to set relationship for character_id={character_id} to {relationship} with score {score} in DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.relationship_table_name)
            try:
                await table.delete_item(
                    Key={
                        "character_id": character_id,
                    }
                )
                self.logger.info(f"Deleting relationship for character_id={character_id} in relationship table.")
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to remove relationship for character_id={character_id} from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.relationship_table_name)
            try:
                response = await table.get_item(
                    Key={
                        "character_id": character_id,
                    }
                )
                if "Item" not in response:
                    return None
                return response["Item"]["relationship"], int(response["Item"]["score"])
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to get relationship for character_id={character_id} from DynamoDB: {traceback_str}"
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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.emotion_table_name)
            try:
                item = {
                    "character_id": character_id,
                    "happiness": happiness,
                    "sadness": sadness,
                    "fear": fear,
                    "anger": anger,
                    "disgust": disgust,
                    "surprise": surprise,
                    "shyness": shyness,
                }
                await table.put_item(Item=item)
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to set emotion for character_id={character_id} from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.emotion_table_name)
            try:
                await table.delete_item(
                    Key={
                        "character_id": character_id,
                    }
                )
                self.logger.info(f"Deleting emotion for character_id={character_id} in emotion table.")
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to remove emotion for character_id={character_id} from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            table = await dynamo_resource.Table(self.emotion_table_name)
            try:
                response = await table.get_item(
                    Key={
                        "character_id": character_id,
                    }
                )
                if "Item" not in response:
                    return None
                ret_dict = dict()
                for key in ("happiness", "sadness", "fear", "anger", "disgust", "surprise", "shyness"):
                    value = response["Item"].get(key, None)
                    if value is not None:
                        ret_dict[key] = int(value)
                return ret_dict
            except ClientError as e:
                traceback_str = traceback.format_exc()
                msg = f"Failed to get emotion for character_id={character_id} from DynamoDB: {traceback_str}"
                self.logger.error(msg)
                raise e
