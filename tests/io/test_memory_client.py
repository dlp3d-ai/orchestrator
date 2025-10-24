import logging
import os
import socket
import time

import pytest

from orchestrator.io.memory.database_memory_client import NoMatchingMediumTermMemoryError
from orchestrator.io.memory.dynamodb_memory_client import DynamoDBMemoryClient
from orchestrator.io.memory.mongodb_memory_client import MongoDBMemoryClient

# MongoDB connection configuration
MONGODB_HOST = os.environ.get("MONGODB_HOST")
MONGODB_PORT = int(os.environ.get("MONGODB_PORT", 27017))
MONGODB_MEMORY_DB = os.environ.get("MONGODB_MEMORY_DB")
MONGODB_AUTH_DATABASE = MONGODB_MEMORY_DB
MONGODB_MEMORY_USER = os.environ.get("MONGODB_MEMORY_USER")
MONGODB_MEMORY_PASSWORD = os.environ.get("MONGODB_MEMORY_PASSWORD")
# DynamoDB
REGION_NAME = "ap-southeast-1"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", None)
# Common
USER_ID = "a9ca656c-9091-70b9-007e-ac639f24845b"
CHARACTER_ID = "88801af2-6d2e-48f0-a413-c0058a448a26"


def is_mongodb_available() -> bool:
    """Check if MongoDB service is available.

    This function verifies if the MongoDB service is running by attempting
    to connect to the MongoDB port.

    Returns:
        bool:
            True if MongoDB port is connectable, False otherwise.
    """
    try:
        # Create socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # Set 5 second timeout

        # Attempt to connect to MongoDB port
        result = sock.connect_ex((MONGODB_HOST, MONGODB_PORT))
        sock.close()

        # If connection successful, result is 0
        return result == 0

    except Exception:
        # Any exception indicates connection failure
        return False


@pytest.fixture
def mongodb_memory_client() -> MongoDBMemoryClient:
    """Fixture for creating a MongoDBMemoryClient instance."""
    return MongoDBMemoryClient(
        host=MONGODB_HOST,
        port=MONGODB_PORT,
        username=MONGODB_MEMORY_USER,
        password=MONGODB_MEMORY_PASSWORD,
        database=MONGODB_MEMORY_DB,
        auth_database=MONGODB_AUTH_DATABASE,
        logger_cfg={"logger_name": "test_mongodb_memory_client", "console_level": logging.DEBUG},
    )


class TestMongoDBMemoryClient:
    """Test class for MongoDBMemoryClient.

    This class contains comprehensive tests for MongoDBMemoryClient
    functionality including emotion, relationship, memory operations, and
    cascade memory retrieval.
    """

    def test_init(self, mongodb_memory_client: MongoDBMemoryClient):
        """Test MongoDBMemoryClient initialization."""
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        assert mongodb_memory_client.host == MONGODB_HOST
        assert mongodb_memory_client.port == MONGODB_PORT
        assert mongodb_memory_client.database_name is not None

    def test_convert_unix_timestamp_to_str(self):
        """Test convert_unix_timestamp_to_str method."""
        test_timestamp_float = 1761276706.296444
        test_timestamp_shanghai_str = "2025-10-24 11:31:46,296"
        test_timestamp_tokyo_str = "2025-10-24 12:31:46,296"
        assert MongoDBMemoryClient.convert_unix_timestamp_to_str(test_timestamp_float) == test_timestamp_shanghai_str
        assert (
            MongoDBMemoryClient.convert_unix_timestamp_to_str(test_timestamp_float, timezone="Asia/Shanghai")
            == test_timestamp_shanghai_str
        )
        assert (
            MongoDBMemoryClient.convert_unix_timestamp_to_str(test_timestamp_float, timezone="Asia/Tokyo")
            == test_timestamp_tokyo_str
        )
        assert (
            MongoDBMemoryClient.convert_unix_timestamp_to_str(test_timestamp_float, timezone="NoSuchPlace")
            == test_timestamp_shanghai_str
        )

    @pytest.mark.asyncio
    async def test_emotion_table_operations(self, mongodb_memory_client: MongoDBMemoryClient):
        """Test emotion table operations: set, get, remove."""
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        await mongodb_memory_client.remove_emotion(CHARACTER_ID)
        result = await mongodb_memory_client.get_emotion(CHARACTER_ID)
        assert result is None

        test_emotion_data = {
            "happiness": 80,
            "sadness": 10,
            "fear": 5,
            "anger": 15,
            "disgust": 8,
            "surprise": 12,
            "shyness": 20,
        }

        await mongodb_memory_client.set_emotion(character_id=CHARACTER_ID, **test_emotion_data)

        retrieved_emotion = await mongodb_memory_client.get_emotion(CHARACTER_ID)
        assert retrieved_emotion is not None
        for key, value in test_emotion_data.items():
            assert retrieved_emotion[key] == value

    @pytest.mark.asyncio
    async def test_relationship_table_operations(self, mongodb_memory_client: MongoDBMemoryClient):
        """Test relationship table operations: set, get, remove."""
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        await mongodb_memory_client.remove_relationship(CHARACTER_ID)

        retrieved_relationship_after_removal = await mongodb_memory_client.get_relationship(CHARACTER_ID)
        assert retrieved_relationship_after_removal is None

        test_relationship = "Friend"
        test_score = 85

        await mongodb_memory_client.set_relationship(
            character_id=CHARACTER_ID, relationship=test_relationship, score=test_score
        )

        retrieved_relationship = await mongodb_memory_client.get_relationship(CHARACTER_ID)
        assert retrieved_relationship is not None
        relationship, score = retrieved_relationship
        assert relationship == test_relationship
        assert score == test_score

    @pytest.mark.asyncio
    async def test_long_term_memory_operations(self, mongodb_memory_client: MongoDBMemoryClient):
        """Test long term memory operations: set, get, remove."""
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        test_content = "This is a test long term memory content"
        test_last_short_term_timestamp = "2025-01-15 10:30:00,123"
        test_last_medium_term_timestamp = "2025-01-15 12:45:00,456"

        await mongodb_memory_client.set_long_term_memory(
            character_id=CHARACTER_ID,
            content=test_content,
            last_short_term_timestamp=test_last_short_term_timestamp,
            last_medium_term_timestamp=test_last_medium_term_timestamp,
        )

        retrieved_memory = await mongodb_memory_client.get_long_term_memory(CHARACTER_ID)
        assert retrieved_memory is not None
        assert retrieved_memory["character_id"] == CHARACTER_ID
        assert retrieved_memory["content"] == test_content
        assert retrieved_memory["last_short_term_timestamp"] == test_last_short_term_timestamp
        assert retrieved_memory["last_medium_term_timestamp"] == test_last_medium_term_timestamp

        test_content = "This is a test long term memory content 2"
        test_last_short_term_timestamp = "2025-01-15 10:30:00,124"
        test_last_medium_term_timestamp = "2025-01-15 12:45:00,457"
        await mongodb_memory_client.set_long_term_memory(
            character_id=CHARACTER_ID,
            content=test_content,
            last_short_term_timestamp=test_last_short_term_timestamp,
            last_medium_term_timestamp=test_last_medium_term_timestamp,
        )
        retrieved_memory = await mongodb_memory_client.get_long_term_memory(CHARACTER_ID)
        assert retrieved_memory is not None
        assert retrieved_memory["character_id"] == CHARACTER_ID
        assert retrieved_memory["content"] == test_content
        assert retrieved_memory["last_short_term_timestamp"] == test_last_short_term_timestamp
        assert retrieved_memory["last_medium_term_timestamp"] == test_last_medium_term_timestamp

        await mongodb_memory_client.remove_long_term_memory(CHARACTER_ID)

        retrieved_memory = await mongodb_memory_client.get_long_term_memory(CHARACTER_ID)
        assert retrieved_memory is None

    @pytest.mark.asyncio
    async def test_profile_memory_operations(self, mongodb_memory_client: MongoDBMemoryClient):
        """Test profile memory operations: set, get, remove."""
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        await mongodb_memory_client.remove_profile_memory(CHARACTER_ID)
        retrieved_memory = await mongodb_memory_client.get_profile_memory(CHARACTER_ID)
        assert retrieved_memory is None

        test_content = "This is a test profile memory content"
        test_last_short_term_timestamp = "2025-01-15 10:30:00,123"

        await mongodb_memory_client.set_profile_memory(
            character_id=CHARACTER_ID,
            content=test_content,
            last_short_term_timestamp=test_last_short_term_timestamp,
        )

        retrieved_memory = await mongodb_memory_client.get_profile_memory(CHARACTER_ID)
        assert retrieved_memory is not None
        assert retrieved_memory["character_id"] == CHARACTER_ID
        assert retrieved_memory["content"] == test_content
        assert retrieved_memory["last_short_term_timestamp"] == test_last_short_term_timestamp

        test_content = "This is a test profile memory content 2, user is a 20 years old male."
        test_last_short_term_timestamp = "2025-01-15 10:30:00,124"
        await mongodb_memory_client.set_profile_memory(
            character_id=CHARACTER_ID,
            content=test_content,
            last_short_term_timestamp=test_last_short_term_timestamp,
        )
        retrieved_memory = await mongodb_memory_client.get_profile_memory(CHARACTER_ID)
        assert retrieved_memory is not None
        assert retrieved_memory["character_id"] == CHARACTER_ID
        assert retrieved_memory["content"] == test_content
        assert retrieved_memory["last_short_term_timestamp"] == test_last_short_term_timestamp

    @pytest.mark.asyncio
    async def test_medium_term_memory_operations(self, mongodb_memory_client: MongoDBMemoryClient):
        """Test medium term memory operations: append, update, get, remove."""
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        test_start_timestamp_0 = "2025-01-15 10:30:00,123"
        test_content_0 = "This is a test medium term memory content 0"
        test_last_short_term_timestamp_0 = "2025-01-15 10:35:00,789"
        test_start_timestamp_1 = "2025-01-15 10:30:01,123"
        test_content_1 = "This is a test medium term memory content 1"
        test_last_short_term_timestamp_1 = "2025-01-15 10:35:01,789"

        await mongodb_memory_client.remove_medium_term_memory_not_after(CHARACTER_ID)

        with pytest.raises(NoMatchingMediumTermMemoryError):
            await mongodb_memory_client.update_medium_term_memory(
                character_id=CHARACTER_ID,
                start_timestamp=test_start_timestamp_0,
                content=test_content_0,
                last_short_term_timestamp=test_last_short_term_timestamp_0,
            )

        await mongodb_memory_client.append_medium_term_memory(
            character_id=CHARACTER_ID, start_timestamp=test_start_timestamp_0
        )
        await mongodb_memory_client.update_medium_term_memory(
            character_id=CHARACTER_ID,
            start_timestamp=test_start_timestamp_0,
            content=test_content_0,
            last_short_term_timestamp=test_last_short_term_timestamp_0,
        )
        await mongodb_memory_client.append_medium_term_memory(
            character_id=CHARACTER_ID, start_timestamp=test_start_timestamp_1
        )
        await mongodb_memory_client.update_medium_term_memory(
            character_id=CHARACTER_ID,
            start_timestamp=test_start_timestamp_1,
            content=test_content_1,
            last_short_term_timestamp=test_last_short_term_timestamp_1,
        )

        memories = await mongodb_memory_client.get_medium_term_memories_after(CHARACTER_ID)
        assert len(memories) == 2
        memories = await mongodb_memory_client.get_medium_term_memories_after(
            CHARACTER_ID, after_timestamp=test_start_timestamp_0
        )
        assert len(memories) == 1
        memories = await mongodb_memory_client.get_medium_term_memories_after(
            CHARACTER_ID, after_timestamp=test_start_timestamp_1
        )
        assert len(memories) == 0

        memories = await mongodb_memory_client.get_medium_term_memories_after(CHARACTER_ID)
        for memory in memories:
            if memory["start_timestamp"] == test_start_timestamp_0:
                assert memory is not None
                assert memory["character_id"] == CHARACTER_ID
                assert memory["content"] == test_content_0
                assert memory["last_short_term_timestamp"] == test_last_short_term_timestamp_0
            elif memory["start_timestamp"] == test_start_timestamp_1:
                assert memory is not None
                assert memory["character_id"] == CHARACTER_ID
                assert memory["content"] == test_content_1
                assert memory["last_short_term_timestamp"] == test_last_short_term_timestamp_1
            else:
                assert False

        await mongodb_memory_client.remove_medium_term_memory_not_after(
            character_id=CHARACTER_ID, not_after_timestamp=test_start_timestamp_0
        )
        memories = await mongodb_memory_client.get_medium_term_memories_after(CHARACTER_ID)
        assert len(memories) == 1
        await mongodb_memory_client.remove_medium_term_memory_not_after(
            character_id=CHARACTER_ID,
        )
        memories = await mongodb_memory_client.get_medium_term_memories_after(CHARACTER_ID)
        assert len(memories) == 0

    @pytest.mark.asyncio
    async def test_chat_history_operations(self, mongodb_memory_client: MongoDBMemoryClient):
        """Test chat history operations: append, get, remove."""
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        current_time = time.time()
        test_timestamp_1 = current_time - 3600
        test_timestamp_2 = current_time - 1800

        await mongodb_memory_client.remove_chat_history_memory(CHARACTER_ID)

        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=test_timestamp_1,
            role="user",
            content="Hello, how are you?",
            relationship="Friend",
        )

        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=test_timestamp_2,
            role="assistant",
            content="I'm doing well, thank you!",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )

        test_timestamp_1_str = MongoDBMemoryClient.convert_unix_timestamp_to_str(test_timestamp_1)
        chat_histories = await mongodb_memory_client.get_chat_histories_after(
            CHARACTER_ID, after_timestamp=test_timestamp_1_str
        )
        assert len(chat_histories) == 1
        chat_histories = await mongodb_memory_client.get_chat_histories_after(CHARACTER_ID)
        assert len(chat_histories) == 2

        user_message = None
        assistant_message = None
        for chat in chat_histories:
            if chat["role"] == "user" and chat["content"] == "Hello, how are you?":
                user_message = chat
            elif chat["role"] == "assistant" and chat["content"] == "I'm doing well, thank you!":
                assistant_message = chat

        assert user_message is not None
        assert user_message["relationship"] == "Friend"
        assert "happiness" not in user_message

        assert assistant_message is not None
        assert assistant_message["happiness"] == 80
        assert assistant_message["sadness"] == 5
        assert "relationship" not in assistant_message

        await mongodb_memory_client.remove_chat_history_memory(CHARACTER_ID)

        chat_histories_after_removal = await mongodb_memory_client.get_chat_histories_after(CHARACTER_ID)
        assert len(chat_histories_after_removal) == 0

    @pytest.mark.asyncio
    async def test_get_cascade_memories(self, mongodb_memory_client: MongoDBMemoryClient):
        """Test get_cascade_memories method with proper time ordering and
        exclusion logic."""
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        await mongodb_memory_client.remove_long_term_memory(CHARACTER_ID)

        await mongodb_memory_client.remove_medium_term_memory_not_after(CHARACTER_ID)
        await mongodb_memory_client.remove_chat_history_memory(CHARACTER_ID)

        current_time = time.time()

        chat_timestamp_1 = current_time - 3600
        chat_timestamp_2 = current_time - 3599
        chat_timestamp_3 = current_time - 3598
        chat_timestamp_4 = current_time - 3597
        chat_timestamp_5 = current_time - 1800
        chat_timestamp_6 = current_time - 1799
        chat_timestamp_7 = current_time - 1798
        chat_timestamp_8 = current_time - 900
        chat_timestamp_9 = current_time - 899
        chat_timestamp_10 = current_time - 898

        medium_term_timestamp_1_str = MongoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_1 - 1)
        last_short_term_timestamp_1_str = MongoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_4)
        medium_term_content_1 = "Medium term memory 1 for chat history 1, 2, 3, 4."
        medium_term_timestamp_2_str = MongoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_5 - 1)
        last_short_term_timestamp_2_str = MongoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_7)
        medium_term_content_2 = "Medium term memory 2 for chat history 5, 6, 7."
        medium_term_timestamp_3_str = MongoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_8 - 1)
        last_short_term_timestamp_3_str = MongoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_9)
        medium_term_content_3 = "Medium term memory 3 for chat history 8, 9."

        last_medium_term_timestamp_str = medium_term_timestamp_1_str
        last_short_term_timestamp_str = MongoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_4)
        long_term_content = "Long term memory content for chat history 1, 2, 3, 4."

        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_1,
            role="user",
            content="User message 1",
            relationship="Friend",
        )
        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_2,
            role="assistant",
            content="Assistant response 2",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )
        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_3,
            role="user",
            content="User message 3",
            relationship="Friend",
        )
        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_4,
            role="assistant",
            content="Assistant response 4",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )
        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_5,
            role="user",
            content="User message 5",
            relationship="Friend",
        )
        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_6,
            role="assistant",
            content="Assistant response 6",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )
        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_7,
            role="user",
            content="User message 7",
            relationship="Friend",
        )
        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_8,
            role="assistant",
            content="Assistant response 8",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )
        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_9,
            role="user",
            content="User message 9",
            relationship="Friend",
        )
        await mongodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_10,
            role="assistant",
            content="Assistant response 10",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )
        cascade_result = await mongodb_memory_client.get_cascade_memories(CHARACTER_ID)

        assert "short_term_memories" in cascade_result
        assert "long_term_memory" not in cascade_result
        assert "medium_term_memories" not in cascade_result

        short_term_memories = cascade_result["short_term_memories"]
        assert len(short_term_memories) == 10

        await mongodb_memory_client.append_medium_term_memory(
            character_id=CHARACTER_ID, start_timestamp=medium_term_timestamp_1_str
        )
        await mongodb_memory_client.update_medium_term_memory(
            character_id=CHARACTER_ID,
            start_timestamp=medium_term_timestamp_1_str,
            content=medium_term_content_1,
            last_short_term_timestamp=last_short_term_timestamp_1_str,
        )
        await mongodb_memory_client.append_medium_term_memory(
            character_id=CHARACTER_ID, start_timestamp=medium_term_timestamp_2_str
        )
        await mongodb_memory_client.update_medium_term_memory(
            character_id=CHARACTER_ID,
            start_timestamp=medium_term_timestamp_2_str,
            content=medium_term_content_2,
            last_short_term_timestamp=last_short_term_timestamp_2_str,
        )
        await mongodb_memory_client.append_medium_term_memory(
            character_id=CHARACTER_ID, start_timestamp=medium_term_timestamp_3_str
        )
        await mongodb_memory_client.update_medium_term_memory(
            character_id=CHARACTER_ID,
            start_timestamp=medium_term_timestamp_3_str,
            content=medium_term_content_3,
            last_short_term_timestamp=last_short_term_timestamp_3_str,
        )
        cascade_result = await mongodb_memory_client.get_cascade_memories(CHARACTER_ID)

        assert "short_term_memories" in cascade_result
        assert "medium_term_memories" in cascade_result
        assert "long_term_memory" not in cascade_result

        short_term_memories = cascade_result["short_term_memories"]
        assert len(short_term_memories) == 1
        medium_term_memories = cascade_result["medium_term_memories"]
        assert len(medium_term_memories) == 3

        await mongodb_memory_client.set_long_term_memory(
            character_id=CHARACTER_ID,
            content=long_term_content,
            last_short_term_timestamp=last_short_term_timestamp_str,
            last_medium_term_timestamp=last_medium_term_timestamp_str,
        )

        cascade_result = await mongodb_memory_client.get_cascade_memories(CHARACTER_ID)

        assert "long_term_memory" in cascade_result
        assert "short_term_memories" in cascade_result
        assert "medium_term_memories" in cascade_result

        long_term_memory = cascade_result["long_term_memory"]
        assert long_term_memory["character_id"] == CHARACTER_ID
        assert long_term_memory["content"] == long_term_content
        assert long_term_memory["last_medium_term_timestamp"] == last_medium_term_timestamp_str

        medium_term_memories = cascade_result["medium_term_memories"]
        assert len(medium_term_memories) == 2
        assert medium_term_memories[0]["content"] == medium_term_content_2
        assert medium_term_memories[1]["content"] == medium_term_content_3

        short_term_memories = cascade_result["short_term_memories"]
        assert len(short_term_memories) == 1
        assert short_term_memories[0]["content"] == "Assistant response 10"


@pytest.fixture
def dynamodb_memory_client() -> DynamoDBMemoryClient:
    """Fixture for creating a DynamoDBMemoryClient instance."""
    return DynamoDBMemoryClient(
        region_name=REGION_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        logger_cfg={"logger_name": "test_dynamodb_memory_client", "console_level": logging.DEBUG},
    )


class TestDynamoDBMemoryClient:
    """Test class for DynamoDBMemoryClient.

    This class contains comprehensive tests for DynamoDBMemoryClient
    functionality including emotion, relationship, memory operations, and
    cascade memory retrieval.
    """

    def test_init(self, dynamodb_memory_client: DynamoDBMemoryClient):
        """Test DynamoDBMemoryClient initialization."""
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        assert dynamodb_memory_client.region_name == REGION_NAME
        assert dynamodb_memory_client.aws_access_key_id == AWS_ACCESS_KEY_ID
        assert dynamodb_memory_client.aws_secret_access_key == AWS_SECRET_ACCESS_KEY
        assert dynamodb_memory_client.session is not None

    def test_convert_unix_timestamp_to_str(self):
        """Test convert_unix_timestamp_to_str method."""
        test_timestamp_float = 1761276706.296444
        test_timestamp_shanghai_str = "2025-10-24 11:31:46,296"
        test_timestamp_tokyo_str = "2025-10-24 12:31:46,296"
        assert MongoDBMemoryClient.convert_unix_timestamp_to_str(test_timestamp_float) == test_timestamp_shanghai_str
        assert (
            MongoDBMemoryClient.convert_unix_timestamp_to_str(test_timestamp_float, timezone="Asia/Shanghai")
            == test_timestamp_shanghai_str
        )
        assert (
            MongoDBMemoryClient.convert_unix_timestamp_to_str(test_timestamp_float, timezone="Asia/Tokyo")
            == test_timestamp_tokyo_str
        )
        assert (
            MongoDBMemoryClient.convert_unix_timestamp_to_str(test_timestamp_float, timezone="NoSuchPlace")
            == test_timestamp_shanghai_str
        )

    @pytest.mark.asyncio
    async def test_emotion_table_operations(self, dynamodb_memory_client: DynamoDBMemoryClient):
        """Test emotion table operations: set, get, remove."""
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        # Remove before testing
        await dynamodb_memory_client.remove_emotion(CHARACTER_ID)
        # Test getting emotion after removal - should raise exception
        result = await dynamodb_memory_client.get_emotion(CHARACTER_ID)
        assert result is None

        # Test data
        test_emotion_data = {
            "happiness": 80,
            "sadness": 10,
            "fear": 5,
            "anger": 15,
            "disgust": 8,
            "surprise": 12,
            "shyness": 20,
        }

        # Test setting emotion
        await dynamodb_memory_client.set_emotion(character_id=CHARACTER_ID, **test_emotion_data)

        # Test getting emotion - should return the same data
        retrieved_emotion = await dynamodb_memory_client.get_emotion(CHARACTER_ID)
        assert retrieved_emotion is not None
        for key, value in test_emotion_data.items():
            assert retrieved_emotion[key] == value
        # Keep the item for manual inspection

    @pytest.mark.asyncio
    async def test_relationship_table_operations(self, dynamodb_memory_client: DynamoDBMemoryClient):
        """Test relationship table operations: set, get, remove."""
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")

        # Remove before testing
        await dynamodb_memory_client.remove_relationship(CHARACTER_ID)

        # Test getting relationship after removal - should return None
        retrieved_relationship_after_removal = await dynamodb_memory_client.get_relationship(CHARACTER_ID)
        assert retrieved_relationship_after_removal is None

        # Test data
        test_relationship = "Friend"
        test_score = 85

        # Test setting relationship
        await dynamodb_memory_client.set_relationship(
            character_id=CHARACTER_ID, relationship=test_relationship, score=test_score
        )

        # Test getting relationship - should return the same data
        retrieved_relationship = await dynamodb_memory_client.get_relationship(CHARACTER_ID)
        assert retrieved_relationship is not None
        relationship, score = retrieved_relationship
        assert relationship == test_relationship
        assert score == test_score
        # Keep the item for manual inspection

    @pytest.mark.asyncio
    async def test_long_term_memory_operations(self, dynamodb_memory_client: DynamoDBMemoryClient):
        """Test long term memory operations: set, get, remove."""
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        # Test data
        test_content = "This is a test long term memory content"
        test_last_short_term_timestamp = "2025-01-15 10:30:00,123"
        test_last_medium_term_timestamp = "2025-01-15 12:45:00,456"

        # Test setting long term memory
        await dynamodb_memory_client.set_long_term_memory(
            character_id=CHARACTER_ID,
            content=test_content,
            last_short_term_timestamp=test_last_short_term_timestamp,
            last_medium_term_timestamp=test_last_medium_term_timestamp,
        )

        # Test getting long term memory - should return the same data
        retrieved_memory = await dynamodb_memory_client.get_long_term_memory(CHARACTER_ID)
        assert retrieved_memory is not None
        assert retrieved_memory["character_id"] == CHARACTER_ID
        assert retrieved_memory["content"] == test_content
        assert retrieved_memory["last_short_term_timestamp"] == test_last_short_term_timestamp
        assert retrieved_memory["last_medium_term_timestamp"] == test_last_medium_term_timestamp

        # Test set again with updated data
        test_content = "This is a test long term memory content 2"
        test_last_short_term_timestamp = "2025-01-15 10:30:00,124"
        test_last_medium_term_timestamp = "2025-01-15 12:45:00,457"
        await dynamodb_memory_client.set_long_term_memory(
            character_id=CHARACTER_ID,
            content=test_content,
            last_short_term_timestamp=test_last_short_term_timestamp,
            last_medium_term_timestamp=test_last_medium_term_timestamp,
        )
        retrieved_memory = await dynamodb_memory_client.get_long_term_memory(CHARACTER_ID)
        assert retrieved_memory is not None
        assert retrieved_memory["character_id"] == CHARACTER_ID
        assert retrieved_memory["content"] == test_content
        assert retrieved_memory["last_short_term_timestamp"] == test_last_short_term_timestamp
        assert retrieved_memory["last_medium_term_timestamp"] == test_last_medium_term_timestamp

        # Test removing long term memory
        await dynamodb_memory_client.remove_long_term_memory(CHARACTER_ID)

        # Test getting long term memory after removal - should raise exception
        retrieved_memory = await dynamodb_memory_client.get_long_term_memory(CHARACTER_ID)
        assert retrieved_memory is None

    @pytest.mark.asyncio
    async def test_profile_memory_operations(self, dynamodb_memory_client: DynamoDBMemoryClient):
        """Test profile memory operations: set, get, remove."""
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        # Remove before testing
        await dynamodb_memory_client.remove_profile_memory(CHARACTER_ID)
        # Test getting profile memory after removal - should raise exception
        retrieved_memory = await dynamodb_memory_client.get_profile_memory(CHARACTER_ID)
        assert retrieved_memory is None

        # Test data
        test_content = "This is a test profile memory content"
        test_last_short_term_timestamp = "2025-01-15 10:30:00,123"

        # Test setting profile memory
        await dynamodb_memory_client.set_profile_memory(
            character_id=CHARACTER_ID,
            content=test_content,
            last_short_term_timestamp=test_last_short_term_timestamp,
        )

        # Test getting profile memory - should return the same data
        retrieved_memory = await dynamodb_memory_client.get_profile_memory(CHARACTER_ID)
        assert retrieved_memory is not None
        assert retrieved_memory["character_id"] == CHARACTER_ID
        assert retrieved_memory["content"] == test_content
        assert retrieved_memory["last_short_term_timestamp"] == test_last_short_term_timestamp
        # keep the item for manual inspection

    @pytest.mark.asyncio
    async def test_medium_term_memory_operations(self, dynamodb_memory_client: DynamoDBMemoryClient):
        """Test medium term memory operations: append, update, get, remove."""
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        # Test data 0
        test_start_timestamp_0 = "2025-01-15 10:30:00,123"
        test_content_0 = "This is a test medium term memory content 0"
        test_last_short_term_timestamp_0 = "2025-01-15 10:35:00,789"
        # Test data 1
        test_start_timestamp_1 = "2025-01-15 10:30:01,123"
        test_content_1 = "This is a test medium term memory content 1"
        test_last_short_term_timestamp_1 = "2025-01-15 10:35:01,789"

        # Remove before testing
        await dynamodb_memory_client.remove_medium_term_memory_not_after(CHARACTER_ID)

        # Test update before appending
        with pytest.raises(NoMatchingMediumTermMemoryError):
            await dynamodb_memory_client.update_medium_term_memory(
                character_id=CHARACTER_ID,
                start_timestamp=test_start_timestamp_0,
                content=test_content_0,
                last_short_term_timestamp=test_last_short_term_timestamp_0,
            )

        # Test appending medium term memory
        await dynamodb_memory_client.append_medium_term_memory(
            character_id=CHARACTER_ID, start_timestamp=test_start_timestamp_0
        )
        # Test updating medium term memory
        await dynamodb_memory_client.update_medium_term_memory(
            character_id=CHARACTER_ID,
            start_timestamp=test_start_timestamp_0,
            content=test_content_0,
            last_short_term_timestamp=test_last_short_term_timestamp_0,
        )
        # Test appending medium term memory
        await dynamodb_memory_client.append_medium_term_memory(
            character_id=CHARACTER_ID, start_timestamp=test_start_timestamp_1
        )
        # Test updating medium term memory
        await dynamodb_memory_client.update_medium_term_memory(
            character_id=CHARACTER_ID,
            start_timestamp=test_start_timestamp_1,
            content=test_content_1,
            last_short_term_timestamp=test_last_short_term_timestamp_1,
        )

        # Test getting medium term memories
        memories = await dynamodb_memory_client.get_medium_term_memories_after(CHARACTER_ID)
        assert len(memories) == 2
        memories = await dynamodb_memory_client.get_medium_term_memories_after(
            CHARACTER_ID, after_timestamp=test_start_timestamp_0
        )
        assert len(memories) == 1
        memories = await dynamodb_memory_client.get_medium_term_memories_after(
            CHARACTER_ID, after_timestamp=test_start_timestamp_1
        )
        assert len(memories) == 0

        # Find our test memory
        memories = await dynamodb_memory_client.get_medium_term_memories_after(CHARACTER_ID)
        for memory in memories:
            if memory["start_timestamp"] == test_start_timestamp_0:
                assert memory is not None
                assert memory["character_id"] == CHARACTER_ID
                assert memory["content"] == test_content_0
                assert memory["last_short_term_timestamp"] == test_last_short_term_timestamp_0
            elif memory["start_timestamp"] == test_start_timestamp_1:
                assert memory is not None
                assert memory["character_id"] == CHARACTER_ID
                assert memory["content"] == test_content_1
                assert memory["last_short_term_timestamp"] == test_last_short_term_timestamp_1
            else:
                assert False

        # Test removing earlier medium term memory
        await dynamodb_memory_client.remove_medium_term_memory_not_after(
            character_id=CHARACTER_ID, not_after_timestamp=test_start_timestamp_0
        )
        memories = await dynamodb_memory_client.get_medium_term_memories_after(CHARACTER_ID)
        assert len(memories) == 1
        # Test removing all medium term memory
        await dynamodb_memory_client.remove_medium_term_memory_not_after(
            character_id=CHARACTER_ID,
        )
        memories = await dynamodb_memory_client.get_medium_term_memories_after(CHARACTER_ID)
        assert len(memories) == 0

    @pytest.mark.asyncio
    async def test_chat_history_operations(self, dynamodb_memory_client: DynamoDBMemoryClient):
        """Test chat history operations: append, get, remove."""
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        # Test data
        current_time = time.time()
        test_timestamp_1 = current_time - 3600  # 1 hour ago
        test_timestamp_2 = current_time - 1800  # 30 minutes ago

        # Remove chat history before testing
        await dynamodb_memory_client.remove_chat_history_memory(CHARACTER_ID)

        # Test appending user chat history
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=test_timestamp_1,
            role="user",
            content="Hello, how are you?",
            relationship="Friend",
        )

        # Test appending assistant chat history
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=test_timestamp_2,
            role="assistant",
            content="I'm doing well, thank you!",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )

        # Test getting chat histories
        test_timestamp_1_str = DynamoDBMemoryClient.convert_unix_timestamp_to_str(test_timestamp_1)
        chat_histories = await dynamodb_memory_client.get_chat_histories_after(
            CHARACTER_ID, after_timestamp=test_timestamp_1_str
        )
        assert len(chat_histories) == 1
        chat_histories = await dynamodb_memory_client.get_chat_histories_after(CHARACTER_ID)
        assert len(chat_histories) == 2

        # Verify user message
        user_message = None
        assistant_message = None
        for chat in chat_histories:
            if chat["role"] == "user" and chat["content"] == "Hello, how are you?":
                user_message = chat
            elif chat["role"] == "assistant" and chat["content"] == "I'm doing well, thank you!":
                assistant_message = chat

        assert user_message is not None
        assert user_message["relationship"] == "Friend"
        assert "happiness" not in user_message

        assert assistant_message is not None
        assert assistant_message["happiness"] == 80
        assert assistant_message["sadness"] == 5
        assert "relationship" not in assistant_message

        # Test removing chat history
        await dynamodb_memory_client.remove_chat_history_memory(CHARACTER_ID)

        # Verify removal
        chat_histories_after_removal = await dynamodb_memory_client.get_chat_histories_after(CHARACTER_ID)
        assert len(chat_histories_after_removal) == 0

    @pytest.mark.asyncio
    async def test_get_cascade_memories(self, dynamodb_memory_client: DynamoDBMemoryClient):
        """Test get_cascade_memories method with proper time ordering and
        exclusion logic."""
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        # Clean up any existing data first
        await dynamodb_memory_client.remove_long_term_memory(CHARACTER_ID)

        await dynamodb_memory_client.remove_medium_term_memory_not_after(CHARACTER_ID)
        await dynamodb_memory_client.remove_chat_history_memory(CHARACTER_ID)

        # Create test data with proper time ordering
        current_time = time.time()

        # Chat history timestamps (after medium term)
        chat_timestamp_1 = current_time - 3600  # 1 hour ago
        chat_timestamp_2 = current_time - 3599
        chat_timestamp_3 = current_time - 3598
        chat_timestamp_4 = current_time - 3597
        chat_timestamp_5 = current_time - 1800  # 30 minutes ago
        chat_timestamp_6 = current_time - 1799
        chat_timestamp_7 = current_time - 1798
        chat_timestamp_8 = current_time - 900  # 15 minutes ago
        chat_timestamp_9 = current_time - 899
        chat_timestamp_10 = current_time - 898
        # Medium term memory timestamps (after long term)
        # 1, 2, 3, 4
        medium_term_timestamp_1_str = DynamoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_1 - 1)
        last_short_term_timestamp_1_str = DynamoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_4)
        medium_term_content_1 = "Medium term memory 1 for chat history 1, 2, 3, 4."
        # 5, 6, 7
        medium_term_timestamp_2_str = DynamoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_5 - 1)
        last_short_term_timestamp_2_str = DynamoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_7)
        medium_term_content_2 = "Medium term memory 2 for chat history 5, 6, 7."
        # 8, 9
        medium_term_timestamp_3_str = DynamoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_8 - 1)
        last_short_term_timestamp_3_str = DynamoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_9)
        medium_term_content_3 = "Medium term memory 3 for chat history 8, 9."

        # Long term memory timestamp (earliest)
        # 1, 2, 3, 4
        last_medium_term_timestamp_str = medium_term_timestamp_1_str
        last_short_term_timestamp_str = DynamoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamp_4)
        long_term_content = "Long term memory content for chat history 1, 2, 3, 4."

        # append chat history
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_1,
            role="user",
            content="User message 1",
            relationship="Friend",
        )
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_2,
            role="assistant",
            content="Assistant response 2",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_3,
            role="user",
            content="User message 3",
            relationship="Friend",
        )
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_4,
            role="assistant",
            content="Assistant response 4",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_5,
            role="user",
            content="User message 5",
            relationship="Friend",
        )
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_6,
            role="assistant",
            content="Assistant response 6",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_7,
            role="user",
            content="User message 7",
            relationship="Friend",
        )
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_8,
            role="assistant",
            content="Assistant response 8",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_9,
            role="user",
            content="User message 9",
            relationship="Friend",
        )
        await dynamodb_memory_client.append_chat_history(
            character_id=CHARACTER_ID,
            unix_timestamp=chat_timestamp_10,
            role="assistant",
            content="Assistant response 10",
            happiness=80,
            sadness=5,
            fear=2,
            anger=3,
            disgust=1,
            surprise=8,
            shyness=15,
        )
        # Verify only chat history
        cascade_result = await dynamodb_memory_client.get_cascade_memories(CHARACTER_ID)

        # Verify structure
        assert "short_term_memories" in cascade_result
        assert "long_term_memory" not in cascade_result
        assert "medium_term_memories" not in cascade_result

        # Verify short term memories (chat histories)
        short_term_memories = cascade_result["short_term_memories"]
        assert len(short_term_memories) == 10

        # append and update medium term memory
        await dynamodb_memory_client.append_medium_term_memory(
            character_id=CHARACTER_ID, start_timestamp=medium_term_timestamp_1_str
        )
        await dynamodb_memory_client.update_medium_term_memory(
            character_id=CHARACTER_ID,
            start_timestamp=medium_term_timestamp_1_str,
            content=medium_term_content_1,
            last_short_term_timestamp=last_short_term_timestamp_1_str,
        )
        await dynamodb_memory_client.append_medium_term_memory(
            character_id=CHARACTER_ID, start_timestamp=medium_term_timestamp_2_str
        )
        await dynamodb_memory_client.update_medium_term_memory(
            character_id=CHARACTER_ID,
            start_timestamp=medium_term_timestamp_2_str,
            content=medium_term_content_2,
            last_short_term_timestamp=last_short_term_timestamp_2_str,
        )
        await dynamodb_memory_client.append_medium_term_memory(
            character_id=CHARACTER_ID, start_timestamp=medium_term_timestamp_3_str
        )
        await dynamodb_memory_client.update_medium_term_memory(
            character_id=CHARACTER_ID,
            start_timestamp=medium_term_timestamp_3_str,
            content=medium_term_content_3,
            last_short_term_timestamp=last_short_term_timestamp_3_str,
        )
        # Verify short term and medium term
        cascade_result = await dynamodb_memory_client.get_cascade_memories(CHARACTER_ID)

        # Verify structure
        assert "short_term_memories" in cascade_result
        assert "medium_term_memories" in cascade_result
        assert "long_term_memory" not in cascade_result

        # Verify short term memories (chat histories)
        short_term_memories = cascade_result["short_term_memories"]
        assert len(short_term_memories) == 1
        # Verify medium term memories
        medium_term_memories = cascade_result["medium_term_memories"]
        assert len(medium_term_memories) == 3

        # Create long term memory
        await dynamodb_memory_client.set_long_term_memory(
            character_id=CHARACTER_ID,
            content=long_term_content,
            last_short_term_timestamp=last_short_term_timestamp_str,
            last_medium_term_timestamp=last_medium_term_timestamp_str,
        )
        # Normally, medium term memory 1 should be removed
        # after it has been concluded by the long term memory.
        # Here we keep it for manual inspection, and it is not expected
        # to be found in the cascade result.

        # Test get_cascade_memories
        cascade_result = await dynamodb_memory_client.get_cascade_memories(CHARACTER_ID)

        # Verify structure
        assert "long_term_memory" in cascade_result
        assert "short_term_memories" in cascade_result
        assert "medium_term_memories" in cascade_result

        # Verify long term memory
        long_term_memory = cascade_result["long_term_memory"]
        assert long_term_memory["character_id"] == CHARACTER_ID
        assert long_term_memory["content"] == long_term_content
        assert long_term_memory["last_medium_term_timestamp"] == last_medium_term_timestamp_str

        # Verify medium term memories
        medium_term_memories = cascade_result["medium_term_memories"]
        assert len(medium_term_memories) == 2
        assert medium_term_memories[0]["content"] == medium_term_content_2
        assert medium_term_memories[1]["content"] == medium_term_content_3

        # Verify short term memories (chat histories)
        short_term_memories = cascade_result["short_term_memories"]
        assert len(short_term_memories) == 1
        assert short_term_memories[0]["content"] == "Assistant response 10"
        # Do not clean up, keep them for manual inspection
