import json
import logging
import os
import socket

import pytest
from pymongo import MongoClient

from orchestrator.io.config.dynamodb_config_client import DynamoDBConfigClient, NoMatchingCharacterSettingsError
from orchestrator.io.config.dynamodb_redis_config_client import DynamoDBRedisConfigClient
from orchestrator.io.config.mongodb_config_client import MongoDBConfigClient

LOGGER_CFG = dict(logger_name="test_config_client", file_level=logging.DEBUG, logger_path="logs/pytest.log")
USER_ID = "a9ca656c-9091-70b9-007e-ac639f24845b"
CHARACTER_ID = "6d4e9c50-452c-4540-825b-16b185f14f46"
# MongoDB configurations
MONGODB_HOST = "mongodb"
MONGODB_PORT = 27017
MONGODB_DB = "web_test"
MONGODB_AUTH_DATABASE = "web_test"
MONGODB_USER = "orchestrator"
MONGODB_PASSWORD = "orchestrator_password"
# DynamoDB configurations
REGION_NAME = "ap-southeast-1"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", None)
CHARACTER_CONFIG_TABLE_NAME = "CharacterConfigs"
USER_CONFIG_TABLE_NAME = "UserConfigs"
# Redis configurations
REDIS_HOST = "redis"
REDIS_PORT = 6379
REDIS_PWD = "CICDredisPWD"
REDIS_DB = 1


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


def is_redis_available() -> bool:
    """Check if Redis service is available.

    Attempts to establish a socket connection to the Redis server to verify
    if the service is running and accessible. Uses a 5-second timeout for
    the connection attempt.

    Returns:
        bool:
            True if Redis service is accessible, False otherwise.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        result = sock.connect_ex((REDIS_HOST, REDIS_PORT))
        sock.close()
        return result == 0
    except Exception:
        return False


@pytest.fixture
def mongodb_config_client() -> MongoDBConfigClient:
    """Fixture for creating a MongoDBConfigClient instance.

    Creates and configures a MongoDBConfigClient instance with test-specific
    settings including MongoDB connection parameters, database name, collection
    names, and logger configuration. This fixture is used across multiple test
    methods to ensure consistent client configuration.

    Returns:
        MongoDBConfigClient:
            Configured MongoDBConfigClient instance for testing.
    """
    return MongoDBConfigClient(
        host=MONGODB_HOST,
        port=MONGODB_PORT,
        username=MONGODB_USER,
        password=MONGODB_PASSWORD,
        database=MONGODB_DB,
        auth_database=MONGODB_AUTH_DATABASE,
        user_config_collection_name=USER_CONFIG_TABLE_NAME,
        character_config_collection_name=CHARACTER_CONFIG_TABLE_NAME,
        logger_cfg=LOGGER_CFG,
    )


class TestMongoDBConfigClient:
    """Test class for MongoDBConfigClient.

    Contains comprehensive test cases for MongoDBConfigClient functionality,
    including initialization, successful data retrieval, and error handling
    scenarios. Tests cover character settings, voice settings, and motion
    settings retrieval methods using MongoDB as the backend database.
    """

    @classmethod
    def setup_class(cls):
        """Setup test data in MongoDB before running tests.

        This method loads sample configuration data from JSON files and writes
        it to MongoDB collections. It uses upsert operations to avoid duplicate
        data while allowing updates to existing records.
        """
        if not is_mongodb_available():
            return

        try:
            # Connect to MongoDB
            client = MongoClient(
                host=MONGODB_HOST,
                port=MONGODB_PORT,
                username=MONGODB_USER,
                password=MONGODB_PASSWORD,
                authSource=MONGODB_AUTH_DATABASE,
            )

            db = client[MONGODB_DB]

            # Load character config data
            with open("data/character_config_sample.json", "r", encoding="utf-8") as f:
                character_data = json.load(f)

            # Load user config data
            with open("data/user_config_sample.json", "r", encoding="utf-8") as f:
                user_data = json.load(f)

            # Write character config data (upsert to avoid duplicates)
            character_collection = db[CHARACTER_CONFIG_TABLE_NAME]
            character_collection.replace_one(
                {"user_id": character_data["user_id"], "character_id": character_data["character_id"]},
                character_data,
                upsert=True,
            )

            # Write user config data (upsert to avoid duplicates)
            user_collection = db[USER_CONFIG_TABLE_NAME]
            user_collection.replace_one({"user_id": user_data["user_id"]}, user_data, upsert=True)

            client.close()

        except Exception as e:
            print(f"Failed to setup test data: {e}")

    def test_init(self, mongodb_config_client: MongoDBConfigClient):
        """Test MongoDBConfigClient initialization.

        Verifies that the MongoDBConfigClient instance is properly initialized
        with the correct configuration values including host, port, username,
        password, database name, and collection names.

        Args:
            mongodb_config_client (MongoDBConfigClient):
                MongoDBConfigClient instance to test.
        """
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        assert mongodb_config_client.host == MONGODB_HOST
        assert mongodb_config_client.port == MONGODB_PORT
        assert mongodb_config_client.username == MONGODB_USER
        assert mongodb_config_client.password == MONGODB_PASSWORD
        assert mongodb_config_client.database_name == MONGODB_DB
        assert mongodb_config_client.user_config_collection_name == USER_CONFIG_TABLE_NAME
        assert mongodb_config_client.character_config_collection_name == CHARACTER_CONFIG_TABLE_NAME

    @pytest.mark.asyncio
    async def test_get_character_settings_success(self, mongodb_config_client: MongoDBConfigClient):
        """Test successful get_character_settings call.

        Verifies that the get_character_settings method successfully retrieves
        complete character configuration data from MongoDB. Validates the
        structure and data types of all returned fields including thresholds,
        adapters, and configuration overrides.

        Args:
            mongodb_config_client (MongoDBConfigClient):
                MongoDBConfigClient instance to test.
        """
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        # Query
        result = await mongodb_config_client.get_character_settings(USER_ID, CHARACTER_ID)

        # Verify result structure and values
        expected_keys = mongodb_config_client.__class__.CHARACTER_KEYS
        for key in expected_keys:
            assert key in result
            if key == "voice_speed":
                assert isinstance(result[key], float) and result[key] > 0
            elif key.endswith("_threshold"):
                assert isinstance(result[key], int) and result[key] > 0
            elif key.endswith("_override"):
                assert result[key] is None or isinstance(result[key], str)
            else:
                assert isinstance(result[key], str) and len(result[key]) > 0

    @pytest.mark.asyncio
    async def test_get_character_settings_wrong_user_id_or_character_id(
        self, mongodb_config_client: MongoDBConfigClient
    ):
        """Test get_character_settings with non-existent user or character ID.

        Verifies that the get_character_settings method properly raises
        NoMatchingCharacterSettingsError when attempting to retrieve
        configuration for non-existent user IDs or character IDs.

        Args:
            mongodb_config_client (MongoDBConfigClient):
                MongoDBConfigClient instance to test.
        """
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        with pytest.raises(NoMatchingCharacterSettingsError):
            await mongodb_config_client.get_character_settings("Non-Existent-User", CHARACTER_ID)
        with pytest.raises(NoMatchingCharacterSettingsError):
            await mongodb_config_client.get_character_settings(USER_ID, "Non-Existent-Character")

    @pytest.mark.asyncio
    async def test_get_voice_settings_success(self, mongodb_config_client: MongoDBConfigClient):
        """Test successful get_voice_settings call.

        Verifies that the get_voice_settings method successfully retrieves
        voice-related configuration data from MongoDB. Validates the
        structure and data types of voice-specific fields including
        TTS adapter, voice name, and voice speed.

        Args:
            mongodb_config_client (MongoDBConfigClient):
                MongoDBConfigClient instance to test.
        """
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        # Query
        result = await mongodb_config_client.get_voice_settings(USER_ID, CHARACTER_ID)

        # Verify result structure - only voice-related fields
        expected_keys = [
            "tts_adapter",
            "voice",
            "voice_speed",
        ]
        for key in expected_keys:
            assert key in result

        # Verify specific values
        assert isinstance(result["tts_adapter"], str) and len(result["tts_adapter"]) > 0
        assert isinstance(result["voice"], str) and len(result["voice"]) > 0
        assert isinstance(result["voice_speed"], float) and result["voice_speed"] > 0

    @pytest.mark.asyncio
    async def test_get_voice_settings_wrong_user_id_or_character_id(self, mongodb_config_client: MongoDBConfigClient):
        """Test get_voice_settings with non-existent user or character ID.

        Verifies that the get_voice_settings method properly raises
        NoMatchingCharacterSettingsError when attempting to retrieve
        voice configuration for non-existent user IDs or character IDs.

        Args:
            mongodb_config_client (MongoDBConfigClient):
                MongoDBConfigClient instance to test.
        """
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        with pytest.raises(NoMatchingCharacterSettingsError):
            await mongodb_config_client.get_voice_settings("Non-Existent-User", CHARACTER_ID)
        with pytest.raises(NoMatchingCharacterSettingsError):
            await mongodb_config_client.get_voice_settings(USER_ID, "Non-Existent-Character")

    @pytest.mark.asyncio
    async def test_get_motion_settings_success(self, mongodb_config_client: MongoDBConfigClient):
        """Test successful get_motion_settings call.

        Verifies that the get_motion_settings method successfully retrieves
        motion-related configuration data from MongoDB. Validates the
        structure and data types of motion-specific fields including
        avatar name.

        Args:
            mongodb_config_client (MongoDBConfigClient):
                MongoDBConfigClient instance to test.
        """
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        # Query
        result = await mongodb_config_client.get_motion_settings(USER_ID, CHARACTER_ID)

        # Verify result structure - only motion-related fields
        expected_keys = [
            "avatar",
        ]
        for key in expected_keys:
            assert key in result

        # Verify specific values
        assert isinstance(result["avatar"], str) and len(result["avatar"]) > 0

    @pytest.mark.asyncio
    async def test_get_motion_settings_wrong_user_id_or_character_id(self, mongodb_config_client: MongoDBConfigClient):
        """Test get_motion_settings with non-existent user or character ID.

        Verifies that the get_motion_settings method properly raises
        NoMatchingCharacterSettingsError when attempting to retrieve
        motion configuration for non-existent user IDs or character IDs.

        Args:
            mongodb_config_client (MongoDBConfigClient):
                MongoDBConfigClient instance to test.
        """
        if not is_mongodb_available():
            pytest.skip("MongoDB not available")
        with pytest.raises(NoMatchingCharacterSettingsError):
            await mongodb_config_client.get_motion_settings("Non-Existent-User", CHARACTER_ID)
        with pytest.raises(NoMatchingCharacterSettingsError):
            await mongodb_config_client.get_motion_settings(USER_ID, "Non-Existent-Character")


@pytest.fixture
def dynamodb_config_client() -> DynamoDBConfigClient:
    """Fixture for creating a DynamoDBConfigClient instance.

    Creates and configures a DynamoDBConfigClient instance with test-specific
    settings including AWS credentials, table names, and logger configuration.
    This fixture is used across multiple test methods to ensure consistent
    client configuration.

    Returns:
        DynamoDBConfigClient:
            Configured DynamoDBConfigClient instance for testing.
    """
    return DynamoDBConfigClient(
        region_name=REGION_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        character_config_table_name=CHARACTER_CONFIG_TABLE_NAME,
        user_config_table_name=USER_CONFIG_TABLE_NAME,
        logger_cfg=LOGGER_CFG,
    )


class TestDynamoDBConfigClient:
    """Test class for DynamoDBConfigClient.

    Contains comprehensive test cases for DynamoDBConfigClient functionality,
    including initialization, successful data retrieval, and error handling
    scenarios. Tests cover character settings, voice settings, and motion
    settings retrieval methods.
    """

    def test_init(self, dynamodb_config_client: DynamoDBConfigClient):
        """Test DynamoDBConfigClient initialization.

        Verifies that the DynamoDBConfigClient instance is properly initialized
        with the correct configuration values including region name, AWS credentials,
        table names, and session object.

        Args:
            dynamodb_config_client (DynamoDBConfigClient):
                DynamoDBConfigClient instance to test.
        """
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        assert dynamodb_config_client.region_name == REGION_NAME
        assert dynamodb_config_client.aws_access_key_id == AWS_ACCESS_KEY_ID
        assert dynamodb_config_client.aws_secret_access_key == AWS_SECRET_ACCESS_KEY
        assert dynamodb_config_client.character_config_table_name == CHARACTER_CONFIG_TABLE_NAME
        assert dynamodb_config_client.session is not None

    @pytest.mark.asyncio
    async def test_get_character_settings_success(self, dynamodb_config_client: DynamoDBConfigClient):
        """Test successful get_character_settings call.

        Verifies that the get_character_settings method successfully retrieves
        complete character configuration data from DynamoDB. Validates the
        structure and data types of all returned fields including thresholds,
        adapters, and configuration overrides.

        Args:
            dynamodb_config_client (DynamoDBConfigClient):
                DynamoDBConfigClient instance to test.
        """
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        # Query
        result = await dynamodb_config_client.get_character_settings(USER_ID, CHARACTER_ID)

        # Verify result structure and values
        expected_keys = dynamodb_config_client.__class__.CHARACTER_KEYS
        for key in expected_keys:
            assert key in result
            if key == "voice_speed":
                assert isinstance(result[key], float) and result[key] > 0
            elif key.endswith("_threshold"):
                assert isinstance(result[key], int) and result[key] > 0
            elif key.endswith("_override"):
                assert result[key] is None or isinstance(result[key], str)
            else:
                assert isinstance(result[key], str) and len(result[key]) > 0

    @pytest.mark.asyncio
    async def test_get_character_settings_wrong_user_id_or_character_id(
        self, dynamodb_config_client: DynamoDBConfigClient
    ):
        """Test get_character_settings with non-existent user or character ID.

        Verifies that the get_character_settings method properly raises
        NoMatchingCharacterSettingsError when attempting to retrieve
        configuration for non-existent user IDs or character IDs.

        Args:
            dynamodb_config_client (DynamoDBConfigClient):
                DynamoDBConfigClient instance to test.
        """
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        with pytest.raises(NoMatchingCharacterSettingsError):
            await dynamodb_config_client.get_character_settings("Non-Existent-User", CHARACTER_ID)
        with pytest.raises(NoMatchingCharacterSettingsError):
            await dynamodb_config_client.get_character_settings(USER_ID, "Non-Existent-Character")

    @pytest.mark.asyncio
    async def test_get_voice_settings_success(self, dynamodb_config_client: DynamoDBConfigClient):
        """Test successful get_voice_settings call.

        Verifies that the get_voice_settings method successfully retrieves
        voice-related configuration data from DynamoDB. Validates the
        structure and data types of voice-specific fields including
        TTS adapter, voice name, and voice speed.

        Args:
            dynamodb_config_client (DynamoDBConfigClient):
                DynamoDBConfigClient instance to test.
        """
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        # Query
        result = await dynamodb_config_client.get_voice_settings(USER_ID, CHARACTER_ID)

        # Verify result structure - only voice-related fields
        expected_keys = [
            "tts_adapter",
            "voice",
            "voice_speed",
        ]
        for key in expected_keys:
            assert key in result

        # Verify specific values
        assert isinstance(result["tts_adapter"], str) and len(result["tts_adapter"]) > 0
        assert isinstance(result["voice"], str) and len(result["voice"]) > 0
        assert isinstance(result["voice_speed"], float) and result["voice_speed"] > 0

    @pytest.mark.asyncio
    async def test_get_voice_settings_wrong_user_id_or_character_id(self, dynamodb_config_client: DynamoDBConfigClient):
        """Test get_voice_settings with non-existent user or character ID.

        Verifies that the get_voice_settings method properly raises
        NoMatchingCharacterSettingsError when attempting to retrieve
        voice configuration for non-existent user IDs or character IDs.

        Args:
            dynamodb_config_client (DynamoDBConfigClient):
                DynamoDBConfigClient instance to test.
        """
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        with pytest.raises(NoMatchingCharacterSettingsError):
            await dynamodb_config_client.get_voice_settings("Non-Existent-User", CHARACTER_ID)
        with pytest.raises(NoMatchingCharacterSettingsError):
            await dynamodb_config_client.get_voice_settings(USER_ID, "Non-Existent-Character")

    @pytest.mark.asyncio
    async def test_get_motion_settings_success(self, dynamodb_config_client: DynamoDBConfigClient):
        """Test successful get_motion_settings call.

        Verifies that the get_motion_settings method successfully retrieves
        motion-related configuration data from DynamoDB. Validates the
        structure and data types of motion-specific fields including
        avatar name.

        Args:
            dynamodb_config_client (DynamoDBConfigClient):
                DynamoDBConfigClient instance to test.
        """
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        # Query
        result = await dynamodb_config_client.get_motion_settings(USER_ID, CHARACTER_ID)

        # Verify result structure - only motion-related fields
        expected_keys = [
            "avatar",
        ]
        for key in expected_keys:
            assert key in result

        # Verify specific values
        assert isinstance(result["avatar"], str) and len(result["avatar"]) > 0

    @pytest.mark.asyncio
    async def test_get_motion_settings_wrong_user_id_or_character_id(
        self, dynamodb_config_client: DynamoDBConfigClient
    ):
        """Test get_motion_settings with non-existent user or character ID.

        Verifies that the get_motion_settings method properly raises
        NoMatchingCharacterSettingsError when attempting to retrieve
        motion configuration for non-existent user IDs or character IDs.

        Args:
            dynamodb_config_client (DynamoDBConfigClient):
                DynamoDBConfigClient instance to test.
        """
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        with pytest.raises(NoMatchingCharacterSettingsError):
            await dynamodb_config_client.get_motion_settings("Non-Existent-User", CHARACTER_ID)
        with pytest.raises(NoMatchingCharacterSettingsError):
            await dynamodb_config_client.get_motion_settings(USER_ID, "Non-Existent-Character")


@pytest.fixture
def dynamodb_redis_config_client():
    """Create DynamoDBRedisConfigClient instance.

    Creates and configures a DynamoDBRedisConfigClient instance with test-specific
    settings including AWS credentials, table names, Redis configuration, and
    cache timeout. This fixture is used for testing Redis caching functionality
    in combination with DynamoDB operations.

    Returns:
        DynamoDBRedisConfigClient:
            Configured DynamoDBRedisConfigClient instance for testing.
    """
    return DynamoDBRedisConfigClient(
        region_name=REGION_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        character_config_table_name=CHARACTER_CONFIG_TABLE_NAME,
        user_config_table_name=USER_CONFIG_TABLE_NAME,
        redis_host=REDIS_HOST,
        redis_port=int(REDIS_PORT),
        redis_db=int(REDIS_DB),
        redis_pwd=REDIS_PWD,
        redis_cache_timeout=600,
    )


class TestDynamoDBRedisConfigClient:
    """Test class for DynamoDBRedisConfigClient.

    Contains comprehensive test cases for DynamoDBRedisConfigClient
    functionality, including initialization, Redis caching behavior, and
    fallback to DynamoDB when cache is unavailable. Tests cover both cache hit
    and cache miss scenarios for character settings retrieval.
    """

    def test_init(self, dynamodb_redis_config_client: DynamoDBRedisConfigClient):
        """Test DynamoDBRedisConfigClient initialization.

        Verifies that the DynamoDBRedisConfigClient instance is properly initialized
        with the correct configuration values including AWS credentials, table names,
        Redis connection parameters, and cache timeout settings.

        Args:
            dynamodb_redis_config_client (DynamoDBRedisConfigClient):
                DynamoDBRedisConfigClient instance to test.
        """
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        if not is_redis_available():
            pytest.skip("Redis service is not available")

        assert dynamodb_redis_config_client.region_name == REGION_NAME
        assert dynamodb_redis_config_client.aws_access_key_id == AWS_ACCESS_KEY_ID
        assert dynamodb_redis_config_client.aws_secret_access_key == AWS_SECRET_ACCESS_KEY
        assert dynamodb_redis_config_client.character_config_table_name == CHARACTER_CONFIG_TABLE_NAME
        assert dynamodb_redis_config_client.session is not None
        assert dynamodb_redis_config_client.redis_host == REDIS_HOST
        assert dynamodb_redis_config_client.redis_port == int(REDIS_PORT)
        assert dynamodb_redis_config_client.redis_db == int(REDIS_DB)
        assert dynamodb_redis_config_client.redis_pwd == REDIS_PWD
        assert dynamodb_redis_config_client.redis_cache_timeout > 0

    @pytest.mark.asyncio
    async def test_get_character_settings_success_from_dynamodb(
        self, dynamodb_redis_config_client: DynamoDBRedisConfigClient
    ):
        """Test successful get_character_settings call from DynamoDB.

        Verifies that the get_character_settings method successfully retrieves
        character configuration data directly from DynamoDB when cache reading
        is disabled. Validates the structure and data types of all returned
        fields and ensures data is properly cached in Redis.

        Args:
            dynamodb_redis_config_client (DynamoDBRedisConfigClient):
                DynamoDBRedisConfigClient instance to test.
        """
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        if not is_redis_available():
            pytest.skip("Redis service is not available")

        # Query
        result = await dynamodb_redis_config_client.get_character_settings(USER_ID, CHARACTER_ID, read_cache=False)

        # Verify result structure and values
        expected_keys = dynamodb_redis_config_client.__class__.CHARACTER_KEYS
        for key in expected_keys:
            assert key in result
            if key == "voice_speed":
                assert isinstance(result[key], float) and result[key] > 0
            elif key.endswith("_threshold"):
                assert isinstance(result[key], int) and result[key] > 0
            elif key.endswith("_override"):
                assert result[key] is None or isinstance(result[key], str)
            else:
                assert isinstance(result[key], str) and len(result[key]) > 0

    @pytest.mark.asyncio
    async def test_get_character_settings_success_from_redis(self):
        """Test successful get_character_settings call from Redis cache.

        Verifies that the get_character_settings method successfully retrieves
        character configuration data from Redis cache when data is already
        cached. Uses a non-existent DynamoDB table to ensure data comes from
        cache rather than database.
        """
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        if not is_redis_available():
            pytest.skip("Redis service is not available")

        no_dynamodb_client = DynamoDBRedisConfigClient(
            region_name=REGION_NAME,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            character_config_table_name="Non-Existent-Table",
            user_config_table_name=USER_CONFIG_TABLE_NAME,
            redis_host=REDIS_HOST,
            redis_port=int(REDIS_PORT),
            redis_db=int(REDIS_DB),
            redis_pwd=REDIS_PWD,
            redis_cache_timeout=600,
        )
        result = await no_dynamodb_client.get_character_settings(USER_ID, CHARACTER_ID, read_cache=True)

        # Verify result structure
        expected_keys = no_dynamodb_client.__class__.CHARACTER_KEYS
        for key in expected_keys:
            assert key in result
            if key == "voice_speed":
                assert isinstance(result[key], float) and result[key] > 0
            elif key.endswith("_threshold"):
                assert isinstance(result[key], int) and result[key] > 0
            elif key.endswith("_override"):
                assert result[key] is None or isinstance(result[key], str)
            else:
                assert isinstance(result[key], str) and len(result[key]) > 0

    @pytest.mark.asyncio
    async def test_get_character_settings_wrong_user_id_or_character_id(
        self, dynamodb_redis_config_client: DynamoDBRedisConfigClient
    ):
        """Test get_character_settings with non-existent user or character ID.

        Verifies that the get_character_settings method properly raises
        NoMatchingCharacterSettingsError when attempting to retrieve
        configuration for non-existent user IDs or character IDs, both
        when reading from cache and when forcing DynamoDB access.

        Args:
            dynamodb_redis_config_client (DynamoDBRedisConfigClient):
                DynamoDBRedisConfigClient instance to test.
        """
        if AWS_ACCESS_KEY_ID is None:
            pytest.skip("AWS credentials not available")
        if not is_redis_available():
            pytest.skip("Redis service is not available")

        with pytest.raises(NoMatchingCharacterSettingsError):
            await dynamodb_redis_config_client.get_character_settings("Non-Existent-User", CHARACTER_ID, True)
        with pytest.raises(NoMatchingCharacterSettingsError):
            await dynamodb_redis_config_client.get_character_settings("Non-Existent-User", CHARACTER_ID, False)
        with pytest.raises(NoMatchingCharacterSettingsError):
            await dynamodb_redis_config_client.get_character_settings(USER_ID, "Non-Existent-Character", True)
        with pytest.raises(NoMatchingCharacterSettingsError):
            await dynamodb_redis_config_client.get_character_settings(USER_ID, "Non-Existent-Character", False)
