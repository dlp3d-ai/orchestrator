from typing import Any, Dict, Union

from redis import asyncio as aioredis

from .dynamodb_config_client import DynamoDBConfigClient


class DynamoDBRedisConfigClient(DynamoDBConfigClient):
    """DynamoDB Redis configuration client class.

    Inherits from DynamoDBConfigClient with added Redis caching functionality.
    Provides asynchronous interaction with AWS DynamoDB while using Redis as a
    caching layer to improve data access performance. Main functionality is to
    retrieve character configuration information, supporting querying user and
    character configuration settings from DynamoDB tables while reducing direct
    DynamoDB access through Redis caching.
    """

    def __init__(
        self,
        region_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        user_config_table_name: str,
        character_config_table_name: str,
        redis_host: str,
        redis_port: int,
        redis_db: int,
        redis_pwd: Union[str, None],
        redis_cache_timeout: int = 600,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the DynamoDB Redis configuration client.

        Args:
            region_name (str):
                AWS region name, e.g., 'us-east-1'.
            aws_access_key_id (str):
                AWS access key ID.
            aws_secret_access_key (str):
                AWS secret access key.
            character_config_table_name (str):
                Name of the character configuration table in DynamoDB.
            redis_host (str):
                Redis server host address.
            redis_port (int):
                Redis server port number.
            redis_db (int):
                Redis database number.
            redis_pwd (Union[str, None]):
                Redis password, None if no password is required.
            redis_cache_timeout (int, optional):
                Redis cache expiration time in seconds. Defaults to 600.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        DynamoDBConfigClient.__init__(
            self,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            user_config_table_name=user_config_table_name,
            character_config_table_name=character_config_table_name,
            logger_cfg=logger_cfg,
        )
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_pwd = redis_pwd
        self.redis_cache_timeout = redis_cache_timeout

    async def get_character_settings(
        self,
        user_id: str,
        character_id: str,
        read_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get complete configuration settings for specified user and
        character.

        First attempts to read configuration information from Redis cache. If the cache doesn't exist
        or read_cache is False, queries from DynamoDB table and updates the result to Redis cache.
        Reduces direct DynamoDB access through caching mechanism to improve data reading performance.

        Args:
            user_id (str):
                User ID for locating user records in DynamoDB.
            character_id (str):
                Character ID for locating character records in DynamoDB.
            read_cache (bool, optional):
                Whether to allow reading from cache. Defaults to True. If False,
                forces connection to DynamoDB to read and updates values to Redis cache.

        Returns:
            Dict[str, Any]:
                Dictionary containing complete character configuration with the following fields:
                - avatar (str): Character name
                - tts_adapter (str): TTS adapter name
                - voice (str): Voice name
                - voice_speed (float): Voice speed
                - asr_adapter (str): ASR adapter name
                - classification_adapter (str): Classification adapter name
                - classification_model_override (str): Classification model override, may be empty string.
                - conversation_adapter (str): Conversation adapter name
                - conversation_model_override (str): Conversation model override, may be empty string.
                - prompt (str): User-edited conversation prompt.
                - reaction_adapter (str): Reaction adapter name
                - reaction_model_override (str): Reaction model override, may be empty string.
                - memory_adapter (str): Memory adapter name
                - memory_model_override (str): Memory model override, may be empty string.
                - acquaintance_threshold (int): Above this threshold, the relationship will be set to Acquaintance.
                - friend_threshold (int): Above this threshold, the relationship will be set to Friend.
                - situationship_threshold (int): Above this threshold, the relationship will be set to Situationship.
                - lover_threshold (int): Above this threshold, the relationship will be set to Lover.
                - neutral_threshold (int): All emotions below this threshold, Neutral is activated.
                - happiness_threshold (int): Above this threshold, the emotion will be activated.
                - sadness_threshold (int): Above this threshold, the emotion will be activated.
                - fear_threshold (int): Above this threshold, the emotion will be activated.
                - anger_threshold (int): Above this threshold, the emotion will be activated.
                - disgust_threshold (int): Above this threshold, the emotion will be activated.
                - surprise_threshold (int): Above this threshold, the emotion will be activated.
                - shyness_threshold (int): Above this threshold, the emotion will be activated.

        Raises:
            NoMatchingCharacterSettingsError:
                When no matching user and character configuration is found in DynamoDB.
        """
        redis_client = await aioredis.from_url(
            f"redis://{self.redis_host}:{self.redis_port}",
            password=self.redis_pwd,
            db=self.redis_db,
        )
        redis_name = f"character_settings:{user_id}:{character_id}"
        sync_from_dynamodb = True
        if read_cache:
            redis_dict = await redis_client.hgetall(redis_name)
            decoded_redis_dict = dict()
            for key, value in redis_dict.items():
                decoded_key = key.decode("utf-8")
                if decoded_key == "voice_speed":
                    decoded_redis_dict[decoded_key] = float(value)
                elif decoded_key.endswith("_threshold"):
                    decoded_redis_dict[decoded_key] = int(value)
                else:
                    decoded_redis_dict[decoded_key] = value.decode("utf-8")
            # check if all keys are in redis_dict
            if all(key in decoded_redis_dict for key in self.__class__.CHARACTER_KEYS):
                sync_from_dynamodb = False
                ret_dict = dict()
                for key in self.__class__.CHARACTER_KEYS:
                    ret_dict[key] = decoded_redis_dict[key]
        if sync_from_dynamodb:
            ret_dict = await super().get_character_settings(user_id, character_id)
            redis_dict = dict()
            for key, value in ret_dict.items():
                redis_dict[key] = value
            await redis_client.hset(redis_name, mapping=redis_dict)
            await redis_client.expire(redis_name, self.redis_cache_timeout)
        await redis_client.aclose()
        return ret_dict

    async def get_user_settings(
        self,
        user_id: str,
        read_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get settings for specified user.

        First attempts to read configuration information from Redis cache. If the cache doesn't exist
        or read_cache is False, queries from DynamoDB table and updates the result to Redis cache.
        Reduces direct DynamoDB access through caching mechanism to improve data reading performance.

        Args:
            user_id (str):
                User ID for locating user records in DynamoDB.
            read_cache (bool, optional):
                Whether to allow reading from cache. Defaults to True. If False,
                forces connection to DynamoDB to read and updates values to Redis cache.

        Returns:
            Dict[str, Any]:
                Dictionary containing user settings with the following fields:
                - openai_api_key (str): OpenAI API key
                - xai_api_key (str): XAI API key
                - anthropic_api_key (str): Anthropic API key
                - gemini_api_key (str): Gemini API key
                - deepseek_api_key (str): DeepSeek API key
                - sensenova_api_key (str): Sensenova API key
                - sensenova_ak (str): Sensenova AK
                - sensenova_sk (str): Sensenova SK
                - sensenovaomni_ak (str): SenseNova Omni AK
                - sensenovaomni_sk (str): SenseNova Omni SK
                - sensechat_ak (str): SenseChat AK
                - sensechat_sk (str): SenseChat SK
                - softsugar_app_id (str): SoftSugar App ID
                - softsugar_app_key (str): SoftSugar App Key
                - huoshan_appid (str): Huoshan App ID
                - huoshan_token (str): Huoshan Token
                - snese_tts_api_key (str): Sensenova TTS API Key
                - nova_tts_api_key (str): Nova TTS API Key
                - elevenlabs_api_key (str): ElevenLabs API Key
                - timezone (str): Timezone name

        Raises:
            NoMatchingUserSettingsError:
                When no matching user configuration is found in DynamoDB.
        """
        redis_client = await aioredis.from_url(
            f"redis://{self.redis_host}:{self.redis_port}",
            password=self.redis_pwd,
            db=self.redis_db,
        )
        redis_name = f"user_settings:{user_id}"
        sync_from_dynamodb = True
        if read_cache:
            redis_dict = await redis_client.hgetall(redis_name)
            decoded_redis_dict = dict()
            for key, value in redis_dict.items():
                decoded_key = key.decode("utf-8")
                decoded_redis_dict[decoded_key] = value.decode("utf-8")
            # check if all keys are in redis_dict
            if all(key in decoded_redis_dict for key in self.__class__.USER_KEYS):
                sync_from_dynamodb = False
                ret_dict = dict()
                for key in self.__class__.USER_KEYS:
                    value = decoded_redis_dict[key]
                    ret_dict[key] = value
        if sync_from_dynamodb:
            ret_dict = await super().get_user_settings(user_id)
            redis_dict = dict()
            for key, value in ret_dict.items():
                if value is None:
                    redis_dict[key] = ""
                else:
                    redis_dict[key] = value
            await redis_client.hset(redis_name, mapping=redis_dict)
            await redis_client.expire(redis_name, self.redis_cache_timeout)
        await redis_client.aclose()
        return ret_dict
