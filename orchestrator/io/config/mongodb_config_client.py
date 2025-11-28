import traceback
from typing import Any, Dict, Union

from pymongo import AsyncMongoClient
from pymongo.errors import PyMongoError

from .database_config_client import DatabaseConfigClient, NoMatchingCharacterSettingsError, NoMatchingUserSettingsError


class MongoDBConfigClient(DatabaseConfigClient):
    """MongoDB configuration client class.

    Inherits from DatabaseConfigClient and implements interaction functionality
    with MongoDB database. Used for retrieving character configuration
    information, including voice settings, motion settings, and complete
    character configurations. Provides methods for querying user and character
    specific configurations from MongoDB collections.
    """

    def __init__(
        self,
        host: str,
        port: int,
        username: Union[str, None] = None,
        password: Union[str, None] = None,
        database: str = "web",
        auth_database: str = "admin",
        user_config_collection_name: str = "UserConfigs",
        character_config_collection_name: str = "CharacterConfigs",
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the MongoDB configuration client.

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
                MongoDB database name. Defaults to "web".
            auth_database (str, optional):
                MongoDB authentication database name. Defaults to "admin".
            user_config_collection_name (str, optional):
                Name of the user configuration collection. Defaults to "UserConfigs".
            character_config_collection_name (str, optional):
                Name of the character configuration collection. Defaults to "CharacterConfigs".
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        DatabaseConfigClient.__init__(self, logger_cfg)
        self.host = host
        self.port = port
        self.database_name = database
        self.username = username
        self.password = password
        self.auth_database_name = auth_database
        self.user_config_collection_name = user_config_collection_name
        self.character_config_collection_name = character_config_collection_name

    async def get_voice_settings(self, user_id: str, character_id: str) -> Dict[str, Any]:
        """Get voice settings for specified user and character.

        Query voice-related configuration information for the specified user and character from MongoDB collection,
        including TTS adapter, voice name, and voice speed.

        Args:
            user_id (str):
                User ID for locating user records in MongoDB.
            character_id (str):
                Character ID for locating character records in MongoDB.

        Returns:
            Dict[str, Any]:
                Dictionary containing character voice settings with the following fields:
                - tts_adapter (str): TTS adapter name
                - voice (str): Voice name
                - voice_speed (float): Voice speed

        Raises:
            NoMatchingCharacterSettingsError:
                When no matching user and character configuration is found in MongoDB.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database_name,
            ) as client:
                db = client[self.database_name]
                col = db[self.character_config_collection_name]
                doc = await col.find_one(
                    {"user_id": user_id, "character_id": character_id},
                    {"_id": 0, "tts_adapter": 1, "voice": 1, "voice_speed": 1},
                )
            if doc is None:
                msg = f"No character settings found for user_id={user_id} and character_id={character_id}"
                self.logger.error(msg)
                raise NoMatchingCharacterSettingsError(msg)
            return_dict = dict(
                tts_adapter=doc["tts_adapter"],
                voice=doc["voice"],
                voice_speed=float(doc["voice_speed"]),
            )
            return return_dict
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to get voice settings from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def get_motion_settings(self, user_id: str, character_id: str) -> Dict[str, Any]:
        """Get motion settings for specified user and character.

        Query motion-related configuration information for the specified user and character from MongoDB collection,
        primarily the character name.

        Args:
            user_id (str):
                User ID for locating user records in MongoDB.
            character_id (str):
                Character ID for locating character records in MongoDB.

        Returns:
            Dict[str, Any]:
                Dictionary containing character motion settings with the following fields:
                - avatar (str): Character name

        Raises:
            NoMatchingCharacterSettingsError:
                When no matching user and character configuration is found in MongoDB.
        """
        try:
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database_name,
            ) as client:
                db = client[self.database_name]
                col = db[self.character_config_collection_name]
                doc = await col.find_one(
                    {"user_id": user_id, "character_id": character_id},
                    {"_id": 0, "avatar": 1},
                )
            if doc is None:
                msg = f"No character settings found for user_id={user_id} and character_id={character_id} in {self.character_config_collection_name}."
                self.logger.error(msg)
                raise NoMatchingCharacterSettingsError(msg)
            return doc
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to get motion settings from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def get_character_settings(self, user_id: str, character_id: str) -> Dict[str, Any]:
        """Get complete configuration settings for specified user and
        character.

        Query complete configuration information for the specified user and character from database,
        including adapter configurations for ASR, classification, conversation, TTS, reaction, and other components.
        This is the main method for retrieving character configuration, returning all configuration information.

        Args:
            user_id (str):
                User ID for locating user records in database.
            character_id (str):
                Character ID for locating character records in database.

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
                When no matching user and character configuration is found in database.
        """
        try:
            return_keys = {"_id": 0}
            for key in self.__class__.CHARACTER_KEYS:
                return_keys[key] = 1
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database_name,
            ) as client:
                db = client[self.database_name]
                col = db[self.character_config_collection_name]
                doc = await col.find_one(
                    {"user_id": user_id, "character_id": character_id},
                    return_keys,
                )
            if doc is None:
                msg = f"No character settings found for user_id={user_id} and character_id={character_id} in {self.character_config_collection_name}."
                self.logger.error(msg)
                raise NoMatchingCharacterSettingsError(msg)
            return_dict = dict()
            for key, value in doc.items():
                if key == "voice_speed":
                    return_dict[key] = float(value)
                elif key.endswith("_threshold"):
                    return_dict[key] = int(value)
                else:
                    return_dict[key] = value
            return return_dict
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to get character settings from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e

    async def get_user_settings(self, user_id: str) -> Dict[str, Any]:
        """Get settings for specified user.

        Query account-related configuration information for the specified user from MongoDB collection,
        including API keys for various paid services. Keys may be empty strings.

        Args:
            user_id (str):
                User ID for locating user records in MongoDB.

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
                When no matching user configuration is found in MongoDB.
        """
        try:
            return_keys = {"_id": 0}
            for key in self.__class__.USER_KEYS:
                return_keys[key] = 1
            async with AsyncMongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.auth_database_name,
            ) as client:
                db = client[self.database_name]
                col = db[self.user_config_collection_name]
                doc = await col.find_one(
                    {"user_id": user_id},
                    return_keys,
                )
            if doc is None:
                msg = f"No user settings found for user_id={user_id} in {self.user_config_collection_name}."
                self.logger.error(msg)
                raise NoMatchingUserSettingsError(msg)
            return doc
        except PyMongoError as e:
            traceback_str = traceback.format_exc()
            msg = f"Failed to get user settings from MongoDB: {traceback_str}"
            self.logger.error(msg)
            raise e
