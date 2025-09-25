from typing import Any, Dict, Union

import aioboto3
import aioboto3.session
from boto3.dynamodb.conditions import Key as DynamoDBKey

from .database_config_client import DatabaseConfigClient, NoMatchingCharacterSettingsError, NoMatchingUserSettingsError


class DynamoDBConfigClient(DatabaseConfigClient):
    """DynamoDB configuration client class.

    Provides asynchronous interaction with AWS DynamoDB specifically for
    retrieving character configuration information. Supports querying user and
    character configuration settings from DynamoDB tables, including voice
    settings, motion settings, and complete character configurations. Offers
    fine-grained configuration retrieval methods to fetch specific types of
    configuration information as needed.
    """

    def __init__(
        self,
        region_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        user_config_table_name: str,
        character_config_table_name: str,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the DynamoDB configuration client.

        Args:
            region_name (str):
                AWS region name, e.g., 'us-east-1'.
            aws_access_key_id (str):
                AWS access key ID.
            aws_secret_access_key (str):
                AWS secret access key.
            user_config_table_name (str):
                Name of the user configuration table in DynamoDB.
            character_config_table_name (str):
                Name of the character configuration table in DynamoDB.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        DatabaseConfigClient.__init__(self, logger_cfg)
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.user_config_table_name = user_config_table_name
        self.character_config_table_name = character_config_table_name
        self.session = aioboto3.Session(
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    async def get_voice_settings(self, user_id: str, character_id: str) -> Dict[str, Any]:
        """Get voice settings for specified user and character.

        Query voice-related configuration information for the specified user and character from DynamoDB table,
        including TTS adapter, voice name, and voice speed.

        Args:
            user_id (str):
                User ID for locating user records in DynamoDB.
            character_id (str):
                Character ID for locating character records in DynamoDB.

        Returns:
            Dict[str, Any]:
                Dictionary containing character voice settings with the following fields:
                - tts_adapter (str): TTS adapter name
                - voice (str): Voice name
                - voice_speed (float): Voice speed

        Raises:
            NoMatchingCharacterSettingsError:
                When no matching user and character configuration is found in DynamoDB.
        """
        async with self.session.resource("dynamodb") as dynamo_resource:
            character_config_table = await dynamo_resource.Table(self.character_config_table_name)
            result = await character_config_table.query(
                KeyConditionExpression=DynamoDBKey("user_id").eq(user_id)
                & DynamoDBKey("character_id").eq(character_id),
                ProjectionExpression="tts_adapter, voice, voice_speed",
            )
        if len(result["Items"]) == 0:
            msg = f"No character settings found for user_id={user_id} and character_id={character_id}"
            self.logger.error(msg)
            raise NoMatchingCharacterSettingsError(msg)
        tts_adapter = result["Items"][0]["tts_adapter"]
        ret_dict = dict(
            tts_adapter=tts_adapter,
            voice=result["Items"][0]["voice"],
            voice_speed=float(result["Items"][0]["voice_speed"]),
        )
        return ret_dict

    async def get_motion_settings(self, user_id: str, character_id: str) -> Dict[str, Any]:
        """Get motion settings for specified user and character.

        Query motion-related configuration information for the specified user and character from DynamoDB table,
        primarily the character name.

        Args:
            user_id (str):
                User ID for locating user records in DynamoDB.
            character_id (str):
                Character ID for locating character records in DynamoDB.

        Returns:
            Dict[str, Any]:
                Dictionary containing character motion settings with the following fields:
                - avatar (str): Character name

        Raises:
            NoMatchingCharacterSettingsError:
                When no matching user and character configuration is found in DynamoDB.
        """
        async with self.session.resource("dynamodb") as dynamo_resource:
            character_config_table = await dynamo_resource.Table(self.character_config_table_name)
            result = await character_config_table.query(
                KeyConditionExpression=DynamoDBKey("user_id").eq(user_id)
                & DynamoDBKey("character_id").eq(character_id),
                ProjectionExpression="avatar",
            )
        if len(result["Items"]) == 0:
            msg = f"No character settings found for user_id={user_id} and character_id={character_id}"
            self.logger.error(msg)
            raise NoMatchingCharacterSettingsError(msg)
        ret_dict = dict(
            avatar=result["Items"][0]["avatar"],
        )
        return ret_dict

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
        async with self.session.resource("dynamodb") as dynamo_resource:
            character_config_table = await dynamo_resource.Table(self.character_config_table_name)
            result = await character_config_table.query(
                KeyConditionExpression=DynamoDBKey("user_id").eq(user_id) & DynamoDBKey("character_id").eq(character_id)
            )
        if len(result["Items"]) == 0:
            msg = f"No character settings found for user_id={user_id} and character_id={character_id}"
            self.logger.error(msg)
            raise NoMatchingCharacterSettingsError(msg)
        classification_model_override = result["Items"][0].get("classification_model_override", "")
        conversation_model_override = result["Items"][0].get("conversation_model_override", "")
        reaction_model_override = result["Items"][0].get("reaction_model_override", "")
        memory_model_override = result["Items"][0].get("memory_model_override", "")
        memory_adapter = result["Items"][0]["memory_adapter"]
        tts_adapter = result["Items"][0]["tts_adapter"]
        prompt = result["Items"][0]["user_prompt"]
        threshold_values = dict()
        for key in self.__class__.CHARACTER_KEYS:
            if key.endswith("_threshold"):
                threshold_values[key] = int(result["Items"][0][key])
        ret_dict = dict(
            # for asr
            asr_adapter=result["Items"][0]["asr_adapter"],
            # for classification
            classification_adapter=result["Items"][0]["classification_adapter"],
            classification_model_override=classification_model_override,
            # for conversation
            conversation_adapter=result["Items"][0]["conversation_adapter"],
            prompt=prompt,
            conversation_model_override=conversation_model_override,
            # for tts
            tts_adapter=tts_adapter,
            voice=result["Items"][0]["voice"],
            voice_speed=float(result["Items"][0]["voice_speed"]),
            # for reaction
            reaction_adapter=result["Items"][0]["reaction_adapter"],
            reaction_model_override=reaction_model_override,
            # for memory
            memory_adapter=memory_adapter,
            memory_model_override=memory_model_override,
            # for s2m and a2f
            avatar=result["Items"][0]["avatar"],
        )
        # for threshold values
        ret_dict.update(threshold_values)
        return ret_dict

    async def get_user_settings(self, user_id: str) -> Dict[str, Any]:
        """Get settings for specified user.

        Query account-related configuration information for the specified user from DynamoDB table,
        including API keys for various paid services. Keys may be empty strings.

        Args:
            user_id (str):
                User ID for locating user records in DynamoDB.

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
                - softsugar_app_id (str): SoftSugar App ID
                - softsugar_app_key (str): SoftSugar App Key
                - huoshan_appid (str): Huoshan App ID
                - huoshan_token (str): Huoshan Token
                - snese_tts_api_key (str): Sensenova TTS API Key
                - nova_tts_api_key (str): Nova TTS API Key
                - elevenlabs_api_key (str): ElevenLabs API Key

        Raises:
            NoMatchingUserSettingsError:
                When no matching user configuration is found in DynamoDB.
        """
        ret_dict = dict()
        async with self.session.resource("dynamodb") as dynamo_resource:
            user_config_table = await dynamo_resource.Table(self.user_config_table_name)
            result = await user_config_table.query(KeyConditionExpression=DynamoDBKey("user_id").eq(user_id))
            if len(result["Items"]) == 0:
                msg = f"No user settings found for user_id={user_id}"
                self.logger.error(msg)
                raise NoMatchingUserSettingsError(msg)
            for key in self.__class__.USER_KEYS:
                value = result["Items"][0].get(key, "")
                value = None if len(value) == 0 else value
                ret_dict[key] = value
        return ret_dict
