from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from ...utils.super import Super


class NoMatchingCharacterSettingsError(Exception):
    """Exception raised when no matching character settings are found."""

    pass


class NoMatchingUserSettingsError(Exception):
    """Exception raised when no matching user settings are found."""

    pass


class DatabaseConfigClient(Super, ABC):
    """Abstract base class for database configuration clients.

    Defines abstract interfaces for retrieving character configuration
    information, used for interacting with different databases. Main
    functionality is to retrieve character configuration information, including
    voice settings, motion settings, and complete character configurations.
    Subclasses need to implement specific database connection and query logic.
    """

    CHARACTER_KEYS = (
        "avatar",
        "tts_adapter",
        "voice",
        "voice_speed",
        "asr_adapter",
        "classification_adapter",
        "classification_model_override",
        "conversation_adapter",
        "conversation_model_override",
        "prompt",
        "reaction_adapter",
        "reaction_model_override",
        "memory_adapter",
        "memory_model_override",
        "acquaintance_threshold",
        "friend_threshold",
        "situationship_threshold",
        "lover_threshold",
        "neutral_threshold",
        "happiness_threshold",
        "sadness_threshold",
        "fear_threshold",
        "anger_threshold",
        "disgust_threshold",
        "surprise_threshold",
        "shyness_threshold",
    )

    USER_KEYS = (
        "openai_api_key",
        "xai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "deepseek_api_key",
        "sensenova_api_key",
        "sensenova_ak",
        "sensenova_sk",
        "softsugar_app_id",
        "softsugar_app_key",
        "huoshan_app_id",
        "huoshan_token",
        "snese_tts_api_key",
        "nova_tts_api_key",
        "elevenlabs_api_key",
        "timezone",
    )

    def __init__(
        self,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the database configuration client.

        Args:
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        Super.__init__(self, logger_cfg)

    @abstractmethod
    async def get_voice_settings(self, user_id: str, character_id: str) -> Dict[str, Any]:
        """Get voice settings for specified user and character.

        Query voice-related configuration information for the specified user and character from database,
        including TTS adapter, voice name, and voice speed.

        Args:
            user_id (str):
                User ID for locating user records in database.
            character_id (str):
                Character ID for locating character records in database.

        Returns:
            Dict[str, Any]:
                Dictionary containing character voice settings with the following fields:
                - tts_adapter (str): TTS adapter name
                - voice (str): Voice name
                - voice_speed (float): Voice speed

        Raises:
            NoMatchingCharacterSettingsError:
                When no matching user and character configuration is found in database.
        """
        pass

    @abstractmethod
    async def get_motion_settings(self, user_id: str, character_id: str) -> Dict[str, Any]:
        """Get motion settings for specified user and character.

        Query motion-related configuration information for the specified user and character from database,
        primarily the character name.

        Args:
            user_id (str):
                User ID for locating user records in database.
            character_id (str):
                Character ID for locating character records in database.

        Returns:
            Dict[str, Any]:
                Dictionary containing character motion settings with the following fields:
                - avatar (str): Character name

        Raises:
            NoMatchingCharacterSettingsError:
                When no matching user and character configuration is found in database.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_user_settings(self, user_id: str) -> Dict[str, Any]:
        """Get settings for specified user.

        Query account-related configuration information for the specified user from database,
        including API keys for various paid services. Keys may be empty strings.

        Args:
            user_id (str):
                User ID for locating user records in database.

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
                - huoshan_app_id (str): Huoshan App ID
                - huoshan_token (str): Huoshan Token
                - snese_tts_api_key (str): Sensenova TTS API Key
                - nova_tts_api_key (str): Nova TTS API Key
                - elevenlabs_api_key (str): ElevenLabs API Key
                - timezone (str): Timezone name

        Raises:
            NoMatchingUserSettingsError:
                When no matching user configuration is found in database.
        """
        pass
