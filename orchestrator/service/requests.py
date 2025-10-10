from typing import Literal, Union

from pydantic import BaseModel


class DirectGenerationRequest(BaseModel):
    """Direct generation request model.

    Used for direct requests for text-to-speech, expression generation and
    motion generation.
    """

    # for text2speech
    speech_text: str
    # for tts
    language: str = "zh"
    # for both tts and reaction
    chunk_n_char_lowerbound: int = 10
    # for both reaction and motion
    first_body_fast_response: bool = False
    # for app
    app_name: Literal["babylon", "python_backend"] = "python_backend"
    # for s2m
    max_front_extension_duration: float = 0.0
    max_rear_extension_duration: float = 0.0
    # for dynamodb
    user_id: Union[str, None] = None
    character_id: Union[str, None] = None
    # without dynamodb
    tts_adapter: Union[str, None] = None
    voice_name: Union[str, None] = None
    voice_speed: Union[float, None] = None
    avatar: Union[str, None] = None


class AudioChatCompleteStartRequestV4(BaseModel):
    """Complete audio chat start request model.

    Used to start the complete audio chat flow, including speech recognition,
    classification, conversation, reaction and speech synthesis.
    """

    # for asr audio stream
    n_channels: int
    sample_width: int
    frame_rate: int
    # for asr, classification, conversation, reaction, tts
    language: str = "zh"
    # for both tts and reaction
    chunk_n_char_lowerbound: int = 10
    # for both reaction and motion
    first_body_fast_response: bool = False
    # for app
    app_name: Literal["babylon", "python_backend"] = "python_backend"
    # for s2m
    max_front_extension_duration: float = 0.0
    max_rear_extension_duration: float = 0.0
    # for dynamodb
    user_id: Union[str, None] = None
    character_id: Union[str, None] = None
    # without dynamodb
    asr_adapter: Union[str, None] = None
    classification_adapter: Union[str, None] = None
    conversation_adapter: Union[str, None] = None
    reaction_adapter: Union[str, None] = None
    tts_adapter: Union[str, None] = None
    voice_name: Union[str, None] = None
    voice_speed: Union[float, None] = None
    avatar: Union[str, None] = None


class AudioChatExpressStartRequestV4(BaseModel):
    """Express audio chat start request model.

    Used to start the express audio chat flow, mainly for English conversations
    and specific character configurations.
    """

    # for audio stream
    n_channels: int
    sample_width: int
    frame_rate: int
    # for conversation
    language: str = "zh"
    # for app
    app_name: Literal["babylon", "python_backend"] = "python_backend"
    # for s2m
    max_front_extension_duration: float = 0.0
    max_rear_extension_duration: float = 0.0
    # for dynamodb
    user_id: Union[str, None] = None
    character_id: Union[str, None] = None
    # without dynamodb
    conversation_adapter: Union[str, None] = None
    voice_name: Union[str, None] = None
    avatar: Union[str, None] = None


class TextChatCompleteRequestV4(BaseModel):
    """Complete text chat request model.

    Used for complete text-only chat flow, including classification,
    conversation, reaction and speech synthesis.
    """

    speech_text: str
    # for classification, conversation, reaction, tts
    language: str = "zh"
    # for both tts and reaction
    chunk_n_char_lowerbound: int = 10
    # for both reaction and motion
    first_body_fast_response: bool = False
    # for app
    app_name: Literal["babylon", "python_backend"] = "python_backend"
    # for s2m
    max_front_extension_duration: float = 0.0
    max_rear_extension_duration: float = 0.0
    # for dynamodb
    user_id: Union[str, None] = None
    character_id: Union[str, None] = None
    # without dynamodb
    classification_adapter: Union[str, None] = None
    conversation_adapter: Union[str, None] = None
    reaction_adapter: Union[str, None] = None
    tts_adapter: Union[str, None] = None
    voice_name: Union[str, None] = None
    voice_speed: Union[float, None] = None
    avatar: Union[str, None] = None


class TextChatExpressRequestV4(BaseModel):
    """Express text chat request model.

    Used for express text-only chat flow, including classification,
    conversation, reaction and speech synthesis.
    """

    speech_text: str
    # for classification, conversation, reaction, tts
    language: str = "zh"
    # for both tts and reaction
    chunk_n_char_lowerbound: int = 10
    # for both reaction and motion
    first_body_fast_response: bool = False
    # for app
    app_name: Literal["babylon", "python_backend"] = "python_backend"
    # for s2m
    max_front_extension_duration: float = 0.0
    max_rear_extension_duration: float = 0.0
    # for dynamodb
    user_id: Union[str, None] = None
    character_id: Union[str, None] = None
    # without dynamodb
    conversation_adapter: Union[str, None] = None
    tts_adapter: Union[str, None] = None
    voice_name: Union[str, None] = None
    avatar: Union[str, None] = None


class AudioChatCompleteStopRequestV4(BaseModel):
    """Complete audio chat stop request model.

    Used to stop the complete audio chat flow.
    """

    signal: Literal["eof"]


class AudioChatExpressStopRequestV4(BaseModel):
    """Express audio chat stop request model.

    Used to stop the express audio chat flow.
    """

    signal: Literal["eof"]
