from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel


class GenerationTaskStartedResponse(BaseModel):
    """Generation task started response."""

    request_id: str


class GenerationTaskStatusResponse(BaseModel):
    """Generation task status response."""

    status: str
    message: Union[str, None] = None


class GenerationTaskResultsResponse(BaseModel):
    """Generation task results response."""

    audio_url: str
    face_url: str
    motion_url: str
    response_type: Literal["normal", "reject", "leave"]


class AdapterChoicesResponse(BaseModel):
    """Adapter choices response."""

    choices: List[str]


class VoiceNamesResponse(BaseModel):
    """Voice names response."""

    voice_names: Dict[str, Any]


class RequestIDResponse(BaseModel):
    """Request ID response."""

    request_id: str


class VoiceSettingsResponse(BaseModel):
    """Voice settings response."""

    tts_adapter: str
    voice: str
    voice_speed: float


class MotionSettingsResponse(BaseModel):
    """Motion settings response."""

    avatar: str


class RelationshipResponse(BaseModel):
    """Relationship response."""

    relationship: str
    score: int


class EmotionResponse(BaseModel):
    """Emotion response."""

    emotions: List[str]
