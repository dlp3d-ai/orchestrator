import io
import wave
from typing import Any, Dict, Union

import httpx
from prometheus_client import Histogram

from .tts_adapter import TextToSpeechAdapter


class ChatterboxTTSClient(TextToSpeechAdapter):
    """Chatterbox TTS client for text-to-speech generation via HTTP API.

    This client implements the TextToSpeechAdapter interface to generate speech
    audio from text by making HTTP requests to a Chatterbox TTS service
    endpoint. It supports streaming and provides audio data in WAV format with
    timing information for each character.

    This client connects to a TTS service deployed using the forked Chatterbox repository
    (https://github.com/LazyBusyYang/chatterbox). When initializing the client,
    provide the `tts_http_url` parameter with the base URL of the deployed service,
    for example: ``https://127.0.0.1:80``.
    """

    AVAILABLE_FOR_STREAM = True

    def __init__(
        self,
        name: str,
        tts_http_url: str,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the Chatterbox TTS client.

        Args:
            name (str):
                Unique name for this TTS adapter instance.
                Used as the logger name for identification.
            tts_http_url (str):
                Base URL of the Chatterbox TTS HTTP service endpoint.
                Should include protocol (http:// or https://) and host.
            queue_size (int, optional):
                Maximum number of requests to queue for processing.
                Defaults to 100.
            sleep_time (float, optional):
                Sleep duration between processing iterations in seconds.
                Defaults to 0.01.
            clean_interval (float, optional):
                Interval in seconds for cleaning expired requests.
                Defaults to 10.0.
            expire_time (float, optional):
                Time in seconds after which requests expire.
                Defaults to 120.0.
            latency_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording request latency distribution
                in seconds. If provided, latency metrics will be collected for monitoring
                purposes. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary. If None, default
                logging configuration is used. Defaults to None.
        """
        TextToSpeechAdapter.__init__(
            self,
            name=name,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            latency_histogram=latency_histogram,
            logger_cfg=logger_cfg,
        )
        self.tts_http_url = tts_http_url

    async def _generate_tts(
        self,
        request_id: str,
        text: str,
        voice_name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate TTS audio from text using Chatterbox HTTP API.

        Sends a POST request to the Chatterbox TTS service to generate audio
        from the input text. The audio is returned in WAV format with character-level
        timing information calculated based on uniform duration distribution.

        Args:
            request_id (str):
                Unique identifier for the TTS request.
            text (str):
                Input text to convert to speech.
            voice_name (str):
                Voice identifier for the Chatterbox TTS service.
            **kwargs (Any):
                Additional keyword arguments (not used in this implementation).

        Returns:
            Dict[str, Any]:
                Dictionary containing TTS results with keys:
                - `audio` (BytesIO): Audio data in WAV format
                - `speech_text` (str): Processed text used for synthesis
                - `speech_time` (List[Tuple[int, float]]): Character timing
                - `duration` (float): Audio duration in seconds
        """
        url = f"{self.tts_http_url}/api/v1/generate_audio"
        request_dict = dict(
            text=text,
            voice_key=voice_name,
        )
        self.logger.info(f"Generating TTS for request {request_id}, text: {text}, voice_name: {voice_name}")
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(url, json=request_dict)
            response.raise_for_status()
            audio_data = response.content
        audio_io = io.BytesIO(audio_data)
        audio_io.seek(0)
        with wave.open(audio_io, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
        audio_io.seek(0)
        speech_text = text
        duration_per_character = duration / len(text)
        speech_time = []
        for i in range(len(text)):
            speech_time.append((i, i * duration_per_character))
        ret_dict = dict(
            audio=audio_io,
            speech_text=speech_text,
            speech_time=speech_time,
            duration=duration,
        )
        return ret_dict

    async def get_voice_names(self, **kwargs: Any) -> Dict[str, Any]:
        """Get available voice names and their descriptions.

        Returns:
            Dict[str, Any]:
                Dictionary mapping voice identifiers to their
                human-readable names or descriptions.
        """
        # Not configured for HTTP, return empty dict
        if not self.tts_http_url.startswith("http://") and not self.tts_http_url.startswith("https://"):
            return dict()
        url = f"{self.tts_http_url}/api/v1/list_voice_names"
        async with httpx.AsyncClient(timeout=1) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                src_voice_names = response.json()
            except Exception as e:
                self.logger.error(f"Failed to get voice names: {e}")
                return dict()
        voice_names = dict()
        for key in src_voice_names["voice_names"].keys():
            voice_names[key] = src_voice_names["voice_names"][key]
        return voice_names
