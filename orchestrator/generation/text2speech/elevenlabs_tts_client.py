import base64
import io
import wave
from typing import Any, Dict, Union

import httpx
from elevenlabs import VoiceSettings
from elevenlabs.client import AsyncElevenLabs
from prometheus_client import Histogram

from ...utils.exception import MissingAPIKeyException
from .tts_adapter import TextToSpeechAdapter


class ElevenLabsTTSClient(TextToSpeechAdapter):
    """ElevenLabs text to speech client.

    This client provides text-to-speech functionality using the ElevenLabs API.
    It supports streaming audio generation with voice customization options
    including speed control and language selection.
    """

    AVAILABLE_FOR_STREAM = True

    def __init__(
        self,
        name: str,
        elevenlabs_model_name: str = "eleven_flash_v2_5",
        proxy_url: Union[None, str] = None,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the ElevenLabs text to speech client.

        Args:
            name (str):
                The name of the text to speech client.
                Logger's name will be set to this name.
            elevenlabs_model_name (str, optional):
                The name of the ElevenLabs model to use.
                Defaults to "eleven_flash_v2_5".
            proxy_url (str, optional):
                The URL of the proxy to use.
                Defaults to None.
            queue_size (int, optional):
                The size of the queue to store the requests.
                Defaults to 100.
            sleep_time (float, optional):
                The time to sleep between requests.
                Defaults to 0.01.
            clean_interval (float, optional):
                The interval to clean the expired requests.
                Defaults to 10.0.
            expire_time (float, optional):
                The time to expire the request.
                Defaults to 120.0.
            latency_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording request latency distribution
                in seconds. If provided, latency metrics will be collected for monitoring
                purposes. Defaults to None.
            logger_cfg (dict, optional):
                The configuration for the logger.
                Defaults to None.
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
        self.elevenlabs_model_name = elevenlabs_model_name
        self.proxy_url = proxy_url

        if self.proxy_url is not None:
            self.http_client = httpx.AsyncClient(proxy=self.proxy_url)
        else:
            self.http_client = None

    async def get_voice_names(self, **kwargs: Any) -> Dict[str, Any]:
        """Get available voice names and their descriptions.

        Returns:
            Dict[str, Any]:
                A dictionary mapping voice IDs to their human-readable names.
                Keys are ElevenLabs voice IDs, values are descriptive names.
        """
        voice_names = {
            "IAfVgyogVSvpCyZeevDo": "Keqing-en",
            "v5WynholsiwnjjP7Iq1L": "Hutao-en",
            "VTpilc4HcK9uYznwQK8c": "Furina-en",
            "EXAVITQu4vr4xnSDxMaL": "Sarah",
            "FGY2WhTYpPnrIDTdsKH5": "Laura",
            "cgSgspJ2msm6clMCkdW9": "Jessica",
        }
        return voice_names

    async def _generate_tts(
        self,
        request_id: str,
        text: str,
        voice_name: str,
        voice_speed: float = 1.0,
        voice_style: Union[None, str] = None,
        language: str = "zh",
        start_time: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate TTS output from text using ElevenLabs API.

        Args:
            request_id (str):
                Unique identifier for the TTS request.
            text (str):
                The input text to convert to speech.
            voice_name (str):
                ElevenLabs voice ID to use for synthesis.
            voice_speed (float, optional):
                Speech rate multiplier. 1.0 is normal speed.
                Defaults to 1.0.
            voice_style (Union[None, str], optional):
                Voice style parameter (currently unused by ElevenLabs).
                Defaults to None.
            language (str, optional):
                Language code for the text. Defaults to "zh".
            start_time (float, optional):
                Start time offset for the generation. Defaults to 0.0.

        Returns:
            Dict[str, Any]:
                Dictionary containing TTS generation results:
                - `audio` (BytesIO): WAV format audio data
                - `speech_text` (str): Processed text used for synthesis
                - `speech_time` (List[Tuple[int, float]]): Character timing
                  information mapping character positions to start times
                - `duration` (float): Total audio duration in seconds
        """
        elevenlabs_api_key = self.input_buffer[request_id].get("api_keys", {}).get("elevenlabs_api_key", "")
        if not elevenlabs_api_key:
            msg = "ElevenLabs API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)
        elevenlabs_client = AsyncElevenLabs(
            api_key=elevenlabs_api_key,
            httpx_client=self.http_client,
        )
        ret_dict = dict()
        audio_io = io.BytesIO()
        alignment = dict(characters=[], character_start_times_seconds=[], character_end_times_seconds=[])

        response = elevenlabs_client.text_to_speech.stream_with_timestamps(
            voice_id=voice_name,
            text=text,
            model_id=self.elevenlabs_model_name,
            language_code=language,
            voice_settings=VoiceSettings(
                speed=voice_speed,
            ),
            output_format="pcm_16000",
            optimize_streaming_latency=4,
        )
        async for chunk in response:
            audio_io.write(base64.b64decode(chunk.audio_base_64))
            if chunk.alignment:
                alignment["characters"].extend(chunk.alignment.characters)
                alignment["character_start_times_seconds"].extend(chunk.alignment.character_start_times_seconds)
                alignment["character_end_times_seconds"].extend(chunk.alignment.character_end_times_seconds)

        # convert pcm data to wav format
        audio_io.seek(0)
        pcm_data = audio_io.read()

        # create a new BytesIO object to store wav file
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)  # single channel
            wav_file.setsampwidth(2)  # 16-bit sampling
            wav_file.setframerate(16000)  # 16kHz sampling rate
            wav_file.writeframes(pcm_data)  # write pcm data

        wav_io.seek(0)
        ret_dict["audio"] = wav_io
        alignment = self.convert_tts_time(text, alignment)
        ret_dict.update(alignment)
        return ret_dict

    @staticmethod
    def convert_tts_time(speech_text: str, alignment: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ElevenLabs alignment data to standardized format.

        Args:
            speech_text (str):
                The original input text used for synthesis.
            alignment (Dict[str, Any]):
                Alignment data from ElevenLabs API containing character
                timing information with keys 'characters',
                'character_start_times_seconds', and
                'character_end_times_seconds'.

        Returns:
            Dict[str, Any]:
                Standardized timing data containing:
                - `speech_text` (str): Original input text
                - `duration` (float): Total audio duration in seconds
                - `speech_time` (List[Tuple[int, float]]): Character timing
                  mapping with (character_index, start_time) pairs
        """
        # get the total duration
        duration = 0.0
        duration = alignment["character_end_times_seconds"][-1]

        # build the mapping from character position to time
        speech_time = []

        characters = alignment["characters"]
        start_times = alignment["character_start_times_seconds"]

        # build the mapping from original text character position to time
        char_pos = 0
        for i, (char, start_time) in enumerate(zip(characters, start_times)):
            # find the corresponding character position in the original text
            if char_pos < len(speech_text) and speech_text[char_pos] == char:
                speech_time.append((char_pos, start_time))
                char_pos += 1
            elif char_pos < len(speech_text):
                # handle possible character differences, skip spaces or special characters
                while char_pos < len(speech_text) and speech_text[char_pos] != char:
                    char_pos += 1
                if char_pos < len(speech_text):
                    speech_time.append((char_pos, start_time))
                    char_pos += 1

        return {
            "speech_text": speech_text,
            "duration": duration,
            "speech_time": speech_time,
        }
