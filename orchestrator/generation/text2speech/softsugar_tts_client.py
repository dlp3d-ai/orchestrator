import base64
import hashlib
import io
import json
import time
import uuid
import wave
from typing import Any, Dict, Union

import httpx
import websockets

from .tts_adapter import TextToSpeechAdapter


class SoftSugarTTSClient(TextToSpeechAdapter):
    """SoftSugar text-to-speech client.

    This client provides text-to-speech functionality using the SoftSugar TTS
    API via WebSocket connection. It supports streaming audio generation with
    voice customization and automatic token management.
    """

    AVAILABLE_FOR_STREAM = True

    def __init__(
        self,
        name: str,
        tts_ws_url: str,
        softsugar_token_url: str,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the SoftSugar text-to-speech client.

        Args:
            name (str):
                Unique name for this TTS client instance.
                Used as the logger name for identification.
            tts_ws_url (str):
                WebSocket URL for the SoftSugar TTS service.
            softsugar_token_url (str):
                URL for obtaining SoftSugar authentication tokens.
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
            logger_cfg=logger_cfg,
        )
        self.tts_ws_url = tts_ws_url
        self.softsugar_token_url = softsugar_token_url

    def create_token_request_data(self, app_id: str, app_key: str) -> Dict[str, Any]:
        """Create authentication request data for SoftSugar token.

        Args:
            app_id (str):
                SoftSugar application ID for authentication.
            app_key (str):
                SoftSugar application key for authentication.

        Returns:
            Dict[str, Any]:
                Authentication request data containing appId, timestamp,
                sign, and grantType fields.
        """
        timestamp = str(int(time.time() * 1000))
        raw_string = f"{app_id}{timestamp}{app_key}"
        sign = hashlib.md5(raw_string.encode()).hexdigest()
        request_data = {"appId": app_id, "timestamp": timestamp, "sign": sign, "grantType": "sign"}
        return request_data

    async def get_response(self, app_id: str, app_key: str) -> Union[Dict[str, Any], None]:
        """Get authentication response from SoftSugar token service.

        Args:
            app_id (str):
                SoftSugar application ID for authentication.
            app_key (str):
                SoftSugar application key for authentication.

        Returns:
            Union[Dict[str, Any], None]:
                Authentication response dictionary if successful,
                None if the request failed.
        """
        headers = {"Content-Type": "application/json"}
        request_data = self.create_token_request_data(app_id, app_key)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.softsugar_token_url, headers=headers, json=request_data)
                response.raise_for_status()  # Raise an exception for HTTP errors
                return response.json()
        except Exception as e:
            self.logger.error(f"Error obtaining token: {e}")
            return None

    async def refresh_token(self, app_id: str, app_key: str) -> str:
        """Refresh the SoftSugar authentication token.

        Args:
            app_id (str):
                SoftSugar application ID for authentication.
            app_key (str):
                SoftSugar application key for authentication.

        Returns:
            str:
                New authentication token string.

        Raises:
            Exception:
                If token refresh fails or returns invalid response.
        """
        try:
            response = await self.get_response(app_id, app_key)
            if response and "data" in response and "accessToken" in response["data"]:
                softsugar_token = response["data"]["accessToken"]
                self.logger.info("SoftSugar token refreshed successfully.")
                return softsugar_token
            else:
                self.logger.error("Failed to refresh SoftSugar token: Invalid response.")
                raise Exception("Failed to refresh SoftSugar token: Invalid response.")
        except Exception as e:
            self.logger.error(f"Error refreshing SoftSugar token: {e}")
            raise e

    async def start(self, websocket, session_id: str, qid: str, speed_ratio: float, language: str) -> None:
        """Send initialization message to SoftSugar TTS service.

        Args:
            websocket:
                WebSocket connection to the SoftSugar TTS service.
            session_id (str):
                Unique session identifier for this TTS request.
            qid (str):
                Voice identifier for the TTS service.
            speed_ratio (float):
                Speech rate multiplier. 1.0 is normal speed.
            language (str):
                Language code for the TTS synthesis.
        """
        starter = {
            "type": "TTS",
            "device": "PAAS_PY_WS_TTS_DEMO",
            "session": session_id,
            "tts": {
                "omit_error": False,
                "format": "wav",  # only pcm supports stream
                "qid": qid,  # voice id
                "speed_ratio": speed_ratio,
                "sentence_time": True,  # sentence timestamp
                "word_time": True,  # word timestamp
                "subtitle": "srt",
                "language": language,
            },
        }
        await websocket.send(json.dumps(starter))
        await websocket.recv()

    async def query(self, websocket, query_input: str) -> None:
        """Send text query to SoftSugar TTS service for synthesis.

        Args:
            websocket:
                WebSocket connection to the SoftSugar TTS service.
            query_input (str):
                Input text to be converted to speech.
        """

        req_id = str(uuid.uuid4())
        req = {
            "id": req_id,
            "query": query_input,
            "ssml": False,
        }
        await websocket.send(json.dumps(req))

    async def get_voice_names(self, **kwargs: Any) -> Dict[str, Any]:
        """Get available voice names and their descriptions.

        Returns:
            Dict[str, Any]:
                Dictionary mapping voice identifiers to their
                human-readable names for SoftSugar TTS service.
        """
        voice_names = {
            "JQb7Qv:AEA_Z10Mqp9GYwDGdLzMvPzEzIqwo": "默认",
            "pgd0Yv:AEAjMt1c3J9GCwDGHKLM93ic0rTKxJyAsK": "爱玩的女孩",
            "lQm7k_:AEAjMt1c3J9GCwDGFLzUzKK0nKLM93j0vMyixOcwo": "体贴的女朋友",
            "EAgnbP:AEAjMt1c3J9GCwDGSpTEn3ji0vzU9KL89zCg": "美丽的女士",
            "WQlHj_:AEAjMt1c3J9GCwDGHKLM93iU1JKixNSM3RS8zO8wo": "善良的女孩",
            "1guW1f:AEAjMt1c3J9GCwDGXKL8tJLEnIL4_Nz8xMyK-NScxNzUtzCg": "闺蜜",
            "-wo4y_:AEAjMt1c3J9GCwDGAyNs5NS8_JzkzLTMjPiDIyMDTNzM0vSwo": "魅惑",
            "8wfZav:AEA_Z10Mqp9GCwDGMrz8xIzi3VScxNzUtLCg": "大学生",
            "YxYlvv:ABAP2ddDKqffEHj8vMTknPyy30ZLAMZikpL0zKT4vPz83PjU0srU0srM1Ny8xMyS-NScxNzUtPiMqrCg": "甜美悦悦",
            "Bxcy1_:ABAYtLdTKTdMrKffEHj8vMTknPyy30ZLAMZikpL0zKT4vPz83PjS_MqyvMzCvMTSyvjUnMTc1LT4jKqwo": "撒娇学妹",
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
        """Generate TTS output from text using SoftSugar API.

        Args:
            request_id (str):
                Unique identifier for the TTS request.
            text (str):
                Input text to convert to speech.
            voice_name (str):
                Voice identifier for the SoftSugar TTS service.
            voice_speed (float, optional):
                Speech rate multiplier. 1.0 is normal speed.
                Defaults to 1.0.
            voice_style (Union[None, str], optional):
                Voice style parameter. Defaults to None.
            language (str, optional):
                Language code for the text. Defaults to "zh".
            start_time (float, optional):
                Start time offset for generation. Defaults to 0.0.

        Returns:
            Dict[str, Any]:
                Dictionary containing TTS generation results:
                - `audio` (BytesIO): WAV format audio data
                - `speech_text` (str): Processed text used for synthesis
                - `speech_time` (List[Tuple[int, float]]): Character timing
                  information mapping character positions to start times
                - `duration` (float): Total audio duration in seconds
        """
        app_id = self.input_buffer[request_id].get("api_keys", {}).get("softsugar_app_id", "")
        app_key = self.input_buffer[request_id].get("api_keys", {}).get("softsugar_app_key", "")
        if not app_id or not app_key:
            msg = "SoftSugar app ID or app key is not found in the API keys."
            self.logger.error(msg)
            raise ValueError(msg)

        response = await self.get_response(app_id, app_key)
        if not response or "data" not in response or "accessToken" not in response["data"]:
            msg = "Failed to obtain SoftSugar token."
            self.logger.error(msg)
            raise ValueError(msg)
        softsugar_token = response["data"]["accessToken"]

        tts_language = "zh-CN" if language == "zh" else "en-US"
        session_id = str(uuid.uuid4())
        max_retries = 2  # Retry once after refreshing token
        for attempt in range(max_retries):
            endpoint = self.tts_ws_url + "?Authorization=Bearer%20" + softsugar_token
            ret_dict = dict()
            audio_io = io.BytesIO()
            subtitle_result = dict()
            try:
                async with websockets.connect(endpoint) as websocket:
                    await self.start(websocket, session_id, voice_name, voice_speed, tts_language)
                    await self.query(websocket, text)
                    buffer = bytearray()
                    timestamp_data = {"sentence_time": [], "word_times": []}
                    with wave.open(audio_io, "wb") as audio:
                        audio.setnchannels(1)
                        audio.setsampwidth(2)
                        audio.setframerate(16000)
                        while True:
                            message = await websocket.recv()
                            done = await self._on_message(
                                message, audio, subtitle_result, timestamp_data, buffer, start_time
                            )
                            if done:
                                break

                self.logger.debug(f"TTS output: {subtitle_result}")
                audio_io.seek(0)
                ret_dict["audio"] = audio_io
                ret_dict.update(subtitle_result)
                return ret_dict

            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"WebSocket error, refreshing token (attempt {attempt + 1}): {e}")
                    softsugar_token = await self.refresh_token(app_id, app_key)
                    continue
                else:
                    self.logger.error(f"WebSocket connection failed after retries: {e}")
                    raise e
        raise RuntimeError("Unexpected end of retry loop")

    async def _on_message(
        self,
        message: Union[str, bytes],
        audio: wave.Wave_write,
        subtitle_result: Dict[str, Any],
        timestamp_data: Dict[str, Any],
        buffer: bytearray,
        start_time: float = 0.0,
    ) -> bool:
        """Process WebSocket messages from SoftSugar TTS service.

        Args:
            message (Union[str, bytes]):
                Raw message received from the WebSocket connection.
            audio (wave.Wave_write):
                Wave file writer object to save audio data.
            subtitle_result (Dict[str, Any]):
                Dictionary to store subtitle and timing information.
            timestamp_data (Dict[str, Any]):
                Dictionary to store sentence and word timing data.
            buffer (bytearray):
                Buffer for handling partial audio data.
            start_time (float, optional):
                Start time offset for timing calculations.
                Defaults to 0.0.

        Returns:
            bool:
                True if the message processing is complete and the
                connection should be closed, False to continue processing.
        """
        try:
            response = json.loads(message)
            if response["status"] != "ok":
                return True
            if "tts" in response:
                tts_response = response["tts"]
                if tts_response["type"] == "eof":
                    subtitle_result.update(self.convert_tts_time(timestamp_data))
                    return True
                if tts_response["type"] == "audio":
                    audio_data = base64.b64decode(tts_response["audio_data"])
                    buffer.clear()
                    buffer.extend(audio_data)
                    audio.writeframes(audio_data)
                if tts_response["type"] == "timestamp":
                    self.logger.debug(f"TTS timestamp: {tts_response}")
                    if "sentence_time" in tts_response:
                        timestamp_data["sentence_time"].append(tts_response["sentence_time"])
                    if "word_times" in tts_response:
                        timestamp_data["word_times"].extend(tts_response["word_times"])
            return False
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return True

    @classmethod
    def convert_tts_time(cls, tts_time: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Convert SoftSugar TTS timing data to standardized format.

        Args:
            tts_time (Dict[str, Any]):
                Timing data from SoftSugar TTS API containing
                'sentence_time' and 'word_times' fields.

        Returns:
            Dict[str, Any]:
                Standardized timing data containing:
                - `speech_text` (str): Concatenated text from all words
                - `speech_time` (List[Tuple[int, float]]): Character timing
                  mapping with (character_index, start_time) pairs
                - `duration` (float): Total audio duration in seconds
        """
        sentence_times = sorted(tts_time["sentence_time"], key=lambda x: x["begin_ms"])
        word_times = sorted(tts_time["word_times"], key=lambda x: x["begin_ms"])

        duration = (sentence_times[-1]["end_ms"] - sentence_times[0]["begin_ms"]) / 1000.0

        speech_text = ""
        speech_time = []
        for word_time in word_times:
            word = word_time["text"]
            word_start_time = word_time["begin_ms"] / 1000
            idx = len(speech_text)  # the word of the tts model may contain multiple characters
            speech_text += word
            speech_time.append((idx, word_start_time))

        return {
            "duration": duration,
            "speech_time": speech_time,
            "speech_text": speech_text,
        }
