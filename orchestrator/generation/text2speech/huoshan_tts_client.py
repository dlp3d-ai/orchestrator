import asyncio
import gzip
import io
import json
import uuid
import wave
from typing import Any, Dict, Union

import websockets

from ...utils.exception import MissingAPIKeyException
from .tts_adapter import TextToSpeechAdapter


class HuoshanTTSClient(TextToSpeechAdapter):
    """Huoshan (Volcano) text-to-speech client.

    This client provides text-to-speech functionality using the Huoshan
    (Volcano) TTS API via WebSocket connection. It supports streaming audio
    generation with voice customization and style options.
    """

    AVAILABLE_FOR_STREAM = True

    def __init__(
        self,
        name: str,
        tts_ws_url: str,
        cluster: str,
        voice_style_file: str = "configs/voice_style.json",
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        max_concurrent_requests: int = 2,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the Huoshan text-to-speech client.

        Args:
            name (str):
                Unique name for this TTS client instance.
                Used as the logger name for identification.
            tts_ws_url (str):
                WebSocket URL for the Huoshan TTS service.
            cluster (str):
                Service cluster identifier (e.g., 'volcano_tts', 'volcano_icl').
            voice_style_file (str, optional):
                Path to JSON file containing voice style configurations.
                Defaults to "configs/voice_style.json".
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
            max_concurrent_requests (int, optional):
                Maximum number of concurrent TTS requests to prevent
                overwhelming the service. Defaults to 2.
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
        self.cluster = cluster
        self.default_header = bytearray(b"\x11\x10\x11\x00")
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        with open(voice_style_file, "r", encoding="utf-8") as f:
            self.voice_style_dict = json.load(f)

    def create_request_json(
        self,
        request_id: str,
        text: str,
        voice_type: str,
        voice_speed: float,
        voice_style: Union[None, str] = None,
    ) -> Dict[str, Any]:
        """Create JSON request payload for Huoshan TTS API.

        Args:
            request_id (str):
                Unique identifier for the TTS request.
            text (str):
                Input text to convert to speech.
            voice_type (str):
                Voice type identifier for the TTS service.
            voice_speed (float):
                Speech rate multiplier. 1.0 is normal speed.
            voice_style (Union[None, str], optional):
                Voice style parameter. If None, default style is used.
                Defaults to None.

        Returns:
            Dict[str, Any]:
                JSON request payload formatted for Huoshan TTS API.
        """
        app_id = self.input_buffer[request_id].get("api_keys", {}).get("huoshan_app_id", "")
        token = self.input_buffer[request_id].get("api_keys", {}).get("huoshan_token", "")
        if not app_id or not token:
            msg = "Huoshan app ID or token is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        audio_dict = {
            "voice_type": voice_type,
            "encoding": "wav",
            "speed_ratio": voice_speed,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
            "language": "cn" if self.sentence_splitter.contains_chinese(text) else "en",
            "rate": 16000,
        }

        if (
            voice_type in self.voice_style_dict
            and voice_style in self.voice_style_dict[voice_type]
            and voice_style != self.voice_style_dict[voice_type][0]
        ):
            audio_dict["style"] = voice_style

        return {
            "app": {"appid": app_id, "token": token, "cluster": self.cluster},
            "user": {"uid": str(uuid.uuid4())},
            "audio": audio_dict,
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": "plain",
                "operation": "submit",
                "with_timestamp": "1",
            },
        }

    async def get_voice_names(self, **kwargs: Any) -> Dict[str, Any]:
        """Get available voice names and their descriptions.

        Returns:
            Dict[str, Any]:
                Dictionary mapping voice identifiers to their
                human-readable names. The available voices depend
                on the configured cluster (volcano_tts or volcano_icl).
        """
        if self.cluster == "volcano_tts":
            voice_names = {
                "BV700_streaming": "火山语音合成-灿灿-中英文",
                "BV421_streaming": "火山语音合成-天才少女-中英文",
                "BV104_streaming": "火山语音合成-温柔淑女",
                "BV405_streaming": "火山语音合成-甜美小源",
                "BV064_streaming": "火山语音合成-小萝莉",
                "zh_female_roumeinvyou_emo_v2_mars_bigtts": "火山语音大模型-柔美女友",
                "ICL_zh_female_chunzhenshaonv_e588402fb8ad_tob": "火山语音大模型-纯真少女",
                "ICL_zh_female_qiuling_v1_tob": "火山语音大模型-倾心少女",
                "ICL_zh_female_yuxin_v1_tob": "火山语音大模型-初恋女友",
                "ICL_zh_female_yry_tob": "火山语音大模型-温柔白月光",
                "ICL_zh_female_wumeiyujie_tob": "火山语音大模型-妩媚御姐",
                "ICL_zh_female_aojiaonvyou_tob": "火山语音大模型-傲娇女友",
                "zh_female_wanwanxiaohe_moon_bigtts": "火山语音大模型-湾湾小何-小智AI",
            }
        elif self.cluster == "volcano_icl":
            voice_names = {
                "S_3hNEIJln1": "火山语音克隆-刻晴",
            }
        else:
            voice_names = {}
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
        """Generate TTS output from text using Huoshan API.

        Args:
            request_id (str):
                Unique identifier for the TTS request.
            text (str):
                Input text to convert to speech.
            voice_name (str):
                Voice identifier for the Huoshan TTS service.
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
        async with self.semaphore:
            submit_request_json = self.create_request_json(request_id, text, voice_name, voice_speed, voice_style)
            payload_bytes = str.encode(json.dumps(submit_request_json))
            payload_bytes = gzip.compress(payload_bytes)
            full_client_request = bytearray(self.default_header)
            full_client_request.extend((len(payload_bytes)).to_bytes(4, "big"))
            full_client_request.extend(payload_bytes)
            token = self.input_buffer[request_id].get("api_keys", {}).get("huoshan_token", None)
            header = {"Authorization": f"Bearer; {token}"}
            ret_dict = dict()
            audio_io = io.BytesIO()
            subtitle_result = dict()
            async with websockets.connect(
                self.tts_ws_url, additional_headers=header, ping_interval=None, max_size=100 * 1024 * 1024
            ) as ws:
                self.logger.debug(f"WebSocket connection established, voice style: {voice_style}, language: {language}")
                await ws.send(full_client_request)
                try:
                    buffer = bytearray()
                    while True:
                        message = await ws.recv()
                        done = await self._on_message(message, audio_io, subtitle_result, buffer)
                        if done:
                            break
                except websockets.exceptions.ConnectionClosed as e:
                    msg = "Connection closed unexpectedly."
                    self.logger.info(msg)
                    raise e
            self.logger.debug(f"TTS output: {subtitle_result}")
            audio_io.seek(0)
            with wave.open(audio_io, "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
            ret_dict["duration"] = duration
            ret_dict["audio"] = audio_io
            ret_dict.update(subtitle_result)
            return ret_dict

    async def _on_message(
        self,
        message: bytes,
        audio_io: io.BytesIO,
        subtitle_result: Dict[str, Any],
        buffer: bytearray,
    ) -> bool:
        """Process WebSocket messages from Huoshan TTS service.

        Args:
            message (bytes):
                Raw message received from the WebSocket connection.
            audio_io (io.BytesIO):
                BytesIO object to write audio data to.
            subtitle_result (Dict[str, Any]):
                Dictionary to store subtitle and timing information.
            buffer (bytearray):
                Buffer for handling partial audio data.

        Returns:
            bool:
                True if the message processing is complete and the
                connection should be closed, False to continue processing.
        """
        try:
            header_size = message[0] & 0x0F
            message_type = message[1] >> 4
            message_type_specific_flags = message[1] & 0x0F
            message_compression = message[2] & 0x0F
            payload = message[header_size * 4 :]

            if message_type == 0xB:
                if message_type_specific_flags == 0:
                    return False
                else:
                    sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                    payload = payload[8:]
                buffer.clear()
                buffer.extend(payload)
                audio_io.write(payload)
                if sequence_number < 0:
                    return True
                else:
                    return False
            elif message_type == 0xF:
                error_msg = payload[8:]
                if message_compression == 1:
                    error_msg = gzip.decompress(error_msg)
                error_msg = str(error_msg, "utf-8")
                self.logger.error(f"Received an error message: {error_msg}")
                return True
            elif message_type == 0xC:
                msg_size = int.from_bytes(payload[:4], "big", signed=False)
                payload = payload[4:]
                if message_compression == 1:
                    payload = gzip.decompress(payload)
                    response_data = json.loads(payload)
                    if "frontend" in response_data:
                        subtitle_result.update(self.convert_tts_time(response_data))
            else:
                self.logger.error(f"Received an unexpected message type: {message_type}")
                return True
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return True

    @classmethod
    def convert_tts_time(cls, tts_time: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Convert Huoshan TTS timing data to standardized format.

        Args:
            tts_time (Dict[str, Any]):
                Timing data from Huoshan TTS API containing 'frontend'
                field with word-level timing information.

        Returns:
            Dict[str, Any]:
                Standardized timing data containing:
                - `speech_text` (str): Concatenated text from all words
                - `speech_time` (List[Tuple[int, float]]): Character timing
                  mapping with (character_index, start_time) pairs
        """
        frontend_data = json.loads(tts_time["frontend"])
        words = frontend_data["words"]

        speech_text = ""
        speech_time = []
        for word_info in words:
            word = word_info["word"]
            word_start_time = word_info["start_time"] / 1000
            idx = len(speech_text)  # the word of the tts model may contain multiple characters
            speech_text += word
            speech_time.append((idx, word_start_time))

        # duration = words[-1]["end_time"] / 1000.0 if words else 0.0

        return {
            "speech_text": speech_text,
            # "duration": duration,
            "speech_time": speech_time,
        }
