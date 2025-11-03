import io
import json
from typing import Any, Dict, Set, Union

import websockets
from prometheus_client import Histogram
from typing_extensions import Buffer

from .tts_adapter import TextToSpeechAdapter


class SensetimeTTSClient(TextToSpeechAdapter):
    """Sensetime text-to-speech client.

    This client provides text-to-speech functionality using the Sensetime TTS
    API via both HTTP and WebSocket connections. It supports streaming audio
    generation with voice customization and style options.
    """

    AVAILABLE_FOR_STREAM = True
    _DEFAULT_PARAMS = dict(
        need_phone=False,
        need_polyphone=False,
        need_subtitle=True,
        ssml=False,
        continuous_synthesis=False,
        language="zh-CN",
        sample_rate=16000,
        volume=5,
        speed_ratio=1.0,
        pitch=0,
        tradition=False,
        voice="baimianmian",
        style="default",
        audio_type="wav",
        batch_file="",
        batch_audio_dir="",
    )

    def __init__(
        self,
        name: str,
        tts_ws_url: str,
        split_marks: Union[Set[str], None] = None,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the Sensetime text-to-speech client.

        Args:
            name (str):
                Unique name for this TTS client instance.
                Used as the logger name for identification.
            tts_ws_url (str):
                WebSocket URL for the Sensetime TTS streaming service.
            split_marks (Union[Set[str], None], optional):
                Custom split marks for text segmentation. If None, uses
                the default split marks from the base class. Defaults to None.
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
        self.tts_ws_url = tts_ws_url
        self.split_marks = split_marks

    async def get_voice_names(self, **kwargs: Any) -> Dict[str, Any]:
        """Get available voice names and their descriptions.

        Returns:
            Dict[str, Any]:
                Dictionary mapping voice identifiers to their
                human-readable names for Sensetime TTS service.
        """
        voice_names = {
            "keqing_0807": "刻晴-音色克隆",
            "funingna_0808_general": "芙宁娜-音色克隆",
            "hutao_0819_general": "胡桃-音色克隆",
            "baimianmian": "白绵绵",
            "xiaotao": "小桃",
            "yuanqishaonv": "元气少女",
            "xiaoning": "小宁",
        }
        return voice_names

    async def _generate_tts(
        self,
        request_id: str,
        text: str,
        voice_name: str,
        voice_speed: float = 1.0,
        voice_style: Union[None, str] = None,
        language: str = "zh-CN",  # only support chinese
        start_time: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate TTS output from text using Sensetime API.

        Args:
            request_id (str):
                Unique identifier for the TTS request.
            text (str):
                Input text to convert to speech.
            voice_name (str):
                Voice identifier for the Sensetime TTS service.
            voice_speed (float, optional):
                Speech rate multiplier. 1.0 is normal speed.
                Defaults to 1.0.
            voice_style (Union[None, str], optional):
                Voice style parameter. Defaults to None.
            language (str, optional):
                Language code for the text. Currently only supports
                Chinese ("zh-CN"). Defaults to "zh-CN".
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
        params = self._DEFAULT_PARAMS.copy()
        params["voice"] = voice_name
        params["speed_ratio"] = voice_speed
        params["language"] = "zh-CN"
        params["query"] = text
        ret_dict = dict()
        audio_io = io.BytesIO()
        subtitle_result = dict()
        async with websockets.connect(self.tts_ws_url, max_size=None) as websocket:
            message = json.dumps(params)
            await websocket.send(message)
            try:
                buffer = bytearray()
                while True:
                    message = await websocket.recv()
                    loop_continue = await self._on_message(message, audio_io, subtitle_result, buffer, start_time)
                    if not loop_continue:
                        break
            except websockets.exceptions.ConnectionClosed as e:
                msg = "Connection closed unexpectedly."
                self.logger.info(msg)
                raise e
        self.logger.debug(f"TTS output: {subtitle_result}")
        audio_io.seek(0)
        ret_dict["audio"] = audio_io
        ret_dict.update(subtitle_result)
        return ret_dict

    async def _on_message(
        self,
        message: Union[str, Buffer],
        audio_io: io.BytesIO,
        subtitle_result: Dict[str, Any],
        buffer: bytearray,
        start_time: float = 0.0,
    ) -> bool:
        """Process the message from the websocket.

        Args:
            message (Union[str, Buffer]):
                The message from the websocket.
            audio_io (io.BytesIO):
                The audio io object
                for the whole query.
            subtitle_result (Dict[str, Any]):
                The subtitle result.
            buffer (bytearray):
                The buffer that has not been
                written to the audio io object.
            start_time (float):
                The start time of the query.
                Defaults to 0.0.

        Returns:
            bool:
                Whether to continue the loop.
        """
        try:
            # if msg_type == websocket.ABNF.OPCODE_TEXT:
            if isinstance(message, str):
                msg = json.loads(message)
                if msg["status"] == "ok":
                    if msg["type"] == "partial":
                        audio_io.write(buffer)
                        buffer.clear()
                    if msg["type"] == "end":
                        audio_io.write(buffer)
                        return False
                    if msg["type"] == "phones":
                        pass
                    if msg["type"] == "subtitle":
                        self.logger.debug(f"TTS timestamp: {msg['subtitles']}")
                        timestamp_data = {"text": "", "duration": 0, "chars": []}
                        for subtitle in msg["subtitles"]:
                            timestamp_data["text"] += subtitle["text"]
                            timestamp_data["duration"] += subtitle["duration"]
                            timestamp_data["chars"].extend(subtitle["chars"])
                        subtitle_result.update(self.convert_tts_time(timestamp_data))
                else:
                    msg = f"Get unexpected TTS result: {msg}"
                    self.logger.error(msg)
                    raise ValueError(msg)
            else:
                # if msg_type == websocket.ABNF.OPCODE_BINARY:
                buffer.extend(message)
        except Exception as e:
            msg = f"Error on_data: {e}"
            self.logger.error(msg)
            return False
        return True

    @classmethod
    def convert_tts_time(cls, tts_time: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Convert the tts time to the speech_time matching the speech_text.

        Args:
            tts_time (Dict[str, Any]):
                The tts time.

        Returns:
            Dict[str, Any]:
                The converted tts time.
        """
        speech_text = tts_time["text"]
        unit = tts_time.get("unit", "ms")
        duration = tts_time["duration"]
        if unit == "ms":
            duration /= 1000
        elif unit == "s":
            pass
        else:
            raise ValueError(f"Invalid unit: {unit}")
        word_times = tts_time["chars"]
        speech_time = []
        cursor = 0
        for word_time in word_times:
            word = word_time["word"]
            if word == "SIL":
                continue
            word_start_time = word_time["start"] / 1000
            idx = speech_text.find(word, cursor)
            if idx != -1:
                speech_time.append((idx, word_start_time))
                cursor = idx + len(word)
        return {
            "speech_text": speech_text,
            "speech_time": speech_time,
            "duration": duration,
        }
