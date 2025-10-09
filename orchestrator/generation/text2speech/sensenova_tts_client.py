import asyncio
import io
import json
import struct
import time
import traceback
import uuid
from typing import Any, Dict, Union

import numpy as np
import soundfile as sf
import websockets
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

from ...data_structures.audio_chunk import AudioWithSubtitleChunkBody
from ...data_structures.process_flow import DAGStatus
from ...utils.exception import MissingAPIKeyException
from .tts_adapter import TextToSpeechAdapter

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\ttts.proto\x12\x03tts"\x8a\x01\n\rSubtitleEntry\x12\x15\n\rstart_time_ms\x18\x01 \x01(\r\x12\x13\n\x0b\x65nd_time_ms\x18\x02 \x01(\r\x12\x0f\n\x07speaker\x18\x03 \x01(\t\x12\r\n\x05style\x18\x04 \x01(\t\x12\x1f\n\x08language\x18\x05 \x01(\x0e\x32\r.tts.Language\x12\x0c\n\x04text\x18\x06 \x01(\t"f\n\nAudioChunk\x12\x12\n\naudio_data\x18\x01 \x01(\x0c\x12\x17\n\x0f\x61udio_chunk_seq\x18\x02 \x01(\x05\x12\x15\n\ris_last_chunk\x18\x03 \x01(\x08\x12\x14\n\x0c\x61udio_format\x18\x04 \x01(\t"\xe8\x03\n\nTtsRequest\x12-\n\x0cmessage_type\x18\x01 \x01(\x0e\x32\x17.tts.RequestMessageType\x12\x0e\n\x06\x61pp_id\x18\x02 \x01(\t\x12\x15\n\rapp_signature\x18\x03 \x01(\t\x12\x0c\n\x04text\x18\x04 \x01(\t\x12\x16\n\x0etext_chunk_seq\x18\x05 \x01(\x05\x12\x1a\n\x12is_last_text_chunk\x18\x06 \x01(\x08\x12 \n\ttext_type\x18\x07 \x01(\x0e\x32\r.tts.TextType\x12\x0f\n\x07speaker\x18\x08 \x01(\t\x12\x1f\n\x08language\x18\t \x01(\x0e\x32\r.tts.Language\x12\r\n\x05style\x18\n \x01(\t\x12\r\n\x05speed\x18\x0b \x01(\x02\x12\x0e\n\x06volume\x18\x0c \x01(\x02\x12\r\n\x05pitch\x18\r \x01(\x02\x12\x15\n\rstream_output\x18\x0e \x01(\x08\x12\x19\n\x11\x61udio_sample_rate\x18\x0f \x01(\x05\x12*\n\x0e\x61udio_encoding\x18\x10 \x01(\x0e\x32\x12.tts.AudioEncoding\x12\x18\n\x10output_subtitles\x18\x11 \x01(\x08\x12\x12\n\nsession_id\x18\x12 \x01(\t\x12%\n\x0cupload_audio\x18\x13 \x01(\x0b\x32\x0f.tts.AudioChunk"\xa1\x02\n\x0bTtsResponse\x12$\n\x0bstatus_code\x18\x01 \x01(\x0e\x32\x0f.tts.StatusCode\x12\x14\n\x0c\x65rror_detail\x18\x02 \x01(\t\x12\x14\n\x0ctime_cost_ms\x18\x03 \x01(\r\x12*\n\x0e\x61udio_encoding\x18\x04 \x01(\x0e\x32\x12.tts.AudioEncoding\x12\x17\n\x0f\x61udio_chunk_seq\x18\x05 \x01(\x05\x12\x12\n\naudio_data\x18\x06 \x01(\x0c\x12\x1b\n\x13is_last_audio_chunk\x18\x07 \x01(\x08\x12\x12\n\nsession_id\x18\x08 \x01(\t\x12%\n\tsubtitles\x18\t \x03(\x0b\x32\x12.tts.SubtitleEntry\x12\x0f\n\x07speaker\x18\n \x01(\t*l\n\x12RequestMessageType\x12\x1c\n\x18\x43LIENT_SYNTHESIS_REQUEST\x10\x00\x12\x19\n\x15\x43LIENT_FINISH_REQUEST\x10\x01\x12\x1d\n\x19\x43LIENT_UPLOAD_CLONE_AUDIO\x10\x02*\x1f\n\x08TextType\x12\t\n\x05PLAIN\x10\x00\x12\x08\n\x04SSML\x10\x01*A\n\x08Language\x12\t\n\x05ZH_CN\x10\x00\x12\t\n\x05\x45N_US\x10\x01\x12\x11\n\rZH_CN_SICHUAN\x10\x02\x12\x0c\n\x08ZH_CN_HK\x10\x03**\n\rAudioEncoding\x12\x07\n\x03PCM\x10\x00\x12\x07\n\x03WAV\x10\x01\x12\x07\n\x03MP3\x10\x02*t\n\nStatusCode\x12\x0b\n\x07SUCCESS\x10\x00\x12\t\n\x05\x45RROR\x10\x01\x12\x0b\n\x07TIMEOUT\x10\x02\x12\x13\n\x0fINVALID_REQUEST\x10\x03\x12\x12\n\x0eINTERNAL_ERROR\x10\x04\x12\x18\n\x14UPLOAD_AUDIO_SUCCESS\x10\x05\x62\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "tts_pb2", _globals)
if _descriptor._USE_C_DESCRIPTORS is False:
    DESCRIPTOR._options = None
    _globals["_REQUESTMESSAGETYPE"]._serialized_start = 1046
    _globals["_REQUESTMESSAGETYPE"]._serialized_end = 1154
    _globals["_TEXTTYPE"]._serialized_start = 1156
    _globals["_TEXTTYPE"]._serialized_end = 1187
    _globals["_LANGUAGE"]._serialized_start = 1189
    _globals["_LANGUAGE"]._serialized_end = 1254
    _globals["_AUDIOENCODING"]._serialized_start = 1256
    _globals["_AUDIOENCODING"]._serialized_end = 1298
    _globals["_STATUSCODE"]._serialized_start = 1300
    _globals["_STATUSCODE"]._serialized_end = 1416
    _globals["_SUBTITLEENTRY"]._serialized_start = 19
    _globals["_SUBTITLEENTRY"]._serialized_end = 157
    _globals["_AUDIOCHUNK"]._serialized_start = 159
    _globals["_AUDIOCHUNK"]._serialized_end = 261
    _globals["_TTSREQUEST"]._serialized_start = 264
    _globals["_TTSREQUEST"]._serialized_end = 752
    _globals["_TTSRESPONSE"]._serialized_start = 755
    _globals["_TTSRESPONSE"]._serialized_end = 1044

# import protobuf generated classes and enums
TtsRequest = _globals["TtsRequest"]
TtsResponse = _globals["TtsResponse"]
SubtitleEntry = _globals["SubtitleEntry"]
AudioChunk = _globals["AudioChunk"]
RequestMessageType = _globals["RequestMessageType"]
TextType = _globals["TextType"]
Language = _globals["Language"]
AudioEncoding = _globals["AudioEncoding"]
StatusCode = _globals["StatusCode"]


class SensenovaTTSClient(TextToSpeechAdapter):
    """Sensenova text-to-speech client.

    This client provides text-to-speech functionality using the Sensenova TTS
    API via WebSocket connection with Protocol Buffers communication. It
    supports streaming audio generation with voice customization and style
    options.
    """

    AVAILABLE_FOR_STREAM = True

    def __init__(
        self,
        name: str,
        tts_ws_url: str,
        voice_style_file: str = "configs/voice_style.json",
        timeout: float = 10.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the Sensenova text-to-speech client.

        Args:
            name (str):
                Unique name for this TTS client instance.
                Used as the logger name for identification.
            tts_ws_url (str):
                WebSocket URL for the Sensenova TTS service.
            voice_style_file (str, optional):
                Path to JSON file containing voice style configurations.
                Defaults to "configs/voice_style.json".
            timeout (float, optional):
                Timeout in seconds for WebSocket operations.
                Defaults to 10.0.
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
        self.timeout = timeout

        with open(voice_style_file, "r", encoding="utf-8") as f:
            self.voice_style_dict = json.load(f)

    def create_synthesis_request(
        self,
        text: str,
        text_chunk_seq: int = 0,
        is_last_text_chunk: bool = False,
        speaker: str = "female_taozi_p2",
        language: str = "zh",
        style: str = "正常",
        speed: float = 1.0,
        volume: float = 0,
        pitch: float = 0,
        session_id: str = "",
        app_id: str = "1d0fb419-a464-4271-bff2-148ca615a297",
    ) -> TtsRequest:
        """Create a TTS synthesis request using Protocol Buffers.

        Args:
            text (str):
                Input text to convert to speech.
            text_chunk_seq (int, optional):
                Sequence number for text chunking. Defaults to 0.
            is_last_text_chunk (bool, optional):
                Whether this is the last text chunk. Defaults to False.
            speaker (str, optional):
                Voice speaker identifier. Defaults to "female_taozi_p2".
            language (str, optional):
                Language code for the text. Defaults to "zh".
            style (str, optional):
                Voice style parameter. Defaults to "正常".
            speed (float, optional):
                Speech rate multiplier. 1.0 is normal speed. Defaults to 1.0.
            volume (float, optional):
                Volume level adjustment. Defaults to 0.
            pitch (float, optional):
                Pitch adjustment. Defaults to 0.
            session_id (str, optional):
                Session identifier for the request. Defaults to "".
            app_id (str, optional):
                Application ID for authentication. Defaults to a predefined value.

        Returns:
            TtsRequest:
                Protocol Buffers TTS request object ready for serialization.
        """
        request = TtsRequest()
        request.message_type = RequestMessageType.CLIENT_SYNTHESIS_REQUEST
        request.text = text
        request.text_chunk_seq = text_chunk_seq
        request.is_last_text_chunk = is_last_text_chunk
        request.text_type = TextType.PLAIN
        request.speaker = speaker
        request.language = Language.ZH_CN if language == "zh" else Language.EN_US
        request.style = style
        request.speed = speed
        request.volume = volume
        request.pitch = pitch
        request.stream_output = True
        request.audio_sample_rate = 16000
        request.audio_encoding = AudioEncoding.PCM
        request.output_subtitles = True
        request.session_id = session_id
        request.app_id = app_id
        return request

    def serialize_request(self, request):
        """Serialize the request.

        Args:
            request (TtsRequest):
                The request to serialize.

        Returns:
            bytes:
                The serialized request.
        """
        request_bytes = request.SerializeToString()
        request_length = struct.pack("!I", len(request_bytes))
        full_request = b"\x01" + request_length + request_bytes
        return full_request

    def parse_response(self, protocol_type: int, data: bytes) -> TtsResponse:
        """Parse Protocol Buffers response from Sensenova TTS service.

        Args:
            protocol_type (int):
                Protocol type identifier from the message header.
            data (bytes):
                Raw response data to parse.

        Returns:
            TtsResponse:
                Parsed Protocol Buffers response object.

        Raises:
            ValueError:
                If the response data cannot be parsed.
        """
        try:
            response = TtsResponse()
            response.ParseFromString(data)
            return response
        except Exception as e:
            raise ValueError(f"Failed to parse response: {str(e)}")

    async def receive_full_message(self, websocket) -> tuple[int, bytes]:
        """Receive complete message from WebSocket connection.

        Args:
            websocket:
                WebSocket connection to receive data from.

        Returns:
            tuple[int, bytes]:
                Tuple containing (protocol_type, data) where protocol_type
                is the protocol identifier and data is the complete message bytes.

        Raises:
            ValueError:
                If the message is invalid, too short, or times out.
        """
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=self.timeout)

            if len(message) < 5:
                raise ValueError("Invalid response: too short")

            protocol_type = message[0]
            if protocol_type != 0x01:
                raise ValueError("Unsupported protocol type")

            protocol_length = struct.unpack("!I", message[1:5])[0]
            data = message[5:]

            if len(data) != protocol_length:
                while len(data) < protocol_length:
                    try:
                        chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=self.timeout)
                        if not chunk:
                            raise ValueError("Server disconnected or sent empty data")
                        data += chunk
                    except asyncio.TimeoutError:
                        raise ValueError(f"Timeout while receiving message. Got {len(data)}/{protocol_length} bytes")

            return protocol_type, data

        except asyncio.TimeoutError:
            raise ValueError(f"Response timed out after {self.timeout} seconds")
        except Exception as e:
            raise ValueError(f"Error receiving data: {str(e)}")

    async def get_voice_names(self, **kwargs: Any) -> Dict[str, Any]:
        """Get available voice names and their descriptions.

        Returns:
            Dict[str, Any]:
                Dictionary mapping voice identifiers to their
                human-readable names for Sensenova TTS service.
        """
        voice_names = {
            "F12": "对话多情感机车少女",
            "female_taozi_p2": "对话多情感可爱桃",
            "female_chunzhen": "纯真少女",
        }
        return voice_names

    async def _stream_generate_task(
        self,
        request_id: str,
        text_segment: str,
        seq_number: int,
    ) -> None:
        """Handle streaming TTS generation task for a text segment.

        Args:
            request_id (str):
                Unique identifier for the TTS request.
            text_segment (str):
                Text segment to convert to speech.
            seq_number (int):
                Sequence number for ordering the generated audio chunks.
        """
        try:
            generation_start_time = time.time()
            ret_dict = await self._generate_tts(
                request_id=request_id,
                text=text_segment,
                voice_name=self.input_buffer[request_id]["voice_name"],
                voice_speed=self.input_buffer[request_id]["voice_speed"],
                voice_style=self.input_buffer[request_id].get("voice_style", "正常"),
                language=self.input_buffer[request_id]["language"],
            )
            duration = ret_dict["duration"]
            generation_end_time = time.time()
            msg = f"Streaming TTS generation spent {generation_end_time - generation_start_time:.3f} seconds"
            msg = msg + f" for request {request_id}"
            msg += f", seq_number: {seq_number}"
            self.logger.debug(msg)
            dag = self.input_buffer[request_id]["dag"]
            while self.input_buffer[request_id]["chunk_sent"] < seq_number:
                if dag.status == DAGStatus.RUNNING:
                    await asyncio.sleep(self.sleep_time)
                else:
                    msg = f"Streaming TTS generation interrupted by DAG status {dag.status}"
                    msg = msg + f" for request {request_id}"
                    msg += f", seq_number: {seq_number}"
                    self.logger.warning(msg)
                    return
            # prepare downstream
            dag = self.input_buffer[request_id]["dag"]
            node_name = self.input_buffer[request_id]["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            downstream_warned = self.input_buffer[request_id]["downstream_warned"]
            if len(downstream_nodes) == 0 and not downstream_warned:
                self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
                self.input_buffer[request_id]["downstream_warned"] = True
                return
            coroutines = list()
            for node in downstream_nodes:
                payload = node.payload
                audio_io = io.BytesIO()
                audio_io.write(ret_dict["audio"].getvalue())
                audio_io.seek(0)
                body_trunk = AudioWithSubtitleChunkBody(
                    request_id=request_id,
                    duration=duration,
                    audio_io=audio_io,
                    seq_number=seq_number,
                    speech_text=text_segment,
                    speech_time=ret_dict["speech_time"],
                )
                coroutines.append(payload.feed_stream(body_trunk))
            asyncio.gather(*coroutines)
            self.input_buffer[request_id]["chunk_sent"] += 1
            first_chunk_handle_time = self.input_buffer[request_id].get("first_chunk_handle_time", None)
            dag_start_time = self.input_buffer[request_id].get("dag_start_time", None)
            if seq_number == 0 and first_chunk_handle_time is not None:
                cur_time = time.time()
                latency = cur_time - first_chunk_handle_time
                msg = (
                    f"Request {request_id} first AudioWithSubtitleChunkBody delay {latency:.2f}s "
                    + "from receiving first TextChunkBody."
                )
                if dag_start_time is not None:
                    latency = cur_time - dag_start_time
                    msg = msg[:-1] + f", delay {latency:.2f}s from dag start."
                self.logger.debug(msg)
        except Exception as e:
            msg = f"Error in streaming TTS: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.FAILED)
            return

    async def _generate_tts(
        self,
        request_id: str,
        text: str,
        voice_name: str,
        voice_speed: float = 1.0,
        voice_style: Union[None, str] = "正常",
        language: str = "ZH-CN",
        start_time: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate TTS output from text using Sensenova API.

        Args:
            request_id (str):
                Unique identifier for the TTS request.
            text (str):
                Input text to convert to speech.
            voice_name (str):
                Voice speaker identifier for the Sensenova TTS service.
            voice_speed (float, optional):
                Speech rate multiplier. 1.0 is normal speed.
                Defaults to 1.0.
            voice_style (Union[None, str], optional):
                Voice style parameter. If None or invalid, defaults to "正常".
                Defaults to "正常".
            language (str, optional):
                Language code for the text. Defaults to "ZH-CN".
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
        apikey = self.input_buffer[request_id].get("api_keys", {}).get("nova_tts_api_key", None)
        if not apikey:
            msg = "Sensenova API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)
        headers = {"apikey": apikey}
        ret_dict = dict()
        audio_io = io.BytesIO()
        speech_time = []

        session_id = str(uuid.uuid4())
        audio_data = b""
        seq = -1
        is_running = True
        start_generation_time = time.time()

        if voice_name not in self.voice_style_dict or voice_style not in self.voice_style_dict[voice_name]:
            voice_style = "正常"

        try:
            async with websockets.connect(self.tts_ws_url, additional_headers=headers) as websocket:
                self.logger.debug(f"WebSocket connection established, voice style: {voice_style}, language: {language}")

                # start receive loop
                async def receive_loop():
                    nonlocal audio_data, seq, is_running, speech_time
                    try:
                        while is_running:
                            try:
                                protocol_type, data = await self.receive_full_message(websocket)
                                response = self.parse_response(protocol_type, data)

                                if response.status_code == StatusCode.SUCCESS:
                                    self.logger.debug(
                                        f"Audio chunk seq:{response.audio_chunk_seq}, is_last:{response.is_last_audio_chunk}"
                                    )

                                    # check sequence number
                                    if response.audio_chunk_seq != seq + 1:
                                        self.logger.error("Invalid sequence number")
                                        is_running = False
                                        break
                                    seq = response.audio_chunk_seq

                                    if response.audio_chunk_seq == 0:
                                        cost = time.time() - start_generation_time
                                        self.logger.debug(f"First chunk received, cost:{cost:.3f}s")

                                    if response.audio_data:
                                        audio_data += response.audio_data
                                        self.logger.debug(f"Audio data length: {len(response.audio_data)} bytes")

                                    # handle subtitle information
                                    if response.subtitles:
                                        self.logger.debug("Subtitles received")
                                        for subtitle in response.subtitles:
                                            self.logger.debug(
                                                f"{subtitle.text} ({subtitle.start_time_ms}-{subtitle.end_time_ms}ms)"
                                            )
                                            # build speech_time list - estimate time based on character position
                                            start_time_s = subtitle.start_time_ms / 1000.0
                                            end_time_s = subtitle.end_time_ms / 1000.0
                                            subtitle_duration = end_time_s - start_time_s
                                            subtitle_len = len(subtitle.text)

                                            for i, char in enumerate(subtitle.text):
                                                # linear interpolation to calculate the start time of each character
                                                if subtitle_len > 1:
                                                    char_time = round(
                                                        start_time_s + (subtitle_duration * i / subtitle_len), 3
                                                    )
                                                else:
                                                    char_time = round(start_time_s, 3)

                                                # speech_time format: (character index in the text, character start time)
                                                char_index = len(speech_time)
                                                speech_time.append((char_index, char_time))

                                    if response.is_last_audio_chunk:
                                        whole_cost = time.time() - start_generation_time
                                        if len(audio_data) > 0:
                                            duration = len(audio_data) / 2 / 16000
                                            rtf = whole_cost / duration
                                            self.logger.debug(
                                                f"Generation complete, total cost {whole_cost:.3f}s, data length {len(audio_data)}, duration {duration:.3f}s, RTF {rtf:.3f}"
                                            )
                                        else:
                                            self.logger.warning("Received empty audio data")
                                        is_running = False
                                else:
                                    self.logger.error(f"Response error {response.status_code}: {response.error_detail}")
                                    is_running = False

                            except asyncio.CancelledError:
                                self.logger.debug("Receive loop cancelled")
                                is_running = False
                                break
                            except Exception as e:
                                self.logger.error(f"Receive loop error: {e}")
                                break
                    except Exception as e:
                        self.logger.error(f"Receive loop terminated: {e}")
                        is_running = False

                receive_task = asyncio.create_task(receive_loop())

                # send text (stream by character)
                text_list = [char for char in text]

                for i, chunk in enumerate(text_list):
                    if not is_running:
                        break
                    is_last = i == len(text_list) - 1
                    try:
                        request = self.create_synthesis_request(
                            text=chunk,
                            text_chunk_seq=i,
                            is_last_text_chunk=is_last,
                            session_id=session_id,
                            speaker=voice_name,
                            language=language,
                            style=voice_style,
                            speed=voice_speed,
                        )
                        full_request = self.serialize_request(request)
                        await websocket.send(full_request)

                    except Exception as e:
                        self.logger.error(f"Error sending text chunk {i}: {e}")
                        is_running = False
                        break

                # wait for receive to complete
                await receive_task

                # handle audio data
                if audio_data:
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    # convert audio to wav format and write to BytesIO
                    sf.write(audio_io, audio_np, samplerate=16000, subtype="PCM_16", format="WAV")
                    audio_io.seek(0)

                    duration = len(audio_data) / 2 / 16000  # 16kHz, 16-bit

                    ret_dict["audio"] = audio_io
                    ret_dict["speech_text"] = text
                    ret_dict["speech_time"] = speech_time
                    ret_dict["duration"] = duration
                else:
                    self.logger.warning("No audio data received")
                    # return empty audio
                    ret_dict["audio"] = audio_io
                    ret_dict["speech_text"] = text
                    ret_dict["speech_time"] = []
                    ret_dict["duration"] = 0.0

        except Exception as e:
            self.logger.error(f"TTS generation error: {e}")
            # return empty audio
            ret_dict["audio"] = audio_io
            ret_dict["speech_text"] = text
            ret_dict["speech_time"] = []
            ret_dict["duration"] = 0.0

        return ret_dict
