import io
from typing import List, Literal, Tuple, Union

from .process_flow import DirectedAcyclicGraph
from .reaction import Reaction


class AudioChunkStart:
    """Start chunk for audio stream processing.

    Represents the beginning of an audio stream with metadata about the audio
    format and processing context. Contains information needed to initialize
    audio processing pipelines.
    """

    def __init__(
        self,
        request_id: str,
        audio_type: Literal["wav", "pcm", "mp3"],
        node_name: str,
        dag: DirectedAcyclicGraph,
        n_channels: Union[int, None] = None,
        sample_width: Union[int, None] = None,
        frame_rate: Union[int, None] = None,
    ):
        """Initialize the audio chunk start.

        Args:
            request_id (str):
                Unique identifier for the request.
            audio_type (Literal["wav", "pcm", "mp3"]):
                Type of audio format being processed.
            node_name (str):
                Name of the processing node.
            dag (DirectedAcyclicGraph):
                Directed acyclic graph for workflow management.
            n_channels (Union[int, None], optional):
                Number of audio channels. Required for PCM audio.
                Defaults to None.
            sample_width (Union[int, None], optional):
                Sample width in bytes. Required for PCM audio.
                Defaults to None.
            frame_rate (Union[int, None], optional):
                Audio frame rate in Hz. Required for PCM audio.
                Defaults to None.
        """
        self.chunk_type = "start"
        self.audio_type = audio_type
        self.request_id = request_id
        self.node_name = node_name
        self.dag = dag
        self.n_channels = n_channels
        self.sample_width = sample_width
        self.frame_rate = frame_rate


class AudioChunkBody:
    """Body chunk containing actual audio data.

    Represents a segment of audio data within a stream, containing the actual
    audio bytes and metadata about the segment's duration and position.
    """

    def __init__(
        self,
        request_id: str,
        duration: float,
        audio_io: io.BytesIO,
        seq_number: int,
    ):
        """Initialize the audio chunk body.

        Args:
            request_id (str):
                Unique identifier for the request.
            duration (float):
                Duration of the audio segment in seconds.
            audio_io (io.BytesIO):
                BytesIO object containing the audio data.
            seq_number (int):
                Sequence number for ordering chunks, starting from 0.
        """
        self.chunk_type = "body"
        self.request_id = request_id
        self.duration = duration
        self.audio_io = audio_io
        self.seq_number = seq_number


class AudioChunkEnd:
    """End chunk signaling completion of audio stream.

    Represents the end of an audio stream, used to signal that no more audio
    data will be sent for the given request.
    """

    def __init__(
        self,
        request_id: str,
    ):
        """Initialize the audio chunk end.

        Args:
            request_id (str):
                Unique identifier for the request.
        """
        self.chunk_type = "end"
        self.request_id = request_id


class AudioWithSubtitleChunkStart(AudioChunkStart):
    """Start chunk for audio streams with subtitle/speech text data.

    Extends AudioChunkStart to support audio streams that include synchronized
    subtitle or speech text information.
    """

    pass


class AudioWithSubtitleChunkBody(AudioChunkBody):
    """Body chunk for audio streams with synchronized subtitle/speech text.

    Extends AudioChunkBody to include speech text and timing information,
    enabling synchronization between audio and text content.
    """

    def __init__(
        self,
        request_id: str,
        duration: float,
        audio_io: io.BytesIO,
        seq_number: int,
        speech_text: str,
        speech_time: List[Tuple[int, float]],
    ):
        """Initialize the audio chunk body with subtitle data.

        Args:
            request_id (str):
                Unique identifier for the request.
            duration (float):
                Duration of the audio segment in seconds.
            audio_io (io.BytesIO):
                BytesIO object containing the audio data.
            seq_number (int):
                Sequence number for ordering chunks, starting from 0.
            speech_text (str):
                Text corresponding to the audio, including phonetic characters
                and punctuation that affects pronunciation duration, in human-readable format.
            speech_time (List[Tuple[int, float]]):
                List of tuples mapping character indices to their pronunciation start times.
                Length is ≤ len(speech_text). Each tuple contains (character_index, time_in_seconds).
                The pronunciation time for character speech_text[speech_time[i][0]]
                starts at speech_time[i][1] seconds.
        """
        AudioChunkBody.__init__(self, request_id, duration, audio_io, seq_number)
        self.speech_text = speech_text
        self.speech_time = speech_time


class AudioWithSubtitleChunkEnd(AudioChunkEnd):
    """End chunk for audio streams with subtitle/speech text data.

    Extends AudioChunkEnd to support audio streams that include synchronized
    subtitle or speech text information.
    """

    pass


class AudioWithReactionChunkStart(AudioWithSubtitleChunkStart):
    """Start chunk for audio streams with reaction data.

    Extends AudioWithSubtitleChunkStart to support audio streams that include
    reaction information along with speech text and audio data.
    """

    pass


class AudioWithReactionChunkBody(AudioWithSubtitleChunkBody):
    """Body chunk for audio streams with reaction data.

    Extends AudioWithSubtitleChunkBody to include reaction information,
    enabling synchronized audio, text, and reaction processing.
    """

    def __init__(
        self,
        request_id: str,
        duration: float,
        audio_io: io.BytesIO,
        seq_number: int,
        speech_text: str,
        speech_time: List[Tuple[int, float]],
        reaction: Reaction,
    ):
        """Initialize the audio chunk body with reaction data.

        Args:
            request_id (str):
                Unique identifier for the request.
            duration (float):
                Duration of the audio segment in seconds.
            audio_io (io.BytesIO):
                BytesIO object containing the audio data.
            seq_number (int):
                Sequence number for ordering chunks, starting from 0.
            speech_text (str):
                Text corresponding to the audio, including phonetic characters
                and punctuation that affects pronunciation duration, in human-readable format.
            speech_time (List[Tuple[int, float]]):
                List of tuples mapping character indices to their pronunciation start times.
                Length is ≤ len(speech_text). Each tuple contains (character_index, time_in_seconds).
                The pronunciation time for character speech_text[speech_time[i][0]]
                starts at speech_time[i][1] seconds.
            reaction (Reaction):
                Reaction object containing emotion, motion, and expression data.
        """
        AudioWithSubtitleChunkBody.__init__(self, request_id, duration, audio_io, seq_number, speech_text, speech_time)
        self.reaction = reaction


class AudioWithReactionChunkEnd(AudioWithSubtitleChunkEnd):
    """End chunk for audio streams with reaction data.

    Extends AudioWithSubtitleChunkEnd to support audio streams that include
    reaction information along with speech text and audio data.
    """

    pass
