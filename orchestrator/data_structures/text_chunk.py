from typing import Union

from .process_flow import DirectedAcyclicGraph


class TextChunkStart:
    """Start chunk for text stream processing.

    Represents the beginning of a text stream with metadata about the
    processing context. Contains information needed to initialize text
    processing pipelines.
    """

    def __init__(self, request_id: str, node_name: str, dag: DirectedAcyclicGraph):
        """Initialize the text chunk start.

        Args:
            request_id (str):
                Unique identifier for the request.
            node_name (str):
                Name of the processing node.
            dag (DirectedAcyclicGraph):
                Directed acyclic graph for workflow management.
        """
        self.chunk_type = "start"
        self.request_id = request_id
        self.node_name = node_name
        self.dag = dag


class TextChunkBody:
    """Body chunk containing text data.

    Represents a segment of text data within a stream, containing the actual
    text content and optional styling information.
    """

    def __init__(self, request_id: str, text_segment: str, style: Union[None, str] = None):
        """Initialize the text chunk body.

        Args:
            request_id (str):
                Unique identifier for the request.
            text_segment (str):
                The text content for this chunk.
            style (Union[None, str], optional):
                Optional styling information for the text.
                Defaults to None.
        """
        self.chunk_type = "body"
        self.request_id = request_id
        self.text_segment = text_segment
        self.style = style


class TextChunkEnd:
    """End chunk signaling completion of text stream.

    Represents the end of a text stream, used to signal that no more text data
    will be sent for the given request.
    """

    def __init__(self, request_id: str):
        """Initialize the text chunk end.

        Args:
            request_id (str):
                Unique identifier for the request.
        """
        self.chunk_type = "end"
        self.request_id = request_id
