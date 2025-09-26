from typing import List, Union

from .process_flow import DirectedAcyclicGraph


class FaceChunkStart:
    """Start chunk for face animation stream processing.

    Represents the beginning of a face animation stream with metadata about
    blendshape names, data types, and processing context. Contains information
    needed to initialize face animation processing pipelines.
    """

    def __init__(
        self,
        request_id: str,
        blendshape_names: List[str],
        dtype: str,
        node_name: str,
        dag: DirectedAcyclicGraph,
        timeline_start_idx: Union[int, None] = None,
    ):
        """Initialize the face chunk start.

        Args:
            request_id (str):
                Unique identifier for the request.
            blendshape_names (List[str]):
                List of blendshape names used in the face animation.
            dtype (str):
                NumPy data type in string format for the face data.
            node_name (str):
                Name of the processing node.
            dag (DirectedAcyclicGraph):
                Directed acyclic graph for workflow management.
            timeline_start_idx (Union[int, None], optional):
                Starting index in the timeline for the face animation.
                Defaults to None, starts from frame 0.
        """
        self.chunk_type = "start"
        self.request_id = request_id
        self.blendshape_names = blendshape_names
        self.dtype = dtype
        self.node_name = node_name
        self.dag = dag
        self.timeline_start_idx = timeline_start_idx


class FaceChunkBody:
    """Body chunk containing face animation data.

    Represents a segment of face animation data within a stream, containing the
    actual blendshape values and metadata about the segment's position.
    """

    def __init__(
        self,
        request_id: str,
        data: bytes,
        seq_number: int,
    ):
        """Initialize the face chunk body.

        Args:
            request_id (str):
                Unique identifier for the request.
            data (bytes):
                Face animation data containing blendshape values.
            seq_number (int):
                Sequence number for ordering chunks, starting from 0.
        """
        self.chunk_type = "body"
        self.request_id = request_id
        self.data = data
        self.seq_number = seq_number


class FaceChunkEnd:
    """End chunk signaling completion of face animation stream.

    Represents the end of a face animation stream, used to signal that no more
    face animation data will be sent for the given request.
    """

    def __init__(
        self,
        request_id: str,
    ):
        """Initialize the face chunk end.

        Args:
            request_id (str):
                Unique identifier for the request.
        """
        self.chunk_type = "end"
        self.request_id = request_id
