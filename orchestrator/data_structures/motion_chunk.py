from typing import List, Union

from .process_flow import DirectedAcyclicGraph


class MotionChunkStart:
    """Start chunk for motion animation stream processing.

    Represents the beginning of a motion animation stream with metadata about
    joint names, rest pose, data types, and processing context. Contains
    information needed to initialize motion animation processing pipelines.
    """

    def __init__(
        self,
        request_id: str,
        node_name: str,
        dag: DirectedAcyclicGraph,
        joint_names: List[str],
        restpose_name: str,
        dtype: str,
        blendshape_names: Union[List[str], None] = None,
        timeline_start_idx: Union[int, None] = None,
    ):
        """Initialize the motion chunk start.

        Args:
            request_id (str):
                Unique identifier for the request.
            node_name (str):
                Name of the processing node.
            dag (DirectedAcyclicGraph):
                Directed acyclic graph for workflow management.
            joint_names (List[str]):
                List of joint names used in the motion animation.
            restpose_name (str):
                Name of the rest pose used as reference for the motion.
            dtype (str):
                NumPy data type in string format for the motion data.
            blendshape_names (Union[List[str], None], optional):
                List of blendshape names if motion includes facial expressions.
                Defaults to None.
            timeline_start_idx (Union[int, None], optional):
                Starting index in the timeline for the motion animation.
                Defaults to None, starts from frame 0.
        """
        self.chunk_type = "start"
        self.request_id = request_id
        self.timeline_start_idx = timeline_start_idx
        self.node_name = node_name
        self.dag = dag
        self.joint_names = joint_names
        self.restpose_name = restpose_name
        self.dtype = dtype
        self.blendshape_names = blendshape_names


class MotionChunkBody:
    """Body chunk containing motion animation data.

    Represents a segment of motion animation data within a stream, containing
    the actual joint transformations and metadata about the segment's position.
    """

    def __init__(
        self,
        request_id: str,
        seq_number: int,
        data: bytes,
    ):
        """Initialize the motion chunk body.

        Args:
            request_id (str):
                Unique identifier for the request.
            seq_number (int):
                Sequence number for ordering chunks, starting from 0.
            data (bytes):
                Motion animation data containing joint transformations,
                root bone translation, cutoff annotations and blendshape values.
        """
        self.chunk_type = "body"
        self.request_id = request_id
        self.seq_number = seq_number
        self.data = data


class MotionChunkEnd:
    """End chunk signaling completion of motion animation stream.

    Represents the end of a motion animation stream, used to signal that no
    more motion animation data will be sent for the given request.
    """

    def __init__(
        self,
        request_id: str,
    ):
        """Initialize the motion chunk end.

        Args:
            request_id (str):
                Unique identifier for the request.
        """
        self.chunk_type = "end"
        self.request_id = request_id
