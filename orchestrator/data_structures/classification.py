from enum import Enum

from .process_flow import DirectedAcyclicGraph


class ClassificationType(Enum):
    """Classification type enumeration.

    Defines the possible classification results for user input processing,
    determining how the system should respond to different types of requests.
    """

    LEAVE = "leave"
    REJECT = "reject"
    ACCEPT = "accept"


class ClassificationChunkStart:
    """Start chunk for classification stream processing.

    Represents the beginning of a classification stream with metadata about the
    processing context. Contains information needed to initialize
    classification processing pipelines.
    """

    def __init__(self, request_id: str, node_name: str, dag: DirectedAcyclicGraph):
        """Initialize the classification chunk start.

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


class ClassificationChunkBody:
    """Body chunk containing classification data.

    Represents a segment of classification data within a stream, containing the
    message to be classified and the resulting classification result.
    """

    def __init__(self, request_id: str, message: str, classification_result: ClassificationType):
        """Initialize the classification chunk body.

        Args:
            request_id (str):
                Unique identifier for the request.
            message (str):
                The message text that was classified.
            classification_result (ClassificationType):
                The result of the classification process.
        """
        self.chunk_type = "body"
        self.request_id = request_id
        self.message = message
        self.classification_result = classification_result


class ClassificationChunkEnd:
    """End chunk signaling completion of classification stream.

    Represents the end of a classification stream, used to signal that no more
    classification data will be sent for the given request.
    """

    def __init__(self, request_id: str):
        """Initialize the classification chunk end.

        Args:
            request_id (str):
                Unique identifier for the request.
        """
        self.chunk_type = "end"
        self.request_id = request_id
