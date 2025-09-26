from enum import Enum
from typing import Any, Dict, List, Union

from ..utils.super import Super


class DAGStatus(Enum):
    """Status enumeration for Directed Acyclic Graph execution states.

    Defines the possible states that a DAG can be in during its lifecycle, from
    initialization through completion or failure.
    """

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DAGNode:
    """Node in a directed acyclic graph representing a processing unit.

    Each node contains a payload (processing logic) and maintains connections
    to upstream and downstream nodes, forming the execution flow of the DAG.
    """

    def __init__(
        self,
        name: str,
        payload: Any,
    ):
        """Initialize the DAG node.

        Args:
            name (str):
                Unique name identifier for the node.
            payload (Any):
                Processing logic or data associated with this node.
        """
        self.name = name
        self.upstreams: List["DAGNode"] = []
        self.downstreams: List["DAGNode"] = []
        self.payload: Any = payload

    def add_upstream(self, upstream_node: "DAGNode"):
        """Add an upstream node to the current node.

        Establishes a dependency relationship where the upstream node
        must be triggered before this node is executed.

        Args:
            upstream_node (DAGNode):
                The upstream node to add as a dependency.
        """
        self.upstreams.append(upstream_node)

    def add_downstream(self, downstream_node: "DAGNode"):
        """Add a downstream node to the current node.

        Establishes a dependency relationship where this node must
        be executed before the downstream node is triggered.

        Args:
            downstream_node (DAGNode):
                The downstream node to add as a dependent.
        """
        self.downstreams.append(downstream_node)


class DirectedAcyclicGraph(Super):
    """Directed Acyclic Graph for orchestrating complex processing workflows.

    Manages a collection of interconnected nodes that represent processing
    steps. Ensures proper execution order through dependency management and
    provides cycle detection to maintain acyclic structure.
    """

    def __init__(self, name: str, conf: Dict[str, Any], logger_cfg: Union[None, Dict[str, Any]] = None):
        """Initialize the directed acyclic graph.

        Args:
            name (str):
                Unique name identifier for the DAG.
            conf (Dict[str, Any]):
                Configuration dictionary containing parameters and settings
                required by the nodes in this DAG.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary. If None, default logging is used.
                Defaults to None.
        """
        super().__init__(logger_cfg=logger_cfg)
        self.name = name
        self.conf = conf
        self.nodes: Dict[str, DAGNode] = {}
        self.status = DAGStatus.IDLE

    def set_status(self, status: DAGStatus):
        """Set the execution status of the DAG.

        Args:
            status (DAGStatus):
                New status to set for the DAG execution.
        """
        self.status = status

    def add_node(self, node: DAGNode):
        """Add a node to the graph.

        Args:
            node (DAGNode):
                The DAG node to add to the graph.
        """
        self.nodes[node.name] = node

    def add_edge(self, upstream_name: str, downstream_name: str):
        """Add an edge between two nodes in the graph.

        Creates a dependency relationship where the upstream node must
        be executed before the downstream node is triggered.

        Args:
            upstream_name (str):
                Name of the upstream node (dependency).
            downstream_name (str):
                Name of the downstream node (dependent).

        Raises:
            ValueError:
                If either node name is not found in the graph.
        """
        if upstream_name not in self.nodes:
            msg = f"Upstream node {upstream_name} not found"
            self.logger.error(msg)
            raise ValueError(msg)
        if downstream_name not in self.nodes:
            msg = f"Downstream node {downstream_name} not found"
            self.logger.error(msg)
            raise ValueError(msg)
        self.nodes[upstream_name].add_downstream(self.nodes[downstream_name])
        self.nodes[downstream_name].add_upstream(self.nodes[upstream_name])

    def get_entry_nodes(self) -> List[DAGNode]:
        """Get all entry nodes (nodes with no upstream dependencies) of the
        graph.

        Returns:
            List[DAGNode]:
                List of nodes that have no upstream dependencies and can be
                executed first in the workflow.
        """
        return [node for node in self.nodes.values() if len(node.upstreams) == 0]

    def get_node(self, name: str) -> DAGNode:
        """Get a node from the graph by name.

        Args:
            name (str):
                Name of the node to retrieve.

        Returns:
            DAGNode:
                The node with the specified name.

        Raises:
            KeyError:
                If no node with the given name exists.
        """
        return self.nodes[name]

    def check_cycle(self) -> bool:
        """Check if the graph contains any cycles.

        Uses depth-first search to detect cycles in the graph structure.
        A cycle would violate the acyclic property required for proper
        execution ordering.

        Returns:
            bool:
                True if the graph contains a cycle, False if it is acyclic.
        """
        visited = set()
        stack = set()

        def dfs(node):
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for neighbor in node.downstreams:
                if dfs(neighbor):
                    return True
            stack.remove(node)

        for node in self.nodes.values():
            if dfs(node):
                return True
        return False
