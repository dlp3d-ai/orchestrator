from typing import Union

from .dynamodb_memory_client import DynamoDBMemoryClient
from .mongodb_memory_client import MongoDBMemoryClient

_DB_MEMORY_CLIENTS = dict(
    DynamoDBMemoryClient=DynamoDBMemoryClient,
    MongoDBMemoryClient=MongoDBMemoryClient,
)


def build_db_memory_client(cfg: dict) -> Union[DynamoDBMemoryClient, MongoDBMemoryClient]:
    """Build a database memory client instance from a configuration dictionary.

    Creates and returns a database memory client instance based on the provided
    configuration. The configuration must include a 'type' field specifying
    which client class to instantiate, along with any required parameters
    for that client.

    Args:
        cfg (dict):
            Configuration dictionary containing:
            - type (str): The client class name to instantiate
            - Additional parameters specific to the chosen client type

    Returns:
        Union[DynamoDBMemoryClient, MongoDBMemoryClient]:
            An instance of the specified database memory client class.

    Raises:
        TypeError:
            If the specified client type is not found in the available clients.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    if cls_name not in _DB_MEMORY_CLIENTS:
        msg = f"Unknown database memory client type: {cls_name}"
        raise TypeError(msg)
    ret_inst = _DB_MEMORY_CLIENTS[cls_name](**cfg)
    return ret_inst
