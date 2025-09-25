from typing import Union

from .dynamodb_config_client import DynamoDBConfigClient
from .dynamodb_redis_config_client import DynamoDBRedisConfigClient
from .mongodb_config_client import MongoDBConfigClient

_DB_CONFIG_CLIENTS = dict(
    DynamoDBConfigClient=DynamoDBConfigClient,
    DynamoDBRedisConfigClient=DynamoDBRedisConfigClient,
    MongoDBConfigClient=MongoDBConfigClient,
)


def build_db_config_client(cfg: dict) -> Union[DynamoDBConfigClient, DynamoDBRedisConfigClient, MongoDBConfigClient]:
    """Build a database configuration client instance from a configuration
    dictionary.

    This function creates an instance of the appropriate database configuration
    client based on the 'type' field in the configuration dictionary. The
    supported client types are DynamoDBConfigClient, DynamoDBRedisConfigClient,
    and MongoDBConfigClient.

    Args:
        cfg (dict):
            Configuration dictionary containing the 'type' field and other
            parameters required for the specific database client. The 'type'
            field will be removed from the dictionary before passing remaining
            parameters to the client constructor.

    Returns:
        Union[DynamoDBConfigClient, DynamoDBRedisConfigClient, MongoDBConfigClient]:
            An instance of the appropriate database configuration client
            initialized with the provided configuration parameters.

    Raises:
        TypeError:
            If the 'type' field in the configuration dictionary does not
            correspond to any supported database client type.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    if cls_name not in _DB_CONFIG_CLIENTS:
        msg = f"Unknown database config client type: {cls_name}"
        raise TypeError(msg)
    ret_inst = _DB_CONFIG_CLIENTS[cls_name](**cfg)
    return ret_inst
