import argparse
import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from pymongo import MongoClient

from orchestrator.proxy import Proxy
from orchestrator.service.server import OrchestratorProxyServer
from orchestrator.utils.config import file2dict
from orchestrator.utils.executor_registry import ExecutorRegistry
from orchestrator.utils.log import setup_logger


def main(args) -> int:
    """Main entry point for the orchestrator proxy server.

    Initializes and starts the orchestrator proxy server with the provided
    configuration. This function handles the complete startup process including:
    creating necessary directories, loading configuration, initializing the
    thread pool executor, setting up the proxy and server components, and
    starting the server.

    Args:
        args:
            Parsed command line arguments containing configuration path
            and max workers settings.

    Returns:
        int:
            Exit code (0 for success).

    Raises:
        ValueError:
            If an invalid proxy or server type is specified in the configuration.
    """
    if not os.path.exists("logs"):
        os.makedirs("logs")
    startup_config = file2dict(args.config_path)
    logger_cfg = startup_config["proxy"]["logger_cfg"].copy()
    logger_cfg["logger_name"] = "main"
    logger = setup_logger(**logger_cfg)
    ensure_mongodb_memory_database(args=args, startup_config=startup_config, logger=logger)
    thread_pool_executor = ThreadPoolExecutor(max_workers=args.max_workers)
    # Initialize proxy
    proxy_cfg = startup_config["proxy"]
    cls_name = proxy_cfg.pop("type")
    if ExecutorRegistry.validate_class(cls_name):
        proxy_cfg["thread_pool_executor"] = thread_pool_executor
    if cls_name == "Proxy":
        proxy = Proxy(**proxy_cfg)
    else:
        raise ValueError(f"Invalid proxy type: {cls_name}")
    # Initialize server
    server_cfg = startup_config["server"]
    cls_name = server_cfg.pop("type")
    if ExecutorRegistry.validate_class(cls_name):
        server_cfg["thread_pool_executor"] = thread_pool_executor
    server_cfg["proxy"] = proxy
    startup_event_listener = server_cfg.pop("startup_event_listener", list())
    startup_event_listener.append(lambda: asyncio.create_task(proxy.maintain_loop()))
    server_cfg["startup_event_listener"] = startup_event_listener
    if cls_name == "OrchestratorProxyServer":
        server = OrchestratorProxyServer(**server_cfg)
    else:
        raise ValueError(f"Invalid server type: {cls_name}")
    server.run()
    return 0


def ensure_mongodb_memory_database(args: argparse.Namespace, startup_config: dict, logger: logging.Logger) -> None:
    """Ensure MongoDB memory database is properly configured and accessible.

    This function checks if the MongoDB memory database connection is working
    properly. If the connection fails, it attempts to set up the database and
    user using admin credentials. The function handles the complete database
    initialization process including connection testing, database creation,
    and user setup with appropriate permissions.

    Args:
        args (argparse.Namespace):
            Parsed command line arguments containing MongoDB admin credentials.
        startup_config (dict):
            Startup configuration dictionary containing database memory settings.
        logger (logging.Logger):
            Logger instance for recording database setup operations.
    """
    db_memory_cfg = startup_config["proxy"]["db_memory_cfg"]
    if db_memory_cfg.get("type") != "MongoDBMemoryClient":
        return
    mongodb_host = db_memory_cfg["host"]
    mongodb_port = db_memory_cfg["port"]
    mongodb_username = db_memory_cfg["username"]
    mongodb_password = db_memory_cfg["password"]
    mongodb_database = db_memory_cfg["database"]
    mongodb_auth_database = db_memory_cfg["auth_database"]
    memory_ok = test_mongodb(
        mongodb_host=mongodb_host,
        mongodb_username=mongodb_username,
        mongodb_password=mongodb_password,
        mongodb_port=mongodb_port,
        mongodb_database=mongodb_database,
        mongodb_auth_database=mongodb_auth_database,
        logger=logger,
    )
    if memory_ok:
        logger.info("MongoDB memory database connection test passed.")
        return
    else:
        logger.info("Cannot connect to MongoDB memory database, trying to setup...")
    if args.mongodb_admin_username is None:
        admin_usename = os.environ.get("MONGODB_ADMIN_USERNAME", "admin")
    else:
        admin_usename = args.mongodb_admin_username
    if args.mongodb_admin_password is None:
        admin_password = os.environ.get("MONGODB_ADMIN_PASSWORD", "")
    else:
        admin_password = args.mongodb_admin_password
    setup_mongodb(
        mongodb_host=mongodb_host,
        mongodb_username=mongodb_username,
        mongodb_password=mongodb_password,
        mongodb_port=mongodb_port,
        mongodb_database=mongodb_database,
        mongodb_auth_database=mongodb_auth_database,
        admin_usename=admin_usename,
        admin_password=admin_password,
        logger=logger,
    )
    logger.info("MongoDB memory database setup completed.")
    return


def test_mongodb(
    mongodb_host: str,
    mongodb_username: str,
    mongodb_password: str,
    mongodb_port: int,
    mongodb_database: str,
    mongodb_auth_database: str,
    logger: logging.Logger,
) -> bool:
    """Test MongoDB connection and permissions for the target user.

    This function tests if the target user can connect to MongoDB using the
    specified auth database and has read/write permissions on the target database.

    Args:
        mongodb_host (str):
            MongoDB server hostname.
        mongodb_username (str):
            Target username to test.
        mongodb_password (str):
            Password for the target user.
        mongodb_port (int):
            MongoDB server port.
        mongodb_database (str):
            Target database name to test permissions on.
        mongodb_auth_database (str):
            Authentication database name.
        logger (logging.Logger):
            Logger instance for recording test results.

    Returns:
        bool:
            True if connection and permissions test passes, False otherwise.
    """
    try:
        # Test connection with target user credentials
        with MongoClient(
            host=mongodb_host,
            port=mongodb_port,
            username=mongodb_username,
            password=mongodb_password,
            authSource=mongodb_auth_database,
        ) as client:
            # Test read permission by listing collections
            db = client[mongodb_database]
            collections = db.list_collection_names()
            logger.debug(f"Successfully listed collections in {mongodb_database}: {collections}")

            # Test write permission by creating a temporary collection
            test_collection = db["_test_permissions"]
            test_collection.insert_one({"test": "permission_check", "timestamp": "temp"})
            test_collection.drop()
            logger.debug(f"Successfully tested write permissions on {mongodb_database}")

            return True

    except Exception as e:
        logger.warning(f"MongoDB connection/permission test failed: {e!s}")
        return False


def setup_mongodb(
    mongodb_host: str,
    mongodb_username: str,
    mongodb_password: str,
    mongodb_port: int,
    mongodb_database: str,
    mongodb_auth_database: str,
    admin_usename: str,
    admin_password: str,
    logger: logging.Logger,
) -> None:
    """Setup MongoDB database and user.

    This function checks if the target database exists and creates it if not.
    It also checks if the target user exists in the auth database and creates
    the user with full permissions on the target database if not.

    Args:
        mongodb_host (str):
            MongoDB server hostname.
        mongodb_username (str):
            Target username to create or verify.
        mongodb_password (str):
            Password for the target user.
        mongodb_port (int):
            MongoDB server port.
        mongodb_database (str):
            Target database name to create or verify.
        mongodb_auth_database (str):
            Authentication database name.
        admin_usename (str):
            Admin username for MongoDB connection.
        admin_password (str):
            Admin password for MongoDB connection.
        logger (logging.Logger):
            Logger instance for recording operations.
    """
    with MongoClient(host=mongodb_host, port=mongodb_port, username=admin_usename, password=admin_password) as client:
        # Check if target database exists
        database_list = client.list_database_names()
        if mongodb_database not in database_list:
            # Create database by creating a collection and then dropping it
            db = client[mongodb_database]
            db.create_collection("_temp_setup")
            db.drop_collection("_temp_setup")
            logger.info(f"Created database: {mongodb_database}")
        else:
            logger.info(f"Database {mongodb_database} already exists, skipping creation")

        # Check if target user exists in auth database
        auth_db = client[mongodb_auth_database]
        try:
            # Try to find the user in the system.users collection
            user_exists = auth_db.command("usersInfo", mongodb_username)
            if user_exists["users"]:
                logger.info(f"User {mongodb_username} already exists " + f"in auth database {mongodb_auth_database}")
            else:
                raise Exception("User not found")
        except Exception:
            # User doesn't exist, create it
            try:
                auth_db.command(
                    "createUser",
                    mongodb_username,
                    pwd=mongodb_password,
                    roles=[{"role": "readWrite", "db": mongodb_database}],
                )
                logger.info(
                    f"Created user {mongodb_username} with " + f"readWrite permissions on database {mongodb_database}"
                )
            except Exception as e:
                logger.error(f"Failed to create user {mongodb_username}: {e!s}")
                raise


def setup_parser():
    """Set up command line argument parser for the orchestrator proxy server.

    Creates and configures an argument parser with the necessary command line
    options for starting the orchestrator proxy server. This includes
    configuration file path and thread pool worker settings.

    Returns:
        argparse.Namespace:
            Parsed command line arguments.
    """
    parser = argparse.ArgumentParser("Start the orchestrator proxy server.")
    # Server configuration arguments
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration file containing server and proxy settings.",
        default="configs/local.py",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        help="Maximum number of worker threads for the thread pool executor.",
        default=4,
    )
    parser.add_argument(
        "--mongodb_admin_username",
        type=str,
        required=False,
        default=None,
        help="MongoDB admin username for database setup operations. "
        + "If not provided, will use MONGODB_ADMIN_USERNAME environment variable "
        + 'or default to "admin". This is used for creating databases and users '
        + "when the target user does not have sufficient permissions.",
    )
    parser.add_argument(
        "--mongodb_admin_password",
        type=str,
        required=False,
        default=None,
        help="MongoDB admin password for database setup operations. "
        + "If not provided, will use MONGODB_ADMIN_PASSWORD environment variable "
        + "or default to empty string. This is used for creating databases and users "
        + "when the target user does not have sufficient permissions.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_parser()
    ret_val = main(args)
    sys.exit(ret_val)
