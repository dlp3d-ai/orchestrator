import argparse
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from orchestrator.proxy import Proxy
from orchestrator.service.server import OrchestratorProxyServer
from orchestrator.utils.config import file2dict
from orchestrator.utils.executor_registry import ExecutorRegistry


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_parser()
    ret_val = main(args)
    sys.exit(ret_val)
