# yapf: disable
import time
from typing import Any, Dict, Union

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..utils.super import Super
from .exceptions import register_error_handlers

# yapf: enable


class BaseFastAPIService(Super):
    """Base class for FastAPI services with common functionality.

    Provides a foundation for building FastAPI-based services with built-in
    features like CORS support, request logging, health checks, and error
    handling. Subclasses can extend this base class to add specific API routes
    and functionality.
    """

    def __init__(
        self,
        name: str,
        enable_cors: bool = False,
        host: str = "0.0.0.0",
        port: int = 80,
        startup_event_listener: Union[None, list] = None,
        shutdown_event_listener: Union[None, list] = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ) -> None:
        """Initialize the base FastAPI service.

        Args:
            name (str):
                Service name for identification and logging.
            enable_cors (bool, optional):
                Whether to enable Cross-Origin Resource Sharing (CORS).
                Defaults to False.
            host (str, optional):
                Host address to bind the service to.
                Defaults to "0.0.0.0".
            port (int, optional):
                Port number to bind the service to.
                Defaults to 80.
            startup_event_listener (Union[None, list], optional):
                List of event listeners to execute on service startup.
                Defaults to None.
            shutdown_event_listener (Union[None, list], optional):
                List of event listeners to execute on service shutdown.
                Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary.
                Defaults to None.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.name = name
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.enable_cors = enable_cors
        if self.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        if startup_event_listener is not None:
            for listener in startup_event_listener:
                self.app.add_event_handler("startup", listener)
        if shutdown_event_listener is not None:
            for listener in shutdown_event_listener:
                self.app.add_event_handler("shutdown", listener)

        register_error_handlers(self.app)
        self.app.middleware("http")(self._print_request_id)

    async def _print_request_id(self, request: Request, call_next):
        """Middleware to log request information and track request IDs.

        Logs request details including method, path, and processing time.
        Tracks requests using x-request-id header for debugging and monitoring.

        Args:
            request (Request):
                FastAPI request object.
            call_next:
                Next middleware or route handler in the chain.

        Returns:
            Response:
                The response from the next handler in the chain.
        """
        # Print x-request-id for tracking request
        x_request_id = request.headers.get("x-request-id")
        start = time.time()
        response = await call_next(request)
        if x_request_id:
            self.logger.info(
                f"x-request-id:{x_request_id} "
                f"{request.method} {request.url.path}, "
                f"cost: {time.time() - start:.4f}s"
            )

        return response

    def _add_api_routes(self, router: APIRouter) -> None:
        """Add common API routes to the router.

        Adds standard routes that are common to all services, such as
        health check endpoints. Subclasses can override this method to
        add additional routes.

        Args:
            router (APIRouter):
                FastAPI router to add routes to.
        """
        router.add_api_route("/health/", endpoint=self.health, status_code=200)

    def run(self) -> None:
        """Run the FastAPI service according to configuration.

        Starts the FastAPI application with the configured host and port.
        Includes all registered routes and middleware.
        """
        router = APIRouter()
        self._add_api_routes(router)
        self.app.include_router(router)
        uvicorn.run(self.app, host=self.host, port=self.port)

    def health(self) -> JSONResponse:
        """Health check endpoint for service monitoring.

        Returns a simple "OK" response to indicate the service is running
        and healthy. Used by load balancers and monitoring systems.

        Returns:
            JSONResponse:
                JSON response with "OK" content and 200 status code.
        """
        resp = JSONResponse(content="OK")
        return resp
