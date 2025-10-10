import os
import traceback
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.utils import is_body_allowed_for_status_code
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..proxy import AdapterNotFoundError
from ..utils.log import get_logger


class APIErrorMessage(BaseModel):
    """API error message.

    Args:
        message (str): The error message.
        code (int): The error code.
        detail (Optional[dict]): The error detail.
    """

    message: str
    code: int
    detail: Optional[dict] = None


OPENAPI_RESPONSE_422 = {"422": {"description": "Validaion Error", "model": APIErrorMessage}}
OPENAPI_RESPONSE_404 = {"404": {"description": "Not found", "model": APIErrorMessage}}
OPENAPI_RESPONSE_400 = {"400": {"description": "Bad Request", "model": APIErrorMessage}}
OPENAPI_RESPONSE_500 = {"500": {"description": "Internal Server Error", "model": APIErrorMessage}}
OPENAPI_RESPONSE_503 = {"503": {"description": "Service Unavailable", "model": APIErrorMessage}}


async def http_exception_handler(request, exc):
    """HTTP exception handler.

    Args:
        request (Request): The request object.
        exc (HTTPException): The HTTP exception.

    Returns:
        JSONResponse: The JSON response.
    """
    headers = getattr(exc, "headers", None)
    if headers is None:
        headers = {}
    if request.headers is not None:
        headers = {**headers, **request.headers}

    if not is_body_allowed_for_status_code(exc.status_code):
        return Response(status_code=exc.status_code, headers=headers)

    content = APIErrorMessage(
        message=f"HTTP Error {exc.status_code}",
        code=exc.status_code,
        detail={"error": exc.detail},
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=content.model_dump(),
        headers=exc.headers,
    )


async def validation_exception_handler(request: Request, exc):
    """Validation exception handler.

    Args:
        request (Request): The request object.
        exc (RequestValidationError): The validation exception.

    Returns:
        JSONResponse: The JSON response.
    """
    content = APIErrorMessage(
        message="Validaion Error",
        code=422,
        detail={"errors": exc.errors(), "body": exc.body},
    )

    return JSONResponse(
        content=content.model_dump(),
        status_code=422,
        headers=request.headers,
    )


async def adapter_not_found_exception_handler(request: Request, exc: AdapterNotFoundError):
    """Adapter not found exception handler.

    Args:
        request (Request): The request object.
        exc (AdapterNotFoundError): The adapter not found exception.

    Returns:
        JSONResponse: The JSON response with 404 status code.
    """
    content = APIErrorMessage(
        message="Adapter Not Found",
        code=404,
        detail={"error": str(exc)},
    )
    return JSONResponse(
        content=content.model_dump(),
        status_code=404,
        headers=request.headers,
    )


async def exception_handler(request: Request, exc):
    """Exception handler.

    Args:
        request (Request): The request object.
        exc (Exception): The exception.

    Returns:
        JSONResponse: The JSON response.
    """
    content = APIErrorMessage(
        message="Internal Server Error",
        code=500,
        detail={"error": f"{str(exc)}, type: {type(exc)}"},
    )
    file_name = os.path.basename(__file__)
    logger = get_logger(file_name)
    logger.error(traceback.format_exc())
    return JSONResponse(
        content=content.model_dump(),
        status_code=500,
        headers=request.headers,
    )


def register_error_handlers(app: FastAPI):
    """Register error handlers.

    Args:
        app (FastAPI): The FastAPI app.
    """
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(AdapterNotFoundError, adapter_not_found_exception_handler)
    app.add_exception_handler(Exception, exception_handler)
