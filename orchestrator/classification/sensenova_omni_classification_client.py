import asyncio
import json
import ssl
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

import jwt
import websockets

from ..data_structures.classification import ClassificationType
from ..utils.exception import MissingAPIKeyException
from ..utils.executor_registry import ExecutorRegistry
from .classification_adapter import ClassificationAdapter


class SenseNovaOmniClassificationClient(ClassificationAdapter):
    """Classification client for SenseNova Omni API using WebSocket connection.

    This client provides text classification functionality through SenseNova's
    Omni API using WebSocket communication. It supports motion keyword-based
    classification and handles authentication via JWT tokens.
    """

    # Class-level SSL context cache
    _ssl_context_cache: Optional[ssl.SSLContext] = None
    ExecutorRegistry.register_class("SenseNovaOmniClassificationClient")

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        wss_url: str = "wss://api-gai.sensetime.com/agent-5o/duplex/ws2",
        proxy_url: Union[None, str] = None,
        timeout: float = 2.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the SenseNova Omni classification client.

        Args:
            name (str):
                The name of the classification client.
            motion_keywords (Union[str, list[str], None]):
                The motion keywords.
            wss_url (str, optional):
                WSS URL for the SenseNova Omni API.
                Defaults to "wss://api-gai.sensetime.com/agent-5o/duplex/ws2".
            proxy_url (Union[None, str], optional):
                The proxy URL for the SenseNova Omni API.
                Defaults to None, use no proxy.
            timeout (float, optional):
                The timeout for the WebSocket connection.
                Defaults to 2.0.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor.
                If None, a new thread pool executor will be created based on
                max_workers. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                The logger configuration. Defaults to None.
        """
        super().__init__(
            name=name,
            motion_keywords=motion_keywords,
            proxy_url=proxy_url,
            logger_cfg=logger_cfg,
        )
        self.wss_url = wss_url
        self.proxy_url = proxy_url
        self.timeout = timeout
        self.executor = (
            thread_pool_executor if thread_pool_executor is not None else ThreadPoolExecutor(max_workers=max_workers)
        )
        self.executor_external = True if thread_pool_executor is not None else False

    def __del__(self) -> None:
        """Destructor, cleanup thread pool executor."""
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    async def _init_llm_client(self, request_id: str) -> None:
        """Initialize the LLM client.

        Args:
            request_id (str):
                The request id.
        """
        pass

    def _gen_token(self, iss: str, secret: str) -> str:
        """Generate a JWT token for the Omni API authentication.

        Args:
            iss (str):
                The API key for the Sensenova API.
            secret (str):
                The API secret key for the Sensenova API.

        Returns:
            str:
                JWT token string for API authentication.
        """
        payload = {"iss": iss, "exp": int(time.time()) + 3600}  # 1 hour expiration
        return jwt.encode(payload, secret, algorithm="HS256")

    async def _ws_create_session(self, ws) -> str:
        """Create a WebSocket session with the SenseNova Omni API.

        Args:
            ws:
                WebSocket connection object.

        Returns:
            str:
                Session ID returned from the API.
        """
        create_session_msg = {"type": "CreateSession", "request_id": uuid.uuid4().hex}
        await ws.send(json.dumps(create_session_msg))
        self.logger.debug("SenseNova sent CreateSession request")

        response = await ws.recv()
        json_res = json.loads(response)
        session_id = json_res.get("session_id")
        self.logger.debug(f"SenseNova created session, ID: {session_id}")
        return session_id

    async def _ws_request_agora_channel_info(self, ws):
        """Request Agora channel information via WebSocket.

        Args:
            ws:
                WebSocket connection object.

        Returns:
            tuple:
                Tuple containing (app_id, channel_id, server_uid).
        """
        request = {"type": "RequestAgoraChannelInfo", "request_id": uuid.uuid4().hex}
        await ws.send(json.dumps(request))
        self.logger.debug("SenseNova sent RequestAgoraChannelInfo request")

        response = await ws.recv()
        json_res = json.loads(response)

        app_id = json_res.get("appid")
        channel_id = json_res.get("channel_id")
        server_uid = json_res.get("server_uid")

        self.logger.debug(f"SenseNova received Agora channel info: AppID={app_id}, ChannelID={channel_id}")
        return app_id, channel_id, server_uid

    async def _ws_request_agora_token(self, ws):
        """Request Agora token via WebSocket.

        Args:
            ws:
                WebSocket connection object.

        Returns:
            tuple:
                Tuple containing (token, user_id).
        """
        request = {"type": "RequestAgoraToken", "duration": 600, "request_id": uuid.uuid4().hex}
        await ws.send(json.dumps(request))
        self.logger.debug("SenseNova sent RequestAgoraToken request")

        response = await ws.recv()
        json_res = json.loads(response)

        token = json_res.get("token")
        user_id = json_res.get("client_uid")

        self.logger.debug(f"SenseNova received Agora token: UserID={user_id}")
        return token, user_id

    async def _ws_message_listener(
        self,
        ws,
    ):
        """WebSocket message listener for receiving responses.

        Args:
            ws:
                WebSocket connection object.

        Returns:
            str:
                Assistant output text from the API response.
        """
        assistant_output = ""
        try:
            while True:
                try:
                    response = await ws.recv()
                    try:
                        json_res = json.loads(response)
                        message_type = json_res.get("type")
                        self.logger.debug(f"{message_type}: {json_res}")
                        if message_type == "ResponseEndTextStream":
                            assistant_output = json_res.get("text")
                            await ws.close()
                            break
                    except json.JSONDecodeError:
                        self.logger.warning(f"SenseNova unable to parse JSON message: {response}")
                except asyncio.TimeoutError:
                    continue  # Continue loop, check status
            return assistant_output
        except websockets.exceptions.ConnectionClosedError:
            self.logger.debug("SenseNova WebSocket connection closed")
            return ""
        except Exception as e:
            self.logger.error(f"SenseNova WebSocket message listener error: {e}")
            return ""

    async def classify(
        self,
        request_id: str,
        prompt: str,
        text: str,
        response_format: Optional[Dict[str, Any]] = None,
        tag_prompt: Optional[str] = None,
    ) -> ClassificationType:
        """Classify the required response type according to user's text input,
        based on LLM.

        Args:
            request_id (str):
                The request id.
            prompt (str):
                The prompt to classify the text.
            text (str):
                The text to classify.
            response_format (Optional[Dict[str, Any]], optional):
                Response format specification. Defaults to None.
            tag_prompt (Optional[str], optional):
                Tag prompt for the LLM. Defaults to None.

        Returns:
            ClassificationType: The classification type.
        """
        iss = self.input_buffer[request_id]["api_keys"].get("sensenova_ak", "")
        secret = self.input_buffer[request_id]["api_keys"].get("sensenova_sk", "")
        if not iss or not secret:
            msg = "SenseNova API key or secret key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        try:
            loop = asyncio.get_running_loop()
            jwt_token = await loop.run_in_executor(self.executor, self._gen_token, iss, secret)
            wss_url_with_token = f"{self.wss_url}?jwt={jwt_token}"
            if self.__class__._ssl_context_cache is None:
                ssl_context = await loop.run_in_executor(self.executor, ssl.create_default_context)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                self.__class__._ssl_context_cache = ssl_context
            else:
                ssl_context = self.__class__._ssl_context_cache
            ws = await websockets.connect(wss_url_with_token, ssl=ssl_context)

            session_id = await self._ws_create_session(ws)

            app_id, channel_id, server_uid = await self._ws_request_agora_channel_info(ws)

            token, user_id = await self._ws_request_agora_token(ws)
            start_serving_msg = {"type": "StartServing"}
            await ws.send(json.dumps(start_serving_msg))

            prompt_msg = {
                "type": "SetSystemPrompt",
                "system_prompt": prompt + "\n" + tag_prompt if tag_prompt else prompt,
            }
            prompt_msg = await loop.run_in_executor(self.executor, _json_dumps_not_ensure_ascii, prompt_msg)
            await ws.send(prompt_msg)

            send_message = {
                "type": "PostMultimodalGenerate",
                "request_id": uuid.uuid4().hex,
                "text": text,
            }
            send_message = await loop.run_in_executor(self.executor, _json_dumps_not_ensure_ascii, send_message)
            await ws.send(send_message)

            response = await self._ws_message_listener(ws)

            if "reject" in response.lower():
                classification_result = "reject"
            elif "leave" in response.lower():
                classification_result = "leave"
            else:
                classification_result = "accept"

            self.logger.debug(f"Classification response: {response}, result: {classification_result}")
            return ClassificationType(classification_result)
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            raise e


def _json_dumps_not_ensure_ascii(obj: Any, **kwargs: Any) -> str:
    """JSON dumps without ensure_ascii."""
    kwargs = kwargs.copy()
    if "ensure_ascii" in kwargs:
        kwargs.pop("ensure_ascii")
    return json.dumps(obj, ensure_ascii=False, **kwargs)
