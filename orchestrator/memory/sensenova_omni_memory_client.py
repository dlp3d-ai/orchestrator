import asyncio
import json
import ssl
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

import jwt
import websockets

from ..io.memory.database_memory_client import DatabaseMemoryClient
from ..utils.executor_registry import ExecutorRegistry
from .memory_adapter import BaseMemoryAdapter


class SenseNovaOmniMemoryClient(BaseMemoryAdapter):
    """SenseNova Omni memory client that implements memory management based on
    SenseNova Omni API.

    This class provides memory management functionality using the SenseNova
    Omni API for LLM calls and memory operations.
    """

    # Class-level SSL context cache
    _ssl_context_cache: Optional[ssl.SSLContext] = None
    ExecutorRegistry.register_class("SenseNovaOmniMemoryClient")

    def __init__(
        self,
        name: str,
        db_client: DatabaseMemoryClient,
        wss_url: str = "wss://api-gai.sensetime.com/agent-5o/duplex/ws2",
        proxy_url: Union[None, str] = None,
        timeout: float = 10.0,
        conversation_char_threshold: int = 10000,
        conversation_char_target: int = 8000,
        short_term_length_threshold: int = 20,
        short_term_target_size: int = 10,
        medium_term_length_threshold: int = 10,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the SenseNova Omni memory client.

        Args:
            name (str):
                Name of the memory client.
            db_client (DatabaseMemoryClient):
                Database client for memory operations.
            wss_url (str, optional):
                WSS URL for the SenseNova Omni API.
            proxy_url (Union[None, str], optional):
                Proxy URL for API requests. Defaults to None.
            timeout (float, optional):
                Request timeout in seconds. Defaults to 10.0.
            conversation_char_threshold (int, optional):
                Character threshold for conversation compression. Defaults to 10000.
            conversation_char_target (int, optional):
                Target character count for conversation compression. Defaults to 8000.
            short_term_length_threshold (int, optional):
                Length threshold for short-term memory compression. Defaults to 20.
            short_term_target_size (int, optional):
                Target size for short-term memory compression. Defaults to 10.
            medium_term_length_threshold (int, optional):
                Length threshold for medium-term memory compression. Defaults to 10.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor.
                If None, a new thread pool executor will be created based on
                max_workers. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
        """
        super().__init__(
            name=name,
            db_client=db_client,
            conversation_char_threshold=conversation_char_threshold,
            conversation_char_target=conversation_char_target,
            short_term_length_threshold=short_term_length_threshold,
            short_term_target_size=short_term_target_size,
            medium_term_length_threshold=medium_term_length_threshold,
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

    def _gen_token(self, iss: str, secret: str) -> str:
        """Generate a JWT token for the SenseNova Omni API authentication.

        Args:
            iss (str):
                SenseNova API iss for authentication.
            secret (str):
                SenseNova API secret for authentication.

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
                        self.logger.warning(f"SenseNova failed to parse JSON message: {response}")
                except asyncio.TimeoutError:
                    continue  # Continue loop, check status
            return assistant_output
        except websockets.exceptions.ConnectionClosedError:
            self.logger.debug(f"SenseNova WebSocket connection closed")
            return ""
        except Exception as e:
            self.logger.error(f"SenseNova WebSocket message listener error: {e}")
            return ""

    async def call_llm(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        tag_prompt: Optional[str] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """Call SenseNova Omni LLM for text generation.

        Args:
            system_prompt (str):
                System prompt for the LLM.
            user_input (str):
                User input for the LLM.
            max_tokens (int):
                Maximum number of tokens to generate.
            response_format (Optional[Dict[str, Any]], optional):
                Response format specification. Defaults to None.
            tag_prompt (Optional[str], optional):
                Tag prompt for the LLM. Defaults to None.
            api_keys (Optional[Dict[str, Any]], optional):
                API keys for the LLM. Defaults to None.
            model_override (Optional[str], optional):
                Model name override. Defaults to None.

        Returns:
            str:
                Generated text content from the SenseNova Omni LLM.
        """
        try:
            if not api_keys:
                raise ValueError("api_keys is required for SenseNova LLM calls")
            iss = api_keys.get("sensenova_ak", "")
            secret = api_keys.get("sensenova_sk", "")
            if not iss or not secret:
                raise ValueError("sensenova api keys not found in api_keys")

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
                "system_prompt": system_prompt + "\n" + tag_prompt if tag_prompt else system_prompt,
            }
            prompt_msg = await loop.run_in_executor(self.executor, _json_dumps_not_ensure_ascii, prompt_msg)
            await ws.send(prompt_msg)

            send_message = {
                "type": "PostMultimodalGenerate",
                "request_id": uuid.uuid4().hex,
                "text": user_input,
            }
            send_message = await loop.run_in_executor(self.executor, _json_dumps_not_ensure_ascii, send_message)
            await ws.send(send_message)

            response = await self._ws_message_listener(ws)
            output = response.split("<output>")[1].split("</output>")[0]
            return output
        except Exception as e:
            self.logger.error(f"SenseNova LLM call failed: {e}")
            raise e


def _json_dumps_not_ensure_ascii(obj: Any, **kwargs: Any) -> str:
    """JSON dumps without ensure_ascii."""
    kwargs = kwargs.copy()
    if "ensure_ascii" in kwargs:
        kwargs.pop("ensure_ascii")
    return json.dumps(obj, ensure_ascii=False, **kwargs)
