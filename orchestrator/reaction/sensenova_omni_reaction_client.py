import asyncio
import json
import re
import ssl
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

import jwt
import websockets

from ..data_structures.reaction import ReactionDelta
from ..utils.exception import MissingAPIKeyException
from ..utils.executor_registry import ExecutorRegistry
from .reaction_adapter import ReactionAdapter


class SenseNovaOmniReactionClient(ReactionAdapter):
    """SenseNova Omni reaction client using SenseNova Omni API for emotion and
    motion analysis.

    This client uses SenseNova Omni multimodal AI platform via WebSocket to
    analyze conversation context and generate appropriate emotional reactions
    and motion suggestions for animated agents.
    """

    # Class-level SSL context cache for WebSocket connections
    _ssl_context_cache: Optional[ssl.SSLContext] = None
    ExecutorRegistry.register_class("SenseNovaOmniReactionClient")

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        wss_url: str = "wss://api-gai.sensetime.com/agent-5o/duplex/ws2",
        proxy_url: Union[None, str] = None,
        timeout: float = 10.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the SenseNova Omni reaction client.

        Args:
            name (str):
                The name of the reaction client.
            motion_keywords (Union[str, list[str], None]):
                The motion keywords.
            wss_url (str, optional):
                The WebSocket URL of the SenseNova Omni API.
                Defaults to "wss://api-gai.sensetime.com/agent-5o/duplex/ws2".
            proxy_url (Union[None, str], optional):
                The proxy URL for the SenseNova Omni API.
                Defaults to None, use no proxy.
            timeout (float, optional):
                The timeout for the SenseNova Omni API requests.
                Defaults to 10.0.
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

        Establishes a new session with the SenseNova Omni API by sending a CreateSession
        request and waiting for the response containing the session ID.

        Args:
            ws:
                WebSocket connection object for communication with the API.

        Returns:
            str:
                Session ID returned from the API for subsequent requests.

        Raises:
            Exception:
                If the session creation fails or response parsing fails.
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

        Requests Agora channel information from the SenseNova Omni API, which is required
        for establishing real-time communication channels.

        Args:
            ws:
                WebSocket connection object for communication with the API.

        Returns:
            tuple:
                Tuple containing (app_id, channel_id, server_uid) for Agora
                channel configuration.

        Raises:
            Exception:
                If the channel info request fails or response parsing fails.
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

        Requests an Agora authentication token from the SenseNova Omni API, which is
        required for joining real-time communication channels.

        Args:
            ws:
                WebSocket connection object for communication with the API.

        Returns:
            tuple:
                Tuple containing (token, user_id) for Agora authentication.

        Raises:
            Exception:
                If the token request fails or response parsing fails.
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

        Listens for incoming messages from the SenseNova Omni API WebSocket connection
        and processes them until a complete response is received. Handles
        various message types and extracts the final assistant output.

        Args:
            ws:
                WebSocket connection object for receiving messages.

        Returns:
            str:
                Assistant output text from the API response, or empty string
                if no valid response is received.

        Raises:
            websockets.exceptions.ConnectionClosedError:
                If the WebSocket connection is closed unexpectedly.
            Exception:
                If message processing fails.
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

    async def get_reaction_delta(
        self,
        request_id: str,
        prompt: str,
        text: str,
        tag: str,
        user_input: str,
        current_emotion: Dict[str, int] | None = None,
        current_relationship: Dict[str, Any] | None = None,
        response_format: Optional[Dict[str, Any]] = None,
        tag_prompt: Optional[str] = None,
    ) -> ReactionDelta:
        """Get the reaction delta according to user's text input using
        SenseNova Omni LLM.

        Analyzes the conversation context including user input, agent response,
        current emotion state, and relationship status to generate appropriate
        emotional reactions and motion suggestions using SenseNova Omni API.

        Args:
            request_id (str):
                The request id.
            prompt (str):
                The system prompt for reaction analysis.
            text (str):
                The agent response text to analyze.
            tag (str):
                The tag associated with the response.
            user_input (str):
                The user input text that triggered the agent response.
            current_emotion (Dict[str, int] | None, optional):
                Current emotion state with emotion names as keys and values as integers.
                Defaults to None.
            current_relationship (Dict[str, Any] | None, optional):
                Current relationship state between user and agent.
                Defaults to None.
            response_format (Optional[Dict[str, Any]], optional):
                JSON schema format for structured response.
                Defaults to None.
            tag_prompt (Optional[str], optional):
                Additional prompt specific to the tag.
                Defaults to None.

        Returns:
            ReactionDelta:
                Reaction delta containing emotion changes, relationship changes,
                and motion suggestions.

        Raises:
            Exception:
                If the SenseNova Omni API request fails, WebSocket connection fails,
                or response parsing fails.
        """
        ws = None
        iss = self.input_buffer[request_id]["api_keys"].get("sensenova_ak", "")
        secret = self.input_buffer[request_id]["api_keys"].get("sensenova_sk", "")
        if not iss or not secret:
            msg = "SenseNova API key or secret key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        try:
            user_message_parts = [f"<user_input>: {user_input}"]
            user_message_parts.append(f"<agent_response>: {text}")

            if current_relationship:
                relationship_str = ", ".join([f"{k}: {v}" for k, v in current_relationship.items()])
                user_message_parts.append(f"<current_relationship>: {relationship_str}")

            if current_emotion:
                emotion_str = ", ".join([f"{k}: {v}" for k, v in current_emotion.items()])
                user_message_parts.append(f"<current_emotion>: {emotion_str}")

            if tag:
                user_message_parts.append(f"<tag>: {tag}")

            user_message = "\n".join(user_message_parts)

            start_time = time.time()
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
                "text": user_message,
            }
            send_message = await loop.run_in_executor(self.executor, _json_dumps_not_ensure_ascii, send_message)
            await ws.send(send_message)

            response = await self._ws_message_listener(ws)

            # Use regex to parse response for improved robustness

            def extract_value(pattern, default=0):
                match = re.search(pattern, response)
                if match:
                    try:
                        return int(match.group(1))
                    except ValueError:
                        return default
                return default

            def extract_text(pattern, default=""):
                match = re.search(pattern, response)
                if match:
                    return match.group(1)
                return default

            happiness_delta = extract_value(r"<happiness_delta>(-?\d+)</happiness_delta>")
            sadness_delta = extract_value(r"<sadness_delta>(-?\d+)</sadness_delta>")
            fear_delta = extract_value(r"<fear_delta>(-?\d+)</fear_delta>")
            anger_delta = extract_value(r"<anger_delta>(-?\d+)</anger_delta>")
            disgust_delta = extract_value(r"<disgust_delta>(-?\d+)</disgust_delta>")
            surprise_delta = extract_value(r"<surprise_delta>(-?\d+)</surprise_delta>")
            shyness_delta = extract_value(r"<shyness_delta>(-?\d+)</shyness_delta>")
            relationship_delta = extract_value(r"<relationship_delta>(-?\d+)</relationship_delta>")
            speech_keywords = extract_text(r"<speech_keywords>(.*?)</speech_keywords>")
            motion_keywords = extract_text(r"<motion_keywords>(.*?)</motion_keywords>")

            # Build data structure matching ReactionDelta expected format
            emotion_delta = {
                "happiness_delta": happiness_delta,
                "sadness_delta": sadness_delta,
                "fear_delta": fear_delta,
                "anger_delta": anger_delta,
                "disgust_delta": disgust_delta,
                "surprise_delta": surprise_delta,
                "shyness_delta": shyness_delta,
            }

            # Build motion array
            motion = []
            if speech_keywords and motion_keywords:
                motion.append({"speech_keywords": speech_keywords, "motion_keywords": motion_keywords})

            response_delta = {
                "emotion_delta": emotion_delta,
                "relationship_delta": relationship_delta,
                "motion": motion,
                "speech_text": text,
            }

            self.logger.debug(
                f"SenseNova Omni spent {time.time() - start_time} seconds to get reaction delta: {response_delta}"
            )
            return ReactionDelta(**response_delta)

        except Exception as e:
            self.logger.error(f"Reaction error: {e}")
            raise e


def _json_dumps_not_ensure_ascii(obj: Any, **kwargs: Any) -> str:
    """JSON dumps without ensure_ascii."""
    kwargs = kwargs.copy()
    if "ensure_ascii" in kwargs:
        kwargs.pop("ensure_ascii")
    return json.dumps(obj, ensure_ascii=False, **kwargs)
