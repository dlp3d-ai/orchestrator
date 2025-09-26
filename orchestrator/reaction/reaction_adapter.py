import asyncio
import re
import time
import traceback
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import httpx

from ..data_structures.conversation import ClassifiedTextChunkBody, ClassifiedTextChunkEnd, ClassifiedTextChunkStart
from ..data_structures.process_flow import DAGStatus
from ..data_structures.reaction import (
    Emotion,
    EmotionDelta,
    Reaction,
    ReactionChunkBody,
    ReactionChunkEnd,
    ReactionChunkStart,
    ReactionDelta,
    ReactionLLM,
    Relationship,
)
from ..utils.log import setup_logger
from ..utils.sentence_splitter import SentenceSplitter
from ..utils.streamable import ChunkWithoutStartError, Streamable
from .prompts import REACTION_PROMPT_EN, REACTION_PROMPT_ZH, RESPONSE_FORMAT, TAG_PROMPT


class ReactionAdapter(Streamable):
    """Generate motion or emotion reaction according to actor's text output,
    based on LLM.

    This adapter processes classified text chunks and generates appropriate
    emotional and motion reactions using large language models. It maintains
    conversation memory, applies emotion decay, and manages relationship
    dynamics between characters.
    """

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None] = None,
        proxy_url: Union[None, str] = None,
        queue_size: int = 200,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        emotion_decay_rate: float = 0.1,
        tts_char_duration_zh: float = 0.2,
        tts_char_duration_en: float = 0.05,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the reaction adapter.

        Args:
            name (str):
                The name of the reaction adapter.
            motion_keywords (Union[str, list[str], None]):
                The motion keywords.
                Defaults to None.
            proxy_url (Union[None, str], optional):
                The proxy URL for the reaction API.
                Defaults to None.
            queue_size (int, optional):
                The size of the processing queue.
                Defaults to 200.
            sleep_time (float, optional):
                Sleep time between processing cycles.
                Defaults to 0.01.
            clean_interval (float, optional):
                The interval to clean expired requests.
                Defaults to 10.0.
            expire_time (float, optional):
                The time to expire requests.
                Defaults to 120.0.
            emotion_decay_rate (float, optional):
                The emotion decay rate towards neutral.
                Defaults to 0.1.
            tts_char_duration_zh (float, optional):
                Chinese TTS character duration in seconds per character.
                Defaults to 0.2.
            tts_char_duration_en (float, optional):
                English TTS character duration in seconds per character.
                Defaults to 0.05.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
        """
        Streamable.__init__(
            self,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            logger_cfg=logger_cfg,
        )
        self.name = name
        self.proxy_url = proxy_url
        self.emotion_decay_rate = emotion_decay_rate
        self.tts_char_duration_zh = tts_char_duration_zh
        self.tts_char_duration_en = tts_char_duration_en
        self.logger_cfg["logger_name"] = name
        self.logger = setup_logger(**self.logger_cfg)
        self.sentence_splitter = SentenceSplitter(logger=self.logger)

        if isinstance(motion_keywords, str):
            response = httpx.get(motion_keywords, verify=False)
            if response.status_code == 200:
                motion_kws = response.json()["motion_keywords"]
        elif isinstance(motion_keywords, list):
            motion_kws = motion_keywords
        elif motion_keywords is None:
            msg = "Motion keywords is None, using empty list."
            self.logger.warning(msg)
            motion_kws = []
        else:
            msg = f"Invalid motion keywords type: {type(motion_keywords)}"
            self.logger.error(msg)
            raise TypeError(msg)
        self.motion_kws: List[str] = motion_kws
        self.logger.info(f"Loaded {len(self.motion_kws)} motion keywords records")

    @abstractmethod
    async def _init_llm_client(self, request_id: str) -> None:
        """Initialize the LLM client.

        Args:
            request_id (str):
                The request id.
        """
        raise NotImplementedError

    @abstractmethod
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
        """Get the reaction delta according to user's text input, based on LLM.

        This method should be implemented by subclasses to call their specific
        LLM API and return the reaction delta containing emotion and relationship
        changes.

        Args:
            request_id (str):
                The request id.
            prompt (str):
                The prompt template for the LLM.
            text (str):
                The text content to analyze.
            tag (str):
                Additional tag information.
            user_input (str):
                The original user input.
            current_emotion (Dict[str, int] | None, optional):
                Current emotion state. Defaults to None.
            current_relationship (Dict[str, Any] | None, optional):
                Current relationship state. Defaults to None.
            response_format (Optional[Dict[str, Any]], optional):
                The response format. Defaults to None.
            tag_prompt (Optional[str], optional):
                The tag prompt. Defaults to None.

        Returns:
            ReactionDelta:
                The reaction delta containing emotion and relationship changes.
        """
        raise NotImplementedError

    async def filter_text(self, text: str) -> tuple[str, str]:
        """Filter the speech text and return filtered text and removed content.

        Removes various types of tags and parentheses from the input text,
        including XML tags, Chinese parentheses, English parentheses, and
        square brackets.

        Args:
            text (str):
                The input text to filter.

        Returns:
            tuple[str, str]:
                A tuple containing (filtered_text, removed_content).
        """

        removed = []

        # Remove all <tag>...</tag> tags
        tag_matches = re.findall(r"<[^<>]+?>.*?</[^<>]+?>", text, flags=re.DOTALL)
        removed.extend(tag_matches)
        text = re.sub(r"<[^<>]+?>.*?</[^<>]+?>", "", text, flags=re.DOTALL)

        # Remove Chinese parentheses content
        cn_paren_matches = re.findall(r"（.*?）", text)
        removed.extend(cn_paren_matches)
        text = re.sub(r"（.*?）", "", text)

        # Remove English parentheses
        en_paren_matches = re.findall(r"\(.*?\)", text)
        removed.extend(en_paren_matches)
        text = re.sub(r"\(.*?\)", "", text)

        # Remove English square brackets
        square_matches = re.findall(r"\[.*?\]", text)
        removed.extend(square_matches)
        text = re.sub(r"\[.*?\]", "", text)

        # Join removed content into a single string
        removed_text = " ".join(removed)

        return text, removed_text

    async def update_reaction(
        self,
        reaction_delta: ReactionDelta,
        current_emotion: Dict[str, int],
        current_relationship: Dict[str, Any],
        acquaintance_threshold: int,
        friend_threshold: int,
        situationship_threshold: int,
        lover_threshold: int,
    ):
        """Update emotion and relationship based on reaction delta.

        Args:
            reaction_delta (ReactionDelta):
                The reaction delta containing changes to apply.
            current_emotion (Dict[str, int]):
                Current emotion state.
            current_relationship (Dict[str, Any]):
                Current relationship state.
            acquaintance_threshold (int):
                Threshold for acquaintance stage.
            friend_threshold (int):
                Threshold for friend stage.
            situationship_threshold (int):
                Threshold for situationship stage.
            lover_threshold (int):
                Threshold for lover stage.

        Returns:
            tuple[Emotion, Relationship]:
                Updated emotion and relationship objects.
        """
        emotion_delta = reaction_delta.emotion_delta
        relationship_delta = reaction_delta.relationship_delta
        new_emotion = self.apply_emotion_delta(Emotion(**current_emotion), emotion_delta)
        new_relationship = self.apply_relationship_delta(
            Relationship(**current_relationship),
            relationship_delta,
            acquaintance_threshold,
            friend_threshold,
            situationship_threshold,
            lover_threshold,
        )
        return new_emotion, new_relationship

    async def get_reaction_llm(
        self,
        request_id: str,
        prompt: str,
        text: str,
        tag: str,
        user_input: str,
        current_emotion: Dict[str, int],
        current_relationship: Dict[str, Any],
        tts_char_duration: float = 0.2,
        timeout_seconds: Union[float, None] = None,
        character_id: Union[str, None] = None,
        memory_db_client: Union[None, Any] = None,
    ) -> ReactionLLM:
        """Get reaction from LLM with timeout handling and memory updates.

        Processes text through LLM to generate reaction, with timeout
        protection and background memory updates for character state.

        Args:
            request_id (str):
                The request id.
            prompt (str):
                The prompt template for the LLM.
            text (str):
                The text content to analyze.
            tag (str):
                Additional tag information.
            user_input (str):
                The original user input.
            current_emotion (Dict[str, int]):
                Current emotion state.
            current_relationship (Dict[str, Any]):
                Current relationship state.
            tts_char_duration (float, optional):
                TTS character duration for timeout calculation.
                Defaults to 0.2.
            timeout_seconds (Union[float, None], optional):
                Override timeout duration. Defaults to None.
            character_id (Union[str, None], optional):
                Character ID for memory updates. Defaults to None.
            memory_db_client (Union[None, Any], optional):
                The memory database client to update. Defaults to None.

        Returns:
            ReactionLLM:
                The generated reaction from LLM.
        """
        if timeout_seconds is None:
            timeout_duration = tts_char_duration * len(text)
        else:
            timeout_duration = timeout_seconds

        # Explicitly start task but do not wait for completion
        task = asyncio.create_task(
            self.get_reaction_delta(
                request_id,
                prompt,
                text,
                tag,
                user_input,
                current_emotion,
                current_relationship,
                RESPONSE_FORMAT,
                TAG_PROMPT,
            )
        )

        try:
            # Wait for specified time, if timeout do not cancel task
            reaction_delta = await asyncio.wait_for(asyncio.shield(task), timeout=timeout_duration)

            # Save threshold values before they might be lost from input_buffer
            acquaintance_threshold = self.input_buffer[request_id]["acquaintance_threshold"]
            friend_threshold = self.input_buffer[request_id]["friend_threshold"]
            situationship_threshold = self.input_buffer[request_id]["situationship_threshold"]
            lover_threshold = self.input_buffer[request_id]["lover_threshold"]

            # Normal completion logic
            emotion, relationship = await self.update_reaction(
                reaction_delta,
                current_emotion,
                current_relationship,
                acquaintance_threshold,
                friend_threshold,
                situationship_threshold,
                lover_threshold,
            )
            reaction_llm = ReactionLLM(
                speech_text=text,
                emotion=emotion,
                relationship=relationship,
                motion=reaction_delta.motion,
            )

            if character_id:
                processed_emotion, processed_relationship = self.apply_memory_processing(
                    reaction_llm.emotion, reaction_llm.relationship
                )
                # Update emotion and relationship
                asyncio.create_task(
                    self._update_emotion_and_relationship(
                        character_id, processed_emotion, processed_relationship, memory_db_client
                    )
                )
                msg = f"Reaction delta updated memory: emotion={processed_emotion.model_dump()}, "
                msg += f"relationship={processed_relationship.model_dump()}"
                self.logger.debug(msg)
            return reaction_llm

        except asyncio.TimeoutError:
            self.logger.warning(
                f"Reaction delta timeout after {timeout_duration:.2f}s, return empty but continue in background"
            )

            # Return default reaction
            empty_reaction_llm = ReactionLLM(
                speech_text=text,
                emotion=Emotion(**current_emotion),
                relationship=Relationship(**current_relationship),
                motion=[],
            )

            # Continue waiting for task, update memory when completed
            async def background_update():
                try:
                    # Save threshold values before they might be lost from input_buffer
                    acquaintance_threshold = self.input_buffer[request_id]["acquaintance_threshold"]
                    friend_threshold = self.input_buffer[request_id]["friend_threshold"]
                    situationship_threshold = self.input_buffer[request_id]["situationship_threshold"]
                    lover_threshold = self.input_buffer[request_id]["lover_threshold"]

                    reaction_delta = await task  # Do not restart, wait for original task
                    emotion, relationship = await self.update_reaction(
                        reaction_delta,
                        current_emotion,
                        current_relationship,
                        acquaintance_threshold,
                        friend_threshold,
                        situationship_threshold,
                        lover_threshold,
                    )
                    if character_id:
                        processed_emotion, processed_relationship = self.apply_memory_processing(emotion, relationship)
                        # Update emotion and relationship
                        asyncio.create_task(
                            self._update_emotion_and_relationship(
                                character_id, processed_emotion, processed_relationship, memory_db_client
                            )
                        )
                        msg = f"Background reaction delta updated memory: emotion={processed_emotion.model_dump()}, "
                        msg += f"relationship={processed_relationship.model_dump()}"
                        self.logger.debug(msg)
                except Exception as e:
                    self.logger.error(f"Background reaction task failed: {e}")

            asyncio.create_task(background_update())
            return empty_reaction_llm

    def apply_emotion_delta(self, emotion: Emotion, emotion_delta: EmotionDelta) -> Emotion:
        """Apply emotion delta with normalization to sum to 100.

        Applies emotion changes and ensures all emotion values are non-negative
        and sum to 100 through normalization.

        Args:
            emotion (Emotion):
                Current emotion state.
            emotion_delta (EmotionDelta):
                Emotion changes to apply.

        Returns:
            Emotion:
                Updated emotion with normalized values.
        """
        new_emotion = emotion.model_copy()
        new_emotion.Happiness += emotion_delta.happiness_delta
        new_emotion.Sadness += emotion_delta.sadness_delta
        new_emotion.Fear += emotion_delta.fear_delta
        new_emotion.Anger += emotion_delta.anger_delta
        new_emotion.Disgust += emotion_delta.disgust_delta
        new_emotion.Surprise += emotion_delta.surprise_delta
        new_emotion.Shyness += emotion_delta.shyness_delta

        # Ensure all emotion values are non-negative
        emotion_dict = new_emotion.model_dump()
        for key in emotion_dict:
            emotion_dict[key] = max(0, emotion_dict[key])

        # Normalize to sum of 100 using more precise method
        total = sum(emotion_dict.values())
        if total > 0:
            # First calculate ideal proportional values
            for key in emotion_dict:
                emotion_dict[key] = emotion_dict[key] * 100 / total

            # Convert to integers and calculate difference
            int_values = {key: int(value) for key, value in emotion_dict.items()}
            current_sum = sum(int_values.values())
            diff = 100 - current_sum

            # Allocate difference to largest dimensions
            if diff != 0:
                sorted_keys = sorted(emotion_dict.keys(), key=lambda k: emotion_dict[k], reverse=True)
                for i in range(abs(diff)):
                    key = sorted_keys[i % len(sorted_keys)]
                    int_values[key] += 1 if diff > 0 else -1
                    int_values[key] = max(0, int_values[key])

            emotion_dict = int_values
        else:
            # If sum is 0, distribute evenly
            emotion_dict = {key: 100 // len(emotion_dict) for key in emotion_dict}
            # Handle remainder
            remainder = 100 % len(emotion_dict)
            keys = list(emotion_dict.keys())
            for i in range(remainder):
                emotion_dict[keys[i]] += 1

        return Emotion(**emotion_dict)

    def apply_relationship_delta(
        self,
        relationship: Relationship,
        relationship_delta: int,
        acquaintance_threshold: int,
        friend_threshold: int,
        situationship_threshold: int,
        lover_threshold: int,
    ) -> Relationship:
        """Apply relationship delta and update relationship stage.

        Updates relationship score and determines appropriate relationship
        stage based on score thresholds.

        Args:
            relationship (Relationship):
                Current relationship state.
            relationship_delta (int):
                Relationship score change to apply.
            acquaintance_threshold (int):
                Threshold for acquaintance stage.
            friend_threshold (int):
                Threshold for friend stage.
            situationship_threshold (int):
                Threshold for situationship stage.
            lover_threshold (int):
                Threshold for lover stage.

        Returns:
            Relationship:
                Updated relationship with new score and stage.
        """
        new_relationship = relationship.model_copy()
        new_relationship.score += relationship_delta
        if new_relationship.score < acquaintance_threshold:
            new_relationship.stage = "Stranger"
        elif new_relationship.score < friend_threshold:
            new_relationship.stage = "Acquaintance"
        elif new_relationship.score < situationship_threshold:
            new_relationship.stage = "Friend"
        elif new_relationship.score < lover_threshold:
            new_relationship.stage = "Situationship"
        else:
            new_relationship.stage = "Lover"
        return new_relationship

    def apply_emotion_decay(self, emotion: Emotion) -> Emotion:
        """Apply emotion decay towards neutral state.

        Emotions farther from average decay more rapidly. Uses deviation-based
        decay rate to gradually return emotions to neutral state.

        Args:
            emotion (Emotion):
                Current emotion state to decay.

        Returns:
            Emotion:
                Decayed emotion with normalized values.
        """
        emotion_dict = emotion.model_dump()

        # Calculate average value (ideally each emotion should be 100/7 ≈ 14.3)
        num_emotions = len(emotion_dict)
        average_value = 100 / num_emotions

        # Apply deviation-based decay to all emotions
        for key in emotion_dict:
            if emotion_dict[key] > 0:
                # Calculate deviation from average
                deviation = abs(emotion_dict[key] - average_value)

                # Base decay rate
                base_decay_rate = self.emotion_decay_rate

                # Deviation decay rate: larger deviation means more additional decay
                # Use square root function to make decay smoother, avoid over-decay
                deviation_factor = (deviation / average_value) ** 0.5
                deviation_decay_rate = base_decay_rate * deviation_factor * 0.5

                # Total decay rate
                total_decay_rate = base_decay_rate + deviation_decay_rate

                # Apply decay
                decay_amount = emotion_dict[key] * total_decay_rate
                emotion_dict[key] = max(0, emotion_dict[key] - decay_amount)

        # Normalize to sum of 100 using more precise method
        total = sum(emotion_dict.values())
        if total > 0:
            # First calculate ideal proportional values
            for key in emotion_dict:
                emotion_dict[key] = emotion_dict[key] * 100 / total

            # Convert to integers and calculate difference
            int_values = {key: int(value) for key, value in emotion_dict.items()}
            current_sum = sum(int_values.values())
            diff = 100 - current_sum

            # Allocate difference to largest dimensions
            if diff != 0:
                sorted_keys = sorted(emotion_dict.keys(), key=lambda k: emotion_dict[k], reverse=True)
                for i in range(abs(diff)):
                    key = sorted_keys[i % len(sorted_keys)]
                    int_values[key] += 1 if diff > 0 else -1
                    int_values[key] = max(0, int_values[key])

            emotion_dict = int_values
        else:
            # If sum is 0, distribute evenly
            emotion_dict = {key: 100 // len(emotion_dict) for key in emotion_dict}
            # Handle remainder
            remainder = 100 % len(emotion_dict)
            keys = list(emotion_dict.keys())
            for i in range(remainder):
                emotion_dict[keys[i]] += 1

        return Emotion(**emotion_dict)

    def apply_memory_processing(self, emotion: Emotion, relationship: Relationship) -> tuple[Emotion, Relationship]:
        """Apply processing before storing to memory (decay, etc.).

        Processes emotion and relationship data before storing to memory,
        including emotion decay and other transformations.

        Args:
            emotion (Emotion):
                Emotion state to process.
            relationship (Relationship):
                Relationship state to process.

        Returns:
            tuple[Emotion, Relationship]:
                Processed emotion and relationship objects.
        """
        # Apply emotion decay
        processed_emotion = self.apply_emotion_decay(emotion)

        # Relationship processing (not implemented yet, can be added here later)
        processed_relationship = relationship

        return processed_emotion, processed_relationship

    async def _update_emotion_and_relationship(
        self, character_id: str, processed_emotion, processed_relationship, memory_db_client
    ):
        """Update emotion and relationship in memory.

        Stores processed emotion and relationship data to the database
        for the specified character.

        Args:
            character_id (str):
                The character ID to update.
            processed_emotion (Emotion):
                Processed emotion data to store.
            processed_relationship (Relationship):
                Processed relationship data to store.
            memory_db_client (Any):
                The memory database client to update.
        """
        try:
            await memory_db_client.set_emotion(
                character_id=character_id,
                happiness=processed_emotion.Happiness,
                sadness=processed_emotion.Sadness,
                fear=processed_emotion.Fear,
                anger=processed_emotion.Anger,
                disgust=processed_emotion.Disgust,
                surprise=processed_emotion.Surprise,
                shyness=processed_emotion.Shyness,
            )
            await memory_db_client.set_relationship(
                character_id=character_id,
                relationship=processed_relationship.stage,
                score=processed_relationship.score,
            )
        except Exception as e:
            self.logger.error(f"Error updating emotion and relationship: {str(e)}")

    async def reaction_llm_to_reaction(
        self,
        request_id: str,
        reaction_llm: ReactionLLM,
    ) -> Reaction:
        """Convert reaction LLM to reaction object.

        Transforms LLM-generated reaction data into a structured reaction
        object with emotion labels, motion keywords, and face emotion.

        Args:
            request_id (str):
                The request id.
            reaction_llm (ReactionLLM):
                The LLM-generated reaction data.

        Returns:
            Reaction:
                Structured reaction object.
        """
        # threshold dictionary
        threshold_dict = {
            "Neutral": self.input_buffer[request_id]["neutral_threshold"],
            "Happiness": self.input_buffer[request_id]["happiness_threshold"],
            "Sadness": self.input_buffer[request_id]["sadness_threshold"],
            "Fear": self.input_buffer[request_id]["fear_threshold"],
            "Anger": self.input_buffer[request_id]["anger_threshold"],
            "Disgust": self.input_buffer[request_id]["disgust_threshold"],
            "Surprise": self.input_buffer[request_id]["surprise_threshold"],
            "Shyness": self.input_buffer[request_id]["shyness_threshold"],
        }

        # label expression
        relationship_str = reaction_llm.relationship.stage
        emotion_list = []
        emotion_scores = reaction_llm.emotion.model_dump()

        # Sort by score from high to low, only add emotions above threshold, ensure first value is strongest non-neutral emotion
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        for key, score in sorted_emotions:
            threshold = threshold_dict[key]
            if score > threshold:
                emotion_list.append(key)

        # If all emotions are below neutral_threshold, add neutral
        # neutral_threshold = 100 // 7 + neutral_epsilon
        if all(score <= threshold_dict["Neutral"] for score in emotion_scores.values()):
            emotion_list.append("Neutral")

        emotion_str = " | ".join(emotion_list)
        label_expression = f"({relationship_str}) & ({emotion_str})"

        # speech text
        speech_text = reaction_llm.speech_text

        # motion keywords
        if not reaction_llm.motion:
            motion_keywords = None
        else:
            motion_keywords = []
            for motion in reaction_llm.motion:
                motion_keyword = motion.motion_keywords
                speech_keyword = motion.speech_keywords
                index = speech_text.find(speech_keyword)
                if index != -1:
                    motion_keywords.append((index, motion_keyword))
        # TODO: Future consideration needed for selecting most suitable emotion for facial expression
        return Reaction(
            speech_text=speech_text,
            label_expression=label_expression,
            motion_keywords=motion_keywords,
            face_emotion=emotion_list[0],
        )

    async def get_empty_reaction(
        self,
        request_id: str,
        text_segment: str,
        current_emotion: Dict[str, int],
        current_relationship: Dict[str, Any],
    ) -> Reaction:
        """Generate empty reaction maintaining current state.

        Creates a reaction object that maintains current emotion and
        relationship without generating new motion or emotion changes.

        Args:
            request_id (str):
                The request id.
            text_segment (str):
                The text segment for the reaction.
            current_emotion (Dict[str, int]):
                Current emotion state to maintain.
            current_relationship (Dict[str, Any]):
                Current relationship state to maintain.

        Returns:
            Reaction:
                Empty reaction maintaining current state.
        """
        """Generate empty reaction."""
        reaction_llm = ReactionLLM(
            speech_text=text_segment,
            emotion=Emotion(**current_emotion),  # Maintain current emotion
            relationship=Relationship(**current_relationship),  # Maintain current relationship
            motion=[],
        )
        return await self.reaction_llm_to_reaction(request_id, reaction_llm)

    async def _handle_start(self, chunk: ClassifiedTextChunkStart, cur_time: float) -> None:
        """Handle the start chunk.

        Initializes request buffer and starts stream processing for
        a new classified text chunk.

        Args:
            chunk (ClassifiedTextChunkStart):
                The start chunk to process.
            cur_time (float):
                Current timestamp.
        """
        dag = chunk.dag
        conf = dag.conf
        memory_adapter = conf.get("memory_adapter", None)
        reaction_model_override = conf.get("reaction_model_override", "")
        api_keys = conf.get("user_settings", {})
        memory_db_client = conf.get("memory_db_client", None)
        chunk_n_char_lowerbound = conf.get("chunk_n_char_lowerbound", 10)
        chunk_n_char_lowerbound_en = conf.get("chunk_n_char_lowerbound_en", 25)
        character_id = conf["character_id"]
        relationship_dynamodb = conf["relationship"]
        relationship = {
            "stage": relationship_dynamodb[0],
            "score": relationship_dynamodb[1],
        }
        emotion = conf["emotion"]
        acquaintance_threshold = conf.get("acquaintance_threshold", 5)
        friend_threshold = conf.get("friend_threshold", 10)
        situationship_threshold = conf.get("situationship_threshold", 15)
        lover_threshold = conf.get("lover_threshold", 20)
        neutral_threshold = conf.get("neutral_threshold", 50)
        happiness_threshold = conf.get("happiness_threshold", 30)
        sadness_threshold = conf.get("sadness_threshold", 30)
        fear_threshold = conf.get("fear_threshold", 30)
        anger_threshold = conf.get("anger_threshold", 30)
        disgust_threshold = conf.get("disgust_threshold", 30)
        surprise_threshold = conf.get("surprise_threshold", 30)
        shyness_threshold = conf.get("shyness_threshold", 30)
        language = conf.get("language", "zh")

        request_id = chunk.request_id
        user_input = chunk.user_input

        # Initialize buffer state using sentence splitter
        buffer_state = self.sentence_splitter.create_buffer_state()
        self.input_buffer[request_id] = {
            "start_time": cur_time,
            "last_update_time": cur_time,
            "dag": chunk.dag,
            "node_name": chunk.node_name,
            "tag": "",
            "chunk_sent": 0,
            "downstream_warned": False,
            "chunk_n_char_lowerbound": chunk_n_char_lowerbound,
            "chunk_n_char_lowerbound_en": chunk_n_char_lowerbound_en,
            **buffer_state,  # Merge sentence splitter buffer state
            "memory_adapter": memory_adapter,
            "llm_client": None,
            "reaction_model_override": reaction_model_override,
            "api_keys": api_keys,
            "memory_db_client": memory_db_client,
            "character_id": character_id,
            "relationship": relationship,
            "emotion": emotion,
            "acquaintance_threshold": acquaintance_threshold,
            "friend_threshold": friend_threshold,
            "situationship_threshold": situationship_threshold,
            "lover_threshold": lover_threshold,
            "neutral_threshold": neutral_threshold,
            "happiness_threshold": happiness_threshold,
            "sadness_threshold": sadness_threshold,
            "fear_threshold": fear_threshold,
            "anger_threshold": anger_threshold,
            "disgust_threshold": disgust_threshold,
            "surprise_threshold": surprise_threshold,
            "shyness_threshold": shyness_threshold,
            "user_input": user_input,
            "language": language,
            "reaction_stream_start_time": 0.0,  # Time when first audio is sent to downstream
            "sent_duration": 0.0,  # Cumulative duration of TTS audio already sent
        }
        asyncio.create_task(self._init_llm_client(request_id))
        asyncio.create_task(self._send_stream_start_task(request_id))

    async def _handle_body(self, chunk: ClassifiedTextChunkBody, cur_time: float) -> None:
        """Handle the body chunk.

        Processes text segments, handles dot sequences, and triggers
        reaction generation when appropriate text boundaries are reached.

        Args:
            chunk (ClassifiedTextChunkBody):
                The body chunk to process.
            cur_time (float):
                Current timestamp.
        """
        request_id = chunk.request_id
        text_segment = chunk.text_segment
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        dag = self.input_buffer[request_id]["dag"]
        if dag.status != DAGStatus.RUNNING:
            msg = f"DAG {dag.name} is not running."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time

        # Determine TTS character duration based on language
        if self.sentence_splitter.contains_chinese(text_segment):
            tts_char_duration = self.tts_char_duration_zh
        else:
            tts_char_duration = self.tts_char_duration_en

        # Define generation callback for sentence splitter
        def generation_callback(segment_text: str, seq_number: int) -> None:
            """Callback function for when a text segment is ready for reaction
            generation."""
            dag = self.input_buffer[request_id]["dag"]
            if dag.status == DAGStatus.RUNNING:
                asyncio.create_task(self._stream_reaction_task(request_id, segment_text, seq_number, tts_char_duration))

        # Use sentence splitter to process the text segment
        await self.sentence_splitter.process_text_segment(
            text_segment=text_segment,
            buffer_state=self.input_buffer[request_id],
            chunk_n_char_lowerbound=self.input_buffer[request_id]["chunk_n_char_lowerbound"],
            chunk_n_char_lowerbound_en=self.input_buffer[request_id]["chunk_n_char_lowerbound_en"],
            generation_callback=generation_callback,
        )

    async def _handle_end(self, chunk: ClassifiedTextChunkEnd, cur_time: float) -> None:
        """Handle the end chunk.

        Processes remaining text and sends stream end signal to
        downstream nodes.

        Args:
            chunk (ClassifiedTextChunkEnd):
                The end chunk to process.
            cur_time (float):
                Current timestamp.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        chunk_for_generation = self.input_buffer[request_id]["text_segments"]
        split_marks = self.sentence_splitter.SPLIT_MARKS
        tts_char_duration = (
            self.tts_char_duration_zh
            if self.sentence_splitter.contains_chinese(chunk_for_generation)
            else self.tts_char_duration_en
        )
        if not all(c in split_marks for c in chunk_for_generation) and len(chunk_for_generation) > 0:
            seq_number = self.input_buffer[request_id]["chunk_received"]
            self.input_buffer[request_id]["chunk_received"] += 1
            asyncio.create_task(
                self._stream_reaction_task(request_id, chunk_for_generation, seq_number, tts_char_duration)
            )
        asyncio.create_task(self._send_stream_end_task(request_id))

    async def _stream_reaction_task(
        self,
        request_id: str,
        text_segment: str,
        seq_number: int,
        tts_char_duration: float = 0.2,
        threshold: float = 0.3,  # Safety threshold, complete reaction threshold seconds before audio finishes
    ) -> None:
        """Stream reaction task.

        Generates and streams reaction for a text segment with timeout
        handling and timing management.

        Args:
            request_id (str):
                The request identifier.
            text_segment (str):
                The text segment to process.
            seq_number (int):
                Sequence number of the chunk.
            tts_char_duration (float, optional):
                TTS character duration for timing. Defaults to 0.2.
            threshold (float, optional):
                Safety threshold for timeout calculation. Defaults to 0.3.
        """
        try:
            generation_start_time = time.time()
            # Get memory_db_client
            memory_db_client = self.input_buffer[request_id]["memory_db_client"]
            # Get user context information
            character_id = self.input_buffer[request_id]["character_id"]
            user_input = self.input_buffer[request_id]["user_input"]
            language = self.input_buffer[request_id]["language"]

            if language == "zh":
                prompt = REACTION_PROMPT_ZH.format(motion_keywords_database=self.motion_kws)
            else:
                prompt = REACTION_PROMPT_EN.format(motion_keywords_database=self.motion_kws)

            # Filter text
            text_segment, removed_text = await self.filter_text(text_segment)
            self.input_buffer[request_id]["tag"] += removed_text

            # Get current emotion and relationship state
            current_emotion = self.input_buffer[request_id]["emotion"]
            current_relationship = self.input_buffer[request_id]["relationship"]

            # Wait for previous chunks to complete sending, ensure correct timing information is read
            dag = self.input_buffer[request_id]["dag"]
            while self.input_buffer[request_id]["chunk_sent"] < seq_number:
                if dag.status == DAGStatus.RUNNING:
                    await asyncio.sleep(self.sleep_time)
                else:
                    msg = f"Streaming reaction interrupted by DAG status {dag.status}"
                    msg = msg + f" for request {request_id}"
                    self.logger.warning(msg)
                    return

            # Calculate timeout based on cumulative audio duration
            current_time = time.time()
            reaction_stream_start_time = self.input_buffer[request_id]["reaction_stream_start_time"]
            sent_duration = self.input_buffer[request_id]["sent_duration"]

            # Check if should timeout (fast processing)
            left_side = current_time - reaction_stream_start_time
            right_side = sent_duration - threshold
            should_timeout = left_side > right_side

            if should_timeout:
                # Fast processing, maintain current emotion and relationship, empty motion keywords
                reaction = await self.get_empty_reaction(
                    request_id,
                    text_segment,
                    current_emotion,
                    current_relationship,
                )
                timeout_reason = (
                    "first_chunk"
                    if seq_number == 0
                    else f"cumulative_audio_timeout ({left_side:.2f}s > {right_side:.2f}s)"
                )
                self.logger.debug(
                    f"Request {request_id}, seq_number {seq_number}: timeout fast processing - {timeout_reason}"
                )
            else:
                # Normal processing, use LLM to generate reaction
                try:
                    # Calculate remaining available time as timeout
                    remaining_time = right_side - left_side
                    reaction_llm = await self.get_reaction_llm(
                        request_id=request_id,
                        prompt=prompt,
                        text=text_segment,
                        tag=self.input_buffer[request_id]["tag"],
                        user_input=user_input,
                        current_emotion=current_emotion,
                        current_relationship=current_relationship,
                        tts_char_duration=tts_char_duration,
                        timeout_seconds=remaining_time,
                        character_id=character_id,
                        memory_db_client=memory_db_client,
                    )
                    reaction = await self.reaction_llm_to_reaction(request_id, reaction_llm)
                    self.logger.debug(
                        f"Request {request_id}, seq_number {seq_number}: normal LLM processing with timeout {remaining_time:.2f}s"
                    )
                except Exception as e:
                    self.logger.error(
                        f"{e} Error in getting reaction for request {request_id}, "
                        + f"seq_number {seq_number}, text_segment {text_segment}, "
                        + "using empty reaction as default."
                    )
                    reaction = await self.get_empty_reaction(
                        request_id,
                        text_segment,
                        current_emotion,
                        current_relationship,
                    )

            generation_end_time = time.time()
            msg = f"Streaming reaction generation spent {generation_end_time - generation_start_time:.3f} seconds"
            msg = msg + f" for request {request_id}"
            msg += f", seq_number: {seq_number}"
            self.logger.debug(msg)

            # Record first chunk send time
            if seq_number == 0:
                self.input_buffer[request_id]["reaction_stream_start_time"] = time.time()

            # Prepare downstream
            dag = self.input_buffer[request_id]["dag"]
            node_name = self.input_buffer[request_id]["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            downstream_warned = self.input_buffer[request_id]["downstream_warned"]
            if len(downstream_nodes) == 0 and not downstream_warned:
                self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
                self.input_buffer[request_id]["downstream_warned"] = True
                return
            coroutines = list()
            for node in downstream_nodes:
                payload = node.payload
                body_chunk = ReactionChunkBody(
                    request_id=request_id,
                    seq_number=seq_number,
                    reaction=reaction,
                )
                coroutines.append(payload.feed_stream(body_chunk))
            asyncio.gather(*coroutines)
            self.input_buffer[request_id]["chunk_sent"] += 1

            # Update cumulative duration of sent audio
            current_audio_duration = tts_char_duration * len(text_segment)
            self.input_buffer[request_id]["sent_duration"] += current_audio_duration
            self.logger.debug(
                f"Request {request_id}, seq_number {seq_number}: updated sent_duration to {self.input_buffer[request_id]['sent_duration']:.2f}s (added {current_audio_duration:.2f}s)"
            )

            time_diff = time.time() - generation_end_time
            msg = f"Streaming reaction sending to downstreams spent {time_diff:.3f} seconds"
            msg = msg + f" for request {request_id}"
            self.logger.debug(msg)
        except Exception as e:
            msg = f"Error in streaming reaction: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.FAILED)
            return

    async def _send_stream_start_task(self, request_id: str) -> None:
        """Send stream start task.

        Sends stream start signal to all downstream nodes for the
        specified request.

        Args:
            request_id (str):
                The request identifier.
        """
        dag = self.input_buffer[request_id]["dag"]
        node_name = self.input_buffer[request_id]["node_name"]
        dag_node = dag.get_node(node_name)
        downstream_nodes = dag_node.downstreams
        downstream_warned = self.input_buffer[request_id]["downstream_warned"]
        if len(downstream_nodes) == 0 and not downstream_warned:
            self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
            self.input_buffer[request_id]["downstream_warned"] = True
            return
        coroutines = list()
        for node in downstream_nodes:
            next_node_name = node.name
            payload = node.payload
            start_trunk = ReactionChunkStart(request_id=request_id, node_name=next_node_name, dag=dag)
            coroutines.append(payload.feed_stream(start_trunk))
        asyncio.gather(*coroutines)

    async def _send_stream_end_task(self, request_id: str) -> None:
        """Send stream end task.

        Waits for all chunks to be sent and then sends stream end
        signal to downstream nodes.

        Args:
            request_id (str):
                The request identifier.
        """
        try:
            dag = self.input_buffer[request_id]["dag"]
        except KeyError:
            msg = f"Request {request_id} not found in input buffer, skip sending stream end task."
            self.logger.warning(msg)
            return
        while self.input_buffer[request_id]["chunk_received"] > self.input_buffer[request_id]["chunk_sent"]:
            if dag.status == DAGStatus.RUNNING:
                await asyncio.sleep(self.sleep_time)
            else:
                msg = f"Reaction sending stream end task interrupted by DAG status {dag.status}"
                msg = msg + f" for request {request_id}"
                self.logger.warning(msg)
                return
        # Prepare downstream
        dag = self.input_buffer[request_id]["dag"]
        node_name = self.input_buffer[request_id]["node_name"]
        dag_node = dag.get_node(node_name)
        downstream_nodes = dag_node.downstreams
        downstream_warned = self.input_buffer[request_id]["downstream_warned"]
        if len(downstream_nodes) == 0 and not downstream_warned:
            self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
            self.input_buffer[request_id]["downstream_warned"] = True
            return
        coroutines = list()
        for node in downstream_nodes:
            payload = node.payload
            end_trunk = ReactionChunkEnd(request_id=request_id)
            coroutines.append(payload.feed_stream(end_trunk))
        asyncio.gather(*coroutines)
        self.input_buffer.pop(request_id)
