"""Sentence splitting utilities for text processing.

This module provides a common sentence splitting functionality used by TTS and
Reaction adapters to handle text segmentation with dot sequence detection and
Chinese/English character boundary handling.
"""

import asyncio
import re
from typing import Awaitable, Callable, Dict, Set, Union


class SentenceSplitter:
    """Handles sentence splitting with dot sequence detection and language-
    aware boundaries.

    This class provides common functionality for processing text segments,
    detecting dot sequences (like "..."), and determining appropriate
    splitting boundaries based on character count thresholds for Chinese
    and English text.
    """

    # Common split marks used by both TTS and Reaction adapters
    SPLIT_MARKS: Set[str] = {"，", ",", "。", ".", "！", "!", "？", "?", "；", ";", "：", ":", "～", "~"}

    def __init__(self, logger=None):
        """Initialize the sentence splitter.

        Args:
            logger: Logger instance for debug messages.
        """
        self.logger = logger

    def contains_chinese(self, text: str) -> bool:
        """Check if the text contains Chinese characters.

        Args:
            text (str): The text to check for Chinese characters.

        Returns:
            bool: True if the text contains at least one Chinese character
                (Unicode range U+4E00 to U+9FFF), False otherwise.
        """
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                return True
        return False

    async def process_text_segment(
        self,
        text_segment: str,
        buffer_state: Dict,
        chunk_n_char_lowerbound: int,
        chunk_n_char_lowerbound_en: int,
        generation_callback: Union[Callable[[str, int], None], Callable[[str, int], Awaitable[None]]],
    ) -> None:
        """Process a text segment and trigger generation when appropriate
        boundaries are reached.

        This method handles the core sentence splitting logic including:
        - Dot sequence detection and correction (e.g., "作为...作为")
        - Character count boundary checking
        - Split mark detection
        - First chunk special handling

        Args:
            text_segment (str): The text segment to process.
            buffer_state (Dict): The request buffer state containing text_segments,
                dealing_with_dots, n_dots, n_char_after_dots, chunk_received, etc.
            chunk_n_char_lowerbound (int): Minimum character count for Chinese text.
            chunk_n_char_lowerbound_en (int): Minimum character count for English text.
            generation_callback (Union[Callable[[str, int], None], Callable[[str, int], Awaitable[None]]]):
                Function to call when a segment is ready for generation.
                Should accept (segment_text, sequence_number) as arguments.
                Can be either sync or async function.
        """
        split_marks = self.SPLIT_MARKS

        # Determine character boundary based on language
        if self.contains_chinese(text_segment):
            char_lowerbound = chunk_n_char_lowerbound
        else:
            char_lowerbound = chunk_n_char_lowerbound_en

        for char in text_segment:
            # Handle dots mode - detecting and correcting patterns like "作为...作为"
            if buffer_state["dealing_with_dots"]:
                await self._handle_dots_mode(char, buffer_state, generation_callback)
            else:
                # Normal mode - regular sentence splitting logic
                await self._handle_normal_mode(char, buffer_state, char_lowerbound, split_marks, generation_callback)

    async def _handle_dots_mode(
        self,
        char: str,
        buffer_state: Dict,
        generation_callback: Union[Callable[[str, int], None], Callable[[str, int], Awaitable[None]]],
    ) -> None:
        """Handle dot sequence detection and correction.

        Args:
            char (str): Current character being processed.
            buffer_state (Dict): Buffer state containing dots-related fields.
            generation_callback (Union[Callable[[str, int], None], Callable[[str, int], Awaitable[None]]]): Callback for generation.
        """
        buffer_state["text_segments"] += char

        if char in {".", "。"}:
            buffer_state["n_dots"] += 1
        else:
            buffer_state["n_char_after_dots"] += 1

            if buffer_state["n_char_after_dots"] < 2:
                return

            if buffer_state["n_char_after_dots"] >= 4:
                buffer_state["dealing_with_dots"] = False
                return

            # Check for repeated pattern and correct if found
            n_char_after_dots = buffer_state["n_char_after_dots"]
            text_segments_len = len(buffer_state["text_segments"])
            n_dots = buffer_state["n_dots"]

            str_after_dots = buffer_state["text_segments"][text_segments_len - n_char_after_dots :]
            str_before_dots = buffer_state["text_segments"][: text_segments_len - n_char_after_dots - n_dots]
            str_before_dots = str_before_dots[max(0, len(str_before_dots) - n_char_after_dots) :]

            if str_before_dots == str_after_dots:
                n_char_to_remove = n_dots - 1 + len(str_after_dots)
                text_segments_src = buffer_state["text_segments"]
                buffer_state["text_segments"] = buffer_state["text_segments"][: text_segments_len - n_char_to_remove]

                if self.logger:
                    msg = (
                        f"Fix text segments from {text_segments_src} to "
                        f"{buffer_state['text_segments']} by dealing_with_dots."
                    )
                    self.logger.info(msg)

                buffer_state["dealing_with_dots"] = False

    async def _handle_normal_mode(
        self,
        char: str,
        buffer_state: Dict,
        char_lowerbound: int,
        split_marks: Set[str],
        generation_callback: Union[Callable[[str, int], None], Callable[[str, int], Awaitable[None]]],
    ) -> None:
        """Handle normal sentence splitting mode.

        Args:
            char (str): Current character being processed.
            buffer_state (Dict): Buffer state containing text processing fields.
            char_lowerbound (int): Minimum character count before splitting.
            split_marks (Set[str]): Set of characters that indicate sentence boundaries.
            generation_callback (Union[Callable[[str, int], None], Callable[[str, int], Awaitable[None]]]): Callback for generation.
        """
        buffer_state["text_segments"] += char

        # Special handling for first chunk ending with split mark
        is_first_chunk = buffer_state["chunk_received"] == 0
        if is_first_chunk and buffer_state["text_segments"] and buffer_state["text_segments"][-1] in split_marks:
            segment_for_generation = buffer_state["text_segments"]
            buffer_state["text_segments"] = ""
            seq_number = buffer_state["chunk_received"]
            buffer_state["chunk_received"] += 1
            if asyncio.iscoroutinefunction(generation_callback):
                await generation_callback(segment_for_generation, seq_number)
            else:
                generation_callback(segment_for_generation, seq_number)
            return

        # Check if we have enough characters to consider splitting
        if len(buffer_state["text_segments"]) < char_lowerbound + 1:
            return

        # Detect dot sequence pattern (e.g., "作为...作为")
        if (
            buffer_state["text_segments"][-1] in {".", "。"}
            and len(buffer_state["text_segments"]) > 3
            and buffer_state["text_segments"][-2] == buffer_state["text_segments"][-1]
        ):
            buffer_state["dealing_with_dots"] = True
            buffer_state["n_char_after_dots"] = 0
            buffer_state["n_dots"] = 2
            return

        # Check if all characters before the last are split marks
        if all(c in split_marks for c in buffer_state["text_segments"][:-1]):
            return

        # Split at the second-to-last character if it's a split mark
        if len(buffer_state["text_segments"]) > 3 and buffer_state["text_segments"][-2] in split_marks:
            segment_for_generation = buffer_state["text_segments"][:-1]
            buffer_state["text_segments"] = buffer_state["text_segments"][-1]
            seq_number = buffer_state["chunk_received"]
            buffer_state["chunk_received"] += 1
            if asyncio.iscoroutinefunction(generation_callback):
                await generation_callback(segment_for_generation, seq_number)
            else:
                generation_callback(segment_for_generation, seq_number)

    def create_buffer_state(self) -> Dict:
        """Create a new buffer state dictionary for a request.

        Returns:
            Dict: Initialized buffer state with all required fields.
        """
        return {
            "text_segments": "",
            "dealing_with_dots": False,
            "n_dots": 0,
            "n_char_after_dots": 0,
            "chunk_received": 0,
        }

    def filter_text(self, text: str) -> str:
        """Filter and clean text by removing HTML tags, parentheses, and
        brackets.

        Args:
            text (str): Input text to be filtered and cleaned.

        Returns:
            str: Cleaned text with HTML tags, parentheses, and brackets removed.
        """
        # Remove HTML tags
        text = re.sub(r"<[^<>]+?>.*?</[^<>]+?>", "", text, flags=re.DOTALL)

        # Remove Chinese parentheses
        text = re.sub(r"（.*?）", "", text)

        # Remove English parentheses
        text = re.sub(r"\(.*?\)", "", text)

        # Remove square brackets
        text = re.sub(r"\[.*?\]", "", text)

        return text
