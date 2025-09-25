from typing import Union

from .blendshapes_aggregator import BlendshapesAggregator
from .callback_aggregator import CallbackAggregator
from .conversation_aggregator import ConversationAggregator
from .tts_reaction_aggregator import TTSReactionAggregator

_AGGREGATORS = dict(
    CallbackAggregator=CallbackAggregator,
    ConversationAggregator=ConversationAggregator,
    TTSReactionAggregator=TTSReactionAggregator,
    BlendshapesAggregator=BlendshapesAggregator,
)


def build_aggregator(
    cfg: dict,
) -> Union[CallbackAggregator, ConversationAggregator, TTSReactionAggregator, BlendshapesAggregator]:
    """Build an aggregator instance from a configuration dictionary.

    Args:
        cfg (dict):
            Configuration dictionary containing the aggregator type and
            initialization parameters. Must contain a 'type' key specifying
            the aggregator class name.

    Returns:
        Union[CallbackAggregator, ConversationAggregator, TTSReactionAggregator, BlendshapesAggregator]:
            An instance of the specified aggregator class initialized
            with the provided configuration parameters.

    Raises:
        TypeError:
            If the specified aggregator type is not recognized or
            if the configuration is invalid.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    if cls_name not in _AGGREGATORS:
        msg = f"Unknown aggregator type: {cls_name}"
        raise TypeError(msg)
    ret_inst = _AGGREGATORS[cls_name](**cfg)
    return ret_inst
