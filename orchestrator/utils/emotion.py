def get_emotion_list(
    happiness_score: int,
    happiness_threshold: int,
    sadness_score: int,
    sadness_threshold: int,
    fear_score: int,
    fear_threshold: int,
    anger_score: int,
    anger_threshold: int,
    disgust_score: int,
    disgust_threshold: int,
    surprise_score: int,
    surprise_threshold: int,
    shyness_score: int,
    shyness_threshold: int,
    neutral_threshold: int,
) -> list[str]:
    """Generate a list of emotions based on scores and thresholds.

    Analyzes emotion scores against their respective thresholds and returns
    a prioritized list of emotions. Emotions are sorted by score in descending
    order, and only emotions above their threshold are included. If the highest
    scoring emotion is below the neutral threshold, "Neutral" is added to the list.

    Args:
        happiness_score (int):
            Happiness emotion score value.
        happiness_threshold (int):
            Minimum threshold for happiness emotion to be included.
        sadness_score (int):
            Sadness emotion score value.
        sadness_threshold (int):
            Minimum threshold for sadness emotion to be included.
        fear_score (int):
            Fear emotion score value.
        fear_threshold (int):
            Minimum threshold for fear emotion to be included.
        anger_score (int):
            Anger emotion score value.
        anger_threshold (int):
            Minimum threshold for anger emotion to be included.
        disgust_score (int):
            Disgust emotion score value.
        disgust_threshold (int):
            Minimum threshold for disgust emotion to be included.
        surprise_score (int):
            Surprise emotion score value.
        surprise_threshold (int):
            Minimum threshold for surprise emotion to be included.
        shyness_score (int):
            Shyness emotion score value.
        shyness_threshold (int):
            Minimum threshold for shyness emotion to be included.
        neutral_threshold (int):
            Maximum threshold below which "Neutral" emotion is added.

    Returns:
        list[str]:
            List of emotion names sorted by score in descending order.
            Includes emotions above their thresholds and "Neutral" if
            the highest score is below neutral_threshold.
    """
    emotion_scores = dict(
        Happiness=happiness_score,
        Sadness=sadness_score,
        Fear=fear_score,
        Anger=anger_score,
        Disgust=disgust_score,
        Surprise=surprise_score,
        Shyness=shyness_score,
    )
    emotion_thresholds = dict(
        Happiness=happiness_threshold,
        Sadness=sadness_threshold,
        Fear=fear_threshold,
        Anger=anger_threshold,
        Disgust=disgust_threshold,
        Surprise=surprise_threshold,
        Shyness=shyness_threshold,
    )
    emotion_list = list()
    # Sort by score from high to low, only add emotions above threshold,
    # ensure first value is strongest non-neutral emotion
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    for key, score in sorted_emotions:
        threshold = emotion_thresholds[key]
        if score > threshold:
            emotion_list.append(key)
    max_score = 0 if len(emotion_list) == 0 else emotion_scores[emotion_list[0]]
    if max_score <= neutral_threshold:
        emotion_list.append("Neutral")
    return emotion_list
