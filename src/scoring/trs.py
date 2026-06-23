import math
from datetime import datetime, timezone

from src.config import DECAY_RATES


def calculate_recency_score(timestamp_str, decay_rate=DECAY_RATES["sensor"]):
    if decay_rate == 0.0:
        return 1.0

    """
    Exponential decay based on the age of the data
    decay_rate controls how fast freshness drops
    """
    chunk_time = datetime.fromisoformat(timestamp_str)
    # Make sure both datetimes are timezone-aware
    if chunk_time.tzinfo is None:
        chunk_time = chunk_time.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    age_in_seconds = (now - chunk_time).total_seconds()
    recency_score = math.exp(-decay_rate * age_in_seconds)
    return recency_score


def calculate_trs(
    semantic_score, timestamp_str, credibility, decay_rate=DECAY_RATES["sensor"]
):
    similarity = 1 / (1 + semantic_score)
    recency = calculate_recency_score(timestamp_str, decay_rate)
    trs = similarity * recency * credibility
    return round(trs, 4)


def rank_chunks_by_trs(chunks_with_scores):
    """
    Takes retrieved chunks with semantic scores,
    calculates TRS for each, returns ranked list.
    """
    ranked = []

    for chunk, semantic_score in chunks_with_scores:
        timestamp = chunk.metadata.get("timestamp", datetime.now().isoformat())
        credibility = chunk.metadata.get("credibility", 0.5)
        document_type = chunk.metadata.get("document_type", "sensor")
        decay_rate = DECAY_RATES.get(document_type, DECAY_RATES["sensor"])
        trs = calculate_trs(semantic_score, timestamp, credibility, decay_rate)
        ranked.append((chunk, trs))

    # sort
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
