import math
from datetime import datetime, timezone

from src.config import ADAPTIVE_DECAY_RATES, DECAY_RATES, INITIAL_BAYESIAN_PRIORS

# In‑memory Bayesian counts (will be initialised lazily)
_bayesian_counts = {}


def get_bayesian_credibility(source_type, sensor_id=None):
    """
    Return the credibility score for a sensor using Bayesian updating.
    Currently uses static priors; counts can be updated dynamically
    via update_bayesian_credibility() when sensor accuracy is evaluated.
    """
    # Build the key – if we have a specific sensor, track per‑sensor;
    # otherwise fall back to the source‑type prior.
    key = sensor_id if sensor_id else source_type

    if key not in _bayesian_counts:
        # Initialise from static prior
        alpha, beta = INITIAL_BAYESIAN_PRIORS.get(source_type, (50, 50))
        _bayesian_counts[key] = {"alpha": alpha, "beta": beta}

    counts = _bayesian_counts[key]
    total = counts["alpha"] + counts["beta"]
    return counts["alpha"] / total if total > 0 else 0.5


def update_bayesian_credibility(source_type, sensor_id, accurate=True):
    """
    Update the Bayesian counts for a sensor.
    Call this when a sensor reading has been evaluated (e.g., via LLM‑as‑judge,
    or by checking against spatial consensus).
    - accurate=True  → increment alpha
    - accurate=False → increment beta
    """
    key = sensor_id if sensor_id else source_type
    if key not in _bayesian_counts:
        alpha, beta = INITIAL_BAYESIAN_PRIORS.get(source_type, (50, 50))
        _bayesian_counts[key] = {"alpha": alpha, "beta": beta}

    if accurate:
        _bayesian_counts[key]["alpha"] += 1
    else:
        _bayesian_counts[key]["beta"] += 1


def get_adaptive_decay_rate(entity_type, document_type, source_type):
    """Pick the best decay rate: entity_type → source_type → generic fallback."""
    if entity_type and entity_type in ADAPTIVE_DECAY_RATES:
        return ADAPTIVE_DECAY_RATES[entity_type]
    if source_type and source_type in ADAPTIVE_DECAY_RATES:
        return ADAPTIVE_DECAY_RATES[source_type]
    return DECAY_RATES.get(document_type, DECAY_RATES["sensor"])


def calculate_recency_score(timestamp_str, decay_rate=DECAY_RATES["sensor"]):
    """
    Exponential decay based on the age of the data
    decay_rate controls how fast freshness drops
    """
    if decay_rate == 0.0:
        return 1.0

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
        sensor_id = chunk.metadata.get("id", None)
        document_type = chunk.metadata.get("document_type", "sensor")
        source_type = chunk.metadata.get("source_type", "government")
        credibility = get_bayesian_credibility(source_type, sensor_id)
        entity_type = chunk.metadata.get("entity_type", None)
        decay_rate = get_adaptive_decay_rate(entity_type, document_type, source_type)

        trs = calculate_trs(semantic_score, timestamp, credibility, decay_rate)
        ranked.append((chunk, trs))

    # sort
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
