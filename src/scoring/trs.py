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


# spatial conflict
def apply_spatial_conflict_penalty(chunk, vectorstore, z_threshold=2.5):
    """
    Check if a numeric sensor reading is a spatial outlier compared
    to nearby sensors of the same metric. If so, penalise the sensor's
    Bayesian credibility and return a penalty multiplier (< 1.0).
    Returns 1.0 if no conflict is detected.
    """
    numeric_metrics = {"AQI", "temperature", "traffic_density", "bin_fill"}
    actual_metric = None
    for m in numeric_metrics:
        if m.lower() in chunk.page_content.lower():
            actual_metric = m
            break
    if not actual_metric:
        return 1.0

    location = chunk.metadata.get("location", None)
    chunk_id = chunk.metadata.get("id", None)
    if not location:
        return 1.0

    # Find neighbors (exclude the current chunk)
    neighbors = vectorstore.similarity_search(actual_metric, k=15)
    neighbor_values = []
    for nb in neighbors:
        nb_id = nb.metadata.get("id")
        if nb_id == chunk_id:
            continue  # skip the chunk we are checking
        nb_loc = nb.metadata.get("location")
        if not nb_loc:
            continue
        lat_diff = abs(location[1] - nb_loc[1])
        lon_diff = abs(location[0] - nb_loc[0])
        if lat_diff < 0.01 and lon_diff < 0.01:
            try:
                parts = nb.page_content.split("|")
                for part in parts:
                    if actual_metric in part:
                        val = float(part.split(":")[-1].strip().split()[0])
                        neighbor_values.append(val)
                        break
            except (ValueError, IndexError):
                continue

    if len(neighbor_values) < 3:
        return 1.0

    mu = sum(neighbor_values) / len(neighbor_values)
    variance = sum((v - mu) ** 2 for v in neighbor_values) / len(neighbor_values)
    sigma = variance**0.5
    if sigma == 0:
        return 1.0

    # Extract this chunk's value
    try:
        parts = chunk.page_content.split("|")
        this_val = None
        for part in parts:
            if actual_metric in part:
                this_val = float(part.split(":")[-1].strip().split()[0])
                break
        if this_val is None:
            return 1.0
    except (ValueError, IndexError):
        return 1.0

    z = (this_val - mu) / sigma

    if abs(z) > z_threshold:
        source_type = chunk.metadata.get("source_type", "government")
        sensor_id = chunk.metadata.get("id", None)
        update_bayesian_credibility(source_type, sensor_id, accurate=False)
        penalty = max(0.1, 1.0 - (abs(z) - z_threshold) / 10)
        return penalty

    update_bayesian_credibility(
        chunk.metadata.get("source_type", "government"),
        chunk.metadata.get("id", None),
        accurate=True,
    )
    return 1.0


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


def rank_chunks_by_trs(chunks_with_scores, vectorstore=None):
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
        if vectorstore is not None:
            penalty = apply_spatial_conflict_penalty(chunk, vectorstore)
            trs = round(trs * penalty, 4)
        ranked.append((chunk, trs))

    # sort
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
