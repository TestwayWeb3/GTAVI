#!/usr/bin/env python3
"""Forecast launch-day Twitch viewership hour by hour, as quantiles.

For each case file found in the input directory, this program reads the
first hour's viewer count, the history of viewer counts before the end of
the first hour, and the hourly counts of comparable past game launches.
It forecasts, for hours 2 through 24 of the launch day, five quantiles of
each hour's viewer count.

Method, in brief: every comparable launch contributes the shape of its own
release day, expressed as the log ratio of each hour's viewers to its
first hour's viewers.  Comparables are weighted by how closely their
pre-launch trajectory (the day before the release day, relative to their
own first hour) resembles the target case's pre-launch trajectory, and by
how close their first-hour scale is to the target's, so launches whose
unlock pattern and size match the target's count most.  For each hour,
the forecast quantiles are the weighted quantiles of the comparable
shapes, anchored on the target's first-hour count.  A spread floor keeps
the bands honest when the comparables happen to agree closely.

The program uses only the standard library, is deterministic, and writes
one forecast file per case file found.
"""

import json
import math
import os
import re
import sys
from datetime import datetime

# Levels requested by the case file; used when the case does not name any.
DEFAULT_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at the default levels, for the fallback spread.
NORMAL_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

# Hour offsets (relative to the release day's start) used to compare a
# launch's pre-launch trajectory with the target's.  Negative offsets are
# hours before the release day begins.
FEATURE_OFFSETS = (-24, -18, -12, -9, -6, -4, -3, -2, -1)
# Feature log-ratios are clamped to this range: counts near zero make raw
# log-ratios meaningless noise.
FEATURE_CLAMP = (-8.0, 2.0)
# Gaussian kernel bandwidth (in log space) for trajectory similarity.
BANDWIDTH = 1.0
# Weight of the launch-scale term in the similarity distance: the squared
# difference of log first-hour counts, halved, added to the mean squared
# trajectory difference.  Launch scale strongly predicts how much a launch
# can still grow after its first hour.
SCALE_WEIGHT = 1.0
# Minimum number of shared feature coordinates for a distance to count.
MIN_SHARED_FEATURES = 3
# Relative weight given to each comparability tier parsed from the unlock
# notes.  All tiers count fully: the trajectory and scale similarity
# already separate the launch regimes, and downweighting tiers hurt
# leave-one-out validation.  The target-history tier is still excluded
# outright, since it holds no release day.
TIER_WEIGHTS = {
    "comparable": 1.0,
    "early-access-contaminated": 1.0,
    "weak": 1.0,
}
# A small floor weight, as a fraction of the largest weight, keeps every
# usable comparable contributing a little, which stabilises the tails.
FLOOR_WEIGHT_FRACTION = 0.02
# Multiplier applied to the spread around the median after smoothing.
SPREAD_MULTIPLIER = 1.0
# Minimum half-width of the 5%-95% band, in log space, growing with the
# hour: uncertainty genuinely grows as the day goes on.
SPREAD_FLOOR_BASE = 0.25
SPREAD_FLOOR_PER_HOUR = 0.02
# Smoothing across neighbouring hours was tried and removed: the weighted
# quantiles are already smooth across hours, and smoothing did not help
# leave-one-out validation.

CAP = 1_000_000_000.0
MIN_QUANTILE = 0.01


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_offset(moment, day_start):
    """Whole-hour offset of moment from day_start, or None if unaligned."""
    delta = (moment - day_start).total_seconds() / 3600.0
    rounded = round(delta)
    if abs(delta - rounded) > 1e-6:
        return None
    return rounded


def safe_log_ratio(viewers, anchor):
    """Log of viewers over anchor, with counts floored at one viewer."""
    return math.log(max(float(viewers), 1.0) / max(float(anchor), 1.0))


def parse_tier(unlock_text):
    """Extract the comparability tier label from a launch's unlock note."""
    if not unlock_text:
        return None
    match = re.search(r"Tier:\s*([A-Za-z\-]+)", unlock_text)
    if not match:
        return None
    return match.group(1).strip().lower()


def hourly_series(rows, day_start):
    """Map whole-hour offsets to viewer counts from hourly rows."""
    series = {}
    for row in rows or []:
        try:
            moment = parse_time(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None:
            continue
        offset = hour_offset(moment, day_start)
        if offset is not None:
            series[offset] = viewers
    return series


def feature_vector(series, anchor):
    """Pre-launch trajectory: clamped log-ratios at the feature offsets."""
    features = {}
    for offset in FEATURE_OFFSETS:
        if offset in series:
            value = safe_log_ratio(series[offset], anchor)
            features[offset] = max(FEATURE_CLAMP[0],
                                   min(FEATURE_CLAMP[1], value))
    return features


def trajectory_distance(target_features, other_features,
                        target_anchor=None, other_anchor=None):
    """Root mean squared difference over shared feature coordinates.

    When both anchors are given and SCALE_WEIGHT is positive, a term for
    the difference in launch scale (log first-hour counts) is added.
    """
    shared = [offset for offset in FEATURE_OFFSETS
              if offset in target_features and offset in other_features]
    if len(shared) < MIN_SHARED_FEATURES:
        return None
    total = sum((target_features[o] - other_features[o]) ** 2
                for o in shared)
    mean_square = total / len(shared)
    if (SCALE_WEIGHT > 0.0 and target_anchor and other_anchor
            and target_anchor > 0 and other_anchor > 0):
        scale_gap = 0.5 * (math.log(float(target_anchor))
                           - math.log(float(other_anchor)))
        mean_square += SCALE_WEIGHT * scale_gap * scale_gap
    return math.sqrt(mean_square)


def weighted_quantile(pairs, level):
    """Weighted quantile of (value, weight) pairs, with interpolation."""
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return None
    threshold = level * total
    cumulative = 0.0
    previous_value = ordered[0][0]
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            if cumulative == weight:
                return value
            # Interpolate within this step for a smoother estimate.
            overshoot = cumulative - threshold
            fraction = 1.0 - overshoot / weight if weight > 0 else 1.0
            return previous_value + fraction * (value - previous_value)
        previous_value = value
    return ordered[-1][0]


def fallback_quantiles(anchor, levels):
    """Flat median at the anchor with a widening spread (baseline-like)."""
    rows = []
    for hour in range(2, 25):
        width = 0.15 * math.sqrt(hour - 1)
        quantiles = []
        for index, _level in enumerate(levels):
            zed = NORMAL_Z[index] if index < len(NORMAL_Z) else 0.0
            quantiles.append(anchor * math.exp(zed * width))
        rows.append((hour, quantiles))
    return rows


def forecast_case(case):
    """Build the 23 forecast rows for one parsed case object."""
    levels = case.get("levels") or list(DEFAULT_LEVELS)
    levels = [float(level) for level in levels]
    if len(levels) != 5 or any(not 0.0 < level < 1.0 for level in levels):
        levels = list(DEFAULT_LEVELS)

    day_start = parse_time(case["day_start"])

    # The anchor: hour 1's print.
    anchor = None
    known = case.get("known")
    if isinstance(known, list) and known:
        anchor = known[0].get("viewers")
    elif isinstance(known, dict):
        anchor = known.get("viewers")
    history = hourly_series(case.get("history_hourly"), day_start)
    if anchor is None and 0 in history:
        anchor = history[0]
    if anchor is None or float(anchor) <= 0:
        anchor = 1.0
    anchor = float(anchor)

    target_features = feature_vector(
        {offset: value for offset, value in history.items() if offset < 0},
        anchor)

    game_name = ""
    game = case.get("game")
    if isinstance(game, dict):
        game_name = game.get("name") or ""

    # Collect comparable launch shapes and their weights.
    shapes = []  # (log-ratio dict by hour, weight)
    for comparable in case.get("comparables") or []:
        try:
            comp_start = parse_time(comparable["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        tier = parse_tier(comparable.get("unlock"))
        if tier == "target-history":
            continue
        if (comparable.get("name") == game_name
                and comparable.get("day_start") == case.get("day_start")):
            continue
        series = hourly_series(comparable.get("hourly"), comp_start)
        if 0 not in series or float(series[0]) <= 0:
            continue
        comp_anchor = float(series[0])
        log_ratios = {}
        for hour in range(2, 25):
            offset = hour - 1
            if offset in series:
                log_ratios[hour] = safe_log_ratio(series[offset], comp_anchor)
        if len(log_ratios) < 12:
            continue
        comp_features = feature_vector(
            {offset: value for offset, value in series.items() if offset < 0},
            comp_anchor)
        distance = trajectory_distance(target_features, comp_features,
                                       anchor, comp_anchor)
        tier_weight = TIER_WEIGHTS.get(tier, 1.0)
        if distance is None:
            weight = tier_weight * 0.1
        else:
            weight = tier_weight * math.exp(-((distance / BANDWIDTH) ** 2))
        shapes.append((log_ratios, weight))

    if not shapes:
        return fallback_quantiles(anchor, levels)

    # Floor weight so every usable comparable contributes a little.
    peak_weight = max(weight for _, weight in shapes)
    floor = FLOOR_WEIGHT_FRACTION * peak_weight
    shapes = [(ratios, max(weight, floor)) for ratios, weight in shapes]

    # Weighted quantiles of the log-ratio shapes, per hour and level.
    median_index = levels.index(0.5) if 0.5 in levels else len(levels) // 2
    log_quantiles = {level: [] for level in levels}
    for hour in range(2, 25):
        pairs = [(ratios[hour], weight) for ratios, weight in shapes
                 if hour in ratios]
        if not pairs:
            pairs = [(0.0, 1.0)]
        for level in levels:
            value = weighted_quantile(pairs, level)
            log_quantiles[level].append(value if value is not None else 0.0)

    # Apply the spread multiplier and floor around the median curve.
    median_curve = log_quantiles[levels[median_index]]
    rows = []
    for position, hour in enumerate(range(2, 25)):
        median = median_curve[position]
        floor_half = SPREAD_FLOOR_BASE + SPREAD_FLOOR_PER_HOUR * (hour - 1)
        quantiles = []
        for level in levels:
            value = log_quantiles[level][position]
            spread = (value - median) * SPREAD_MULTIPLIER
            quantiles.append(median + spread)
        # Enforce the minimum 5%-95% half-width.
        low = quantiles[0]
        high = quantiles[-1]
        if high - low < 2.0 * floor_half:
            middle = 0.5 * (low + high)
            quantiles[0] = middle - floor_half
            quantiles[-1] = middle + floor_half
        # Convert back to viewer counts, enforce order and bounds.
        counts = []
        running = 0.0
        for value in quantiles:
            count = anchor * math.exp(value)
            count = max(MIN_QUANTILE, min(CAP, count))
            count = round(count, 2)
            count = max(count, running)
            running = count
            counts.append(count)
        rows.append((hour, counts))
    return rows


def write_forecast(output_path, rows):
    """Write one forecast file: a single object with the hours key."""
    payload = {"hours": [{"hour": hour, "q": counts}
                         for hour, counts in rows]}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)


def safe_forecast(case):
    """Forecast a case, falling back to a valid flat forecast on error."""
    try:
        return forecast_case(case)
    except Exception:
        anchor = 1.0
        try:
            known = case.get("known")
            if isinstance(known, list) and known:
                anchor = float(known[0].get("viewers") or 1.0)
            elif isinstance(known, dict):
                anchor = float(known.get("viewers") or 1.0)
        except Exception:
            anchor = 1.0
        if anchor <= 0:
            anchor = 1.0
        return fallback_quantiles(anchor, list(DEFAULT_LEVELS))


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    try:
        names = sorted(name for name in os.listdir(input_dir)
                       if name.endswith(".in"))
    except OSError:
        return 0
    for name in names:
        input_path = os.path.join(input_dir, name)
        stem = name[:-3]
        output_path = os.path.join(output_dir, stem + ".out")
        try:
            with open(input_path, "r", encoding="utf-8") as handle:
                case = json.load(handle)
            rows = safe_forecast(case)
        except Exception:
            rows = fallback_quantiles(1.0, list(DEFAULT_LEVELS))
        write_forecast(output_path, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
