"""
utils/explainability.py
=======================
Helper functions for:
  1. Waste Pressure Index (WPI) — converts MT predictions into action levels
  2. Explainability summaries — human-readable reasons from attention weights
  3. Decision recommendations — what the municipality should do per zone
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ─── WPI Thresholds ───────────────────────────────────────────────────────────
# Based on historical Chennai data: mean ≈ 340 MT/zone/year → ~0.93 MT/day
# Thresholds divide into 5 operational levels

WPI_LEVELS = [
    {"label": "Critical",  "min_factor": 1.30, "color": "#D32F2F", "emoji": "🔴",
     "action": "Dispatch extra trucks immediately. Alert supervisors. All hands on deck."},
    {"label": "High",      "min_factor": 1.15, "color": "#F57C00", "emoji": "🟠",
     "action": "Increase collection frequency. Pre-position standby trucks."},
    {"label": "Elevated",  "min_factor": 1.05, "color": "#FBC02D", "emoji": "🟡",
     "action": "Monitor closely. Ensure standard trucks are ready."},
    {"label": "Normal",    "min_factor": 0.90, "color": "#388E3C", "emoji": "🟢",
     "action": "Standard collection schedule. No extra action needed."},
    {"label": "Low",       "min_factor": 0.00, "color": "#1976D2", "emoji": "🔵",
     "action": "Reduce trips if possible. Reallocate resources to higher zones."},
]


def compute_wpi(predictions_mt: np.ndarray,
                zone_baselines: Dict[str, float]) -> List[Dict]:
    """
    Compute Waste Pressure Index for each zone.

    WPI = predicted_MT / zone_baseline_MT

    Args:
        predictions_mt : array of shape [N_zones] with predicted waste in MT
        zone_baselines : dict {zone_name: baseline_daily_MT}

    Returns:
        List of dicts, one per zone, with WPI score + level info
    """
    results = []
    zones = list(zone_baselines.keys())

    for i, zone in enumerate(zones):
        pred   = float(predictions_mt[i])
        base   = zone_baselines[zone]
        wpi    = pred / base if base > 0 else 1.0

        # Find level
        level_info = WPI_LEVELS[-1]   # default: Low
        for level in WPI_LEVELS:
            if wpi >= level["min_factor"]:
                level_info = level
                break

        results.append({
            "zone":        zone,
            "predicted_mt": round(pred, 2),
            "baseline_mt":  round(base, 2),
            "wpi_score":    round(wpi, 3),
            "wpi_percent":  round((wpi - 1.0) * 100, 1),  # % above/below baseline
            "level":        level_info["label"],
            "color":        level_info["color"],
            "emoji":        level_info["emoji"],
            "action":       level_info["action"],
        })

    return results


def explain_prediction(zone: str,
                       predicted_mt: float,
                       baseline_mt: float,
                       temporal_weights: np.ndarray,
                       spatial_weights: np.ndarray,
                       zones: List[str],
                       zone_idx: int,
                       context: Dict) -> Dict:
    """
    Generate a human-readable explanation for a zone's prediction.

    Args:
        zone              : zone name
        predicted_mt      : predicted waste in MT
        baseline_mt       : expected normal waste
        temporal_weights  : [N_zones, window] attention weights
        spatial_weights   : [N_zones, N_zones] adjacency attention
        zones             : list of all zone names
        zone_idx          : index of this zone in the zones list
        context           : dict with {is_festival, festival_name, is_weekend, ...}

    Returns:
        dict with explanation text and contributing factors
    """
    reasons   = []
    factors   = {}
    deviation = predicted_mt - baseline_mt
    pct_dev   = (deviation / baseline_mt * 100) if baseline_mt > 0 else 0

    # ── Reason 1: Temporal pattern ────────────────────────────────────────────
    if temporal_weights is not None and zone_idx < len(temporal_weights):
        z_temp_w = temporal_weights[zone_idx]          # [window]
        top_day  = int(np.argmax(z_temp_w))
        window   = len(z_temp_w)
        days_ago = window - top_day
        if days_ago == 0:
            day_str = "yesterday"
        else:
            day_str = f"{days_ago} day{'s' if days_ago > 1 else ''} ago"
        reasons.append(
            f"Temporal pattern: The model is most influenced by waste data from "
            f"**{day_str}** (attention weight: {z_temp_w[top_day]:.2f})"
        )
        factors["top_influential_day"] = day_str

    # ── Reason 2: Spatial spillover ───────────────────────────────────────────
    if spatial_weights is not None and zone_idx < len(spatial_weights):
        row       = spatial_weights[zone_idx]           # influence ON this zone
        # Exclude self
        row_no_self = row.copy()
        row_no_self[zone_idx] = 0
        if row_no_self.max() > 0.05:
            top_neighbour_idx = int(np.argmax(row_no_self))
            top_neighbour     = zones[top_neighbour_idx]
            top_weight        = float(row_no_self[top_neighbour_idx])
            reasons.append(
                f"Spatial influence: **{top_neighbour}** is the strongest "
                f"neighbouring zone influencing this prediction "
                f"(spatial weight: {top_weight:.2f})"
            )
            factors["top_spatial_neighbour"] = top_neighbour

    # ── Reason 3: Festival / holiday ─────────────────────────────────────────
    if context.get("is_festival"):
        fest_name = context.get("festival_name", "festival")
        reasons.append(
            f"Festival effect: **{fest_name}** typically increases waste "
            f"generation by 20–40%"
        )
        factors["festival"] = fest_name

    # ── Reason 4: Weekend / weekday ───────────────────────────────────────────
    if context.get("is_weekend"):
        reasons.append(
            "Weekend effect: Residential waste tends to be ~10% higher on weekends"
        )
        factors["day_type"] = "weekend"

    # ── Reason 5: Deviation summary ───────────────────────────────────────────
    if abs(pct_dev) > 5:
        direction = "above" if pct_dev > 0 else "below"
        reasons.append(
            f"Overall deviation: Predicted waste is **{abs(pct_dev):.1f}% {direction}** "
            f"the historical baseline of {baseline_mt:.1f} MT"
        )

    return {
        "zone":          zone,
        "predicted_mt":  round(predicted_mt, 2),
        "baseline_mt":   round(baseline_mt, 2),
        "deviation_pct": round(pct_dev, 1),
        "reasons":       reasons,
        "factors":       factors,
    }


def compute_zone_baselines(daily_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the average daily waste per zone from historical data.
    Used as the denominator in WPI calculation.
    """
    baselines = (
        daily_df.groupby("zone")["generated_mt"]
        .mean()
        .to_dict()
    )
    return baselines
