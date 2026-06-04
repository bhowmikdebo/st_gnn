"""
utils/predictor.py
==================
Loads the trained ST-GNN model and runs next-day predictions.

Used by the Streamlit frontend — keeps all model loading/inference
logic separate from the UI code.
"""

import torch
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import ST_GNN, normalise_adjacency
from models.dataset import FEATURE_COLS
from utils.explainability import compute_wpi, explain_prediction, compute_zone_baselines


# ─── Predictor Class ──────────────────────────────────────────────────────────

class WastePredictor:
    """
    Wraps the trained ST-GNN for inference.

    Usage:
        predictor = WastePredictor()
        predictor.load()
        results = predictor.predict_next_day(recent_data, context)
    """

    def __init__(self,
                 models_dir:    str = "models/saved",
                 processed_dir: str = "data/processed"):

        self.models_dir    = Path(models_dir)
        self.processed_dir = Path(processed_dir)
        self.device        = torch.device("cpu")   # CPU for inference
        self.model         = None
        self.scaler        = None
        self.adj_norm      = None
        self.zones         = None
        self.baselines     = None

    def load(self):
        """Load model, scaler, adjacency, and zone baselines."""

        # ── Load trained model ────────────────────────────────────────────────
        ckpt_path = self.models_dir / "best_stgnn.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {ckpt_path}. "
                "Please run train.py first."
            )

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.zones = checkpoint["zones"]

        self.model = ST_GNN(
            n_zones    = len(self.zones),
            n_features = checkpoint["n_features"],
            window     = checkpoint["window"],
            hidden_dim = checkpoint["hidden_dim"],
            n_layers   = checkpoint["n_layers"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.window = checkpoint["window"]

        # ── Load scaler ───────────────────────────────────────────────────────
        scaler_path = self.processed_dir / "scaler.pkl"
        self.scaler = joblib.load(scaler_path)

        # ── Load adjacency ────────────────────────────────────────────────────
        adj = np.load(self.processed_dir / "adjacency.npy")
        self.adj_norm = normalise_adjacency(adj)

        # ── Load baselines ────────────────────────────────────────────────────
        daily_df = pd.read_csv(
            self.processed_dir / "daily_waste.csv",
            parse_dates=["date"]
        )
        self.baselines = compute_zone_baselines(daily_df)
        self.daily_df  = daily_df

        print(f"[Predictor] Model loaded successfully. "
              f"Window={self.window}, Zones={len(self.zones)}")
        return self

    def get_recent_window(self,
                          real_time_entries: Optional[Dict[str, List[float]]] = None
                          ) -> np.ndarray:
        """
        Build the input window [N_zones, window, n_features] from:
          1. Historical data (last W days from the processed CSV)
          2. Any real-time entries the user has typed in (override historical)

        Args:
            real_time_entries : {zone_name: [list of daily MT values, newest last]}

        Returns:
            numpy array [N_zones, window, n_features]
        """
        # Get last W days from historical data
        dates = sorted(self.daily_df["date"].unique())
        last_dates = dates[-self.window:]

        window_data = np.zeros((len(self.zones), self.window, len(FEATURE_COLS)))

        for z_idx, zone in enumerate(self.zones):
            zone_df = (self.daily_df[self.daily_df["zone"] == zone]
                       .sort_values("date")
                       .set_index("date"))

            for t_idx, date in enumerate(last_dates):
                if date in zone_df.index:
                    row = zone_df.loc[date]
                    for f_idx, feat in enumerate(FEATURE_COLS):
                        window_data[z_idx, t_idx, f_idx] = row[feat]

        # Override with real-time entries if provided
        if real_time_entries:
            for zone, values in real_time_entries.items():
                if zone not in self.zones:
                    continue
                z_idx = self.zones.index(zone)
                # values is a list of daily MT, newest last
                n_vals = min(len(values), self.window)
                for t_offset in range(n_vals):
                    # Place from the end of the window
                    t_idx = self.window - n_vals + t_offset
                    # Override the waste value (feature 0)
                    window_data[z_idx, t_idx, 0] = values[t_offset]

        # Scale waste values (feature 0) with the trained scaler
        waste_vals = window_data[:, :, 0].reshape(-1, 1)
        scaled     = self.scaler.transform(waste_vals)
        window_data[:, :, 0] = scaled.reshape(len(self.zones), self.window)

        # Normalise other features (same as in dataset.py)
        window_data[:, :, 1] /= 6.0    # day_of_week
        window_data[:, :, 2] /= 12.0   # month
        # features 3,4 (is_weekend, is_festival) already 0/1
        window_data[:, :, 5] /= 15.0   # pop_proxy

        return window_data

    def predict_next_day(self,
                         real_time_entries: Optional[Dict[str, List[float]]] = None,
                         festival_info:     Optional[Dict] = None
                         ) -> Dict:
        """
        Run next-day prediction for all zones.

        Args:
            real_time_entries : {zone: [daily MT values for past W days]}
            festival_info     : {"is_festival": True/False,
                                  "festival_name": "...",
                                  "affected_zones": ["all"] or list of zone names}

        Returns:
            dict with:
                predictions    : {zone: predicted_mt}
                wpi_results    : list of WPI dicts (one per zone)
                explanations   : list of explanation dicts (one per zone)
                attention      : {temporal: ..., spatial: ...}
        """
        # ── Build input window ────────────────────────────────────────────────
        window_np = self.get_recent_window(real_time_entries)

        # Apply festival boost to the window if specified
        if festival_info and festival_info.get("is_festival"):
            affected = festival_info.get("affected_zones", ["all"])
            for z_idx, zone in enumerate(self.zones):
                if "all" in affected or zone in affected:
                    # Boost last time step's waste by festival factor
                    window_np[z_idx, -1, 0] *= (1 + festival_info.get("spike", 0.25))

        # ── Run model ─────────────────────────────────────────────────────────
        x_tensor = torch.FloatTensor(window_np).unsqueeze(0)  # [1, N, W, F]

        with torch.no_grad():
            preds_scaled = self.model(x_tensor, self.adj_norm)  # [1, N]

        preds_scaled_np = preds_scaled.squeeze(0).numpy()       # [N]

        # Inverse scale back to MT
        preds_mt = self.scaler.inverse_transform(
            preds_scaled_np.reshape(-1, 1)
        ).flatten()

        # Apply festival boost to predictions if specified
        if festival_info and festival_info.get("is_festival"):
            affected = festival_info.get("affected_zones", ["all"])
            spike    = festival_info.get("spike", 0.25)
            for z_idx, zone in enumerate(self.zones):
                if "all" in affected or zone in affected:
                    preds_mt[z_idx] *= (1 + spike)

        # ── Get attention weights ─────────────────────────────────────────────
        attn = self.model.get_attention_weights()
        temporal_w = (attn["temporal"].squeeze(0).numpy()
                      if attn["temporal"] is not None else None)
        spatial_w  = (attn["spatial"].numpy()
                      if attn["spatial"] is not None else None)

        # ── Compute WPI ───────────────────────────────────────────────────────
        wpi_results = compute_wpi(preds_mt, self.baselines)

        # ── Build explanations ────────────────────────────────────────────────
        explanations = []
        for z_idx, zone in enumerate(self.zones):
            context = {
                "is_festival":   festival_info.get("is_festival", False) if festival_info else False,
                "festival_name": festival_info.get("festival_name", "") if festival_info else "",
                "is_weekend":    False,   # frontend can pass this
            }
            exp = explain_prediction(
                zone          = zone,
                predicted_mt  = float(preds_mt[z_idx]),
                baseline_mt   = self.baselines.get(zone, 1.0),
                temporal_weights = temporal_w,
                spatial_weights  = spatial_w,
                zones         = self.zones,
                zone_idx      = z_idx,
                context       = context,
            )
            explanations.append(exp)

        # ── Package results ───────────────────────────────────────────────────
        return {
            "predictions":  {zone: round(float(preds_mt[i]), 2)
                             for i, zone in enumerate(self.zones)},
            "wpi_results":  wpi_results,
            "explanations": explanations,
        }
