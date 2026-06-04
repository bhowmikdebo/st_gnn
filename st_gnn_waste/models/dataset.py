
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path


# ─── Feature columns used as input to the model 
# These match the paper's feature vector F
FEATURE_COLS = [
    "generated_mt",    # past waste (main signal)
    "day_of_week",     # 0–6
    "month",           # 1–12
    "is_weekend",      # 0 or 1
    "is_festival",     # 0 or 1
    "pop_proxy",       # relative population size
]


class WasteDataset(Dataset):
    

    def __init__(self,
                 daily_df: pd.DataFrame,
                 zones:    list,
                 window:   int = 7,
                 scaler:   MinMaxScaler = None,
                 fit_scaler: bool = False):

        self.window = window
        self.zones  = zones
        self.n_zones = len(zones)

        # ── Pivot: rows=date, columns=zone, values=feature ───────────────────
        # We'll build a tensor of shape [T, N, F]
        # where T = number of days, N = zones, F = features

        dates = sorted(daily_df["date"].unique())
        self.T = len(dates)
        self.n_features = len(FEATURE_COLS)

        # Build the full [T, N, F] array
        data_array = np.zeros((self.T, self.n_zones, self.n_features))

        for z_idx, zone in enumerate(zones):
            zone_df = daily_df[daily_df["zone"] == zone].sort_values("date")
            zone_df = zone_df.set_index("date").reindex(dates)

            # Fill missing with forward-fill then backward-fill
            zone_df = zone_df.ffill().bfill()

            data_array[:, z_idx, :] = zone_df[FEATURE_COLS].values

        # ── Scale only generated_mt (feature index 0) with MinMaxScaler ──────
        # Other features are already in [0,1] or small integer range
        waste_values = data_array[:, :, 0].reshape(-1, 1)  # [T*N, 1]

        if fit_scaler:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(waste_values)
        else:
            self.scaler = scaler

        scaled_waste = self.scaler.transform(waste_values).reshape(self.T, self.n_zones)
        data_array[:, :, 0] = scaled_waste

        # Normalise other features to [0,1] range
        # day_of_week (0-6) → divide by 6
        data_array[:, :, 1] /= 6.0
        # month (1-12) → divide by 12
        data_array[:, :, 2] /= 12.0
        # is_weekend, is_festival already 0/1
        # pop_proxy (1-15) → divide by 15
        data_array[:, :, 5] /= 15.0

        self.data = torch.FloatTensor(data_array)  # [T, N, F]

        # ── Valid window indices ───────────────────────────────────────────────
        # We need at least 'window' past days + 1 future day
        self.valid_indices = list(range(window, self.T - 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]

        # x_window: past W days → shape [N, W, F]
        x_window = self.data[t - self.window : t, :, :]   # [W, N, F]
        x_window = x_window.permute(1, 0, 2)              # [N, W, F]

        # y_target: next day's waste for all zones → shape [N]
        y_target = self.data[t + 1, :, 0]                 # [N] (waste only)

        return x_window, y_target


def create_dataloaders(processed_dir: str = "data/processed",
                       window:        int  = 7,
                       batch_size:    int  = 16,
                       train_ratio:   float = 0.70,
                       val_ratio:     float = 0.10):
    """
    Load data, split into train/val/test, return DataLoaders.

    Split is time-based (not random) to prevent data leakage:
      70% train | 10% val | 20% test
    """
    processed_dir = Path(processed_dir)

    import json
    daily_df = pd.read_csv(processed_dir / "daily_waste.csv", parse_dates=["date"])
    with open(processed_dir / "zones.json") as f:
        zones = json.load(f)

    # ── Time-based split ─────────────────────────────────────────────────────
    dates = sorted(daily_df["date"].unique())
    T = len(dates)

    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))

    train_dates = dates[:train_end]
    val_dates   = dates[train_end:val_end]
    test_dates  = dates[val_end:]

    train_df = daily_df[daily_df["date"].isin(train_dates)]
    val_df   = daily_df[daily_df["date"].isin(val_dates)]
    test_df  = daily_df[daily_df["date"].isin(test_dates)]

    print(f"[Dataset] Train: {len(train_dates)} days | "
          f"Val: {len(val_dates)} days | "
          f"Test: {len(test_dates)} days")

    # ── Create datasets (fit scaler on train only to prevent leakage) ─────────
    train_dataset = WasteDataset(train_df, zones, window, fit_scaler=True)
    val_dataset   = WasteDataset(val_df,   zones, window,
                                  scaler=train_dataset.scaler, fit_scaler=False)
    test_dataset  = WasteDataset(test_df,  zones, window,
                                  scaler=train_dataset.scaler, fit_scaler=False)

    # Save scaler for use in the frontend
    joblib.dump(train_dataset.scaler, processed_dir / "scaler.pkl")
    print(f"[Dataset] Saved scaler → {processed_dir}/scaler.pkl")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.scaler, zones
