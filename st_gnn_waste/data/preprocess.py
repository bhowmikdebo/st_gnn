"""
preprocess.py
=============
Transforms the Chennai annual waste CSV into a daily time-series
that the ST-GNN can learn from.

What this file does:
  1. Loads the raw annual data (15 zones × 3 years)
  2. Fixes known data issues (e.g. Madavaram 1111 typo → 111)
  3. Expands each annual total into realistic daily values by
     injecting day-of-week rhythms, seasonal patterns, festival spikes
  4. Builds the spatial adjacency matrix (which zones are neighbours)
  5. Saves everything as clean CSVs + a numpy adjacency matrix

Run:
    python data/preprocess.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# ─── Paths ────────────────────────────────────────────────────────────────────
RAW_CSV    = Path("data/raw/SolidWasteGeneratedCollectedProcessedDataChennai2015to2018.csv")
OUT_DIR    = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Zone metadata: approximate lat/lon centroids for Chennai zones ───────────
ZONE_META = {
    "Thiruvotriyur":    {"lat": 13.157, "lon": 80.313, "type": "industrial"},
    "Manali":           {"lat": 13.165, "lon": 80.264, "type": "industrial"},
    "Madavaram":        {"lat": 13.142, "lon": 80.245, "type": "residential"},
    "Thondiyarpet":     {"lat": 13.118, "lon": 80.296, "type": "commercial"},
    "Royapuram":        {"lat": 13.112, "lon": 80.293, "type": "commercial"},
    "Thiru Vi Ka Nagar":{"lat": 13.105, "lon": 80.261, "type": "residential"},
    "Ambattur":         {"lat": 13.114, "lon": 80.155, "type": "industrial"},
    "Anna Nagar":       {"lat": 13.085, "lon": 80.210, "type": "residential"},
    "Teynampet":        {"lat": 13.043, "lon": 80.250, "type": "commercial"},
    "Kodambakkam":      {"lat": 13.051, "lon": 80.224, "type": "residential"},
    "Valasaravakkam":   {"lat": 13.050, "lon": 80.178, "type": "residential"},
    "Alandur":          {"lat": 13.002, "lon": 80.200, "type": "residential"},
    "Adayar":           {"lat": 13.001, "lon": 80.256, "type": "residential"},
    "Perungudi":        {"lat": 12.961, "lon": 80.240, "type": "industrial"},
    "Sholinganalur":    {"lat": 12.900, "lon": 80.228, "type": "industrial"},
}

# ─── Known Chennai festivals with rough dates (month, day) ───────────────────
# Waste typically spikes 20-40% during these periods
FESTIVAL_CALENDAR = [
    {"name": "Pongal",        "month": 1,  "day": 14, "duration": 3, "spike": 0.30},
    {"name": "Republic Day",  "month": 1,  "day": 26, "duration": 1, "spike": 0.10},
    {"name": "Tamil New Year","month": 4,  "day": 14, "duration": 2, "spike": 0.25},
    {"name": "Eid",           "month": 4,  "day": 21, "duration": 2, "spike": 0.20},  # approx
    {"name": "Independence",  "month": 8,  "day": 15, "duration": 1, "spike": 0.10},
    {"name": "Ganesh Chathurthi","month": 9,"day": 10,"duration": 3, "spike": 0.35},
    {"name": "Dussehra",      "month": 10, "day": 5,  "duration": 2, "spike": 0.25},
    {"name": "Deepavali",     "month": 10, "day": 24, "duration": 3, "spike": 0.40},
    {"name": "Christmas",     "month": 12, "day": 25, "duration": 2, "spike": 0.15},
    {"name": "New Year",      "month": 12, "day": 31, "duration": 2, "spike": 0.20},
]


def load_and_clean_raw_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the raw CSV and fix known data quality issues.
    Returns a tidy DataFrame with columns:
        zone, year, generated_mt, collected_mt
    """
    df = pd.read_csv(csv_path)

    # Rename columns to shorter names for easier handling
    df = df.rename(columns={
        "Zone Name": "zone",
        "Total quantum of MSW generated in the city (in Metric tonnes) - 2015-16":  "gen_2015",
        "Total quantum of MSW generated in the city (in Metric tonnes) - 2016-17":  "gen_2016",
        "Total quantum of MSW generated in the city (in Metric tonnes) - 2017-18":  "gen_2017",
        "Total quantum of MSW collected by the ULB or private operator (in Metric tonnes) - 2015-16": "col_2015",
        "Total quantum of MSW collected by the ULB or private operator (in Metric tonnes) - 2016-17": "col_2016",
        "Total quantum of MSW collected by the ULB or private operator (in Metric tonnes) - 2017-18": "col_2017",
    })

    df = df[["zone", "gen_2015", "gen_2016", "gen_2017",
             "col_2015", "col_2016", "col_2017"]].copy()

    # ── Fix outlier: Madavaram 2016-17 shows 1111 instead of 111 ──────────────
    # (visible in raw data; likely a data entry error — median of adjacent years ≈ 104)
    mask = (df["zone"] == "Madavaram") & (df["gen_2016"] > 500)
    if mask.any():
        corrected = int((df.loc[df["zone"]=="Madavaram","gen_2015"].values[0] +
                         df.loc[df["zone"]=="Madavaram","gen_2017"].values[0]) / 2)
        df.loc[mask, "gen_2016"] = corrected
        df.loc[mask, "col_2016"] = corrected
        print(f"[Preprocess] Fixed Madavaram 2016 outlier → {corrected} MT")

    return df


def annual_to_daily(zone: str, annual_mt: float, year: int,
                    zone_type: str, rng: np.random.Generator) -> pd.Series:
    """
    Convert one zone's annual total (metric tonnes) into 365 daily values.

    Logic:
      - Base daily = annual / 365
      - Day-of-week multiplier: weekends generate ~10% more waste
      - Monsoon multiplier (June–Sept): slight dip due to operational issues
      - Festival spikes: short bursts of 20-40% extra
      - Light Gaussian noise for realism
    """
    dates = pd.date_range(start=f"{year}-04-01", periods=365, freq="D")
    base  = annual_mt / 365.0

    daily = np.full(365, base)

    # Day-of-week effect
    for i, d in enumerate(dates):
        if d.weekday() >= 5:          # Saturday / Sunday
            daily[i] *= 1.10
        elif d.weekday() == 0:         # Monday (after weekend)
            daily[i] *= 1.05

    # Seasonal: monsoon (Jun-Sep) → slight dip; summer (Mar-May) → slight rise
    for i, d in enumerate(dates):
        if d.month in [6, 7, 8, 9]:    # monsoon — collection disruption
            daily[i] *= 0.95
        elif d.month in [3, 4, 5]:     # summer — more outdoor activity
            daily[i] *= 1.05

    # Festival spikes
    for fest in FESTIVAL_CALENDAR:
        for i, d in enumerate(dates):
            if d.month == fest["month"]:
                day_diff = abs(d.day - fest["day"])
                if day_diff < fest["duration"]:
                    daily[i] *= (1 + fest["spike"] * (1 - day_diff / fest["duration"]))

    # Zone-type effect: commercial zones spike more on weekdays
    if zone_type == "commercial":
        for i, d in enumerate(dates):
            if d.weekday() < 5:
                daily[i] *= 1.08
    elif zone_type == "industrial":
        for i, d in enumerate(dates):
            if d.weekday() >= 5:       # less waste on industrial zone weekends
                daily[i] *= 0.85

    # Add realistic Gaussian noise (±3%)
    noise = rng.normal(1.0, 0.03, 365)
    daily = daily * noise

    # Ensure no negative values
    daily = np.clip(daily, 0.1, None)

    return pd.Series(daily, index=dates, name=zone)


def build_daily_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand all zones across all 3 years into a single daily DataFrame.
    Columns: date, zone, generated_mt, day_of_week, month, is_weekend,
             is_festival, festival_name, zone_type, population_proxy
    """
    rng = np.random.default_rng(seed=42)   # fixed seed for reproducibility
    all_records = []

    # Approximate population proxy (1 = smallest zone)
    pop_proxy = {z: i+1 for i, z in enumerate(ZONE_META.keys())}

    for _, row in raw_df.iterrows():
        zone = row["zone"]
        zone_type = ZONE_META.get(zone, {}).get("type", "residential")

        for year_label, col_gen in [(2015, "gen_2015"), (2016, "gen_2016"), (2017, "gen_2017")]:
            annual_total = row[col_gen]
            daily_series = annual_to_daily(zone, annual_total, year_label,
                                           zone_type, rng)

            for date, val in daily_series.items():
                # Check if this date is near a festival
                is_fest, fest_name = False, ""
                for fest in FESTIVAL_CALENDAR:
                    if date.month == fest["month"] and abs(date.day - fest["day"]) < fest["duration"]:
                        is_fest   = True
                        fest_name = fest["name"]
                        break

                all_records.append({
                    "date":           date,
                    "zone":           zone,
                    "generated_mt":   round(val, 3),
                    "day_of_week":    date.weekday(),        # 0=Mon … 6=Sun
                    "month":          date.month,
                    "is_weekend":     int(date.weekday() >= 5),
                    "is_festival":    int(is_fest),
                    "festival_name":  fest_name,
                    "zone_type":      zone_type,
                    "pop_proxy":      pop_proxy.get(zone, 1),
                })

    df = pd.DataFrame(all_records)
    df = df.sort_values(["zone", "date"]).reset_index(drop=True)
    return df


def build_adjacency_matrix(zones: list) -> np.ndarray:
    
    n = len(zones)
    coords = np.array([[ZONE_META[z]["lat"], ZONE_META[z]["lon"]] for z in zones])

    # Compute pairwise Euclidean distances (in degrees, approx)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dlat = coords[i, 0] - coords[j, 0]
            dlon = coords[i, 1] - coords[j, 1]
            dist_matrix[i, j] = np.sqrt(dlat**2 + dlon**2)

    sigma = dist_matrix[dist_matrix > 0].std()
    kappa = dist_matrix[dist_matrix > 0].mean()  # threshold = mean distance

    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                adj[i, j] = 1.0                  # self-loop
            elif dist_matrix[i, j] <= kappa:
                adj[i, j] = np.exp(-(dist_matrix[i, j]**2) / (sigma**2))

    return adj


def main():
    print("=" * 60)
    print("  Chennai Waste Data Preprocessor")
    print("=" * 60)

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    print("\n[1/4] Loading raw data...")
    raw_df = load_and_clean_raw_data(RAW_CSV)
    print(f"      Loaded {len(raw_df)} zones")

    # ── Step 2: Expand to daily ───────────────────────────────────────────────
    print("\n[2/4] Expanding annual data to daily time-series...")
    daily_df = build_daily_dataframe(raw_df)
    print(f"      Generated {len(daily_df):,} daily records "
          f"({daily_df['zone'].nunique()} zones × ~1095 days)")

    # ── Step 3: Save daily CSV ────────────────────────────────────────────────
    out_path = OUT_DIR / "daily_waste.csv"
    daily_df.to_csv(out_path, index=False)
    print(f"\n[3/4] Saved daily data → {out_path}")

    # ── Step 4: Build and save adjacency matrix ───────────────────────────────
    zones = sorted(daily_df["zone"].unique().tolist())
    adj   = build_adjacency_matrix(zones)
    np.save(OUT_DIR / "adjacency.npy", adj)

    # Also save zone list so we always know the order
    with open(OUT_DIR / "zones.json", "w") as f:
        json.dump(zones, f, indent=2)

    print(f"[4/4] Saved adjacency matrix ({len(zones)}×{len(zones)}) → data/processed/adjacency.npy")
    print(f"      Saved zone list → data/processed/zones.json")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n── Data Summary ──────────────────────────────────────────")
    print(daily_df.groupby("zone")["generated_mt"]
          .agg(["mean", "min", "max"])
          .round(2)
          .to_string())
    print("\n[Done] Preprocessing complete.\n")


if __name__ == "__main__":
    main()
