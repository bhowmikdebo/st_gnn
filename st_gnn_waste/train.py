
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import json
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.model   import ST_GNN, normalise_adjacency
from models.dataset import create_dataloaders, FEATURE_COLS

# ─── Config ───────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
PLOTS_DIR     = Path("outputs/training_plots")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW     = 7       # how many past days to use
BATCH_SIZE = 16
EPOCHS     = 60
LR         = 1e-3
HIDDEN_DIM = 64
N_LAYERS   = 2
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Loss Function (from paper: MSE + L2 regularisation) ─────────────────────

def loss_fn(predictions, targets, model, lambda_reg=1e-4):
    """
    L(Θ) = (1/N) Σ (y_true - y_pred)² + λ‖Θ‖²
    """
    mse_loss = nn.MSELoss()(predictions, targets)
    l2_reg   = sum(p.pow(2).sum() for p in model.parameters())
    return mse_loss + lambda_reg * l2_reg


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, adj_norm):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch  = x_batch.to(DEVICE)
        y_batch  = y_batch.to(DEVICE)
        adj_norm = adj_norm.to(DEVICE)

        optimiser.zero_grad()
        preds = model(x_batch, adj_norm)
        loss  = loss_fn(preds, y_batch, model)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, adj_norm, scaler):
    """Run model on a dataloader, return MAE and RMSE in original units (MT)."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch  = x_batch.to(DEVICE)
            adj_norm = adj_norm.to(DEVICE)
            preds    = model(x_batch, adj_norm).cpu().numpy()
            targets  = y_batch.numpy()
            all_preds.append(preds)
            all_targets.append(targets)

    all_preds   = np.concatenate(all_preds,   axis=0)   # [samples, N_zones]
    all_targets = np.concatenate(all_targets, axis=0)

    # Inverse transform to get real MT values
    shape = all_preds.shape
    preds_mt   = scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(shape)
    targets_mt = scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(shape)

    mae  = mean_absolute_error(targets_mt.flatten(), preds_mt.flatten())
    rmse = np.sqrt(mean_squared_error(targets_mt.flatten(), preds_mt.flatten()))

    return mae, rmse, preds_mt, targets_mt


# ─── Baselines ────────────────────────────────────────────────────────────────

def moving_average_baseline(test_loader, scaler, window=7):
    
    all_preds, all_targets = [], []

    for x_batch, y_batch in test_loader:
        # x_batch: [B, N, W, F]; last W values of waste are feature index 0
        waste_window = x_batch[:, :, :, 0].numpy()  # [B, N, W]
        preds        = waste_window.mean(axis=2)     # [B, N]
        all_preds.append(preds)
        all_targets.append(y_batch.numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)

    shape = preds.shape
    preds_mt   = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(shape)
    targets_mt = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(shape)

    mae  = mean_absolute_error(targets_mt.flatten(), preds_mt.flatten())
    rmse = np.sqrt(mean_squared_error(targets_mt.flatten(), preds_mt.flatten()))
    return mae, rmse


def random_forest_baseline(train_loader, test_loader, scaler, n_zones):
    
    print("  [RF] Collecting training data for Random Forest...")
    X_train, y_train = [], []

    for x_batch, y_batch in train_loader:
        B, N, W, F = x_batch.shape
        X_train.append(x_batch.numpy().reshape(B, N * W * F))
        y_train.append(y_batch.numpy().reshape(B, N))

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    rf = RandomForestRegressor(n_estimators=100, max_depth=8,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    X_test, y_test = [], []
    for x_batch, y_batch in test_loader:
        B, N, W, F = x_batch.shape
        X_test.append(x_batch.numpy().reshape(B, N * W * F))
        y_test.append(y_batch.numpy().reshape(B, N))

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    preds = rf.predict(X_test)

    shape = preds.shape
    preds_mt  = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(shape)
    y_test_mt = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(shape)

    mae  = mean_absolute_error(y_test_mt.flatten(), preds_mt.flatten())
    rmse = np.sqrt(mean_squared_error(y_test_mt.flatten(), preds_mt.flatten()))
    return mae, rmse


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_maes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("ST-GNN Training Progress", fontsize=14, fontweight="bold")

    ax1.plot(train_losses, color="#2196F3", linewidth=2)
    ax1.set_title("Training Loss (MSE + L2)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_maes, color="#4CAF50", linewidth=2)
    ax2.set_title("Validation MAE (Metric Tonnes)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (MT)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Plot] Saved training curves → {PLOTS_DIR}/training_curves.png")


def plot_spatial_heatmap(model, zones):
    """
    Visualise the spatial attention weights:
    - Rows = zones being PREDICTED
    - Cols = zones INFLUENCING the prediction
    - Colour = attention strength
    """
    weights = model.get_attention_weights()["spatial"]
    if weights is None:
        print("[Plot] No spatial weights available yet.")
        return

    W = weights.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        W,
        xticklabels=zones,
        yticklabels=zones,
        cmap="Blues",
        ax=ax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Attention Weight (influence strength)"}
    )
    ax.set_title("ST-GNN Spatial Attention Weights\n(How much each zone influences another)", 
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Influencing Zone")
    ax.set_ylabel("Zone Being Predicted")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "spatial_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Plot] Saved spatial heatmap → {PLOTS_DIR}/spatial_heatmap.png")


def plot_temporal_attention(model, zones, window=7):
    """
    Visualise temporal attention: for each zone, which past day matters most.
    """
    weights = model.get_attention_weights()["temporal"]
    if weights is None:
        print("[Plot] No temporal weights available yet.")
        return

    # Average across batches
    W = weights.mean(dim=0).cpu().numpy()  # [N, window]

    day_labels = [f"t-{window - i}" for i in range(window)]
    day_labels[-1] = "t (today)"

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        W,
        xticklabels=day_labels,
        yticklabels=zones,
        cmap="YlOrRd",
        ax=ax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Temporal Attention Weight"}
    )
    ax.set_title("Temporal Attention Weights\n(Which past days matter most per zone)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Past Time Steps")
    ax.set_ylabel("Zone")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "temporal_attention.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Plot] Saved temporal attention → {PLOTS_DIR}/temporal_attention.png")


def plot_prediction_vs_actual(preds_mt, targets_mt, zones, n_samples=60):
    """Plot predicted vs actual waste for a few zones."""
    n_zones_to_show = min(4, len(zones))
    fig, axes = plt.subplots(n_zones_to_show, 1,
                              figsize=(14, 3 * n_zones_to_show))

    for i in range(n_zones_to_show):
        ax = axes[i] if n_zones_to_show > 1 else axes
        ax.plot(targets_mt[:n_samples, i], label="Actual",
                color="#2196F3", linewidth=1.5)
        ax.plot(preds_mt[:n_samples, i],   label="Predicted",
                color="#FF5722", linewidth=1.5, linestyle="--")
        ax.set_title(f"Zone: {zones[i]}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Waste (MT)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("ST-GNN: Predicted vs Actual Waste Generation", 
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "prediction_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Plot] Saved prediction vs actual → {PLOTS_DIR}/prediction_vs_actual.png")


# ─── Main Training Script ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ST-GNN Training Script")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, scaler, zones = create_dataloaders(
        processed_dir=PROCESSED_DIR,
        window=WINDOW,
        batch_size=BATCH_SIZE,
    )
    n_zones = len(zones)

    # ── Load adjacency matrix ─────────────────────────────────────────────────
    adj = np.load(PROCESSED_DIR / "adjacency.npy")
    adj_norm = normalise_adjacency(adj).to(DEVICE)
    print(f"[Data] Adjacency matrix: {adj.shape}")

    # ── Initialise model ──────────────────────────────────────────────────────
    model = ST_GNN(
        n_zones    = n_zones,
        n_features = len(FEATURE_COLS),
        window     = WINDOW,
        hidden_dim = HIDDEN_DIM,
        n_layers   = N_LAYERS,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] ST-GNN parameters: {n_params:,}")

    optimiser = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.5)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[Training] Running {EPOCHS} epochs...")
    train_losses = []
    val_maes     = []
    best_val_mae = float("inf")
    best_epoch   = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimiser, adj_norm)
        val_mae, val_rmse, _, _ = evaluate(model, val_loader, adj_norm, scaler)

        train_losses.append(train_loss)
        val_maes.append(val_mae)
        scheduler.step()

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "val_mae":    val_mae,
                "zones":      zones,
                "window":     WINDOW,
                "n_features": len(FEATURE_COLS),
                "hidden_dim": HIDDEN_DIM,
                "n_layers":   N_LAYERS,
            }, MODELS_DIR / "best_stgnn.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val MAE: {val_mae:.2f} MT | "
                  f"Val RMSE: {val_rmse:.2f} MT"
                  + (" ← best" if epoch == best_epoch else ""))

    print(f"\n[Training] Best model saved at epoch {best_epoch} "
          f"(Val MAE = {best_val_mae:.2f} MT)")

    # ── Load best model for evaluation ───────────────────────────────────────
    checkpoint = torch.load(MODELS_DIR / "best_stgnn.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Run one forward pass to populate attention weights
    for x_b, _ in test_loader:
        with torch.no_grad():
            model(x_b.to(DEVICE), adj_norm)
        break

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n[Evaluation] Computing test metrics...")
    stgnn_mae, stgnn_rmse, preds_mt, targets_mt = evaluate(
        model, test_loader, adj_norm, scaler
    )

    # ── Baseline comparisons ──────────────────────────────────────────────────
    print("[Evaluation] Running Moving Average baseline...")
    ma_mae, ma_rmse = moving_average_baseline(test_loader, scaler, WINDOW)

    print("[Evaluation] Running Random Forest baseline...")
    rf_mae, rf_rmse = random_forest_baseline(train_loader, test_loader, scaler, n_zones)

    # ── Results Table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Model Comparison (Test Set)")
    print("=" * 55)
    print(f"  {'Model':<22}  {'MAE (MT)':>10}  {'RMSE (MT)':>10}")
    print("-" * 55)
    print(f"  {'Moving Average'::<22}  {ma_mae:>10.2f}  {ma_rmse:>10.2f}")
    print(f"  {'Random Forest'::<22}  {rf_mae:>10.2f}  {rf_rmse:>10.2f}")
    print(f"  {'ST-GNN (Ours)'::<22}  {stgnn_mae:>10.2f}  {stgnn_rmse:>10.2f}")
    print("=" * 55)

    ma_improve = (ma_mae - stgnn_mae) / ma_mae * 100
    rf_improve = (rf_mae - stgnn_mae) / rf_mae * 100
    print(f"\n  ST-GNN reduces MAE by {ma_improve:.1f}% vs Moving Average")
    print(f"  ST-GNN reduces MAE by {rf_improve:.1f}% vs Random Forest")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "stgnn":          {"mae": round(stgnn_mae, 3), "rmse": round(stgnn_rmse, 3)},
        "moving_average": {"mae": round(ma_mae,    3), "rmse": round(ma_rmse,    3)},
        "random_forest":  {"mae": round(rf_mae,    3), "rmse": round(rf_rmse,    3)},
    }
    import json
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[Plots] Generating visualisations...")
    plot_training_curves(train_losses, val_maes)
    plot_spatial_heatmap(model, zones)
    plot_temporal_attention(model, zones, WINDOW)
    plot_prediction_vs_actual(preds_mt, targets_mt, zones)

    print("\n[Done] Training complete!")
    print(f"  Best model → {MODELS_DIR}/best_stgnn.pt")
    print(f"  Metrics    → {MODELS_DIR}/metrics.json")
    print(f"  Plots      → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
