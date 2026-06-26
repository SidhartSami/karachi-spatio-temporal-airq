"""
Honest regeneration of 07_horizon_degradation.png using proper
RECURSIVE FORECASTING on the existing 1-day LSTM (no retraining).

For each test sample at day t (last day of a 14-day feature window):
  - h=1: predict pm25(t+1) from features(t-13 .. t)        [standard]
  - h=2: predict pm25(t+2) from features(t-12 .. t+1)
  - h=3: predict pm25(t+3) from features(t-11 .. t+2)
  - ...
  - h=7: predict pm25(t+7) from features(t-6  .. t+6)

All shifted-window features are REAL (satellite retrievals + calendar
fields from the modeling dataset). Only the pm25 output is predicted.
This is the standard recursive-forecast approach: each step uses a
sliding window whose last day is the target day of the previous step.

Output:
  - notebooks/outputs/07_horizon_degradation.png   (the fixed figure)
  - notebooks/outputs/07_horizon_metrics.csv       (the source table)
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "processed" / "modeling_dataset.csv"
MODEL_PT = ROOT / "notebooks" / "models" / "lstm_model.pt"
OUT_DIR  = ROOT / "notebooks" / "outputs"

TARGET     = "pm25"
SEQ_LEN    = 14
HORIZON    = 7

plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#0e1117",
    "savefig.facecolor":"#0e1117",
    "axes.edgecolor":   "#888888",
    "axes.labelcolor":  "#cccccc",
    "xtick.color":      "#cccccc",
    "ytick.color":      "#cccccc",
    "text.color":       "#cccccc",
    "grid.color":       "#333333",
    "font.family":      "DejaVu Sans",
})


# ── Architecture (matches 07_lstm_digital_twin.ipynb PM25_LSTM) ──────────────
class PM25_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3,
                 dropout=0.25, horizon=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.horizon     = horizon
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1,
                             batch_first=True, bidirectional=False)
        self.bn1   = nn.BatchNorm1d(hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size,
                             num_layers=max(1, num_layers - 1),
                             batch_first=True, bidirectional=False,
                             dropout=dropout if num_layers > 2 else 0)
        self.bn2   = nn.BatchNorm1d(hidden_size)
        self.drop2 = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, horizon),
        )

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.bn1(out1.permute(0, 2, 1)).permute(0, 2, 1)
        out1 = self.drop1(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.bn2(out2.permute(0, 2, 1)).permute(0, 2, 1)
        out2 = self.drop2(out2)
        attn = torch.softmax(self.attention(out2), dim=1)
        ctx  = (attn * out2).sum(dim=1)
        return self.fc(ctx)


def build_per_station_arrays(df, features):
    """Return per-station:
        station_dates:    (n,) datetime array
        station_features: (n, n_features) raw feature array
        station_target:   (n,) pm25 raw array
    Plus a global (station, date) -> index dict for fast lookup.
    """
    by_station = {}
    lookup = {}  # (station, date_str) -> (station_idx, row_idx)
    for s_idx, (stn, g) in enumerate(df.sort_values("date").groupby("station")):
        g = g.sort_values("date").reset_index(drop=True)
        dates  = g["date"].values
        feats  = g[features].values.astype(np.float32)
        target = g[TARGET].values.astype(np.float32)
        by_station[stn] = {
            "dates": dates,
            "features": feats,
            "target": target,
            "scaler_y": MinMaxScaler().fit(target.reshape(-1, 1)),
            "scaler_x": MinMaxScaler().fit(feats),
        }
        for i, d in enumerate(dates):
            lookup[(stn, pd.Timestamp(d))] = (stn, i)
    return by_station, lookup


def forecast_one_horizon(model, X_batch, device):
    """Single forward pass."""
    with torch.no_grad():
        out = model(torch.from_numpy(X_batch).to(device)).cpu().numpy()
    return out[:, 0]  # horizon=1 model returns 1 value per sample


def main():
    print("Loading data ...")
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    df = df.dropna(subset=[TARGET]).copy()

    print("Loading LSTM checkpoint ...")
    ckpt = torch.load(MODEL_PT, weights_only=False)
    features = ckpt["features"]
    seq_len  = ckpt["seq_len"]
    horizon  = ckpt["horizon"]
    print(f"  features: {features}")
    print(f"  seq_len : {seq_len}, horizon : {horizon}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PM25_LSTM(input_size=len(features), horizon=horizon).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Building per-station feature/target arrays + lookup ...")
    by_station, lookup = build_per_station_arrays(df, features)

    # ── Define test set: 2023 only ──────────────────────────────────────────
    test_dates_by_station = {}
    for stn, info in by_station.items():
        mask = pd.to_datetime(info["dates"]).year == 2023
        test_dates_by_station[stn] = info["dates"][mask]

    # ── For each test sample at station s, last-day-of-window date d,
    #    produce predictions for horizons 1..7 ────────────────────────────────
    print("Recursive forecasting horizons 1..7 ...")
    y_true = []   # list of (station, last_date, [pm25_h1..pm25_h7])
    y_pred = []   # list of (station, last_date, [pm25_h1..pm25_h7])

    for stn, info in by_station.items():
        if stn not in test_dates_by_station:
            continue
        dates  = pd.to_datetime(info["dates"])
        feats  = info["features"]
        target = info["target"]
        scaler_x = info["scaler_x"]
        scaler_y = info["scaler_y"]
        n = len(dates)

        for d in test_dates_by_station[stn]:
            d = pd.Timestamp(d)
            # Find position of d in dates (must be at least SEQ_LEN from start)
            try:
                last_idx = np.where(dates == d)[0][0]
            except IndexError:
                continue
            if last_idx < SEQ_LEN - 1:
                continue  # not enough history

            # For each horizon h=1..7:
            #   build a 14-day window ending at d + (h-1) days,
            #   where every feature is REAL (scaled per-station).
            #   Predict pm25 at d + h days (no peeking at future pm25).
            preds_h = np.zeros(HORIZON, dtype=np.float64)
            truths_h = np.full(HORIZON, np.nan, dtype=np.float64)

            # Get the target_date = d + h days
            target_dates = [d + pd.Timedelta(days=h) for h in range(1, HORIZON + 1)]

            ok = True
            for h in range(1, HORIZON + 1):
                # Window end index in dates: last_idx + (h-1)
                end_idx = last_idx + (h - 1)
                start_idx = end_idx - SEQ_LEN + 1
                if end_idx >= n:
                    ok = False
                    break
                # Look up real features at start_idx..end_idx
                window_feats_raw = feats[start_idx:end_idx + 1]  # (SEQ_LEN, n_features)
                # Scale using this station's scaler
                window_feats_scl = scaler_x.transform(window_feats_raw)
                # Predict
                pred_scaled = forecast_one_horizon(
                    model, window_feats_scl[np.newaxis, ...], device
                )[0]
                # Inverse-transform back to µg/m³
                preds_h[h - 1] = scaler_y.inverse_transform(
                    np.array([[pred_scaled]])
                )[0, 0]
                # Ground truth
                target_date = d + pd.Timedelta(days=h)
                key = (stn, target_date)
                if key in lookup:
                    _, idx = lookup[key]
                    truths_h[h - 1] = target[idx]
                else:
                    truths_h[h - 1] = np.nan

            if not ok:
                continue
            # Need ground truth for ALL 7 horizons
            if np.isnan(truths_h).any() or np.isnan(preds_h).any():
                if len(y_true) < 3:
                    print(f"  [skip] {stn} {d.date()}  truths={truths_h}  preds={preds_h}")
                continue
            y_pred.append((stn, d, preds_h))
            y_true.append((stn, d, truths_h))

    print(f"  test samples with full 7-horizon ground truth: {len(y_pred):,}")

    # ── Compute per-horizon metrics ────────────────────────────────────────
    Y_true = np.array([t[2] for t in y_true])  # (n, 7)
    Y_pred = np.array([t[2] for t in y_pred])  # (n, 7)
    rows = []
    for h in range(HORIZON):
        yt = Y_true[:, h]
        yp = Y_pred[:, h]
        rmse_v = float(np.sqrt(mean_squared_error(yt, yp)))
        mae_v  = float(mean_absolute_error(yt, yp))
        r2_v   = float(r2_score(yt, yp))
        mape_v = float(np.mean(np.abs((yt - yp) / np.maximum(yt, 1))) * 100)
        rows.append({"horizon": h + 1, "RMSE": rmse_v, "MAE": mae_v,
                     "R2": r2_v, "MAPE": mape_v})
        print(f"  h={h+1}:  RMSE={rmse_v:7.3f}   MAE={mae_v:7.3f}   "
              f"R²={r2_v:+.4f}   MAPE={mape_v:6.2f}%")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "07_horizon_metrics.csv", index=False)
    print(f"  ✓ Wrote {OUT_DIR / '07_horizon_metrics.csv'}")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    h_x = metrics_df["horizon"].values

    panels = [
        ("RMSE vs Forecast Horizon", "RMSE (µg/m³)", metrics_df["RMSE"], "#3b8bd6"),
        ("R² vs Forecast Horizon",   "R²",           metrics_df["R2"],   "#e88a3a"),
        ("MAPE vs Forecast Horizon", "MAPE (%)",     metrics_df["MAPE"], "#3ab97a"),
    ]
    for ax, (title, ylabel, ydata, color) in zip(axes, panels):
        ax.plot(h_x, ydata, marker="o", markersize=10, linewidth=2.5,
                color=color, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.fill_between(h_x, ydata, alpha=0.15, color=color)
        ax.set_xticks(h_x)
        ax.set_xlabel("Forecast Horizon (days ahead)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, pad=10)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xlim(0.7, 7.3)
        if title.startswith("R²"):
            ax.axhline(0, color="white", linewidth=0.8, alpha=0.5,
                       linestyle="--")

    fig.suptitle("Forecast Skill Degradation over Horizon — LSTM "
                 "(recursive forecast on real features)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out_png = OUT_DIR / "07_horizon_degradation.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Wrote {out_png}")


if __name__ == "__main__":
    main()