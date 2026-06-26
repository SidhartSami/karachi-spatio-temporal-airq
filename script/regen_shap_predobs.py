"""
Regenerate two figures cleanly:
  - 05_shap_analysis.png         (Random Forest SHAP, was XGBoost + label overlap)
  - 05_predictions_vs_actual.png (4 panels: Observed + RF + XGB + LGB + SVR)

Loads the saved RF/XGB/LGB/SVR models and the modeling_dataset.csv,
re-derives the 13-feature matrix the same way 05_models.ipynb does
(6 consensus + 7 lag/rolling), and rebuilds both PNGs with clean
matplotlib layout (no overlapping x-axis labels).
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib

# Same dark style the repo already uses
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#0e1117",
    "savefig.facecolor": "#0e1117",
    "axes.edgecolor":   "#888888",
    "axes.labelcolor":  "#cccccc",
    "xtick.color":      "#cccccc",
    "ytick.color":      "#cccccc",
    "text.color":       "#cccccc",
    "grid.color":       "#333333",
    "font.family":      "DejaVu Sans",
})

ROOT     = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "processed" / "modeling_dataset.csv"
MODEL_RF = ROOT / "notebooks" / "models" / "random_forest.pkl"
MODEL_XGB = ROOT / "notebooks" / "models" / "xgboost.pkl"
MODEL_LGB = ROOT / "notebooks" / "models" / "lightgbm.pkl"
MODEL_SVR = ROOT / "notebooks" / "models" / "svr.pkl"
OUT_DIR  = ROOT / "notebooks" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "pm25"
# Same 13 features 05_models.ipynb uses
FEATURE_COLS = [
    "Optical_Depth_055", "month_cos", "month", "wind_speed",
    "month_sin", "day_of_week",
    "pm25_lag1", "pm25_lag3", "pm25_lag7",
    "pm25_roll7", "pm25_roll14", "pm25_roll30", "aod_roll7",
]


def load_data():
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    df = df.sort_values(["station", "date"]).reset_index(drop=True)

    # Re-derive lag/rolling with shift(1) — same as notebook cell 4
    for lag in [1, 3, 7]:
        df[f"pm25_lag{lag}"] = df.groupby("station")[TARGET].shift(lag)
    df["pm25_roll7"]  = df.groupby("station")[TARGET].transform(
        lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
    df["pm25_roll14"] = df.groupby("station")[TARGET].transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    df["pm25_roll30"] = df.groupby("station")[TARGET].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    df["aod_roll7"]   = df.groupby("station")["Optical_Depth_055"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean())

    return df


def split(df):
    """Same temporal split as 05_models.ipynb: train 2019-2022, test 2023."""
    df = df.dropna(subset=FEATURE_COLS + [TARGET]).copy()
    train = df[df["date"].dt.year < 2023]
    test  = df[df["date"].dt.year == 2023]
    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET].values
    dates   = test["date"].values
    return X_train, y_train, X_test, y_test, dates


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def gen_shap_figure(rf, X_test, dates):
    """SHAP on the Random Forest (NOT XGBoost — that's the previous bug)."""
    import shap

    rng = np.random.default_rng(42)
    # Use a smaller sample (200) so we don't OOM on Windows
    sample_idx = rng.choice(len(X_test), size=min(200, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]

    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={"wspace": 0.45})

    # LEFT: beeswarm — drawn manually to avoid SHAP's tight default x-label
    plt.sca(axes[0])
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=FEATURE_COLS,
        max_display=15, show=False, plot_type="dot",
    )
    # Force an explicit, short x-axis label (the full string was overlapping
    # the right-panel label even at wspace=0.55, so we keep this terse).
    axes[0].set_xlabel("SHAP value (impact on RF output)", fontsize=11, labelpad=10)
    axes[0].set_title("SHAP Feature Impact (Random Forest)", fontsize=13, pad=12)

    # RIGHT: mean |SHAP| bar chart
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)
    axes[1].barh(
        np.array(FEATURE_COLS)[order],
        mean_abs[order],
        color="#3b8bd6", edgecolor="#1a3a5c",
    )
    axes[1].set_xlabel("mean(|SHAP value|)  (avg impact on RF output)", fontsize=11, labelpad=10)
    axes[1].set_title("Mean |SHAP| Feature Importance", fontsize=13, pad=12)
    axes[1].grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle(
        "SHAP Explainability Analysis — Random Forest PM$_{2.5}$ Model",
        fontsize=15, y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "05_shap_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Wrote {out}")


def gen_predobs_figure(models, X_test, y_test, dates):
    """5 stacked panels: Observed, RF, XGB, LGB, Prophet.

    Prophet and SVR are omitted because they were trained on different
    feature sets (Prophet uses only the date; SVR uses 31 features).
    The three tree-based models share the 13-feature matrix the
    paper documents, so they are the apples-to-apples comparison
    shown here. Prophet and SVR metrics are in Table 3.
    """
    preds = {"Random Forest": models["rf"].predict(X_test)}
    preds["XGBoost"]  = models["xgb"].predict(X_test)
    preds["LightGBM"] = models["lgb"].predict(X_test)

    panel_specs = [("Observed PM2.5", None)] + [(name, preds[name]) for name in preds]
    n = len(panel_specs)

    fig, axes = plt.subplots(n, 1, figsize=(16, 2.6 * n), sharex=True)

    # Reduce to daily citywide mean so the figure is legible
    daily = pd.DataFrame({"date": pd.to_datetime(dates), "y": y_test})
    for label, yhat in panel_specs:
        daily[label] = yhat if yhat is not None else daily["y"]
    daily_mean = daily.groupby("date").mean(numeric_only=True).sort_index()

    for i, (label, _) in enumerate(panel_specs):
        ax = axes[i]
        if label == "Observed PM2.5":
            ax.plot(daily_mean.index, daily_mean[label], color="#dddddd", linewidth=1.4)
            ax.set_ylabel("Observed\nPM$_{2.5}$\n(µg/m$^3$)", fontsize=10)
        else:
            ax.plot(daily_mean.index, daily_mean["y"], color="#888888",
                    linewidth=0.6, alpha=0.6, label="Observed")
            yhat_daily = daily_mean[label]
            metric_r2  = r2(daily_mean["y"], yhat_daily)
            metric_rmse = rmse(daily_mean["y"], yhat_daily)
            ax.plot(daily_mean.index, yhat_daily,
                    color="#3b8bd6", linewidth=1.3,
                    label=f"{label}  (RMSE={metric_rmse:.2f}, R²={metric_r2:.3f})")
            ax.set_ylabel(f"{label}\n(µg/m$^3$)", fontsize=10)
            ax.legend(loc="upper right", fontsize=8, facecolor="#0e1117", edgecolor="#444")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)

    axes[-1].set_xlabel("2023 (test year)")
    fig.suptitle(
        "Predicted vs. Actual PM$_{2.5}$ — Karachi 2023 Test Set",
        fontsize=15, y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = OUT_DIR / "05_predictions_vs_actual.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Wrote {out}")


def main():
    print("Loading data ...")
    df = load_data()
    X_train, y_train, X_test, y_test, dates = split(df)
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    print("Loading models ...")
    models = {
        "rf":  joblib.load(MODEL_RF),
        "xgb": joblib.load(MODEL_XGB),
        "lgb": joblib.load(MODEL_LGB),
        "svr": joblib.load(MODEL_SVR),
    }

    print("Regenerating SHAP figure (Random Forest) ...")
    gen_shap_figure(models["rf"], X_test, dates)

    print("Regenerating predictions-vs-actual figure (with Random Forest) ...")
    gen_predobs_figure(models, X_test, y_test, dates)

    print("Done.")


if __name__ == "__main__":
    main()
