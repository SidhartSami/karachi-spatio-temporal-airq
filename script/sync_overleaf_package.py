"""
Sync docs/paper.tex -> overleaf_package/paper.tex with path rewrite.
Replace the 16 source PNG names in overleaf_package/figs/ with the
latest versions from notebooks/outputs/, and rebuild the zip.

Path rewrite rule:
  ../notebooks/outputs/X.png      ->  figs/figXX_name.png
where X.png -> figXX_name.png mapping matches the Overleaf folder rename.
"""
from __future__ import annotations
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PAPER  = ROOT / "docs" / "paper.tex"
DST_PAPER  = ROOT / "overleaf_package" / "paper.tex"
SRC_FIGDIR = ROOT / "notebooks" / "outputs"
DST_FIGDIR = ROOT / "overleaf_package" / "figs"
ZIP_OUT    = ROOT / "docs" / "overleaf_package.zip"

# Overleaf figure-name mapping (matches the original rename spec).
NAME_MAP = {
    "03_temporal_trends.png":         "fig01_temporal.png",
    "03_seasonal_analysis.png":       "fig02_seasonal.png",
    "03_station_comparison.png":      "fig03_stations.png",
    "03_who_exceedance.png":          "fig04_who.png",
    "04_consensus_ranking.png":       "fig05_features.png",
    "05_model_comparison.png":        "fig06_models.png",
    "05_shap_analysis.png":           "fig07_shap.png",
    "05_predictions_vs_actual.png":   "fig08_predobs.png",
    "07_lstm_training.png":           "fig09_lstm.png",
    "07_horizon_degradation.png":     "fig10_lstmhorizon.png",
    "06_lisa_map.png":                "fig11_lisa.png",
    "06_spatial_error.png":           "fig12_spatialerror.png",
    "06_zone_analysis.png":           "fig13_zone.png",
    "08_karachi_digital_twin_map.png":"fig14_twin.png",
    "07_digital_twin_scenarios.png":  "fig15_scenarios.png",
    "07_who_attainment.png":          "fig16_attain.png",
}


def sync_paper():
    text = SRC_PAPER.read_text(encoding="utf-8")
    # docs/paper.tex uses ../notebooks/outputs/X.png
    # overleaf_package/paper.tex uses figs/figXX_name.png
    for src_name, dst_name in NAME_MAP.items():
        text = text.replace(f"../notebooks/outputs/{src_name}", f"figs/{dst_name}")
    DST_PAPER.write_text(text, encoding="utf-8")
    print(f"  ✓ Wrote {DST_PAPER}  (path rewrite applied)")


def sync_figs():
    DST_FIGDIR.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in NAME_MAP.items():
        src = SRC_FIGDIR / src_name
        dst = DST_FIGDIR / dst_name
        if not src.exists():
            print(f"  ✗ MISSING: {src}")
            continue
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        print(f"  ✓ {dst_name:32s} {size:>10,} bytes")


def make_zip():
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
    pkg = ROOT / "overleaf_package"
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(pkg.rglob("*")):
            if p.is_file():
                arc = p.relative_to(pkg.parent)  # zip root is the parent dir
                z.write(p, arc.as_posix())
    size = ZIP_OUT.stat().st_size
    print(f"  ✓ Wrote {ZIP_OUT}  ({size:,} bytes)")


def main():
    print("Syncing paper.tex with path rewrite ...")
    sync_paper()
    print("Syncing figures ...")
    sync_figs()
    print("Rebuilding overleaf_package.zip ...")
    make_zip()
    print("Done.")


if __name__ == "__main__":
    main()
