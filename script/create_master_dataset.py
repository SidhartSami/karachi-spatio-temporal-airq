"""
DEPRECATED — REMOVED 2026-06-22
=================================
This script fabricated the PM2.5 target as a hand-coded linear combination of
the very features the model was meant to learn from:

    pm25 = 45 + Optical_Depth_055*80 + no2*15 + so2*10
         + max(0, 15-wind_speed)*2 + humidity_contrib
         + seasonal_contrib + N(0, 5)

See ISSUES_FOUND.md findings:
  C1 — Data leakage / fabrication (the model's headline R²=0.61 came from
       this leakage, not from real learning).
  H1 — np.random.normal(0, 5) was called with no seed, so even the broken
       baseline was non-reproducible.

DO NOT RUN. Use the real pipeline instead:

    python script/rebuild_master_dataset.py [--start 2019-01-01] [--end 2024-12-31]

The orchestrator will use OpenAQ ground sensors, fall back to the US Consulate
anchor, then to MERRA-2 citywide, and never fabricate a target value.
"""
import sys


def main() -> None:
    sys.stderr.write(
        "ERROR: script/create_master_dataset.py was removed on 2026-06-22.\n"
        "       It fabricated the PM2.5 target (see ISSUES_FOUND.md C1).\n"
        "       Use: python script/rebuild_master_dataset.py\n"
    )
    raise NotImplementedError(
        "create_master_dataset.py removed — see deprecation notice in docstring"
    )


if __name__ == "__main__":
    main()
