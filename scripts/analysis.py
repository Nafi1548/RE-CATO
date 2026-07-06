"""
HVI & Pareto Domination Analysis
=================================
Compares two multi-objective optimisation frameworks across multiple runs.

Usage
-----
Set BO_DIR and BASELINE_DIR at the top of the script, then run:
    python hvi_pareto_analysis.py

Assumptions
-----------
  - Each directory contains one or more CSV files.
  - Every CSV has columns `neg_f1_score` (minimise) and `compute_cost` (minimise).
  - All CSVs within a directory have the same number of rows (iterations).
  - Files are matched by sort order; unpaired extras are ignored.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── USER CONFIGURATION ────────────────────────────────────────────────────────

BO_DIR       = Path("/path/to/bo_results")           # your framework CSVs
BASELINE_DIR = Path("/path/to/post_output_samples")  # baseline CSVs

OBJ_COLS = ["neg_f1_score", "compute_cost"]  # both minimised
HVI_REF  = (0.0, 10_000.0)                  # reference point — must be weakly dominated by no Pareto pt

SAVE_FIGS    = True          # also save PNGs alongside this script
FIG_DPI      = 180
BO_LABEL     = "TRBO-GP (ours)"
BASELINE_LABEL = "HyperMapper (baseline)"

# ── PALETTE ──────────────────────────────────────────────────────────────────

C_BO   = "#2a78d6"
C_BASE = "#eda100"
C_GREEN = "#1baf7a"
C_RED   = "#e34948"

# ── PARETO & HVI UTILITIES ───────────────────────────────────────────────────

def _pareto_update(front: list, pt: list) -> list:
    """Return updated Pareto front after adding pt (incremental, minimisation)."""
    if not front:
        return [pt]
    arr = np.asarray(front)
    npt = np.asarray(pt)
    # Is pt dominated by any existing front point?
    if np.any(np.all(arr <= npt, axis=1) & np.any(arr < npt, axis=1)):
        return front
    # Remove existing points dominated by pt
    keep = ~(np.all(npt <= arr, axis=1) & np.any(npt < arr, axis=1))
    return list(arr[keep]) + [pt]


def hypervolume_2d(front: list, ref: tuple) -> float:
    """Exact 2-D hypervolume for a minimisation Pareto front."""
    if not front:
        return 0.0
    pts = sorted(front, key=lambda p: p[0])
    hv = 0.0
    for i, pt in enumerate(pts):
        w = pts[i + 1][0] - pt[0] if i + 1 < len(pts) else ref[0] - pt[0]
        hv += w * (ref[1] - pt[1])
    return hv


def running_hvi(df: pd.DataFrame, ref: tuple = HVI_REF) -> np.ndarray:
    """HVI after each successive iteration (cumulative best-so-far)."""
    vals = df[OBJ_COLS].values.tolist()
    front, curve = [], []
    for pt in vals:
        front = _pareto_update(front, pt)
        curve.append(hypervolume_2d(front, ref))
    return np.array(curve)


def final_pareto(df: pd.DataFrame) -> np.ndarray:
    """Return the final Pareto front as an (n, 2) array."""
    vals = df[OBJ_COLS].values.tolist()
    front = []
    for pt in vals:
        front = _pareto_update(front, pt)
    return np.asarray(front)


def _is_dominated_by_any(pt: list, candidates: np.ndarray) -> bool:
    return bool(np.any(
        np.all(candidates <= pt, axis=1) & np.any(candidates < pt, axis=1)
    ))


def combined_pareto_stats(bo_front: np.ndarray, base_front: np.ndarray) -> dict:
    """Domination stats for one run-pair."""
    all_pts = np.vstack([bo_front, base_front])
    n_bo, n_all = len(bo_front), len(all_pts)
    dominated = np.zeros(n_all, dtype=bool)
    for i in range(n_all):
        if dominated[i]:
            continue
        for j in range(n_all):
            if i == j or dominated[j]:
                continue
            if np.all(all_pts[j] <= all_pts[i]) and np.any(all_pts[j] < all_pts[i]):
                dominated[i] = True
                break
    src = ["BO"] * n_bo + ["Base"] * len(base_front)
    combined_src = [src[i] for i in range(n_all) if not dominated[i]]
    n_combined = len(combined_src)
    n_bo_in   = combined_src.count("BO")
    bo_dom_by_base = sum(
        _is_dominated_by_any(p, base_front) for p in bo_front.tolist()
    )
    base_dom_by_bo = sum(
        _is_dominated_by_any(p, bo_front) for p in base_front.tolist()
    )
    return {
        "combined_size":      n_combined,
        "bo_in_combined":     n_bo_in,
        "bo_in_combined_pct": 100 * n_bo_in / n_combined,
        "base_dom_by_bo_pct": 100 * base_dom_by_bo / len(base_front),
        "bo_dom_by_base_pct": 100 * bo_dom_by_base / len(bo_front),
        "bo_front":   bo_front,
        "base_front": base_front,
    }


def global_pareto(all_bo: np.ndarray, all_base: np.ndarray):
    """Non-dominated front across all runs for both methods."""
    all_pts = np.vstack([all_bo, all_base])
    n_bo, n_all = len(all_bo), len(all_pts)
    dominated = np.zeros(n_all, dtype=bool)
    for i in range(n_all):
        if dominated[i]:
            continue
        for j in range(n_all):
            if i == j or dominated[j]:
                continue
            if np.all(all_pts[j] <= all_pts[i]) and np.any(all_pts[j] < all_pts[i]):
                dominated[i] = True
                break
    pts = all_pts[~dominated]
    src = (["BO"] * n_bo + ["Base"] * len(all_base))
    src_filt = [src[i] for i in range(n_all) if not dominated[i]]
    idx = np.argsort(pts[:, 0])
    return pts[idx], [src_filt[i] for i in idx]


# ── DATA LOADING ─────────────────────────────────────────────────────────────

def load_csvs(directory: Path) -> list[pd.DataFrame]:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return [pd.read_csv(f) for f in files]


# ── PLOTTING ─────────────────────────────────────────────────────────────────

def _ax_style(ax, xlabel="", ylabel=""):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", labelsize=9)
    ax.set_xlabel(xlabel, fontsize=9, color="#555")
    ax.set_ylabel(ylabel, fontsize=9, color="#555")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_hvi_convergence(ax, bo_hvs: np.ndarray, base_hvs: np.ndarray):
    iters = np.arange(1, bo_hvs.shape[1] + 1)
    for hvs, color, label, ls in [
        (bo_hvs,   C_BO,   BO_LABEL,       "-"),
        (base_hvs, C_BASE, BASELINE_LABEL, "--"),
    ]:
        mean = hvs.mean(axis=0)
        std  = hvs.std(axis=0)
        ax.fill_between(iters, mean - std, mean + std,
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(iters, mean, color=color, lw=2.2, ls=ls, label=label, zorder=3)
    _ax_style(ax, xlabel="Iteration", ylabel="HVI")
    ax.set_xlim(1, bo_hvs.shape[1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=8.5, frameon=False)


def plot_pareto_scatter(ax, run_stats: list, gf_pts: np.ndarray, gf_src: list):
    # All individual Pareto points
    bo_pts   = np.vstack([s["bo_front"]   for s in run_stats])
    base_pts = np.vstack([s["base_front"] for s in run_stats])
    # Convert: x = F1 (negate neg_f1), y = cost
    ax.scatter(-bo_pts[:, 0],   bo_pts[:, 1],
               c=C_BO,   alpha=0.55, s=28, marker="o",
               linewidths=0.5, edgecolors=C_BO, label=BO_LABEL, zorder=3)
    ax.scatter(-base_pts[:, 0], base_pts[:, 1],
               c=C_BASE, alpha=0.55, s=28, marker="^",
               linewidths=0.5, edgecolors=C_BASE, label=BASELINE_LABEL, zorder=3)
    # Global non-dominated step-line
    sorted_gf = sorted(zip(-gf_pts[:, 0], gf_pts[:, 1], gf_src),
                       key=lambda t: -t[0])   # descending F1
    sx = [p[0] for p in sorted_gf]
    sy = [p[1] for p in sorted_gf]
    step_x, step_y = [], []
    for i, (x, y, _) in enumerate(sorted_gf):
        step_x.append(x);  step_y.append(y)
        if i + 1 < len(sorted_gf):
            step_x.append(sorted_gf[i + 1][0]);  step_y.append(y)
    ax.plot(step_x, step_y, color="#888", lw=1.2, ls=(0, (3, 2)),
            zorder=2, label="Global non-dominated front")
    _ax_style(ax, xlabel="F1 score  (↑)", ylabel="Compute cost  (↓)")
    ax.legend(fontsize=8, frameon=False)


def plot_domination_bars(ax, run_stats: list):
    labels   = [f"Run {i}" for i in range(len(run_stats))]
    bo_in_c  = [s["bo_in_combined_pct"]  for s in run_stats]
    base_dom = [s["base_dom_by_bo_pct"]  for s in run_stats]
    bo_dom   = [s["bo_dom_by_base_pct"]  for s in run_stats]
    n = len(labels)
    x = np.arange(n)
    w = 0.22
    ax.bar(x - w,   bo_in_c,  width=w, color=C_BO,    alpha=0.82,
           label="BO in combined front", zorder=3, clip_on=False)
    ax.bar(x,       base_dom, width=w, color=C_GREEN,  alpha=0.82,
           label="Baseline pts dominated by BO", zorder=3, clip_on=False)
    ax.bar(x + w,   bo_dom,   width=w, color=C_RED,    alpha=0.80,
           label="BO pts dominated by baseline", zorder=3, clip_on=False)
    ax.set_xticks(x);  ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _ax_style(ax, ylabel="Percentage (%)")
    ax.legend(fontsize=7.5, frameon=False, ncol=1)


# ── AGGREGATE STATS PRINTER ───────────────────────────────────────────────────

def print_stats(run_stats: list, bo_hvs: np.ndarray, base_hvs: np.ndarray,
                gf_pts: np.ndarray, gf_src: list):
    n_runs = len(run_stats)
    bo_in  = [s["bo_in_combined_pct"]  for s in run_stats]
    bd     = [s["base_dom_by_bo_pct"]  for s in run_stats]
    bod    = [s["bo_dom_by_base_pct"]  for s in run_stats]
    n_bo_g = gf_src.count("BO")
    n_g    = len(gf_src)

    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  HVI & Pareto Domination Summary  ({n_runs} runs)")
    print(sep)
    print(f"  Final HVI")
    print(f"    {BO_LABEL:30s}  {bo_hvs[:,-1].mean():.2f}  ±{bo_hvs[:,-1].std():.2f}")
    print(f"    {BASELINE_LABEL:30s}  {base_hvs[:,-1].mean():.2f}  ±{base_hvs[:,-1].std():.2f}")
    print()
    print(f"  Pareto domination (per-run mean ± std)")
    print(f"    BO pts in combined front      {np.mean(bo_in):.1f}%  ±{np.std(bo_in):.1f} pp")
    print(f"    Baseline pts dominated by BO  {np.mean(bd):.1f}%  ±{np.std(bd):.1f} pp")
    print(f"    BO pts dominated by baseline   {np.mean(bod):.1f}%  ±{np.std(bod):.1f} pp")
    print()
    print(f"  Global cross-run combined Pareto  ({n_g} pts total)")
    print(f"    BO:       {n_bo_g}  ({100*n_bo_g/n_g:.1f}%)")
    print(f"    Baseline: {n_g-n_bo_g}  ({100*(n_g-n_bo_g)/n_g:.1f}%)")
    print()
    print(f"  Per-run breakdown")
    hdr = f"    {'Run':>4}  {'BO in comb':>10}  {'Base dom by BO':>14}  {'BO dom by base':>14}"
    print(hdr)
    for i, s in enumerate(run_stats):
        print(f"    {i:>4}  {s['bo_in_combined_pct']:>10.1f}%  "
              f"{s['base_dom_by_bo_pct']:>13.1f}%  "
              f"{s['bo_dom_by_base_pct']:>13.1f}%")
    print(sep + "\n")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Load
    print(f"Loading BO results from:       {BO_DIR}")
    bo_dfs   = load_csvs(BO_DIR)
    print(f"Loading baseline results from: {BASELINE_DIR}")
    base_dfs = load_csvs(BASELINE_DIR)
    n_runs = min(len(bo_dfs), len(base_dfs))
    bo_dfs, base_dfs = bo_dfs[:n_runs], base_dfs[:n_runs]
    print(f"Using {n_runs} run(s), {len(bo_dfs[0])} iterations each.\n")

    # HVI curves
    print("Computing running HVI curves...")
    bo_hvs   = np.stack([running_hvi(df) for df in bo_dfs])   # (n_runs, iters)
    base_hvs = np.stack([running_hvi(df) for df in base_dfs])

    # Per-run Pareto stats
    print("Computing Pareto domination stats...")
    run_stats = [
        combined_pareto_stats(final_pareto(bo_dfs[i]), final_pareto(base_dfs[i]))
        for i in range(n_runs)
    ]

    # Global Pareto
    all_bo   = np.vstack([s["bo_front"]   for s in run_stats])
    all_base = np.vstack([s["base_front"] for s in run_stats])
    gf_pts, gf_src = global_pareto(all_bo, all_base)

    # Print
    print_stats(run_stats, bo_hvs, base_hvs, gf_pts, gf_src)

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                          left=0.07, right=0.97, top=0.94, bottom=0.08)

    ax_hvi  = fig.add_subplot(gs[0, :])    # full top row
    ax_scat = fig.add_subplot(gs[1, 0])
    ax_dom  = fig.add_subplot(gs[1, 1])

    # Titles
    ax_hvi.set_title("HVI convergence  —  mean ± 1 std across runs",
                     fontsize=11, fontweight="normal", pad=8, color="#333")
    ax_scat.set_title("Final Pareto fronts  —  all runs combined",
                      fontsize=10, fontweight="normal", pad=6, color="#333")
    ax_dom.set_title("Per-run Pareto domination",
                     fontsize=10, fontweight="normal", pad=6, color="#333")

    plot_hvi_convergence(ax_hvi,  bo_hvs,   base_hvs)
    plot_pareto_scatter(ax_scat, run_stats, gf_pts, gf_src)
    plot_domination_bars(ax_dom, run_stats)

    # Aggregate annotation on HVI plot
    hvi_diff = bo_hvs[:, -1].mean() - base_hvs[:, -1].mean()
    ax_hvi.annotate(
        f"Final HVI:  {BO_LABEL} {bo_hvs[:,-1].mean():.1f} ± {bo_hvs[:,-1].std():.1f}  |  "
        f"{BASELINE_LABEL} {base_hvs[:,-1].mean():.1f} ± {base_hvs[:,-1].std():.1f}",
        xy=(0.01, 0.03), xycoords="axes fraction",
        fontsize=8.5, color="#666",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd", lw=0.7)
    )

    # Global Pareto annotation on scatter
    n_bo_g = gf_src.count("BO")
    n_g    = len(gf_src)
    ax_scat.annotate(
        f"Global front: {n_bo_g}/{n_g} pts from {BO_LABEL}  ({100*n_bo_g/n_g:.0f}%)",
        xy=(0.02, 0.97), xycoords="axes fraction",
        va="top", fontsize=8, color="#555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd", lw=0.7)
    )

    # Mean lines on domination bars
    bo_in_mean   = np.mean([s["bo_in_combined_pct"] for s in run_stats])
    base_dom_mean = np.mean([s["base_dom_by_bo_pct"] for s in run_stats])
    for val, color in [(bo_in_mean, C_BO), (base_dom_mean, C_GREEN)]:
        ax_dom.axhline(val, color=color, lw=1.0, ls=":", alpha=0.6, zorder=1)

    ax_dom.annotate(
        f"Mean: BO in front {bo_in_mean:.1f}%  |  Baseline dominated {base_dom_mean:.1f}%",
        xy=(0.02, 0.97), xycoords="axes fraction",
        va="top", fontsize=7.5, color="#555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd", lw=0.7)
    )

    if SAVE_FIGS:
        out_path = Path(__file__).parent / "hvi_pareto_analysis.png"
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight",
                    facecolor="white")
        print(f"Figure saved → {out_path}")

    plt.show()


if __name__ == "__main__":
    main()