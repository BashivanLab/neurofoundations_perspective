"""
plot_results.py
---------------
Loads counts.json produced by fetch_counts.py and generates
publication-ready figures.

Usage:
    python plot_results.py                    # reads counts.json, saves PNGs
    python plot_results.py --data my.json     # custom input file
    python plot_results.py --fmt pdf          # save as PDF instead of PNG
    python plot_results.py --dpi 300          # resolution (default 200)
    python plot_results.py --out figures/     # output directory

Denominators
------------
  TOTAL  — raw subfield total (e.g. all "working memory"[tiab] papers).
            Shown for context only.  NOT used for percentages because most
            of those papers never name a specific paradigm in their abstract.

  UNION  — OR of all paradigm queries.  The number of papers that mention
            at least one of our tracked paradigms.  Used as the denominator
            for concentration percentages and the Gini / HHI metrics.

  Coverage = UNION / TOTAL — what fraction of the subfield our search covers.

Output figures
--------------
  00_coverage.png               — % of subfield literature covered by named paradigms
  01_summary_concentration.png  — Top-1 / Top-3 bar, Gini, HHI (3-panel)
  02_lorenz_curves.png          — Lorenz curves for all subfields
  03_cumulative_concentration.png — Cumulative % vs number of paradigms
  04_working_memory.png         — Bar + pie for working memory
  05_decision_making.png        — Bar + pie for decision making
  06_spatial_navigation.png     — Bar + pie for spatial navigation
  07_attention.png              — Bar + pie for attention
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from queries import SUBFIELD_COLORS

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def gini(values):
    v = sorted(float(x) for x in values)
    n = len(v)
    if n == 0 or sum(v) == 0:
        return 0.0
    cum = sum((i + 1) * vi for i, vi in enumerate(v))
    return (2 * cum - (n + 1) * sum(v)) / (n * sum(v))


def normalized_hhi(values):
    total = sum(values)
    if total == 0:
        return 0.0
    shares = [v / total for v in values]
    hhi = sum(s ** 2 for s in shares)
    n = len(shares)
    return (hhi - 1 / n) / (1 - 1 / n) if n > 1 else 1.0


def subfield_stats(paradigm_counts: dict, union: int) -> dict:
    """
    paradigm_counts : {label: count}  — individual paradigm counts only
                      (no TOTAL, no UNION).
    union           : UNION count — used as the percentage denominator.
                      If 0 or smaller than the largest individual paradigm
                      (UNION query failed), falls back to sum of paradigm
                      counts and sets union_fallback=True in the result.
    """
    items = sorted(paradigm_counts.items(), key=lambda x: -x[1])
    if not items:
        return {}

    paradigm_sum = sum(paradigm_counts.values())
    fallback = False

    # UNION should be >= every individual paradigm count.  If it's 0 or
    # implausibly small the long OR query likely failed URL-encoding; fall back.
    if union == 0 or union < items[0][1]:
        union = paradigm_sum
        fallback = True

    top1_label, top1_n = items[0]
    top3_sum = sum(v for _, v in items[:3])
    vals = list(paradigm_counts.values())

    # How many paradigms to reach 50 % / 80 % of the denominator
    cum = 0
    n_50 = n_80 = len(items)
    for i, (_, v) in enumerate(items):
        cum += v
        if cum / union >= 0.50 and n_50 == len(items):
            n_50 = i + 1
        if cum / union >= 0.80 and n_80 == len(items):
            n_80 = i + 1
            break

    return {
        "items_sorted":   items,
        "union_fallback": fallback,
        "union":          union,
        "top1_label":     top1_label,
        "top1_pct":       top1_n / union * 100,
        "top3_pct":       top3_sum / union * 100,
        "gini":           gini(vals),
        "hhi_norm":       normalized_hhi(vals),
        "n_50":           n_50,
        "n_80":           n_80,
    }


# ---------------------------------------------------------------------------
# Figure 00 — Coverage
# ---------------------------------------------------------------------------

def plot_coverage(subfields, totals, unions, out_path):
    """
    Two-panel figure showing:
      Left  — absolute counts: TOTAL vs UNION (stacked bar)
      Right — coverage percentage (UNION / TOTAL * 100)
    """
    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(14, 5))

    colors = [SUBFIELD_COLORS[s] for s in subfields]
    x = np.arange(len(subfields))

    # --- left: stacked bar UNION vs unnamed remainder ---
    named   = [unions[s] for s in subfields]
    unnamed = [max(totals[s] - unions[s], 0) for s in subfields]

    b1 = ax_abs.bar(x, named,   color=colors, alpha=0.9, label="Named-paradigm studies (UNION)")
    b2 = ax_abs.bar(x, unnamed, bottom=named, color="#dddddd", alpha=0.8, label="Other / unnamed")
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(subfields, fontsize=11)
    ax_abs.set_ylabel("Number of PubMed publications", fontsize=12)
    ax_abs.set_title("How many subfield studies name\na tracked paradigm?",
                     fontsize=13, fontweight="bold")
    ax_abs.legend(fontsize=10)

    for bar, n in zip(b1, named):
        ax_abs.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{n:,}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    for bar, u, n in zip(b1, unnamed, named):
        ax_abs.text(bar.get_x() + bar.get_width() / 2,
                    n + u / 2,
                    f"{u:,}", ha="center", va="center",
                    fontsize=9, color="#555555")

    # --- right: coverage % ---
    pcts = [unions[s] / totals[s] * 100 if totals[s] else 0 for s in subfields]
    bars = ax_pct.bar(subfields, pcts, color=colors, alpha=0.85)
    ax_pct.set_ylabel("Coverage (%)", fontsize=12)
    ax_pct.set_ylim(0, max(pcts) * 1.35)
    ax_pct.set_title("Coverage: named-paradigm studies\nas % of subfield total",
                     fontsize=13, fontweight="bold")
    for bar, pct in zip(bars, pcts):
        ax_pct.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(pcts) * 0.02,
                    f"{pct:.1f}%", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=args_dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 04-07 — Individual subfield bar + pie
# ---------------------------------------------------------------------------

def plot_subfield(paradigm_counts, title, out_path, color, union):
    """
    paradigm_counts : individual paradigm counts (no TOTAL/UNION).
    union           : UNION count — used as the percentage denominator.
    """
    items_asc = sorted(paradigm_counts.items(), key=lambda x: x[1])  # ascending for barh
    names  = [x[0] for x in items_asc]
    counts = [x[1] for x in items_asc]

    base_rgba = mcolors.to_rgba(color)
    n = len(items_asc)
    bar_colors = [(*base_rgba[:3], 0.35 + 0.65 * (i / max(n - 1, 1)))
                  for i in range(n)]

    fig, (ax_bar, ax_pie) = plt.subplots(
        1, 2, figsize=(16, 7),
        gridspec_kw={"width_ratios": [2, 1]}
    )

    # --- bar chart ---
    bars = ax_bar.barh(range(n), counts,
                       color=bar_colors, edgecolor="white", linewidth=0.5)
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels(names, fontsize=11)
    ax_bar.set_xlabel("Number of PubMed publications [tiab]", fontsize=12)
    ax_bar.set_title(
        f"{title}\nParadigm Distribution  "
        f"(denominator = UNION = {union:,} papers naming any paradigm)",
        fontsize=13, fontweight="bold"
    )

    for bar, cnt in zip(bars, counts):
        pct = cnt / union * 100
        ax_bar.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:,}  ({pct:.1f}% of named)",
            va="center", fontsize=10,
        )
    ax_bar.set_xlim(0, max(counts) * 1.40)

    # --- pie chart: top-3 vs rest of named ---
    items_desc = sorted(paradigm_counts.items(), key=lambda x: -x[1])
    top3   = items_desc[:3]
    rest   = union - sum(v for _, v in top3)

    pie_labels = [k for k, _ in top3] + ["All other\nnamed paradigms"]
    pie_sizes  = [v for _, v in top3] + [max(rest, 0)]
    pie_colors = [mcolors.to_rgba(color, a) for a in [1.0, 0.70, 0.45]] + ["#dddddd"]

    _, _, autotexts = ax_pie.pie(
        pie_sizes, labels=pie_labels, autopct="%1.1f%%",
        colors=pie_colors, startangle=90,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax_pie.set_title("Top-3 share of\nnamed-paradigm studies",
                     fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=args_dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 01 — Summary 3-panel
# ---------------------------------------------------------------------------

def plot_summary(all_stats, subfields, out_path):
    colors = [SUBFIELD_COLORS[s] for s in subfields]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(subfields))
    w = 0.35

    top1 = [all_stats[s]["top1_pct"] for s in subfields]
    top3 = [all_stats[s]["top3_pct"] for s in subfields]
    b1 = axes[0].bar(x - w / 2, top1, w, label="Top-1 paradigm",  color=colors, alpha=0.95)
    b2 = axes[0].bar(x + w / 2, top3, w, label="Top-3 paradigms", color=colors, alpha=0.45)
    axes[0].set_ylabel("% of named-paradigm studies (UNION)", fontsize=12)
    axes[0].set_title("Paradigm Concentration\n(denominator = UNION)",
                      fontsize=13, fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(subfields, fontsize=10)
    axes[0].legend(fontsize=10)
    for b, v in zip(b1, top1):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.5,
                     f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    for b, v in zip(b2, top3):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.5,
                     f"{v:.1f}%", ha="center", fontsize=9)

    ginis = [all_stats[s]["gini"] for s in subfields]
    bars  = axes[1].bar(subfields, ginis, color=colors, alpha=0.85)
    axes[1].set_ylabel("Gini coefficient", fontsize=12)
    axes[1].set_title("Inequality of Paradigm Usage\n(Gini; 1 = maximum skew)",
                      fontsize=13, fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.5, color="gray", ls="--", alpha=0.5, label="Moderate inequality")
    axes[1].legend(fontsize=9)
    for b, v in zip(bars, ginis):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.01,
                     f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")

    hhis = [all_stats[s]["hhi_norm"] for s in subfields]
    bars = axes[2].bar(subfields, hhis, color=colors, alpha=0.85)
    axes[2].set_ylabel("Normalized HHI", fontsize=12)
    axes[2].set_title("Market Concentration Index\n(Normalized HHI)",
                      fontsize=13, fontweight="bold")
    axes[2].set_ylim(0, max(hhis) * 1.4)
    axes[2].axhline(0.25, color="gray", ls="--", alpha=0.5, label="High concentration")
    axes[2].legend(fontsize=9)
    for b, v in zip(bars, hhis):
        axes[2].text(b.get_x() + b.get_width()/2, v + max(hhis) * 0.02,
                     f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=args_dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 02 — Lorenz curves
# ---------------------------------------------------------------------------

def plot_lorenz(all_data, subfields, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    for sf in subfields:
        vals = sorted(all_data[sf].values())
        n = len(vals); total = sum(vals)
        if total == 0: continue
        cum = np.insert(np.cumsum(vals) / total, 0, 0)
        pop = np.arange(n + 1) / n
        ax.plot(pop, cum, "-o", color=SUBFIELD_COLORS[sf],
                label=sf, linewidth=2, markersize=4)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect equality")
    ax.set_xlabel("Cumulative share of paradigm types (sorted by frequency)", fontsize=12)
    ax.set_ylabel("Cumulative share of named-paradigm publications", fontsize=12)
    ax.set_title("Lorenz Curves of Paradigm Usage\nacross Neuroscience Subfields",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=args_dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 03 — Cumulative concentration
# ---------------------------------------------------------------------------

def plot_cumulative(all_data, unions, subfields, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for sf in subfields:
        vals_sorted = sorted(all_data[sf].values(), reverse=True)
        denom = unions[sf]
        if denom == 0: continue
        cum = np.cumsum(vals_sorted) / denom * 100
        ax.plot(range(1, len(cum) + 1), cum, "-o",
                color=SUBFIELD_COLORS[sf], label=sf, linewidth=2.5, markersize=6)

    ax.axhline(50, color="gray", ls="--", alpha=0.5)
    ax.axhline(80, color="gray", ls=":",  alpha=0.5)
    ax.text(0.6, 51, "50%", fontsize=9, color="gray")
    ax.text(0.6, 81, "80%", fontsize=9, color="gray")
    ax.set_xlabel("Number of top paradigms (ranked by frequency)", fontsize=12)
    ax.set_ylabel("Cumulative % of named-paradigm studies (UNION denominator)", fontsize=12)
    ax.set_title("Cumulative Paradigm Concentration\n"
                 "(How many paradigms account for X% of named-paradigm studies?)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0.5, max(len(all_data[s]) for s in subfields) + 0.5)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=args_dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

args_dpi = 200


def main():
    global args_dpi

    parser = argparse.ArgumentParser(description="Plot PubMed paradigm counts.")
    parser.add_argument("--data", default="counts.json")
    parser.add_argument("--out",  default="figures")
    parser.add_argument("--fmt",  default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi",  type=int, default=200)
    args = parser.parse_args()

    args_dpi = args.dpi
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_raw  = json.loads(Path(args.data).read_text())

    # Separate reserved keys from paradigm counts
    data   = {}   # {subfield: {paradigm: count}}
    totals = {}   # {subfield: TOTAL count}
    unions = {}   # {subfield: UNION count}

    for sf, counts in data_raw.items():
        totals[sf] = counts.get("TOTAL") or 0
        unions[sf] = counts.get("UNION") or 0
        data[sf]   = {k: v for k, v in counts.items()
                      if k not in ("TOTAL", "UNION") and v is not None}

    subfields = list(data.keys())

    # Compute stats using UNION as denominator
    all_stats = {sf: subfield_stats(data[sf], unions[sf]) for sf in subfields}

    # Use the (possibly corrected) union value from stats for all downstream work
    for sf in subfields:
        if all_stats[sf]:
            unions[sf] = all_stats[sf]["union"]

    # Console summary
    print("\n" + "=" * 70)
    print("PARADIGM CONCENTRATION — SUMMARY STATISTICS")
    print("(percentages relative to UNION = named-paradigm studies)")
    print("=" * 70)
    for sf in subfields:
        s = all_stats[sf]
        if not s:
            print(f"\n{sf}  — no data")
            continue
        cov = unions[sf] / totals[sf] * 100 if totals[sf] else 0
        fallback_note = "  ⚠ UNION query failed; using sum of paradigms as fallback" if s.get("union_fallback") else ""
        print(f"\n{sf}")
        print(f"  Subfield total (TOTAL)          : {totals[sf]:>10,}")
        print(f"  Named-paradigm studies (UNION)  : {unions[sf]:>10,}  ({cov:.1f}% coverage){fallback_note}")
        print(f"  Top paradigm : {s['top1_label']}  ({s['top1_pct']:.1f}% of UNION)")
        print(f"  Top-3 share  : {s['top3_pct']:.1f}% of UNION")
        print(f"  Gini         : {s['gini']:.3f}")
        print(f"  HHI (norm)   : {s['hhi_norm']:.4f}")
        print(f"  Paradigms for 50% of UNION : {s['n_50']}")
        print(f"  Paradigms for 80% of UNION : {s['n_80']}")

    def fp(name):
        return out_dir / f"{name}.{args.fmt}"

    print("\nGenerating figures …")

    # Only plot subfields that have valid stats
    valid_subfields = [sf for sf in subfields if all_stats.get(sf)]
    if len(valid_subfields) < len(subfields):
        skipped = set(subfields) - set(valid_subfields)
        print(f"\n  ⚠ Skipping figures for subfields with no data: {', '.join(skipped)}")

    plot_coverage(subfields, totals, unions, fp("00_coverage"))

    labels = {
        "Working Memory":    "04_working_memory",
        "Decision Making":   "05_decision_making",
        "Spatial Navigation":"06_spatial_navigation",
        "Attention":         "07_attention",
    }
    for sf in valid_subfields:
        plot_subfield(
            dict(data[sf]),
            sf,
            fp(labels.get(sf, sf.lower().replace(" ", "_"))),
            SUBFIELD_COLORS.get(sf, "#607D8B"),
            union=unions[sf],
        )

    plot_summary(all_stats, valid_subfields, fp("01_summary_concentration"))
    plot_lorenz(data, valid_subfields, fp("02_lorenz_curves"))
    plot_cumulative(data, unions, valid_subfields, fp("03_cumulative_concentration"))

    print(f"\n✓ All figures saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
