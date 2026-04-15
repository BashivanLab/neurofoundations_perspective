"""
5_plot_results.py
-----------------
Generates publication-ready figures from pubmed_counts.json.
Identical in spirit to the plot_results.py in paradigm_analysis/ but
also overlays the LLM mention frequencies alongside PubMed counts.

Input:
    pubmed_counts.json        (from 4_fetch_task_counts.py)
    task_frequencies.json     (from 3_aggregate_tasks.py)   [optional]

Output figures (in figures/ by default):
    00_coverage.png
    01_summary_concentration.png
    02_lorenz_curves.png
    03_cumulative_concentration.png
    04_working_memory.png  …  07_attention.png
    08_llm_vs_pubmed.png      (scatter: LLM mention rank vs PubMed count rank)

Usage:
    python 5_plot_results.py
    python 5_plot_results.py --data pubmed_counts.json --llm task_frequencies.json
    python 5_plot_results.py --fmt pdf --dpi 300
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

SUBFIELD_COLORS = {
    "Working Memory":    "#2196F3",
    "Decision Making":   "#FF5722",
    "Spatial Navigation":"#4CAF50",
    "Attention":         "#9C27B0",
}

_DPI = 200


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def gini(values):
    v = sorted(float(x) for x in values)
    n = len(v)
    if n == 0 or sum(v) == 0:
        return 0.0
    return (2 * sum((i+1)*vi for i,vi in enumerate(v)) - (n+1)*sum(v)) / (n*sum(v))


def norm_hhi(values):
    t = sum(values)
    if t == 0: return 0.0
    s = [v/t for v in values]
    n = len(s)
    return (sum(x**2 for x in s) - 1/n) / (1 - 1/n) if n > 1 else 1.0


def subfield_stats(counts: dict, union: int) -> dict:
    items = sorted(counts.items(), key=lambda x: -x[1])
    if not items: return {}
    vals  = list(counts.values())
    psum  = sum(vals)
    # Fallback if union query failed
    fallback = union == 0 or union < items[0][1]
    if fallback:
        union = psum
    top3 = sum(v for _,v in items[:3])
    cum, n50, n80 = 0, len(items), len(items)
    for i, (_,v) in enumerate(items):
        cum += v
        if cum/union >= 0.5 and n50 == len(items): n50 = i+1
        if cum/union >= 0.8 and n80 == len(items): n80 = i+1; break
    return dict(
        items_sorted=items, union=union, union_fallback=fallback,
        top1_label=items[0][0], top1_pct=items[0][1]/union*100,
        top3_pct=top3/union*100,
        gini=gini(vals), hhi_norm=norm_hhi(vals),
        n_50=n50, n_80=n80,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fp(out_dir, name, fmt):
    return out_dir / f"{name}.{fmt}"


def plot_coverage(subfields, totals, unions, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = [SUBFIELD_COLORS[s] for s in subfields]
    x = np.arange(len(subfields))
    named   = [unions[s] for s in subfields]
    unnamed = [max(totals[s]-unions[s], 0) for s in subfields]
    b1 = a1.bar(x, named,   color=colors, alpha=0.9, label="Named-task studies (UNION)")
    a1.bar(x, unnamed, bottom=named, color="#ddd", alpha=0.8, label="Other / unnamed")
    a1.set_xticks(x); a1.set_xticklabels(subfields, fontsize=11)
    a1.set_ylabel("PubMed publications"); a1.legend(fontsize=10)
    a1.set_title("How many subfield studies name\na tracked task?", fontsize=13, fontweight="bold")
    for bar, n, u in zip(b1, named, unnamed):
        a1.text(bar.get_x()+bar.get_width()/2, n/2, f"{n:,}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    pcts = [unions[s]/totals[s]*100 if totals[s] else 0 for s in subfields]
    bars = a2.bar(subfields, pcts, color=colors, alpha=0.85)
    a2.set_ylabel("Coverage (%)"); a2.set_ylim(0, max(pcts)*1.35)
    a2.set_title("Named-task studies as\n% of empirical subfield total", fontsize=13, fontweight="bold")
    for b, p in zip(bars, pcts):
        a2.text(b.get_x()+b.get_width()/2, b.get_height()+max(pcts)*0.02,
                f"{p:.1f}%", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=_DPI, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_subfield(counts, title, path, color, union):
    items = sorted(counts.items(), key=lambda x: x[1])
    names, vals = [i[0] for i in items], [i[1] for i in items]
    n = len(items)
    base = mcolors.to_rgba(color)
    bar_colors = [(*base[:3], 0.35+0.65*(i/max(n-1,1))) for i in range(n)]
    fig, (ab, ap) = plt.subplots(1, 2, figsize=(16, max(6, n*0.38)),
                                  gridspec_kw={"width_ratios":[2,1]})
    bars = ab.barh(range(n), vals, color=bar_colors, edgecolor="white", linewidth=0.5)
    ab.set_yticks(range(n)); ab.set_yticklabels(names, fontsize=10)
    ab.set_xlabel("PubMed publications [tiab]")
    ab.set_title(f"{title}\n(denominator = UNION = {union:,} papers naming any task)",
                 fontsize=13, fontweight="bold")
    for bar, v in zip(bars, vals):
        pct = v/union*100
        ab.text(bar.get_width()+max(vals)*0.01, bar.get_y()+bar.get_height()/2,
                f"{v:,}  ({pct:.1f}%)", va="center", fontsize=9)
    ab.set_xlim(0, max(vals)*1.40)
    # Pie: top-3 vs rest
    top3  = sorted(counts.items(), key=lambda x:-x[1])[:3]
    rest  = union - sum(v for _,v in top3)
    sizes = [v for _,v in top3] + [max(rest,0)]
    pcolors = [mcolors.to_rgba(color, a) for a in [1.0,0.70,0.45]] + ["#ddd"]
    _, _, ats = ap.pie(sizes, labels=[k for k,_ in top3]+["All other\ntasks"],
                       autopct="%1.1f%%", colors=pcolors, startangle=90,
                       textprops={"fontsize":10})
    for at in ats: at.set_fontsize(11); at.set_fontweight("bold")
    ap.set_title("Top-3 task\nconcentration", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=_DPI, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_summary(stats, subfields, path):
    colors = [SUBFIELD_COLORS[s] for s in subfields]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x, w = np.arange(len(subfields)), 0.35
    t1 = [stats[s]["top1_pct"] for s in subfields]
    t3 = [stats[s]["top3_pct"] for s in subfields]
    b1 = axes[0].bar(x-w/2, t1, w, color=colors, alpha=0.95, label="Top-1")
    b2 = axes[0].bar(x+w/2, t3, w, color=colors, alpha=0.45, label="Top-3")
    axes[0].set_xticks(x); axes[0].set_xticklabels(subfields, fontsize=10)
    axes[0].set_ylabel("% of named-task studies (UNION)"); axes[0].legend(fontsize=10)
    axes[0].set_title("Task Concentration\n(denominator = UNION)", fontsize=13, fontweight="bold")
    for b,v in zip(b1,t1): axes[0].text(b.get_x()+b.get_width()/2, v+.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    for b,v in zip(b2,t3): axes[0].text(b.get_x()+b.get_width()/2, v+.5, f"{v:.1f}%", ha="center", fontsize=9)
    gs = [stats[s]["gini"] for s in subfields]
    bars = axes[1].bar(subfields, gs, color=colors, alpha=0.85)
    axes[1].set_ylim(0,1); axes[1].axhline(0.5,color="gray",ls="--",alpha=0.5)
    axes[1].set_ylabel("Gini coefficient")
    axes[1].set_title("Inequality of Task Usage\n(Gini; 1 = maximum skew)", fontsize=13, fontweight="bold")
    for b,v in zip(bars,gs): axes[1].text(b.get_x()+b.get_width()/2, v+.01, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    hs = [stats[s]["hhi_norm"] for s in subfields]
    bars = axes[2].bar(subfields, hs, color=colors, alpha=0.85)
    axes[2].set_ylim(0, max(hs)*1.4); axes[2].axhline(0.25,color="gray",ls="--",alpha=0.5)
    axes[2].set_ylabel("Normalized HHI")
    axes[2].set_title("Market Concentration Index\n(Normalized HHI)", fontsize=13, fontweight="bold")
    for b,v in zip(bars,hs): axes[2].text(b.get_x()+b.get_width()/2, v+max(hs)*.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=_DPI, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_lorenz(data, subfields, path):
    fig, ax = plt.subplots(figsize=(8,8))
    for sf in subfields:
        v = sorted(data[sf].values()); t = sum(v); n = len(v)
        if t == 0: continue
        ax.plot(np.arange(n+1)/n, np.insert(np.cumsum(v)/t,0,0), "-o",
                color=SUBFIELD_COLORS[sf], label=sf, lw=2, ms=4)
    ax.plot([0,1],[0,1],"k--",alpha=0.4,label="Perfect equality")
    ax.set_xlabel("Cumulative share of task types (sorted by frequency)")
    ax.set_ylabel("Cumulative share of publications")
    ax.set_title("Lorenz Curves of Task Usage\nacross Neuroscience Subfields",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left"); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=_DPI, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_cumulative(data, unions, subfields, path):
    fig, ax = plt.subplots(figsize=(10,6))
    for sf in subfields:
        vals = sorted(data[sf].values(), reverse=True)
        denom = unions[sf]
        if denom == 0: continue
        ax.plot(range(1,len(vals)+1), np.cumsum(vals)/denom*100, "-o",
                color=SUBFIELD_COLORS[sf], label=sf, lw=2.5, ms=6)
    ax.axhline(50,color="gray",ls="--",alpha=0.5); ax.axhline(80,color="gray",ls=":",alpha=0.5)
    ax.text(0.6,51,"50%",fontsize=9,color="gray"); ax.text(0.6,81,"80%",fontsize=9,color="gray")
    ax.set_xlabel("Number of top tasks (ranked by frequency)")
    ax.set_ylabel("Cumulative % of named-task studies")
    ax.set_title("Cumulative Task Concentration", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.set_ylim(0,110); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(path, dpi=_DPI, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_llm_vs_pubmed(data, llm_freqs, subfields, path):
    """Scatter plot: LLM mention rank vs PubMed count rank per subfield."""
    fig, axes = plt.subplots(1, len(subfields),
                              figsize=(5*len(subfields), 5), squeeze=False)
    for ax, sf in zip(axes[0], subfields):
        pm   = data.get(sf, {})
        llm  = llm_freqs.get(sf, {})
        # Only tasks that appear in both
        common = sorted(set(pm) & set(llm))
        if not common:
            ax.set_visible(False); continue
        pm_ranks  = {t: r for r,(t,_) in enumerate(sorted(pm.items(),  key=lambda x:-x[1]), 1)}
        llm_ranks = {t: r for r,(t,_) in enumerate(sorted(llm.items(), key=lambda x:-x[1]), 1)}
        xs = [llm_ranks[t] for t in common]
        ys = [pm_ranks[t]  for t in common]
        ax.scatter(xs, ys, color=SUBFIELD_COLORS[sf], alpha=0.7, s=40)
        # Label top-5 by PubMed count
        for t in sorted(common, key=lambda t: pm_ranks[t])[:5]:
            ax.annotate(t, (llm_ranks[t], pm_ranks[t]),
                        fontsize=7, xytext=(4,2), textcoords="offset points")
        # Spearman r
        from scipy.stats import spearmanr
        r, p = spearmanr(xs, ys)
        ax.set_title(f"{sf}\nSpearman r={r:.2f} (p={p:.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("LLM mention rank"); ax.set_ylabel("PubMed count rank")
        # Diagonal reference (perfect agreement)
        lim = max(max(xs), max(ys)) + 1
        ax.plot([1,lim],[1,lim],"k--",alpha=0.3)
        ax.invert_xaxis(); ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(path, dpi=_DPI, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _DPI
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="pubmed_counts.json")
    parser.add_argument("--llm",   default="task_frequencies.json",
                        help="LLM mention frequencies (optional, enables scatter plot)")
    parser.add_argument("--out",   default="figures")
    parser.add_argument("--fmt",   default="png", choices=["png","pdf","svg"])
    parser.add_argument("--dpi",   type=int, default=200)
    args = parser.parse_args()

    _DPI = args.dpi
    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)

    raw = json.loads(Path(args.data).read_text())

    totals, unions, data = {}, {}, {}
    for sf, counts in raw.items():
        totals[sf] = counts.get("TOTAL") or 0
        unions[sf] = counts.get("UNION") or 0
        data[sf]   = {k:v for k,v in counts.items()
                      if k not in ("TOTAL","UNION") and v is not None}

    subfields = list(data.keys())
    stats = {sf: subfield_stats(data[sf], unions[sf]) for sf in subfields}

    # Update unions with fallback values
    for sf in subfields:
        if stats[sf]: unions[sf] = stats[sf]["union"]

    # Console summary
    print("\n" + "="*70)
    print("TASK CONCENTRATION — SUMMARY (LLM-discovered tasks, PubMed counts)")
    print("="*70)
    for sf in subfields:
        s = stats.get(sf, {})
        if not s: continue
        cov = unions[sf]/totals[sf]*100 if totals[sf] else 0
        fb  = "  ⚠ UNION fallback" if s.get("union_fallback") else ""
        print(f"\n{sf}")
        print(f"  TOTAL (empirical filter)   : {totals[sf]:>10,}")
        print(f"  UNION (named-task studies) : {unions[sf]:>10,}  ({cov:.1f}% coverage){fb}")
        print(f"  Top task    : {s['top1_label']}  ({s['top1_pct']:.1f}%)")
        print(f"  Top-3 share : {s['top3_pct']:.1f}%")
        print(f"  Gini        : {s['gini']:.3f}   HHI: {s['hhi_norm']:.4f}")
        print(f"  Paradigms for 50%: {s['n_50']}   for 80%: {s['n_80']}")

    def f(name): return fp(out_dir, name, args.fmt)

    valid = [sf for sf in subfields if stats.get(sf)]
    labels = {"Working Memory":"04_working_memory","Decision Making":"05_decision_making",
              "Spatial Navigation":"06_spatial_navigation","Attention":"07_attention"}

    print("\nGenerating figures …")
    plot_coverage(subfields, totals, unions, f("00_coverage"))
    for sf in valid:
        plot_subfield(dict(data[sf]), sf,
                      f(labels.get(sf, sf.lower().replace(" ","_"))),
                      SUBFIELD_COLORS.get(sf,"#607D8B"), union=unions[sf])
    plot_summary(stats, valid, f("01_summary_concentration"))
    plot_lorenz(data, valid, f("02_lorenz_curves"))
    plot_cumulative(data, unions, valid, f("03_cumulative_concentration"))

    # LLM vs PubMed scatter (optional)
    llm_path = Path(args.llm)
    if llm_path.exists():
        try:
            llm_freqs = json.loads(llm_path.read_text())
            plot_llm_vs_pubmed(data, llm_freqs, valid, f("08_llm_vs_pubmed"))
        except Exception as e:
            print(f"  (skipping LLM scatter: {e})")

    print(f"\n✓ Figures saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
