#!/usr/bin/env python3
"""Figure 1 Technical Roadmap – v7
Fixes: (1) BOX_PAD-corrected arrow endpoints (exact visual edge touch)
       (2) Font sizes dramatically increased (19-22 pt main, 17-19 pt sub)
       (3) Box heights recalculated to match new fonts
"""
from __future__ import annotations
import shutil, warnings
from pathlib import Path
import matplotlib, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR  = REPO_ROOT / "outputs" / "figures"
PLOS_DIR = REPO_ROOT / "outputs" / "plos_submission"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PLOS_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 18, "axes.titlesize": 19, "figure.dpi": 120,
})
DPI = 600

C = {
    "navy":"#2B4590","teal":"#1B7A6E","amber":"#C98A00","gray":"#7A8695",
    "red":"#B83232","purple":"#5E3A8C","brown":"#7A4A10","pink":"#A02070",
    "orange":"#B85000","violet":"#5830A0","cyan":"#1A6878","gold":"#C09000",
    "coral":"#B84040","green":"#1E6E3C",
    "bg_blue":"#D6E8F8","bg_green":"#D0EDD8","bg_yellow":"#FFF0B8","bg_pink":"#F8D8E8",
}

# ── KEY CONSTANT: matches boxstyle pad so arrows touch visual edge exactly ──
BOX_PAD = 0.010

def vtop(cy, h): return cy + h/2 + BOX_PAD   # visual top  edge
def vbot(cy, h): return cy - h/2 - BOX_PAD   # visual bottom edge

def _dark(hx):
    r,g,b = int(hx[1:3],16),int(hx[3:5],16),int(hx[5:7],16)
    return 0.299*r+0.587*g+0.114*b < 140

def _box(ax, cx, cy, w, h, color, *, text="", fontsize=19,
         text_color=None, lw=2.5, alpha=0.92, zorder=3):
    ax.add_patch(FancyBboxPatch(
        (cx-w/2+0.004, cy-h/2-0.004), w, h,
        boxstyle=f"round,pad={BOX_PAD}", facecolor="#AAAAAA",
        edgecolor="none", alpha=0.20, zorder=zorder-1))
    ax.add_patch(FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h,
        boxstyle=f"round,pad={BOX_PAD}", facecolor=color,
        edgecolor="#1C2333", linewidth=lw, alpha=alpha, zorder=zorder))
    if text:
        tc = text_color or ("white" if _dark(color) else "#1C2333")
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=tc,
                linespacing=1.38, multialignment="center", zorder=zorder+1)

def _arrow(ax, x0, y0, x1, y1, *, lw=2.5, color="#2C3E50", zorder=10):
    """shrinkA/B = 0 because callers pass exact visual-edge coordinates."""
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                                mutation_scale=24,
                                connectionstyle="arc3,rad=0",
                                shrinkA=0, shrinkB=0),
                zorder=zorder)

def _sidebar(ax, cx, cy, text, bg, edge, fontsize=17):
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=bg,
                      edgecolor=edge, linewidth=2.8, alpha=0.97),
            zorder=20)

def build_figure():
    fig, ax = plt.subplots(figsize=(22, 34))
    ax.set_xlim(0, 1); ax.set_ylim(0.01, 1.00); ax.axis("off")

    # ── Y row centres (computed so visual edges never touch) ─────────────────
    # visual half-extent = h/2 + BOX_PAD; gap between rows ≥ 0.018
    H    = 0.044   # 2-line main box
    H3   = 0.074   # 3-line wide box
    H_MD = 0.074   # model boxes
    H_SM = 0.110   # 6-line eval boxes
    H_EX = 0.082   # 4-line extra boxes
    H_RC = 0.059   # recommend box

    # vhalf helpers
    vh    = lambda h: h/2 + BOX_PAD

    # Build Y from top down, leaving ≥0.018 gap between visual edges
    GAP = 0.020
    yc   = 0.955                          # collect
    ypr  = yc  - vh(H)   - GAP - vh(H)   # preproc
    ysp  = ypr - vh(H)   - GAP - vh(H)   # split
    ymo  = ysp - vh(H)   - GAP - vh(H_MD)# models
    yhp  = ymo - vh(H_MD)- GAP - vh(H3)  # hpopt
    yev  = yhp - vh(H3)  - GAP - vh(H_SM)# eval
    yin  = yev - vh(H_SM)- GAP - vh(H3)  # integrate
    yex  = yin - vh(H3)  - GAP - vh(H_EX)# extra
    yrec = yex - vh(H_EX)- GAP - vh(H_RC)# recommend

    Y = dict(collect=yc,preproc=ypr,split=ysp,models=ymo,
             hpopt=yhp,eval=yev,integrate=yin,extra=yex,recommend=yrec)

    W_MAIN = 0.44; W3 = 0.485; W_MD = 0.185; W_SM = 0.155; W_EX = 0.220

    # ── [1] Data Collection ──────────────────────────────────────────────────
    _box(ax, 0.5, Y["collect"], W_MAIN, H, C["navy"],
         text="Data Collection\n9 Datasets (7 QA-Approved)  |  159,811 Total Samples",
         fontsize=22)

    # ── [2] Preprocessing ────────────────────────────────────────────────────
    _box(ax, 0.5, Y["preproc"], W_MAIN, H, C["teal"],
         text="Data Preprocessing & Quality Assurance\n"
              "Harmonise variables · Remove outliers · Exclude phyto_wide & phyto_long",
         fontsize=19)

    # ── [3] Split ────────────────────────────────────────────────────────────
    _box(ax, 0.5, Y["split"], W_MAIN, H, C["amber"],
         text="Data Split Strategy  (Leakage Prevention)\n"
              "Time-series: chronological 70/15/15%   ·   Cross-sectional: stratified 70/15/15%",
         fontsize=18, text_color="#1C2333")

    # spine arrows  1→2→3
    for ya, yb in [(Y["collect"],Y["preproc"]),(Y["preproc"],Y["split"])]:
        _arrow(ax, 0.5, vbot(ya,H), 0.5, vtop(yb,H))

    # ── [4] Three model boxes ────────────────────────────────────────────────
    MX = [0.175, 0.500, 0.825]
    MT = ["Baseline Models\nMEAN · Ridge · LASSO",
          "Traditional ML\nSVR · RF · XGBoost",
          "Deep Learning\nLSTM · Transformer\n(5 DL datasets)"]
    MC = [C["gray"], C["red"], C["purple"]]
    for mx,mt,mc in zip(MX,MT,MC):
        _box(ax, mx, Y["models"], W_MD, H_MD, mc, text=mt, fontsize=20)
        _arrow(ax, 0.5, vbot(Y["split"],H), mx, vtop(Y["models"],H_MD))

    # ── [5] Hpopt ────────────────────────────────────────────────────────────
    _box(ax, 0.5, Y["hpopt"], W3, H3, C["brown"],
         text="Hyperparameter Optimisation  (Grid Search on Validation Set)\n"
              "Ridge/LASSO: α   ·   SVR: C, γ   ·   RF/XGBoost: n_trees, depth, lr\n"
              "LSTM/Trans.: hidden, dropout, early-stop",
         fontsize=19)
    for mx in MX:
        _arrow(ax, mx, vbot(Y["models"],H_MD), 0.5, vtop(Y["hpopt"],H3))

    # ── [6] Four eval boxes ──────────────────────────────────────────────────
    EX = [0.125, 0.370, 0.630, 0.875]
    ET = ["Performance\nEvaluation\nR\u00b2, \u0394R\u00b2, MAE\nRMSE, NRMSE, NSE\nPrec/Recall/F1*",
          "Statistical\nSignificance\nPermutation Test\nK=10,000 (XGBoost)\np-value report",
          "Robustness\nAnalysis\nRadar chart\n5 dimensions:\nMean·Best·Stab\nConsist·Sig.Rate",
          "Feature\nImportance\nPermut. Imp.\n(RF / XGBoost)\nSHAP values\n(LSTM / Trans.)"]
    EC = [C["pink"],C["orange"],C["violet"],C["cyan"]]
    for ex,et,ec in zip(EX,ET,EC):
        _box(ax, ex, Y["eval"], W_SM, H_SM, ec, text=et, fontsize=17)
        _arrow(ax, 0.5, vbot(Y["hpopt"],H3), ex, vtop(Y["eval"],H_SM))

    # ── [7] Integrate ────────────────────────────────────────────────────────
    _box(ax, 0.5, Y["integrate"], W3, H3, C["gold"],
         text="Results Integration & Analysis\n"
              "Cross-dataset heatmap · Model distributions · Robustness radar\n"
              "Difficulty vs size · Feature profiles",
         fontsize=19, text_color="#1C2333")
    for ex in EX:
        _arrow(ax, ex, vbot(Y["eval"],H_SM), 0.5, vtop(Y["integrate"],H3))

    # ── [8] Two extra boxes ──────────────────────────────────────────────────
    EXX = [0.280, 0.720]
    EXT = ["[NEW]  Downsampling Ablation\nera5_daily  |  n in {1k,5k,10k,50k,100k}\n"
           "Diminishing returns >~10k-50k\n(S4 Fig · S4 Table)",
           "[NEW]  Cross-Dataset Transfer Pilot\ncleaned_data  <->  rolling_mean\n"
           "Zero-shot · R\u00b2=0.646-0.729\nDomain shift confirmed  (S8 Table)"]
    EXC = [C["coral"],C["green"]]
    for ex,et,ec in zip(EXX,EXT,EXC):
        _box(ax, ex, Y["extra"], W_EX, H_EX, ec, text=et, fontsize=18)
        _arrow(ax, 0.5, vbot(Y["integrate"],H3), ex, vtop(Y["extra"],H_EX))

    # ── [9] Recommend ────────────────────────────────────────────────────────
    _box(ax, 0.5, Y["recommend"], W3, H_RC, C["navy"],
         text="Best Practices & Recommendations\n"
              "Model-selection landscape · Cross-dataset comparisons · "
              "Leakage-aware benchmarking roadmap",
         fontsize=19)
    for ex in EXX:
        _arrow(ax, ex, vbot(Y["extra"],H_EX), 0.5, vtop(Y["recommend"],H_RC))

    # ── Sidebars ─────────────────────────────────────────────────────────────
    # Middle sidebars relocated to bottom-corner empty space (y≈0.107)
    SB_MID = max(Y["recommend"] - 0.06, 0.08)

    _sidebar(ax, 0.080, Y["collect"],
             "INPUT\n• 9 Datasets\n• 7 QA-Pass\n• 159,811 samples\n• Multi-domain",
             C["bg_blue"], C["navy"], fontsize=17)
    _sidebar(ax, 0.080, SB_MID,          # ← MODELS moved to bottom area
             "MODELS  (8 types)\n• MEAN · Ridge · LASSO\n"
             "• SVR · RF · XGBoost\n• LSTM · Transformer",
             C["bg_green"], C["teal"], fontsize=17)
    _sidebar(ax, 0.080, Y["integrate"],
             "METRICS  (UPDATED)\n• R²  ·  ΔR²  ·  MAE\n"
             "• RMSE · NRMSE · NSE\n• Bootstrap CI  (95%)\n• Prec/Recall/F1*",
             C["bg_pink"], C["pink"], fontsize=17)

    _sidebar(ax, 0.920, Y["collect"],
             "KEY RESULTS\n• RF: R²=0.872\n  (rolling_mean)\n"
             "• LSTM: Best on 3/7\n• 5/7: p < 1e-4",
             C["bg_yellow"], C["amber"], fontsize=17)
    _sidebar(ax, 0.920, SB_MID,           # ← DATASETS moved to bottom area
             "DATASETS  (7 QA-Pass)\n• rolling_mean   Easy\n"
             "• cleaned_data   Medium\n• era5_daily     Medium\n"
             "• processed_seq Medium\n• hydrographic  Hard\n"
             "• biotoxin          Hard\n• cast               Hard",
             C["bg_green"], C["teal"], fontsize=16)
    _sidebar(ax, 0.920, Y["integrate"],
             "FINDINGS\n• Quality > Quantity\n• DL wins on temporal\n"
             "• Baselines essential\n• Transfer ≠ free",
             C["bg_blue"], C["navy"], fontsize=17)

    # ── Footnote ─────────────────────────────────────────────────────────────
    ax.text(0.5, vbot(Y["recommend"],H_RC) - 0.010,
            "* Event-based Precision/Recall/F1 at 90th-percentile training threshold; "
            "reported for all datasets, primary focus on biotoxin.",
            ha="center", va="top", fontsize=13, color="#555555", style="italic")

    plt.tight_layout(pad=0.4)
    return fig

def save_all(fig, stem):
    base = FIG_DIR / stem
    for ext in ("pdf","png","tiff"):
        out = base.with_suffix(f".{ext}")
        kw = dict(dpi=DPI, bbox_inches="tight", facecolor="white")
        if ext=="tiff": kw["format"]="tiff"
        elif ext=="pdf": kw["format"]="pdf"
        fig.savefig(out, **kw)
        print(f"  Saved: {out.relative_to(REPO_ROOT)}")
    shutil.copyfile(base.with_suffix(".tiff"), PLOS_DIR/"Fig1.tif")
    print(f"  Copied: {(PLOS_DIR/'Fig1.tif').relative_to(REPO_ROOT)}")

def main():
    print("="*60)
    print("  generate_figure1_updated.py  –  v7")
    print("="*60)
    fig = build_figure()
    save_all(fig, "figure1_technical_roadmap_final")
    plt.close(fig)
    print("\n✅  Done.")

if __name__ == "__main__":
    main()
