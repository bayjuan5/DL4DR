"""
evaluate.py
===========
Comprehensive evaluation of a trained Two-Tower Residual Late Fusion checkpoint
across all three required splits (Section 15).

Outputs
-------
  results/random_split_scatter.png      — Figure 9 in v18
  results/compound_out_bar.png          — Figure 10
  results/cellline_out_scatter.png      — Figure 11
  results/cellline_r2_table.csv         — per-cell-line R² vs ECFP-Only baseline
  results/eval_summary.txt              — Table 9 in v18

Usage
-----
  # Evaluate random split checkpoint (already trained):
  python evaluate.py \
      --data       data/BREAST-136344-56786-51.txt \
      --genomic    genomic_images \
      --ckpt       checkpoints/best_random.pt \
      --split      all              # evaluate all three splits
"""

import os
import argparse
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import r2_score

from dataset import (load_records, compute_baselines,
                      random_split, leave_compound_out_split,
                      leave_cell_line_out_splits,
                      DrugResponseDataset, preload_genomic_images)
from model   import DrugResponseModel, DrugResponseConfig
from train   import collate_fn, forward_batch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")


# ─────────────────────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues, achs, smis = [], [], [], []
    for batch in loader:
        p, _ = forward_batch(model, batch, device)
        preds.extend(p.cpu().tolist())
        trues.extend(batch["target"].tolist())
        achs.extend(batch["ach_id"])
        smis.extend(batch["smi_id"])
    return np.array(preds), np.array(trues), achs, smis


# ─────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────

HIGHLIGHT_CLS = ["MCF7", "MDAMB231", "SKBR3", "T47D", "BT549", "HS578T"]

def plot_random_scatter(preds, trues, achs, save_path):
    """Figure 9 in v18: 6-panel scatter for representative cell lines."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # Group by cell line
    cl_data = defaultdict(lambda: {"p": [], "t": []})
    for p, t, a in zip(preds, trues, achs):
        cl_data[a]["p"].append(p)
        cl_data[a]["t"].append(t)

    plotted = 0
    for cl in HIGHLIGHT_CLS:
        if cl not in cl_data or plotted >= 6:
            continue
        p = np.array(cl_data[cl]["p"])
        t = np.array(cl_data[cl]["t"])
        r2 = r2_score(t, p) if len(t) > 1 else float("nan")
        ax = axes[plotted]
        ax.scatter(t, p, alpha=0.35, s=8, c="steelblue", rasterized=True)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
        ax.set_title(f"{cl}\nn={len(t)}  R²={r2:.2f}", fontsize=9)
        ax.set_xlabel("Observed ln(IC₅₀)", fontsize=8)
        ax.set_ylabel("Predicted",          fontsize=8)
        plotted += 1

    # Fill any remaining panels with whatever cell lines we have
    remaining = [c for c in sorted(cl_data.keys()) if c not in HIGHLIGHT_CLS]
    for cl in remaining:
        if plotted >= 6:
            break
        p = np.array(cl_data[cl]["p"])
        t = np.array(cl_data[cl]["t"])
        r2 = r2_score(t, p) if len(t) > 1 else float("nan")
        ax = axes[plotted]
        ax.scatter(t, p, alpha=0.35, s=8, c="steelblue", rasterized=True)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
        ax.set_title(f"{cl}\nn={len(t)}  R²={r2:.2f}", fontsize=9)
        ax.set_xlabel("Observed ln(IC₅₀)", fontsize=8)
        ax.set_ylabel("Predicted",          fontsize=8)
        plotted += 1

    fig.suptitle("Random Split: Predicted vs. Observed ln(IC₅₀)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", save_path)


def plot_compound_out_bar(model_r2s, base_r2s, cls, save_path):
    """Figure 10 in v18: grouped bar chart, Two-Tower vs additive baseline."""
    x   = np.arange(len(cls))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, base_r2s,  w, label="Additive baseline", color="grey",  alpha=0.7)
    ax.bar(x + w/2, model_r2s, w, label="Two-Tower RLF",     color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cls, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("Leave-Compound-Out R²: Two-Tower RLF vs. Additive Baseline",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", save_path)


def plot_cellline_out_scatter(ecfp_r2s, best_r2s, save_path):
    """Figure 11 in v18: diagonal scatter across 51 cell lines."""
    above = [(e, b) for e, b in zip(ecfp_r2s, best_r2s) if b >= e]
    below = [(e, b) for e, b in zip(ecfp_r2s, best_r2s) if b <  e]
    fig, ax = plt.subplots(figsize=(6, 6))
    if above:
        ax.scatter(*zip(*above), color="steelblue", alpha=0.7, s=25, label="Residual > ECFP-Only")
    if below:
        ax.scatter(*zip(*below), color="grey",      alpha=0.7, s=25, label="Residual ≤ ECFP-Only")
    lo = min(min(ecfp_r2s), min(best_r2s))
    hi = max(max(ecfp_r2s), max(best_r2s))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    ax.set_xlabel("ECFP-Only MLP R²")
    ax.set_ylabel("Best Residual Fusion R²")
    ax.set_title(f"Leave-Cell-Line-Out: {len(above)}/{len(above)+len(below)} above diagonal",
                 fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", save_path)


# ─────────────────────────────────────────────────────────────
# Split-specific evaluation routines
# ─────────────────────────────────────────────────────────────

def eval_random(model, records, genomic_dir, baselines, device, out_dir):
    _, _, test_recs = random_split(records)
    cell_cache = preload_genomic_images(genomic_dir)
    ds = DrugResponseDataset(test_recs, genomic_dir, cell_img_cache=cell_cache)
    dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_fn, num_workers=2)
    preds, trues, achs, smis = predict(model, dl, device)
    r2   = r2_score(trues, preds)
    rmse = float(np.sqrt(np.mean((preds - trues)**2)))
    log.info("Random split  →  R²=%.4f  RMSE=%.4f", r2, rmse)
    plot_random_scatter(preds, trues, achs,
                        os.path.join(out_dir, "random_split_scatter.png"))
    return {"split": "random", "r2": r2, "rmse": rmse, "n": len(trues)}


def eval_compound_out(model, records, genomic_dir, baselines, device, out_dir):
    _, _, test_recs = leave_compound_out_split(records)
    cell_cache = preload_genomic_images(genomic_dir)

    ds = DrugResponseDataset(test_recs, genomic_dir, cell_img_cache=cell_cache)
    dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_fn, num_workers=2)
    preds, trues, achs, smis = predict(model, dl, device)

    # Per-cell-line R² vs additive baseline
    cl_data = defaultdict(lambda: {"p": [], "t": [], "a": []})
    for p, t, a, s in zip(preds, trues, achs, smis):
        add = (baselines["per_cl_mean"].get(a, baselines["global_mean"]) +
               baselines["per_cmp_mean"].get(s, baselines["global_mean"]) -
               baselines["global_mean"])
        cl_data[a]["p"].append(p)
        cl_data[a]["t"].append(t)
        cl_data[a]["a"].append(add)

    cls_sorted = sorted(cl_data.keys(),
                        key=lambda c: len(cl_data[c]["p"]), reverse=True)[:6]
    model_r2s = [r2_score(cl_data[c]["t"], cl_data[c]["p"]) for c in cls_sorted]
    base_r2s  = [r2_score(cl_data[c]["t"], cl_data[c]["a"]) for c in cls_sorted]

    r2   = r2_score(trues, preds)
    rmse = float(np.sqrt(np.mean((preds - trues)**2)))
    log.info("Leave-compound-out  →  R²=%.4f  RMSE=%.4f", r2, rmse)

    plot_compound_out_bar(model_r2s, base_r2s, cls_sorted,
                          os.path.join(out_dir, "compound_out_bar.png"))
    return {"split": "compound_out", "r2": r2, "rmse": rmse, "n": len(trues)}


def eval_cellline_out(model, records, genomic_dir, baselines, device, out_dir):
    """
    Run all 5 GroupKFold folds.  For each fold, compare Two-Tower vs ECFP-Only.
    ECFP-Only baseline: use per-compound-mean + per-CL-mean - global_mean
    (the best non-interaction baseline, tighter upper bound than just CL-mean).
    """
    folds = leave_cell_line_out_splits(records)
    cell_cache = preload_genomic_images(genomic_dir)

    all_preds, all_trues, all_achs = [], [], []
    fold_r2s = []

    for fold_i, (train_recs, test_recs) in enumerate(folds):
        ds = DrugResponseDataset(test_recs, genomic_dir, cell_img_cache=cell_cache)
        dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_fn, num_workers=2)
        preds, trues, achs, _ = predict(model, dl, device)
        all_preds.extend(preds)
        all_trues.extend(trues)
        all_achs.extend(achs)
        r2 = r2_score(trues, preds)
        fold_r2s.append(r2)
        log.info("  Fold %d  R²=%.4f", fold_i + 1, r2)

    # Per-cell-line R² — model vs ECFP-only (additive) baseline
    cl_model = defaultdict(lambda: {"p": [], "t": []})
    cl_add   = defaultdict(lambda: {"p": [], "t": []})
    for p, t, a in zip(all_preds, all_trues, all_achs):
        add = baselines["per_cl_mean"].get(a, baselines["global_mean"])
        cl_model[a]["p"].append(p);  cl_model[a]["t"].append(t)
        cl_add[a]["p"].append(add);  cl_add[a]["t"].append(t)

    rows = []
    for a in sorted(cl_model.keys()):
        m_r2   = r2_score(cl_model[a]["t"], cl_model[a]["p"])
        ecfp_r2= r2_score(cl_add[a]["t"],   cl_add[a]["p"])
        rows.append({"ach_id": a, "model_r2": m_r2, "ecfp_r2": ecfp_r2,
                     "delta_r2": m_r2 - ecfp_r2, "n": len(cl_model[a]["t"])})

    rows.sort(key=lambda r: r["delta_r2"], reverse=True)

    # Write per-CL table
    import csv as _csv
    table_path = os.path.join(out_dir, "cellline_r2_table.csv")
    with open(table_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["ach_id","model_r2","ecfp_r2","delta_r2","n"])
        w.writeheader(); w.writerows(rows)
    log.info("Per-cell-line R² table → %s", table_path)

    ecfp_r2s = [r["ecfp_r2"]  for r in rows]
    best_r2s = [r["model_r2"] for r in rows]
    plot_cellline_out_scatter(ecfp_r2s, best_r2s,
                              os.path.join(out_dir, "cellline_out_scatter.png"))

    mean_r2  = float(np.mean(fold_r2s))
    mean_dr2 = float(np.mean([r["delta_r2"] for r in rows]))
    n_above  = sum(1 for r in rows if r["delta_r2"] > 0)
    log.info("Leave-cell-line-out  →  mean_R²=%.4f  mean_ΔR²=%.4f  "
             "%d/%d above ECFP baseline", mean_r2, mean_dr2, n_above, len(rows))

    return {"split": "cellline_out", "r2": mean_r2,
            "delta_r2": mean_dr2, "n_above": n_above, "n_total": len(rows)}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Two-Tower RLF model.")
    p.add_argument("--smiles",  required=True,
                   help="Path to CompoundSmiles_full_140474.txt")
    p.add_argument("--data",    required=True)
    p.add_argument("--genomic", required=True)
    p.add_argument("--ckpt",    required=True,
                   help="Checkpoint .pt file (from train.py)")
    p.add_argument("--split",   default="all",
                   choices=["all", "random", "compound_out", "cellline_out"])
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model ───────────────────────────────────────────
    ckpt  = torch.load(args.ckpt, map_location=device)
    cfg   = ckpt.get("config", DrugResponseConfig())
    model = DrugResponseModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Loaded checkpoint: %s  (epoch %d, val_R²=%.4f)",
             args.ckpt, ckpt.get("epoch", -1), ckpt.get("val_r2", float("nan")))

    # ── Data ─────────────────────────────────────────────────
    records   = load_records(args.data, args.smiles)
    baselines = compute_baselines(records)

    # ── Evaluate ─────────────────────────────────────────────
    results = []
    do = lambda s: args.split in ("all", s)

    if do("random"):
        results.append(eval_random(model, records, args.genomic, baselines, device, args.out_dir))
    if do("compound_out"):
        results.append(eval_compound_out(model, records, args.genomic, baselines, device, args.out_dir))
    if do("cellline_out"):
        results.append(eval_cellline_out(model, records, args.genomic, baselines, device, args.out_dir))

    # ── Summary table ────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "eval_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("Two-Tower Residual Late Fusion — Evaluation Summary\n")
        fh.write("=" * 55 + "\n")
        for r in results:
            fh.write(f"\n[{r['split']}]\n")
            for k, v in r.items():
                if k != "split":
                    fh.write(f"  {k:15s} = {v}\n")
    log.info("Summary written → %s", summary_path)


if __name__ == "__main__":
    main()
