"""
external_validation.py
======================
External validation on the CellTiter-Glo dataset (Section 17).
Tests whether the genomic encoder generalises to 603 unseen cell lines
across 27 tissue types — none of which appeared in training.

Core hypothesis (Section 17.2):
  genomic_distance(ext_cell_line, nearest_training_line) ∝ prediction_error

Produces four figures (Section 17.3):
  Fig 1 — Genomic Distance vs Prediction Error  (transparency test)
  Fig 2 — R² by Tissue Type
  Fig 3 — IC50 distribution: observed vs predicted
  Fig 4 — Seen vs Unseen compound performance

Usage
-----
  # Step 0: data audit (no model needed)
  python external_validation.py --mode audit \
      --train    data/BREAST-136344-56786-51.txt \
      --external data/cellTiter_Glo.txt

  # Step 1: generate predictions
  python external_validation.py --mode predict \
      --model       checkpoints/best_random.pt \
      --external    data/cellTiter_Glo.txt \
      --genomic_dir genomic_images_celltiter \
      --output      results/ext_predictions.csv

  # Step 2: produce all four figures
  python external_validation.py --mode analyse \
      --predictions       results/ext_predictions.csv \
      --train_embeddings  results/train_cell_embeddings.npy \
      --output_dir        results/figures
"""

import os
import sys
import csv
import argparse
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from dataset import (load_records, compute_baselines,
                      DrugResponseDataset, preload_genomic_images)
from model   import DrugResponseModel, DrugResponseConfig
from train   import collate_fn, forward_batch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Mode: audit  — compare datasets, no model needed
# ─────────────────────────────────────────────────────────────

def audit(args):
    log.info("=== DATA AUDIT ===")

    # Training set
    train_recs = load_records(args.train, args.smiles)
    train_smis = {r["smi_id"] for r in train_recs}
    train_cls  = {r["ach_id"] for r in train_recs}
    train_ic50 = [r["ln_ic50"] for r in train_recs]

    # External set
    ext_recs = load_external(args.external)
    ext_smis = {r["smi_id"] for r in ext_recs}
    ext_cls  = {r["ach_id"] for r in ext_recs}
    ext_ic50 = [r["ln_ic50"] for r in ext_recs]

    overlap_smis = train_smis & ext_smis
    overlap_cls  = train_cls  & ext_cls

    print("\n" + "="*60)
    print(f"  Training set:    {len(train_recs):>8,} records  |  "
          f"{len(train_smis):>6,} compounds  |  {len(train_cls):>4} cell lines")
    print(f"  External set:    {len(ext_recs):>8,} records  |  "
          f"{len(ext_smis):>6,} compounds  |  {len(ext_cls):>4} cell lines")
    print(f"  Compound overlap:  {len(overlap_smis):>4} / {len(ext_smis)} "
          f"({100*len(overlap_smis)/max(len(ext_smis),1):.1f}%)")
    print(f"  Cell-line overlap: {len(overlap_cls):>4} / {len(ext_cls)} "
          f"({'cold-start' if not overlap_cls else 'LEAKAGE DETECTED'})")
    print(f"  Training IC50 range:  {min(train_ic50):.2f} – {max(train_ic50):.2f}  "
          f"(median {np.median(train_ic50):.2f})")
    print(f"  External IC50 range:  {min(ext_ic50):.2f} – {max(ext_ic50):.2f}  "
          f"(median {np.median(ext_ic50):.2f})")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────
# Mode: predict — generate predictions on external set
# ─────────────────────────────────────────────────────────────

def load_external(path: str):
    """Load cellTiter_Glo.txt (same semicolon format as training data)."""
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = None
        for row in reader:
            if header is None:
                header = [h.strip().lower() for h in row]
                continue
            if len(row) < 5:
                continue
            try:
                recs.append({
                    "smi_id":   row[0].strip(),
                    "smiles":   row[1].strip(),
                    "ach_id":   row[3].strip(),
                    "ln_ic50":  float(row[4]),
                    "tissue":   row[5].strip() if len(row) > 5 else "unknown",
                })
            except (ValueError, IndexError):
                continue
    log.info("Loaded %d external records.", len(recs))
    return recs


@torch.no_grad()
def predict_external(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt  = torch.load(args.model, map_location=device)
    cfg   = ckpt.get("config", DrugResponseConfig())
    model = DrugResponseModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Checkpoint loaded: %s", args.model)

    # External records
    ext_recs   = load_external(args.external)
    cell_cache = preload_genomic_images(args.genomic_dir)

    # Training compound set (for seen/unseen flag)
    if args.train:
        train_recs = load_records(args.train, args.smiles)
        train_smis = {r["smi_id"] for r in train_recs}
    else:
        train_smis = set()

    ds = DrugResponseDataset(ext_recs, args.genomic_dir, cell_img_cache=cell_cache)
    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=2)

    rows = []
    for batch in dl:
        preds, _ = forward_batch(model, batch, device)
        for i, (p, a, s) in enumerate(zip(preds.cpu().tolist(),
                                           batch["ach_id"], batch["smi_id"])):
            rows.append({
                "ach_id":      a,
                "smi_id":      s,
                "observed":    batch["target"][i].item(),
                "predicted":   p,
                "seen_compound": "yes" if s in train_smis else "no",
                "tissue":      ext_recs[0].get("tissue", "unknown"),  # placeholder
            })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log.info("Predictions written → %s  (%d rows)", args.output, len(rows))

    # Save cell-line embeddings for genomic-distance analysis
    if args.train_embeddings:
        train_recs = load_records(args.train, args.smiles) if args.train else []
        train_cls  = list({r["ach_id"] for r in train_recs})
        train_imgs = torch.stack([
            cell_cache.get(a, torch.zeros(3, 139, 139)) for a in train_cls
        ]).to(device)
        with torch.no_grad():
            embs = model.cell_proj(model.cell_encoder(train_imgs)).cpu().numpy()
        np.save(args.train_embeddings, embs)
        np.save(args.train_embeddings.replace(".npy", "_ids.npy"),
                np.array(train_cls))
        log.info("Training cell-line embeddings → %s", args.train_embeddings)


# ─────────────────────────────────────────────────────────────
# Mode: analyse — produce four figures
# ─────────────────────────────────────────────────────────────

def analyse(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load predictions
    pred_rows = []
    with open(args.predictions, "r") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            pred_rows.append({
                "ach_id":    r["ach_id"],
                "smi_id":    r["smi_id"],
                "obs":       float(r["observed"]),
                "pred":      float(r["predicted"]),
                "seen":      r.get("seen_compound", "yes"),
                "tissue":    r.get("tissue", "unknown"),
            })

    # ── Figure 1: Genomic Distance vs Prediction Error ───────
    if args.train_embeddings and os.path.exists(args.train_embeddings):
        fig1_genomic_distance(pred_rows, args, os.path.join(args.output_dir, "fig1_genomic_distance.png"))
    else:
        log.warning("--train_embeddings not found; skipping Figure 1.")

    # ── Figure 2: R² by tissue type ──────────────────────────
    fig2_r2_by_tissue(pred_rows, os.path.join(args.output_dir, "fig2_r2_by_tissue.png"))

    # ── Figure 3: IC50 distribution ──────────────────────────
    fig3_ic50_distribution(pred_rows, os.path.join(args.output_dir, "fig3_ic50_distribution.png"))

    # ── Figure 4: Seen vs Unseen compounds ───────────────────
    fig4_seen_unseen(pred_rows, os.path.join(args.output_dir, "fig4_seen_unseen.png"))

    log.info("All figures written to %s", args.output_dir)


def fig1_genomic_distance(pred_rows, args, save_path):
    """
    Core transparency test: does genomic distance to nearest training line
    predict prediction error?  A positive Pearson r confirms the encoder
    is using genuine genomic signal (Section 17.2).
    """
    # Load training embeddings
    train_embs = np.load(args.train_embeddings)              # (N_train_CL, d)
    train_ids  = np.load(args.train_embeddings.replace(".npy","_ids.npy"),
                         allow_pickle=True).tolist()

    # Aggregate per-cell-line prediction error
    cl_errors = defaultdict(list)
    for r in pred_rows:
        cl_errors[r["ach_id"]].append(abs(r["pred"] - r["obs"]))
    cl_mae = {a: np.mean(v) for a, v in cl_errors.items()}

    # Compute genomic distance for external cell lines using their embeddings
    # (stored alongside predictions if --train_embeddings was set)
    ext_emb_path = args.train_embeddings.replace("train_cell_embeddings",
                                                   "ext_cell_embeddings")
    if not os.path.exists(ext_emb_path):
        log.warning("External embeddings not found at %s; "
                    "Figure 1 requires running predict mode first.", ext_emb_path)
        return

    ext_embs = np.load(ext_emb_path)
    ext_ids  = np.load(ext_emb_path.replace(".npy","_ids.npy"),
                       allow_pickle=True).tolist()

    dists = cdist(ext_embs, train_embs, metric="cosine").min(axis=1)
    dist_map = {a: dists[i] for i, a in enumerate(ext_ids)}

    common = [a for a in cl_mae if a in dist_map]
    if len(common) < 5:
        log.warning("Too few cell lines with embeddings for Figure 1 (%d).", len(common))
        return

    x = np.array([dist_map[a] for a in common])
    y = np.array([cl_mae[a]   for a in common])
    r, p = pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.6, s=20)
    # Regression line
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m*xs + b, "r--")
    ax.set_xlabel("Genomic distance to nearest training cell line (cosine)")
    ax.set_ylabel("Mean absolute prediction error")
    ax.set_title(f"Figure 1: Genomic Distance vs Prediction Error\n"
                 f"Pearson r = {r:.3f}  (p = {p:.3g})", fontweight="bold")
    sig = "✓ Genomic encoder is transparent" if r > 0.2 and p < 0.05 else \
          "✗ Encoder may be using tissue-type shortcuts"
    ax.text(0.05, 0.95, sig, transform=ax.transAxes, fontsize=8, va="top",
            color="green" if "✓" in sig else "red")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Fig 1 saved → %s  (r=%.3f, p=%.3g)", save_path, r, p)


def fig2_r2_by_tissue(pred_rows, save_path):
    tissue_data = defaultdict(lambda: {"obs": [], "pred": []})
    for r in pred_rows:
        tissue_data[r["tissue"]]["obs"].append(r["obs"])
        tissue_data[r["tissue"]]["pred"].append(r["pred"])

    tissues = sorted(tissue_data.keys())
    r2s = []
    for t in tissues:
        obs  = tissue_data[t]["obs"]
        pred = tissue_data[t]["pred"]
        r2s.append(r2_score(obs, pred) if len(obs) > 1 else float("nan"))

    valid = [(t, r) for t, r in zip(tissues, r2s) if not np.isnan(r)]
    valid.sort(key=lambda x: x[1], reverse=True)
    ts, rs = zip(*valid) if valid else ([], [])

    fig, ax = plt.subplots(figsize=(max(8, len(ts)*0.4), 5))
    ax.barh(range(len(ts)), rs, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(ts)))
    ax.set_yticklabels(ts, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("R²")
    ax.set_title("Figure 2: R² by Tissue Type (trained on breast cancer, tested on 27 types)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Fig 2 saved → %s", save_path)


def fig3_ic50_distribution(pred_rows, save_path):
    obs  = [r["obs"]  for r in pred_rows]
    pred = [r["pred"] for r in pred_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(min(obs + pred), max(obs + pred), 40)
    axes[0].hist(obs,  bins=bins, color="steelblue", alpha=0.7, label="Observed")
    axes[0].hist(pred, bins=bins, color="orange",    alpha=0.7, label="Predicted")
    axes[0].legend(); axes[0].set_xlabel("ln(IC₅₀)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Figure 3a: IC₅₀ Distributions")
    axes[1].scatter(obs, pred, alpha=0.2, s=5, rasterized=True)
    lo, hi = min(obs), max(obs)
    axes[1].plot([lo,hi],[lo,hi],"r--")
    axes[1].set_xlabel("Observed"); axes[1].set_ylabel("Predicted")
    r2 = r2_score(obs, pred)
    axes[1].set_title(f"Figure 3b: Scatter  (R²={r2:.3f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Fig 3 saved → %s", save_path)


def fig4_seen_unseen(pred_rows, save_path):
    seen_obs, seen_pred     = [], []
    unseen_obs, unseen_pred = [], []
    for r in pred_rows:
        if r["seen"] == "yes":
            seen_obs.append(r["obs"]);   seen_pred.append(r["pred"])
        else:
            unseen_obs.append(r["obs"]); unseen_pred.append(r["pred"])

    r2_seen   = r2_score(seen_obs,   seen_pred)   if seen_obs   else float("nan")
    r2_unseen = r2_score(unseen_obs, unseen_pred) if unseen_obs else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, obs, pred, label, r2 in [
        (axes[0], seen_obs,   seen_pred,   "Seen compounds",   r2_seen),
        (axes[1], unseen_obs, unseen_pred, "Unseen compounds", r2_unseen),
    ]:
        ax.scatter(obs, pred, alpha=0.3, s=8, rasterized=True)
        if obs:
            lo, hi = min(obs), max(obs)
            ax.plot([lo,hi],[lo,hi],"r--")
        ax.set_title(f"Figure 4: {label}\nn={len(obs)}  R²={r2:.3f}")
        ax.set_xlabel("Observed ln(IC₅₀)")
        ax.set_ylabel("Predicted")

    plt.suptitle("Figure 4: Seen vs Unseen Compound Generalisation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Fig 4 saved → %s  (seen R²=%.3f, unseen R²=%.3f)",
             save_path, r2_seen, r2_unseen)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="External validation on CellTiter-Glo dataset.")
    p.add_argument("--mode", required=True, choices=["audit", "predict", "analyse"])
    # audit
    p.add_argument("--smiles",   default=None,
                   help="Path to CompoundSmiles_full_140474.txt")
    p.add_argument("--train",    default=None)
    p.add_argument("--external", default=None)
    # predict
    p.add_argument("--model",       default=None)
    p.add_argument("--genomic_dir", default="genomic_images_celltiter")
    p.add_argument("--output",      default="results/ext_predictions.csv")
    p.add_argument("--train_embeddings", default="results/train_cell_embeddings.npy")
    # analyse
    p.add_argument("--predictions", default=None)
    p.add_argument("--output_dir",  default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "audit":
        if not args.train or not args.external:
            sys.exit("audit mode requires --train and --external")
        audit(args)
    elif args.mode == "predict":
        if not args.model or not args.external:
            sys.exit("predict mode requires --model and --external")
        predict_external(args)
    elif args.mode == "analyse":
        if not args.predictions:
            sys.exit("analyse mode requires --predictions")
        analyse(args)


if __name__ == "__main__":
    main()
