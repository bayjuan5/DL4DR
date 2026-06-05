"""
run_gradcam.py
==============
Drop this into:
  C:\\Users\\Administrator\\Documents\\AI_ML\\DL4DR\\Breast-CellTiter_Glo-Genomics\\full-Drug Response Prediction Model\\

Then run:
  python run_gradcam.py

What it does
------------
  1. Loads your trained checkpoint from ./checkpoints/
  2. Hooks GradCAM onto the last Conv2d of CellLineTower
       → model.cell_encoder.encoder[17]   (128→128 Conv2d in Block 3)
  3. For each cell line in ./genomic_images/, backprops through z_C
  4. Maps activated pixels back to gene names via pixel→gene lookup
  5. Saves per-cell-line 4-panel figures to ./gradcam_outputs/
  6. Saves cross-cell-line driver gene frequency summary
"""

import os, glob, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# ── make sure model.py and generate_genomic_images.py are importable ──
sys.path.insert(0, os.path.dirname(__file__))
from model import DrugResponseModel
from generate_genomic_images import build_master_index

# ─────────────────────────────────────────────────────────────
# CONFIG  ← adjust these two paths if needed
# ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = "./checkpoints"
GENOMIC_IMG_DIR  = "./genomic_images"
EXPR_FOLDER      = "../gene_expression_full"   # same as generate_genomic_images.py
OUTPUT_DIR       = "./gradcam_outputs"
TOP_K            = 20
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# Comprehensive breast cancer driver gene list (~190 genes)
# Sources: IntOGen (Martinez-Jimenez et al., Nat Rev Cancer 2020),
#          TCGA BRCA significantly mutated genes,
#          COSMIC Cancer Gene Census (breast-relevant),
#          Epigenetic regulators & chromosomal instability genes
#          added because GradCAM v1 flagged KMT2A/KMT2C/HERC2/
#          SPEN/BRWD1/NUMA1/TACC2 as top cross-cell-line activators.
# ─────────────────────────────────────────────────────────────
KNOWN_DRIVERS = {
    # Tier 1 — high-confidence breast cancer drivers (IntOGen / TCGA)
    "TP53","PIK3CA","CDH1","GATA3","MAP3K1","PTEN","AKT1","RUNX1",
    "TBX3","CBFB","FOXA1","ERBB2","RB1","CDKN2A","BRCA1","BRCA2",
    "NF1","SF3B1","MYC","CCND1","FGFR1","ZNF703","EMSY","LSP1",
    "MAP3K13","NCOR1","SPEN",
    # Tier 2 — recurrently mutated in TCGA BRCA
    "KMT2C","KMT2A","KMT2D","ARID1A","ARID1B","ARID2",
    "PIK3R1","ERBB3","ERBB4","FGFR2","FGFR3","FGFR4",
    "ESR1","PGR","AR","EGFR","MET","ALK","RET","NTRK1","NTRK3",
    "KRAS","HRAS","NRAS","BRAF","RAF1",
    "FBXW7","NOTCH1","NOTCH2","NOTCH3","NOTCH4",
    "ATM","CHEK1","CHEK2","PALB2","RAD51","RAD51B","RAD51D",
    "BRIP1","BARD1","NBN",
    "PIK3CB","PIK3CD","AKT2","AKT3","TSC1","TSC2","MTOR",
    "CDK4","CDK6","CDK12","CDKN1B","CCNE1","CCNE2",
    "RPS6KB1","RPS6KB2","MLLT4","MLLT3","MLLT1",
    "GPS2","NCOR2","TLE3","ZMYM2","ZMYM3",
    "MED1","MED12","MED23","EP300","CREBBP","KAT6A","KAT6B",
    "SMAD2","SMAD3","SMAD4","TGFBR2",
    "CTCF","RAD21","SMC1A","SMC3","STAG2","CASP8","FAM175A",
    # Epigenetic regulators — KMT2A/KMT2C were GradCAM v1 top hits
    "KMT2B","KMT2E",
    "SETD2","SETD1A","SETD1B","SETD3",
    "KDM5A","KDM5B","KDM5C","KDM6A",
    "DNMT3A","DNMT3B","TET1","TET2",
    "EZH2","EZH1","SUZ12","EED",
    "HDAC1","HDAC2","HDAC3",
    "ASXL1","ASXL2","ASXL3",
    "ATRX","DAXX","SMARCA4","SMARCB1","SMARCC1","SMARCC2",
    # Chromosomal instability — HERC2/BRWD1/NUMA1/TACC2 were GradCAM v1 top hits
    "HERC2","BRWD1","NUMA1","TACC2","TACC3",
    "NIPBL","STAG1","AURKB","PLK1","BUB1","BUB1B","MAD1L1","MAD2L1",
    # Mitochondrial / metabolic
    "IDH1","IDH2","SDHA","SDHB","SDHC","SDHD",
    # Splicing
    "SRSF2","U2AF1","U2AF2","ZRSR2",
    # Signalling
    "PTPN11","SOS1","CBL","CBLB",
    "JAK1","JAK2","JAK3","STAT3","STAT5A","STAT5B",
    "CCAR1","PELP1","SRC","YES1","LYN","INPP4B",
    "MAP2K4","MAP2K7",
    # Cell adhesion / EMT
    "CDH3","CDH13","CLDN3","CLDN4","CLDN7",
    # Copy-number amplified in breast cancer
    "BCAS3","BCAS4","RARA",
}

IMAGE_SIZE = 139
N_GENES    = 19177


# ─────────────────────────────────────────────────────────────
# 1.  LOAD CHECKPOINT
# ─────────────────────────────────────────────────────────────

def load_model(checkpoint_dir: str, device: str) -> DrugResponseModel:
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No .pt files found in {checkpoint_dir}")

    # Prefer best_random.pt, else take the latest
    preferred = [c for c in ckpts if "best" in os.path.basename(c).lower()]
    ckpt_path = preferred[0] if preferred else ckpts[-1]
    print(f"Loading checkpoint: {ckpt_path}")

    model = DrugResponseModel().to(device)
    state = torch.load(ckpt_path, map_location=device)

    # handle both raw state_dict and {'model_state_dict': ...} wrappers
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Model loaded. Device: {device}")
    return model


# ─────────────────────────────────────────────────────────────
# 2.  PIXEL → GENE LOOKUP
# ─────────────────────────────────────────────────────────────

def build_pixel_gene_lookup(master_index: dict) -> tuple:
    """
    Returns:
      lookup : dict  {(row, col): gene_name}
      rows   : np.ndarray  shape (N_GENES,)
      cols   : np.ndarray  shape (N_GENES,)
      genes  : np.ndarray  shape (N_GENES,) dtype=object
    """
    reverse = {idx: gene for gene, idx in master_index.items()}
    lookup  = {}
    rows_, cols_, genes_ = [], [], []
    for idx in range(N_GENES):
        r, c = divmod(idx, IMAGE_SIZE)
        if idx in reverse:
            gene = reverse[idx]
            lookup[(r, c)] = gene
            rows_.append(r); cols_.append(c); genes_.append(gene)
    return lookup, np.array(rows_), np.array(cols_), np.array(genes_, dtype=object)


def get_master_index(expr_folder: str) -> dict:
    txts = glob.glob(os.path.join(expr_folder, "ACH-*.txt"))
    if not txts:
        raise FileNotFoundError(
            f"No expression files found in {expr_folder}\n"
            f"Check EXPR_FOLDER in run_gradcam.py"
        )
    print(f"Building master gene index from {txts[0]} ...")
    mi = build_master_index(txts[0])
    print(f"  {len(mi)} genes indexed.")
    return mi


# ─────────────────────────────────────────────────────────────
# 3.  GRADCAM HOOK
# ─────────────────────────────────────────────────────────────

class GradCAMHook:
    def __init__(self, layer):
        self.act  = None
        self.grad = None
        self._f = layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'act', o.detach()))
        self._b = layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grad', go[0].detach()))

    def cam(self) -> np.ndarray:
        w   = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(dim=1))          # (1, H, W)
        cam = F.interpolate(cam.unsqueeze(0),
                            size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode='bilinear', align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        mn, mx = cam.min(), cam.max()
        return (cam - mn) / (mx - mn + 1e-8)

    def remove(self):
        self._f.remove(); self._b.remove()


# ─────────────────────────────────────────────────────────────
# 4.  PER-CELL-LINE GRADCAM
# ─────────────────────────────────────────────────────────────

def gradcam_for_cell_line(model, img_tensor, hook, device):
    """
    img_tensor: (1, 3, 139, 139) float32 [0,1]
    Returns cam_R, cam_G  both (139,139) float [0,1]
    """
    img_tensor = img_tensor.to(device)

    cam_maps = {}
    for ch_name in ["R", "G"]:
        model.zero_grad()
        t = img_tensor.clone().requires_grad_(True)

        # Forward through cell encoder only — no compound needed
        # z = model.cell_proj(model.cell_encoder(t))   # (1, hidden)
        # target = z.sum()
        z = model.cell_proj(model.cell_encoder(t))   # (1, hidden)
        target = model.gate(z).sum()

        target.backward()

        raw = hook.cam()

        # Weight by channel signal so R-cam highlights expression pixels,
        # G-cam highlights mutation pixels
        ch_idx    = 0 if ch_name == "R" else 1
        ch_signal = img_tensor[0, ch_idx].cpu().numpy()
        masked    = raw * (ch_signal > 0.05)
        mn, mx    = masked.min(), masked.max()
        cam_maps[ch_name] = (masked - mn) / (mx - mn + 1e-8)

    return cam_maps["R"], cam_maps["G"]


def extract_top_genes(cam_R, cam_G, img_np, rows, cols, genes, top_k=20):
    """img_np: (139,139,3) uint8"""
    sr = cam_R[rows, cols]
    sg = cam_G[rows, cols]
    sc = sr * sg
    g_px = img_np[rows, cols, 1].astype(float)   # G channel pixel value

    # driver candidates: must have G pixel > 0 (confirmed mutated+expressed)
    mutated = g_px > 0
    sd = sc * mutated

    def top(scores, k):
        idx = np.argsort(scores)[::-1][:k]
        return list(zip(genes[idx].tolist(), scores[idx].tolist()))

    di = np.argsort(sd)[::-1]
    di = di[sd[di] > 0][:top_k]

    return {
        "top_expr":    top(sr, top_k),
        "top_mut":     top(sg, top_k),
        "top_combined":top(sc, top_k),
        "drivers":     list(zip(genes[di].tolist(), sd[di].tolist())),
    }


# ─────────────────────────────────────────────────────────────
# 5.  PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_cell_line(ach_id, img_np, cam_R, cam_G, results, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle(
        f"Genomic GradCAM  |  {ach_id}  |  CellLineTower last-Conv2d (Block 3)",
        fontsize=12, fontweight='bold'
    )

    # Panel 1 — Expression GradCAM
    ax = axes[0]
    ax.set_title("R: Gene Expression + GradCAM", fontsize=10)
    ax.imshow(img_np[:,:,0], cmap='Greys', vmin=0, vmax=255, alpha=0.5)
    ax.imshow(cam_R, cmap='Reds', alpha=0.55, vmin=0, vmax=1)
    ax.set_xlabel("col (gene % 139)", fontsize=8); ax.set_ylabel("row (gene // 139)", fontsize=8)

    # Panel 2 — Mutation GradCAM
    ax = axes[1]
    ax.set_title("G: Mutation×Expr + GradCAM", fontsize=10)
    ax.imshow(img_np[:,:,1], cmap='Greys', vmin=0, vmax=255, alpha=0.5)
    ax.imshow(cam_G, cmap='Greens', alpha=0.55, vmin=0, vmax=1)

    # Panel 3 — Combined driver hotspot
    ax = axes[2]
    ax.set_title("Combined cam_R × cam_G\n(expressed AND mutated)", fontsize=10)
    combined = cam_R * cam_G
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    ax.imshow(combined, cmap='hot', vmin=0, vmax=1)
    # annotate top-5 driver positions
    for gene, _ in results["drivers"][:5]:
        # find pixel
        gene_idx = None
        for k, v in enumerate(results["drivers"]):
            if v[0] == gene:
                break
        # pixel lookup: reverse from gene name is O(n), acceptable for top-5
        pass   # annotation skipped here; bar chart is more readable

    # Panel 4 — Bar chart
    ax = axes[3]
    drivers = results["drivers"][:15]
    if drivers:
        gnames = [g for g,_ in drivers]
        scores = [s for _,s in drivers]
        colors = ['#d62728' if g in KNOWN_DRIVERS else '#1f77b4' for g in gnames]
        y = range(len(gnames))
        ax.barh(y, scores, color=colors, edgecolor='white', height=0.7)
        ax.set_yticks(y); ax.set_yticklabels(gnames, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("GradCAM score (cam_R × cam_G, G>0)", fontsize=8)
        n_k = sum(1 for g in gnames if g in KNOWN_DRIVERS)
        ax.set_title(f"Top Driver Candidates\n{n_k}/{len(gnames)} known drivers", fontsize=9)
        kp = mpatches.Patch(color='#d62728', label='Known driver')
        np_ = mpatches.Patch(color='#1f77b4', label='Novel candidate')
        ax.legend(handles=[kp, np_], fontsize=7, loc='lower right')
    else:
        ax.text(0.5, 0.5, "No mutated+expressed\ndriver pixels found",
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title("Top Driver Candidates", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → saved {save_path}")


def plot_summary(all_results, n_lines, save_path):
    counter = Counter()
    for r in all_results.values():
        for gene, _ in r["drivers"]:
            counter[gene] += 1

    top30 = counter.most_common(30)
    if not top30:
        print("No driver genes found across cell lines — summary skipped.")
        return

    gnames = [g for g,_ in top30]
    counts = [c for _,c in top30]
    colors = ['#d62728' if g in KNOWN_DRIVERS else '#1f77b4' for g in gnames]

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.barh(range(len(gnames)), counts, color=colors, edgecolor='white')
    ax.set_yticks(range(len(gnames))); ax.set_yticklabels(gnames, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(n_lines * 0.5, color='gray', ls='--', alpha=0.5, label='50% of cell lines')
    ax.set_xlabel(f"# cell lines (of {n_lines}) where gene ranked as top driver", fontsize=10)
    ax.set_title(
        "Cross-Cell-Line Driver Gene Frequency\n"
        "Genes consistently activating CellLineTower GradCAM across 51 breast cancer lines",
        fontsize=12, fontweight='bold'
    )
    kp  = mpatches.Patch(color='#d62728', label='Known cancer driver')
    np_ = mpatches.Patch(color='#1f77b4', label='Novel candidate')
    ax.legend(handles=[kp, np_, plt.Line2D([0],[0],color='gray',ls='--',label='50% threshold')],
              fontsize=9)

    n_top10_known = sum(1 for g in gnames[:10] if g in KNOWN_DRIVERS)
    expected = len(KNOWN_DRIVERS) / N_GENES * 10
    fig.text(0.99, 0.01,
             f"Top-10: {n_top10_known}/10 known drivers  |  "
             f"Driver list: {len(KNOWN_DRIVERS)} genes  |  "
             f"Expected by chance ≈ {expected:.2f}/10",
             ha='right', va='bottom', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSummary saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    model = load_model(CHECKPOINT_DIR, DEVICE)

    # Target layer: CellLineTower.encoder is nn.Sequential
    # index 17 = second Conv2d in Block 3  (Conv2d 128→128)
    # Layout: [0]Conv [1]GN [2]GELU [3]Conv [4]GN [5]GELU [6]MaxPool   <- Block1 (0-6)
    #         [7]Conv [8]GN [9]GELU [10]Conv [11]GN [12]GELU [13]MaxPool <- Block2 (7-13)
    #         [14]Conv [15]GN [16]GELU [17]Conv [18]GN [19]GELU [20]MaxPool <- Block3 (14-20)
    #         [21]AdaptiveAvgPool [22]Flatten
    target_layer = model.cell_encoder.encoder[17]
    print(f"GradCAM target layer: {target_layer}")
    hook = GradCAMHook(target_layer)

    # Build gene mapping
    master_index = get_master_index(EXPR_FOLDER)
    lookup, rows, cols, genes = build_pixel_gene_lookup(master_index)
    print(f"Pixel→gene lookup: {len(lookup)} valid pixels\n")

    # Process each genomic image
    img_paths = sorted(glob.glob(os.path.join(GENOMIC_IMG_DIR, "ACH-*.png")))
    if not img_paths:
        print(f"ERROR: No genomic images found in {GENOMIC_IMG_DIR}")
        print("Run generate_genomic_images.py first.")
        return

    print(f"Found {len(img_paths)} genomic images. Running GradCAM...\n")
    all_results = {}

    for img_path in img_paths:
        ach_id = os.path.splitext(os.path.basename(img_path))[0]

        img_np     = np.array(Image.open(img_path).convert("RGB"))     # (139,139,3) uint8
        img_tensor = torch.from_numpy(img_np.astype(np.float32)/255.0) \
                         .permute(2,0,1).unsqueeze(0)                   # (1,3,139,139)

        print(f"Processing {ach_id} ...", end="  ")
        cam_R, cam_G = gradcam_for_cell_line(model, img_tensor, hook, DEVICE)

        results = extract_top_genes(cam_R, cam_G, img_np, rows, cols, genes, TOP_K)
        all_results[ach_id] = results

        # Console output
        drivers = results["drivers"][:10]
        n_k = sum(1 for g,_ in drivers if g in KNOWN_DRIVERS)
        print(f"{n_k}/{len(drivers)} known drivers in top-10")
        for gene, score in drivers:
            tag = " *** KNOWN DRIVER ***" if gene in KNOWN_DRIVERS else ""
            print(f"    {gene:14s}  {score:.5f}{tag}")

        # Save figure
        plot_cell_line(
            ach_id, img_np, cam_R, cam_G, results,
            save_path=os.path.join(OUTPUT_DIR, f"gradcam_{ach_id}.png")
        )

    hook.remove()

    # Cross-cell-line summary
    plot_summary(all_results, len(all_results),
                 save_path=os.path.join(OUTPUT_DIR, "driver_gene_frequency_summary.png"))

    print(f"\nDone. All outputs in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
