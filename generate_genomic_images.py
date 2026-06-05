"""
generate_genomic_images.py
==========================
Converts DepMap gene expression and mutation data into 139×139 RGB genomic images
for use as CellLineTower input.

Channel layout (matches ORNN_DMPNN_Theory_v18, Section 13.2):
  R  — gene expression level  (99th-percentile normalised to [0, 255])
  G  — expression × mutation severity  (WT=0, Missense=85, Splice=170, Truncating=255)
  B  — reserved for copy-number variation (zero-filled until CNV data is added)

Image size: 139 × 139 = 19,321 pixels ≥ 19,177 genes.
Genes are placed in a deterministic row-major order derived from the master index
built from the first expression file encountered (alphabetical sort of gene names).
Pixels beyond the last gene are left as zero.

Usage
-----
  # Build all images from DepMap bulk RNA-seq and mutation files:
  python generate_genomic_images.py \
      --expr_dir   data/gene_expression_full \
      --mut_file   data/mutations.csv \
      --output_dir genomic_images

  # Dry-run: build index only, print stats, no images written:
  python generate_genomic_images.py --expr_dir data/gene_expression_full --dry_run

Expected input formats
----------------------
Expression files  (one per cell line, e.g. ACH-000001.txt):
  Tab-separated, first column = gene symbol, second column = log2(TPM+1) or similar.
  Header row optional (detected automatically).

Mutation file  (OmicsSomaticMutations.csv from DepMap):
  Must contain columns: ModelID, HugoSymbol, VariantClassification
  VariantClassification values mapped to severity:
    Missense_Mutation              → 1  (G = 85)
    Splice_Site, Splice_Region     → 2  (G = 170)
    Nonsense_Mutation, Frame_Shift_*, In_Frame_*, Nonstop_Mutation,
    Translation_Start_Site         → 3  (G = 255)
    Everything else                → 0  (WT, silent, UTR, intron)
"""

import os
import sys
import argparse
import glob
import csv
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

IMAGE_SIZE  = 139
N_PIXELS    = IMAGE_SIZE * IMAGE_SIZE   # 19,321
N_GENES_MAX = 19_177

# Severity mapping for DepMap VariantClassification column
MUT_SEVERITY = {
    "Missense_Mutation":        1,
    "Splice_Site":              2,
    "Splice_Region":            2,
    "Nonsense_Mutation":        3,
    "Frame_Shift_Del":          3,
    "Frame_Shift_Ins":          3,
    "In_Frame_Del":             3,
    "In_Frame_Ins":             3,
    "Nonstop_Mutation":         3,
    "Translation_Start_Site":   3,
}
SEVERITY_TO_G = {0: 0, 1: 85, 2: 170, 3: 255}


# ─────────────────────────────────────────────────────────────
# 1.  Master gene index  (gene → pixel index, 0-based)
# ─────────────────────────────────────────────────────────────

def build_master_index(expr_file: str) -> dict:
    """
    Read one expression file, sort gene names alphabetically,
    assign each a sequential pixel index 0 … N_GENES-1.
    Returns dict {gene_symbol: pixel_index}.
    """
    genes = []
    with open(expr_file, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            # skip header rows
            if gene.lower() in ("gene", "gene_id", "symbol", "hugo_symbol", ""):
                continue
            try:
                float(row[1])           # second column must be numeric
                genes.append(gene)
            except (IndexError, ValueError):
                continue
    genes = sorted(set(genes))
    if len(genes) > N_PIXELS:
        log.warning("Gene count %d exceeds image capacity %d; "
                    "truncating to first %d genes.", len(genes), N_PIXELS, N_PIXELS)
        genes = genes[:N_PIXELS]
    log.info("Master index: %d unique genes.", len(genes))
    return {g: i for i, g in enumerate(genes)}


# ─────────────────────────────────────────────────────────────
# 2.  Load expression for one cell line
# ─────────────────────────────────────────────────────────────

def load_expression(expr_file: str, master_index: dict) -> np.ndarray:
    """
    Returns float32 array of shape (N_PIXELS,) with raw expression values
    for genes in master_index; unmapped pixels are 0.
    """
    vals = np.zeros(N_PIXELS, dtype=np.float32)
    with open(expr_file, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            if gene not in master_index:
                continue
            try:
                vals[master_index[gene]] = float(row[1])
            except (IndexError, ValueError):
                pass
    return vals


def normalise_expression(vals: np.ndarray) -> np.ndarray:
    """
    99th-percentile normalise to [0, 255], clip, return uint8.
    A per-cell-line normalisation ensures that the dynamic range of each
    image reflects intra-cell-line relative expression.
    """
    p99 = np.percentile(vals, 99)
    if p99 < 1e-6:
        return np.zeros_like(vals, dtype=np.uint8)
    normed = np.clip(vals / p99, 0.0, 1.0) * 255.0
    return normed.astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# 3.  Load mutations for all cell lines
# ─────────────────────────────────────────────────────────────

def load_mutations(mut_file: str) -> dict:
    """
    Returns dict {ach_id: {gene: severity_level (0-3)}}.
    Worst-case severity per (cell line, gene) is kept.
    """
    mut = defaultdict(lambda: defaultdict(int))
    with open(mut_file, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        # normalise column names
        fieldnames = [f.strip() for f in reader.fieldnames]
        id_col   = next((f for f in fieldnames if "modelid"   in f.lower()), None)
        gene_col = next((f for f in fieldnames if "hugosymbol" in f.lower() or
                                                   "gene" in f.lower()), None)
        vc_col   = next((f for f in fieldnames if "variantclassification" in f.lower() or
                                                   "variant" in f.lower()), None)
        if not all([id_col, gene_col, vc_col]):
            raise ValueError(
                f"Could not find required columns in {mut_file}.\n"
                f"  Detected: {fieldnames}\n"
                f"  Need: ModelID, HugoSymbol, VariantClassification"
            )
        for row in reader:
            ach  = row[id_col].strip()
            gene = row[gene_col].strip()
            vc   = row[vc_col].strip()
            sev  = MUT_SEVERITY.get(vc, 0)
            if sev > mut[ach][gene]:
                mut[ach][gene] = sev
    log.info("Mutations loaded: %d cell lines.", len(mut))
    return dict(mut)


# ─────────────────────────────────────────────────────────────
# 4.  Build one genomic image
# ─────────────────────────────────────────────────────────────

def build_image(expr_vals_raw: np.ndarray,
                mut_map: dict,
                master_index: dict) -> np.ndarray:
    """
    Returns uint8 array of shape (IMAGE_SIZE, IMAGE_SIZE, 3).

    R = normalised expression
    G = R * mutation_severity_pixel  (expressed-AND-mutated signal)
    B = 0  (reserved for CNV)
    """
    R_flat = normalise_expression(expr_vals_raw)          # uint8, (N_PIXELS,)

    # Build severity map
    G_sev_flat = np.zeros(N_PIXELS, dtype=np.uint8)
    for gene, sev in mut_map.items():
        if gene in master_index:
            G_sev_flat[master_index[gene]] = SEVERITY_TO_G[sev]

    # G channel = expression × severity, re-normalised to [0,255]
    G_flat = (R_flat.astype(np.float32) / 255.0 *
              G_sev_flat.astype(np.float32)).astype(np.uint8)

    B_flat = np.zeros(N_PIXELS, dtype=np.uint8)

    # Stack and reshape
    rgb_flat = np.stack([R_flat, G_flat, B_flat], axis=1)  # (N_PIXELS, 3)
    img = np.zeros((IMAGE_SIZE * IMAGE_SIZE, 3), dtype=np.uint8)
    img[:N_PIXELS] = rgb_flat
    img = img.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    return img


# ─────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate genomic images for CellLineTower.")
    p.add_argument("--expr_dir",   required=True,
                   help="Directory containing per-cell-line expression files (ACH-*.txt).")
    p.add_argument("--mut_file",   default=None,
                   help="DepMap OmicsSomaticMutations.csv. If omitted, G channel = 0.")
    p.add_argument("--output_dir", default="genomic_images",
                   help="Directory to write PNG images (default: ./genomic_images).")
    p.add_argument("--dry_run",    action="store_true",
                   help="Build index and print stats only; write no images.")
    return p.parse_args()


def main():
    args = parse_args()

    expr_files = sorted(glob.glob(os.path.join(args.expr_dir, "ACH-*.txt")))
    if not expr_files:
        log.error("No expression files (ACH-*.txt) found in %s", args.expr_dir)
        sys.exit(1)
    log.info("Found %d expression files.", len(expr_files))

    # Build master gene index from the first file
    master_index = build_master_index(expr_files[0])

    if args.dry_run:
        log.info("Dry run complete.  Index has %d genes.  "
                 "Image size: %d × %d.  No images written.", len(master_index),
                 IMAGE_SIZE, IMAGE_SIZE)
        return

    # Load mutations (optional)
    all_muts: dict = {}
    if args.mut_file:
        all_muts = load_mutations(args.mut_file)
    else:
        log.warning("--mut_file not provided.  G channel will be zero for all images.")

    os.makedirs(args.output_dir, exist_ok=True)

    n_written = 0
    for ef in expr_files:
        ach_id = os.path.splitext(os.path.basename(ef))[0]   # e.g. ACH-000001
        out_path = os.path.join(args.output_dir, f"{ach_id}.png")
        if os.path.exists(out_path):
            continue   # skip already-generated images

        expr_raw = load_expression(ef, master_index)
        mut_map  = all_muts.get(ach_id, {})
        img_arr  = build_image(expr_raw, mut_map, master_index)

        Image.fromarray(img_arr, mode="RGB").save(out_path)
        n_written += 1
        if n_written % 10 == 0:
            log.info("Written %d / %d images …", n_written, len(expr_files))

    log.info("Done.  %d new images written to %s", n_written, args.output_dir)


if __name__ == "__main__":
    main()
