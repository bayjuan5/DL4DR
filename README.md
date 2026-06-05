# DL4DR — Residual Multi-Modal Learning for Drug Response Prediction

**From single-cell-line memorisation to multi-cell-line genomic image conditioning**

> MD Anderson Cancer Center · DL4DR Project  
> Theory document: `ORNN_DMPNN_Theory_v18.pdf`

---

## Overview

This repository implements the **Two-Tower Residual Late Fusion (RLF)** architecture for
predicting cancer drug response (IC₅₀) across multiple cell lines. The model jointly encodes:

| Modality | Tower | Encoder |
|---|---|---|
| Molecular structure (SMILES) | Compound | D-MPNN (directed message-passing) |
| 2D molecular depiction | Compound | ORNN (octave residual CNN) |
| ECFP fingerprint | Hard-memory head | Fixed MLP |
| Genomic image (139×139×3 RGB) | Cell Line | CNN (no ID lookup) |

The fusion equation is:

```
f(compound, cell_line) = f_hard(x_ECFP ⊕ z_C)
                       + λ(z_C) · f_residual( CrossAttn(Q=z_C, KV=[z_ORNN, z_DMPNN]) )
```

**Key results (51 breast cancer cell lines, DepMap):**
- Residual Fusion outperforms ECFP-Only in **48 / 51 cell lines (94.1%)**
- Mean ΔR² = +0.016 across all 51 lines
- Random-split R² = 0.61–0.69 across representative cell lines
- CellLineTower GradCAM independently recovers **PIK3CA** as the top genomic discriminator (30× enrichment over chance for known cancer drivers)

---

## Repository Structure

```
DL4DR/
├── generate_genomic_images.py   # Step 1: DepMap expression/mutation → 139×139 RGB images
├── model.py                     # Two-Tower RLF architecture (full PyTorch)
├── dataset.py                   # Data loading, three splits, sqrt-temperature sampler
├── train.py                     # Training loop (SmoothL1, AdamW, cosine LR)
├── evaluate.py                  # Three-split evaluation + baseline comparison
├── external_validation.py       # CellTiter-Glo external validation (603 cell lines)
├── run_gradcam.py               # CellLineTower GradCAM interpretability
└── README.md
```

---

## Data Sources

Full provenance, column layouts, DOIs, and download links are in **[DATA_SOURCES.md](DATA_SOURCES.md)**.

| File | Size | Origin |
|---|---|---|
| `BREAST-136344-56786-51.txt` | 136,344 rows | [GDSC](https://www.cancerrxgene.org/downloads/bulk_download) filtered to 51 DepMap breast cancer lines |
| `CompoundSmiles_full_140474.txt` | 140,474 rows | ChEMBL + PubChem, matched to GDSC drug identifiers |
| `genomic_images/ACH-*.png` | 51 files (generated) | Built from DepMap `OmicsExpression` + `OmicsSomaticMutations` via `generate_genomic_images.py` |
| `cellTiter_Glo.txt` | 51,118 rows (external val.) | DepMap PRISM CellTiter-Glo, 603 cell lines / 27 tissue types |

The two `.txt` files are included in this repository (or tracked via Git LFS).
Genomic images must be generated locally — see Step 3 below.

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision numpy scipy scikit-learn pillow matplotlib rdkit-pypi
```

For D-MPNN graph featurisation (optional but recommended):
```bash
pip install chemprop
```

### 2. Prepare data

Download from [DepMap portal](https://depmap.org/portal/):
- `OmicsExpressionProteinCodingGenesTPMLogp1.csv` — bulk RNA-seq (one file per cell line)
- `OmicsSomaticMutations.csv` — somatic mutations
- Drug response: search for GDSC IC₅₀ data → `BREAST-136344-56786-51.txt` (semicolon-delimited)

For external validation, download `CellTiter_Glo_viability.csv` from the same portal.

### 3. Generate genomic images

```bash
python generate_genomic_images.py \
    --expr_dir  data/gene_expression_full \
    --mut_file  data/OmicsSomaticMutations.csv \
    --output_dir genomic_images
```

Each cell line produces one `ACH-XXXXXX.png` (139×139 RGB):
- **R channel** — gene expression (99th-percentile normalised)
- **G channel** — expression × mutation severity (WT=0, Missense=85, Splice=170, Truncating=255)
- **B channel** — reserved for copy-number variation

Dry-run (index only, no images):
```bash
python generate_genomic_images.py --expr_dir data/gene_expression_full --dry_run
```

### 4. Train

```bash
# Random split (upper bound, fastest convergence)
python train.py \
    --smiles  data/CompoundSmiles_full_140474.txt \
    --data    data/BREAST-136344-56786-51.txt \
    --genomic genomic_images \
    --split   random \
    --epochs  60 \
    --batch   256 \
    --lr      3e-4

# Leave-compound-out (tests compound tower generalisation)
python train.py --smiles data/CompoundSmiles_full_140474.txt --data ... --genomic ... --split compound_out

# Leave-cell-line-out (tests genomic encoder generalisation — the key split)
python train.py --smiles data/CompoundSmiles_full_140474.txt --data ... --genomic ... --split cellline_out
```

Checkpoints are saved to `checkpoints/` automatically:
- `best_random.pt`
- `best_compound_out.pt`
- `best_cellline_out.pt`

### 5. Evaluate (all three splits)

```bash
python evaluate.py \
    --smiles  data/CompoundSmiles_full_140474.txt \
    --data    data/BREAST-136344-56786-51.txt \
    --genomic genomic_images \
    --ckpt    checkpoints/best_random.pt \
    --split   all
```

Outputs in `results/`:

| File | Description |
|---|---|
| `random_split_scatter.png` | 6-panel scatter (Figure 9 in theory doc) |
| `compound_out_bar.png` | Two-Tower vs additive baseline (Figure 10) |
| `cellline_out_scatter.png` | 51-cell-line diagonal scatter (Figure 11) |
| `cellline_r2_table.csv` | Per-cell-line R² with ΔR² vs ECFP-Only |
| `eval_summary.txt` | Summary table (Table 9) |

### 6. External validation (CellTiter-Glo)

```bash
# Step 0: audit data overlap
python external_validation.py --mode audit \
    --train    data/BREAST-136344-56786-51.txt \
    --external data/cellTiter_Glo.txt

# Step 1: generate predictions (build genomic images for the 603 ext cell lines first)
python external_validation.py --mode predict \
    --model       checkpoints/best_random.pt \
    --external    data/cellTiter_Glo.txt \
    --genomic_dir genomic_images_celltiter \
    --output      results/ext_predictions.csv \
    --train_embeddings results/train_cell_embeddings.npy

# Step 2: produce all four figures
python external_validation.py --mode analyse \
    --predictions       results/ext_predictions.csv \
    --train_embeddings  results/train_cell_embeddings.npy \
    --output_dir        results/figures
```

**Success criterion (Section 17.5):**
- Fig 1 Pearson r > 0.2, p < 0.05 → genomic encoder is transparent
- Fig 2 epithelial cancers rank higher than haematopoietic → biology generalises
- Fig 4 seen-compound R² ≥ unseen-compound R² → compound tower generalises

### 7. CellLineTower GradCAM interpretability

```bash
python run_gradcam.py
```

Configuration at the top of `run_gradcam.py`:
```python
CHECKPOINT_DIR  = "./checkpoints"
GENOMIC_IMG_DIR = "./genomic_images"
EXPR_FOLDER     = "../gene_expression_full"
OUTPUT_DIR      = "./gradcam_outputs"
TOP_K           = 20
```

Outputs per cell line: 4-panel figure (expression GradCAM | mutation GradCAM | combined | driver bar chart).  
Cross-cell-line summary: `driver_gene_frequency_summary.png` — frequency of each gene being a top activator across all 51 cell lines.

**Verified finding:** PIK3CA appears as the top cross-cell-line activator (10/51 cell lines, 30× enrichment over the chance expectation for known cancer drivers), without any pathway prior being injected during training.

---

## Evaluation Splits — Critical Notes

The three splits test fundamentally different things. **Never report only the random split.**

| Split | How | Tests | Valid claim |
|---|---|---|---|
| Random | Row-level 80/10/10 | Upper bound (leaks) | "The model fits the data" |
| Leave-compound-out | Group by `smi_id` | Compound tower | "Novel molecules are encodable" |
| **Leave-cell-line-out** | GroupKFold by ACH ID | **Genomic encoder** | **"Unseen cell lines are predictable"** |

The additive baseline (`cl_mean + cp_mean − global_mean`) must be beaten on the leave-cell-line-out split to claim interaction-level learning. Many published DRP models fail this test.

---

## Model Architecture

```
SMILES ──► D-MPNN ──► z_DMPNN ──┐
         ► ORNN   ──► z_ORNN  ──┤──► CrossAttn(Q=z_C) ──► λ·f_residual ──┐
ECFP ────────────────────────────┤                                         ├──► f (ln IC50)
                                 └──► f_hard(ECFP ⊕ z_C) ─────────────────┘
Genomic image ──► CellLineTower ──► z_C ──► λ(z_C) gate
```

Key design choices:
- **Residual structure** prevents the learnable branch from overwriting ECFP memorisation (Eq. 5)
- **Content-based cell line encoding** (no ID lookup) → generalises to unseen cell lines
- **GroupNorm** in CellLineTower (effective sample size = n_cell_lines, not n_records)
- **sqrt-temperature sampling** (α=0.5) balances 2050:1 cell-line record imbalance

---

## Interpretability Framework

Three independent interpretability axes (Section 9 of theory document):

| Case | Path | Tool | Question |
|---|---|---|---|
| Case 1 | ECFP hard-memory | Gradient × input on atom features | Which atoms drive predicted potency? |
| Case 2 | ORNN compound image | GradCAM (last Conv2d, high-freq branch) | Which structural regions drive the residual correction? |
| Case 3 | CellLineTower genomic | GradCAM (Block 3, `encoder[17]`) | Which genomic loci define cell line identity? |

---

## Requirements

```
python      >= 3.9
torch       >= 2.0
torchvision >= 0.15
rdkit-pypi  >= 2023.3
numpy       >= 1.24
scipy       >= 1.10
scikit-learn>= 1.2
pillow      >= 9.0
matplotlib  >= 3.7
chemprop    >= 2.0    (optional, for full D-MPNN graph batching)
```

---

## Citation

If you use this code or the theoretical framework, please cite:

```bibtex
@techreport{dl4dr2024,
  title  = {Residual Multi-Modal Learning for Drug Response Prediction:
            From Single-Cell-Line Memorization to Multi-Cell-Line Genomic Image Conditioning},
  author = {DL4DR Project, MD Anderson Cancer Center},
  year   = {2024},
  note   = {ORNN\_DMPNN\_Theory\_v18}
}
```

Driver gene annotation uses:
```bibtex
@article{martinez2020comprehensive,
  author  = {Mart{\'i}nez-Jim{\'e}nez, Francisco and others},
  title   = {A compendium of mutational cancer driver genes},
  journal = {Nature Reviews Cancer},
  volume  = {20}, pages = {555--572}, year = {2020}
}
```

---

## Acknowledgements

Computational resources: Texas Advanced Computing Center (TACC), allocation MCB23032.  
Model training: Google Colaboratory (NVIDIA Tesla T4).  
Data: DepMap consortium and GDSC project.
