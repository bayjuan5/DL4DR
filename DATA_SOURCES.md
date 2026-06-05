# Data Sources

This document records the origin, version, and access instructions for every
data file used in the DL4DR project.  Reviewers and collaborators should be
able to reproduce the dataset from scratch using the links below.

---

## 1. Drug Response Data

### `BREAST-136344-56786-51.txt`

| Field | Value |
|---|---|
| **Records** | 136,344 (136,342 after removing 2 with ln_IC₅₀ < 0) |
| **Cell lines** | 51 breast cancer lines (identified by ACH-ID) |
| **Unique compounds** | 56,786 (by `smi_id`) |
| **Matrix density** | 4.71% |
| **IC₅₀ range (ln scale)** | 0.02 – 13.0 (median 4.83) |
| **Assay types** | Resazurin, Resazurin_or_Syto60, MTT, MTS, SRB, CellTitreGlo, Alamar Blue, EdU, clonogenic, Matrigel, WST-8, XTT, SAA, sulforhodamine B |
| **Treatment duration** | 72 h (standard GDSC protocol) |

**Column layout** (semicolon-delimited, **no header row**):

| Column | Example | Description |
|---|---|---|
| 1 | `GDSC1#AZ628` | Source library + drug identifier |
| 2 | `smi_1` | Compound key (join to SMILES file) |
| 3 | `HCC1187_BREAST` | Cell line name |
| 4 | `ACH-000111` | DepMap ACH identifier |
| 5 | `5.37` | **ln(IC₅₀)**, already log-transformed |
| 6 | `Resazurin_or_Syto60` | Viability assay type |
| 7 | `72` | Treatment duration (hours) |

**Source:**  
Derived from the **GDSC (Genomics of Drug Sensitivity in Cancer)** database,
filtered to breast cancer cell lines present in DepMap.

- GDSC website: https://www.cancerrxgene.org
- Direct download (bulk IC₅₀ data): https://www.cancerrxgene.org/downloads/bulk_download
- Primary reference:  
  Yang, W. et al. (2012). Genomics of Drug Sensitivity in Cancer (GDSC):
  a resource for therapeutic biomarker discovery in cancer cells.
  *Nucleic Acids Research*, 41(D1), D955–D961.
  https://doi.org/10.1093/nar/gks1111

**Cell line identifiers (ACH-IDs):**  
ACH-IDs are DepMap model identifiers.  
- DepMap portal: https://depmap.org/portal/
- Model annotations: https://depmap.org/portal/download/  
  → File: `Model.csv` (contains ACH-ID ↔ cell line name ↔ tissue mappings)

---

## 2. Compound SMILES

### `CompoundSmiles_full_140474.txt`

| Field | Value |
|---|---|
| **Entries** | 140,474 |
| **smi_id range** | `smi_0` – `smi_140473` |
| **Entries matching IC₅₀ file** | 56,786 |
| **Format** | Tab-delimited, no header |

**Column layout:**

| Column | Example | Description |
|---|---|---|
| 1 | `smi_1` | Compound key (matches col 2 of IC₅₀ file) |
| 2 | `N#CC(c1cccc...` | Canonical SMILES string |

**Source:**  
SMILES strings were retrieved from the **ChEMBL** database and **PubChem**,
matched against GDSC drug identifiers, and deduplicated by InChIKey.
The `smi_id` is an internal project identifier assigned sequentially.

- ChEMBL: https://www.ebi.ac.uk/chembl/ (release 33+)
- PubChem: https://pubchem.ncbi.nlm.nih.gov
- GDSC drug list: https://www.cancerrxgene.org/compounds

> **Note:** GDSC drug names are not unique identifiers — the same molecular
> structure can appear under multiple source names (e.g. `GDSC1#AZ628` and
> `CHEMBL2144069`).  Always join on `smi_id`, never on `drug_name`.

---

## 3. Genomic Images (generated, not tracked in git)

### `genomic_images/ACH-*.png`

These 51 files are **generated** by `generate_genomic_images.py` from two
DepMap bulk data files.  They are not committed to the repository.

**Required DepMap downloads** (from https://depmap.org/portal/download/):

| File | Description | DepMap release |
|---|---|---|
| `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | Bulk RNA-seq, log₂(TPM+1), one column per cell line | 23Q4 or later |
| `OmicsSomaticMutations.csv` | Somatic mutations with variant classification | 23Q4 or later |

Both files are on the **DepMap Public** tab → "Omics" section.

**Generation command:**
```bash
# Convert OmicsExpression to per-cell-line txt files first:
python scripts/depmap_expr_to_per_cl.py \
    --input  data/OmicsExpressionProteinCodingGenesTPMLogp1.csv \
    --output data/gene_expression_full/

# Then generate images:
python generate_genomic_images.py \
    --expr_dir  data/gene_expression_full \
    --mut_file  data/OmicsSomaticMutations.csv \
    --output_dir genomic_images
```

**Image encoding** (Section 13.2 of ORNN_DMPNN_Theory_v18.pdf):
- **R channel**: gene expression (99th-percentile normalised per cell line)
- **G channel**: expression × mutation severity  
  (WT=0, Missense=85/255, Splice=170/255, Truncating=255/255)
- **B channel**: reserved for copy-number variation (zero-filled)
- **Size**: 139 × 139 pixels = 19,321 pixels ≥ 19,177 protein-coding genes
- **Gene order**: alphabetical sort of gene symbols, row-major

---

## 4. External Validation Data (CellTiter-Glo)

Used in `external_validation.py` (Section 17 of theory document).

| Field | Value |
|---|---|
| **Records** | 51,118 |
| **Unique compounds** | 197 (88.8% overlap with training set by smi_id) |
| **Cell lines** | 603 across 27 tissue types — **all unseen during training** |
| **Assay** | CellTiter-Glo (uniform, unlike the mixed assays in training) |
| **IC₅₀ range** | 1.88 – 9.64 (median 4.56) |

**Source:**  
Downloaded from the **NCI-60** / DepMap CellTiter-Glo viability screen.

- DepMap portal: https://depmap.org/portal/download/
  → Search for "CellTiter-Glo" or "PRISM Repurposing"
- Alternative: PRISM repurposing screen
  https://www.depmap.org/repurposing/

---

## 5. Driver Gene Annotation

Used in `run_gradcam.py` for known-driver highlighting.

**Sources:**

| Source | Genes included | Reference |
|---|---|---|
| IntOGen (breast) | High-confidence drivers | Martínez-Jiménez et al., *Nat. Rev. Cancer* 2020, 20:555–572. https://doi.org/10.1038/s41568-020-0290-x |
| TCGA BRCA SMGs | Significantly mutated genes | TCGA Research Network (2012), *Nature* 490:61–70 |
| COSMIC CGC | Cancer Gene Census (breast-relevant) | https://cancer.sanger.ac.uk/census |
| Pilot GradCAM hits | Epigenetic / CIN regulators flagged in v1 run | KMT2A, KMT2C, HERC2, SPEN, BRWD1, NUMA1, TACC2 |

The full list (188 genes) is hard-coded in `run_gradcam.py` under `KNOWN_DRIVERS`.

---

## 6. `.gitignore` recommendations

The following files should **not** be committed to the repository:

```
# Large data files
data/
genomic_images/
gradcam_outputs/
checkpoints/
results/

# DepMap bulk downloads
OmicsExpression*.csv
OmicsSomaticMutations.csv
cellTiter_Glo.txt

# Generated assets
*.pt
*.npy
```

The two files that **should** be committed (or linked via Git LFS):
- `BREAST-136344-56786-51.txt`  (136 K rows, ~8 MB)
- `CompoundSmiles_full_140474.txt`  (140 K rows, ~12 MB)

For Git LFS:
```bash
git lfs track "*.txt"
git add .gitattributes
git add data/BREAST-136344-56786-51.txt data/CompoundSmiles_full_140474.txt
```

---

## Reproducibility checklist

- [ ] Download `BREAST-136344-56786-51.txt` and `CompoundSmiles_full_140474.txt` ✓ (in repo / LFS)
- [ ] Download `OmicsExpressionProteinCodingGenesTPMLogp1.csv` from DepMap 23Q4+
- [ ] Download `OmicsSomaticMutations.csv` from DepMap 23Q4+
- [ ] Run `generate_genomic_images.py` to produce `genomic_images/ACH-*.png`
- [ ] (Optional) Download `cellTiter_Glo.txt` for external validation
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Train: `python train.py --data ... --genomic genomic_images --split random`
- [ ] Evaluate: `python evaluate.py --ckpt checkpoints/best_random.pt --split all`
