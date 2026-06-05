"""
dataset.py
==========
Data loading, three evaluation splits, and sqrt-temperature sampling
for the Two-Tower Residual Late Fusion drug response model.

Sections referenced: 11 (Data Summary), 12 (Preprocessing),
                     14 (sqrt-Temperature Sampling), 15.1 (Data Splits).

Actual file formats (verified against uploaded data):
─────────────────────────────────────────────────────
BREAST-136344-56786-51.txt  (semicolon-delimited, NO header):
  col 1: drug_name   e.g. "GDSC1#AZ628"
  col 2: smi_id      e.g. "smi_1"
  col 3: cl_name     e.g. "HCC1187_BREAST"
  col 4: ach_id      e.g. "ACH-000111"
  col 5: ln_ic50     e.g. "5.37"
  col 6: assay_type  e.g. "Resazurin_or_Syto60"
  col 7: duration_h  e.g. "72"

CompoundSmiles_full_140474.txt  (tab-delimited, NO header):
  col 1: smi_id      e.g. "smi_0"
  col 2: smiles      e.g. "COCCOc1cc2..."
  Total: 140,474 entries (smi_0 … smi_140473)
  Note: 56,786 smi_ids appear in the IC50 file; the rest are additional
        compounds not tested on these 51 cell lines.
"""

import os
import csv
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import GroupKFold
from collections import defaultdict
from typing import Optional, Tuple, List, Dict
from PIL import Image
import torchvision.transforms as T

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1.  Load SMILES lookup  (CompoundSmiles_full_140474.txt)
# ─────────────────────────────────────────────────────────────

def load_smiles_lookup(smiles_file: str) -> Dict[str, str]:
    """
    Returns {smi_id: smiles_string} from CompoundSmiles_full_140474.txt.
    File is tab-delimited, no header.
    """
    lookup = {}
    with open(smiles_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                lookup[parts[0].strip()] = parts[1].strip()
    log.info("SMILES lookup: %d entries loaded from %s",
             len(lookup), os.path.basename(smiles_file))
    return lookup


# ─────────────────────────────────────────────────────────────
# 2.  Raw data loading & preprocessing  (Section 12)
# ─────────────────────────────────────────────────────────────

def load_records(data_file: str, smiles_file: str) -> List[Dict]:
    """
    Load BREAST-136344-56786-51.txt and join SMILES from
    CompoundSmiles_full_140474.txt.

    Column layout (no header, semicolon-delimited):
        0: drug_name   (GDSC source + drug identifier)
        1: smi_id      (compound key — use this, NOT drug_name)
        2: cl_name     (cell line name, e.g. HCC1187_BREAST)
        3: ach_id      (DepMap ACH identifier)
        4: ln_ic50     (already log-transformed)
        5: assay_type
        6: duration_h

    Preprocessing (Section 12):
      - Drop records where ln_ic50 < 0  (2 measurement errors)
      - Deduplicate (smi_id, ach_id) pairs by taking the mean  (30 duplicates)
      - Skip smi_ids with no entry in smiles_file  (should be zero for this dataset)
    """
    smiles_lookup = load_smiles_lookup(smiles_file)

    raw, skipped_smiles = [], 0
    with open(data_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            row = line.split(";")
            if len(row) < 5:
                continue
            try:
                smi_id  = row[1].strip()
                ach_id  = row[3].strip()
                ln_ic50 = float(row[4].strip())
            except (ValueError, IndexError):
                continue

            if smi_id not in smiles_lookup:
                skipped_smiles += 1
                continue

            raw.append({
                "drug_name":  row[0].strip(),
                "smi_id":     smi_id,
                "cl_name":    row[2].strip(),
                "ach_id":     ach_id,
                "ln_ic50":    ln_ic50,
                "assay_type": row[5].strip() if len(row) > 5 else "",
                "smiles":     smiles_lookup[smi_id],
            })

    if skipped_smiles:
        log.warning("Skipped %d records with no SMILES entry.", skipped_smiles)

    # Drop negative IC50
    n_before = len(raw)
    raw = [r for r in raw if r["ln_ic50"] >= 0]
    log.info("Dropped %d records with ln_ic50 < 0.", n_before - len(raw))

    # Deduplicate (smi_id, ach_id) — take mean of ln_ic50
    groups: Dict[tuple, List[float]] = defaultdict(list)
    meta:   Dict[tuple, Dict]        = {}
    for r in raw:
        key = (r["smi_id"], r["ach_id"])
        groups[key].append(r["ln_ic50"])
        meta[key] = r

    deduped = []
    for key, vals in groups.items():
        rec = dict(meta[key])
        rec["ln_ic50"] = float(np.mean(vals))
        deduped.append(rec)

    n_dups = len(raw) - len(deduped)
    log.info("Loaded %d records  →  %d after deduplication (%d duplicates merged).",
             len(raw), len(deduped), n_dups)
    log.info("Unique smi_ids: %d  |  Unique ACH ids: %d",
             len({r["smi_id"] for r in deduped}),
             len({r["ach_id"] for r in deduped}))
    return deduped


# ─────────────────────────────────────────────────────────────
# 3.  Additive and simple baselines  (Section 15.2)
# ─────────────────────────────────────────────────────────────

def compute_baselines(records: List[Dict]) -> Dict:
    """
    Returns:
        global_mean   : float
        per_cl_mean   : {ach_id: float}
        per_cmp_mean  : {smi_id: float}

    Additive baseline:
        pred = per_cl_mean[ach] + per_cmp_mean[smi] - global_mean
    """
    all_ic = [r["ln_ic50"] for r in records]
    g_mean = float(np.mean(all_ic))

    cl_vals:  Dict[str, list] = defaultdict(list)
    cmp_vals: Dict[str, list] = defaultdict(list)
    for r in records:
        cl_vals[r["ach_id"]].append(r["ln_ic50"])
        cmp_vals[r["smi_id"]].append(r["ln_ic50"])

    baselines = {
        "global_mean":  g_mean,
        "per_cl_mean":  {k: float(np.mean(v)) for k, v in cl_vals.items()},
        "per_cmp_mean": {k: float(np.mean(v)) for k, v in cmp_vals.items()},
    }
    log.info("Baselines — global_mean=%.4f  |  %d cell lines  |  %d compounds",
             g_mean, len(baselines["per_cl_mean"]), len(baselines["per_cmp_mean"]))
    return baselines


def additive_pred(rec: Dict, baselines: Dict) -> float:
    cl  = baselines["per_cl_mean"].get(rec["ach_id"],  baselines["global_mean"])
    cp  = baselines["per_cmp_mean"].get(rec["smi_id"], baselines["global_mean"])
    return cl + cp - baselines["global_mean"]


# ─────────────────────────────────────────────────────────────
# 4.  Three splits  (Section 15.1)
# ─────────────────────────────────────────────────────────────

def random_split(records: List[Dict],
                 train_frac: float = 0.8,
                 val_frac:   float = 0.1,
                 seed: int = 42) -> Tuple[List, List, List]:
    """Row-level 80/10/10 random split — upper bound (leaks)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n_tr = int(len(records) * train_frac)
    n_va = int(len(records) * val_frac)
    return ([records[i] for i in idx[:n_tr]],
            [records[i] for i in idx[n_tr: n_tr + n_va]],
            [records[i] for i in idx[n_tr + n_va:]])


def leave_compound_out_split(records: List[Dict],
                             seed: int = 42) -> Tuple[List, List, List]:
    """
    Group by smi_id, split compound groups 80/10/10.
    Tests compound tower generalisation to novel molecular structures.
    """
    rng    = np.random.default_rng(seed)
    groups: Dict[str, list] = defaultdict(list)
    for r in records:
        groups[r["smi_id"]].append(r)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    n_tr = int(len(keys) * 0.8)
    n_va = int(len(keys) * 0.1)

    def flatten(ks):
        out = []
        for k in ks: out.extend(groups[k])
        return out

    return (flatten(keys[:n_tr]),
            flatten(keys[n_tr: n_tr + n_va]),
            flatten(keys[n_tr + n_va:]))


def leave_cell_line_out_splits(records: List[Dict],
                               n_folds: int = 5,
                               seed:    int = 42) -> List[Tuple[List, List]]:
    """
    GroupKFold by ach_id.
    The scientifically decisive split — tests whether the genomic encoder
    generalises to entirely unseen cell lines.
    Returns list of (train_records, test_records) per fold.
    """
    gkf    = GroupKFold(n_splits=n_folds)
    groups = [r["ach_id"] for r in records]
    X      = np.arange(len(records))
    return [([records[i] for i in tr], [records[i] for i in te])
            for tr, te in gkf.split(X, groups=groups)]


# ─────────────────────────────────────────────────────────────
# 5.  sqrt-Temperature sampler  (Section 14)
# ─────────────────────────────────────────────────────────────

def make_sampler(records: List[Dict], alpha: float = 0.5) -> WeightedRandomSampler:
    """
    P(cell line c) ∝ n_c^alpha.
    Row weight = n_c^(alpha - 1).

    alpha = 0.5  →  sqrt-temperature  (recommended)
    alpha = 1.0  →  uniform random sampling
    alpha = 0.0  →  uniform per cell line

    The 2050:1 record imbalance becomes ~45:1 at alpha=0.5.
    IMPORTANT: compute counts from training records ONLY.
    """
    counts: Dict[str, int] = defaultdict(int)
    for r in records:
        counts[r["ach_id"]] += 1

    weights = np.array(
        [counts[r["ach_id"]] ** (alpha - 1.0) for r in records],
        dtype=np.float64,
    )
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(records),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────
# 6.  ECFP + molecular image featurisation  (requires RDKit)
# ─────────────────────────────────────────────────────────────

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import MolToImage
    _RDKIT = True
except ImportError:
    _RDKIT = False
    log.warning("RDKit not found — ECFP and molecular images will be zero tensors.")


def smiles_to_ecfp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    if not _RDKIT:
        return np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    for b in fp.GetOnBits():
        arr[b] = 1.0
    return arr


def smiles_to_image(smiles: str, size: int = 224) -> torch.Tensor:
    """Returns (3, size, size) float32 tensor in [0, 1]."""
    if not _RDKIT:
        return torch.zeros(3, size, size)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(3, size, size)
    pil = MolToImage(mol, size=(size, size))
    arr = np.array(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ─────────────────────────────────────────────────────────────
# 7.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────

_CELL_TRANSFORM = T.ToTensor()   # uint8 HWC → float32 CHW in [0, 1]


class DrugResponseDataset(Dataset):
    """
    One item = one (compound, cell_line) pair.

    Parameters
    ----------
    records       : list of dicts from load_records()
    genomic_dir   : path to ACH-*.png genomic images
    ecfp_dim      : ECFP bit-vector length
    mol_img_size  : size of 2D molecular depiction (pixels)
    cell_img_cache: optional pre-loaded {ach_id: Tensor} for speed
    """

    def __init__(self,
                 records: List[Dict],
                 genomic_dir: str,
                 ecfp_dim: int = 2048,
                 mol_img_size: int = 224,
                 cell_img_cache: Optional[Dict[str, torch.Tensor]] = None):
        self.records      = records
        self.genomic_dir  = genomic_dir
        self.ecfp_dim     = ecfp_dim
        self.mol_img_size = mol_img_size
        self.cell_cache   = cell_img_cache or {}

        all_cl = sorted({r["ach_id"] for r in records})
        self.cl2idx = {cl: i for i, cl in enumerate(all_cl)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r      = self.records[idx]
        smiles = r["smiles"]
        ach    = r["ach_id"]

        ecfp    = torch.from_numpy(smiles_to_ecfp(smiles, n_bits=self.ecfp_dim))
        mol_img = smiles_to_image(smiles, size=self.mol_img_size)

        if ach in self.cell_cache:
            cell_img = self.cell_cache[ach]
        else:
            p = os.path.join(self.genomic_dir, f"{ach}.png")
            cell_img = (_CELL_TRANSFORM(Image.open(p).convert("RGB"))
                        if os.path.exists(p) else torch.zeros(3, 139, 139))

        return {
            "ecfp":     ecfp,
            "mol_img":  mol_img,
            "cell_img": cell_img,
            "cell_idx": torch.tensor(self.cl2idx[ach], dtype=torch.long),
            "target":   torch.tensor(r["ln_ic50"], dtype=torch.float32),
            "smi_id":   r["smi_id"],
            "ach_id":   ach,
        }


def preload_genomic_images(genomic_dir: str,
                           ach_ids: Optional[List[str]] = None
                           ) -> Dict[str, torch.Tensor]:
    """Load all 51 genomic images into RAM once at training start."""
    ids = ach_ids or [
        os.path.splitext(f)[0]
        for f in os.listdir(genomic_dir) if f.endswith(".png")
    ]
    cache = {}
    for ach in ids:
        p = os.path.join(genomic_dir, f"{ach}.png")
        if os.path.exists(p):
            cache[ach] = _CELL_TRANSFORM(Image.open(p).convert("RGB"))
    log.info("Preloaded %d genomic images into RAM.", len(cache))
    return cache
