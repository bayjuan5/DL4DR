"""
eval_novelty_and_conditioning.py

Follow-up evaluation for the conditional SMILES VAE (generative/best_vae.pt
and generative/final_vae.pt), beyond raw validity rate.

Run from inside generative/, e.g.:

    python eval_novelty_and_conditioning.py \
        --vae_ckpt checkpoints_gen/best_vae.pt \
        --dl4dr_ckpt ../checkpoints/best_random.pt \
        --data ../data/BREAST-136344-56786-51.txt \
        --smiles ../data/CompoundSmiles_full_140474.txt \
        --genomic ../genomic_images \
        --n_samples 200

It answers three questions:

  Q1. NOVELTY   - Of the valid molecules generated, how many are exact
                  duplicates of a training-set compound vs. genuinely new?
                  Also reports nearest-neighbor Tanimoto similarity to the
                  training set for the "new" ones, so we know if they're
                  trivial near-copies (e.g. sim > 0.95) or real novel
                  scaffolds.

  Q2. UNIQUENESS - Of the valid molecules generated for a single cell line,
                  how many are duplicates of each other (mode collapse
                  check)?

  Q3. CONDITIONAL SENSITIVITY - Sample from 3 different cell lines' z_C.
                  If the generated sets barely differ (high overlap,
                  near-identical validity/uniqueness), the model is likely
                  ignoring the conditioning signal and just generating a
                  generic "good enough" SMILES distribution regardless of
                  z_C. We want LOW overlap between cell lines here.

Needs: rdkit, torch, pillow, numpy (same env as train.py).
"""
import argparse
import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

from model import SmilesVocab, FrozenCellLineEncoder, ConditionalSmilesVAE

MAX_LEN = 120


def load_root_dataset(repo_root: Path):
    spec = importlib.util.spec_from_file_location("dl4dr_root_dataset", repo_root / "dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def canonical(smiles: str):
    """Return RDKit canonical SMILES, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def sample_for_cell_line(vae, cell_encoder, vocab, img_tensor, n_samples, device):
    """Generate n_samples SMILES (raw, undecoded validity not yet checked) for one cell line image."""
    with torch.no_grad():
        z_c = cell_encoder(img_tensor.unsqueeze(0).to(device))
        tokens = vae.sample(z_c, n_samples=n_samples)
    out = []
    for row in tokens:
        smi = vocab.decode(row.tolist())
        out.append(smi)
    return out


def evaluate_cell_line(raw_smiles_list, train_canon_set, train_fps, train_fp_smiles):
    """Given raw generated SMILES strings for one cell line, compute validity/uniqueness/novelty."""
    valid_canon = []
    for smi in raw_smiles_list:
        c = canonical(smi)
        if c is not None:
            valid_canon.append(c)

    validity = len(valid_canon) / max(1, len(raw_smiles_list))
    unique_valid = set(valid_canon)
    uniqueness = len(unique_valid) / max(1, len(valid_canon)) if valid_canon else 0.0

    exact_dupes = [s for s in unique_valid if s in train_canon_set]
    novel = [s for s in unique_valid if s not in train_canon_set]
    novelty_rate = len(novel) / max(1, len(unique_valid)) if unique_valid else 0.0

    # Nearest-neighbor Tanimoto similarity for the "novel" set, to catch
    # trivial near-copies (e.g. a single substituent swapped).
    nn_sims = []
    for smi in novel:
        fp = morgan_fp(smi)
        if fp is None or not train_fps:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        nn_sims.append(max(sims) if sims else 0.0)

    return {
        "n_raw": len(raw_smiles_list),
        "validity": validity,
        "n_valid_unique": len(unique_valid),
        "uniqueness": uniqueness,
        "n_exact_train_dupes": len(exact_dupes),
        "novelty_rate": novelty_rate,
        "novel_smiles": novel,
        "nn_tanimoto_mean": float(np.mean(nn_sims)) if nn_sims else None,
        "nn_tanimoto_over_0.95": sum(1 for s in nn_sims if s > 0.95),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_ckpt", required=True, help="e.g. checkpoints_gen/best_vae.pt")
    ap.add_argument("--dl4dr_ckpt", required=True, help="frozen DL4DR checkpoint, e.g. ../checkpoints/best_random.pt")
    ap.add_argument("--data", required=True, help="../data/BREAST-136344-56786-51.txt")
    ap.add_argument("--smiles", required=True, help="../data/CompoundSmiles_full_140474.txt")
    ap.add_argument("--genomic", required=True, help="../genomic_images")
    ap.add_argument("--n_samples", type=int, default=200, help="samples to draw per cell line")
    ap.add_argument("--n_cell_lines", type=int, default=3, help="how many distinct cell lines to compare")
    ap.add_argument("--zc_dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = Path(args.data).resolve().parent.parent
    root_dataset = load_root_dataset(repo_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Rebuild vocab EXACTLY as in train.py (same data files, same order)
    records_all = root_dataset.load_records(args.data, args.smiles)
    vocab = SmilesVocab([r["smiles"] for r in records_all])
    print(f"Vocab size: {len(vocab)} | training records: {len(records_all)}")

    # Training-set canonical SMILES set + fingerprints, for novelty checks
    train_canon_set = set()
    train_fps = []
    train_fp_smiles = []
    for r in records_all:
        c = canonical(r["smiles"])
        if c is not None and c not in train_canon_set:
            train_canon_set.add(c)
            fp = morgan_fp(c)
            if fp is not None:
                train_fps.append(fp)
                train_fp_smiles.append(c)
    print(f"Unique valid training compounds: {len(train_canon_set)}")

    # Pick N distinct cell lines (by ach_id) to compare conditioning sensitivity
    all_ach_ids = sorted(set(r["ach_id"] for r in records_all))
    random.shuffle(all_ach_ids)
    chosen = all_ach_ids[: args.n_cell_lines]
    print(f"Comparing cell lines: {chosen}")

    # Load models
    cell_encoder = FrozenCellLineEncoder(args.dl4dr_ckpt, device=device).to(device)
    vae = ConditionalSmilesVAE(vocab_size=len(vocab), zc_dim=args.zc_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()

    results = {}
    for ach_id in chosen:
        img_path = Path(args.genomic) / f"{ach_id}.png"
        if not img_path.exists():
            print(f"  [skip] no genomic image for {ach_id}")
            continue
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).permute(2, 0, 1)

        raw = sample_for_cell_line(vae, cell_encoder, vocab, img_tensor, args.n_samples, device)
        res = evaluate_cell_line(raw, train_canon_set, train_fps, train_fp_smiles)
        results[ach_id] = res

        print(f"\n=== {ach_id} ===")
        print(f"  Q1 novelty     : validity={res['validity']:.2%} | "
              f"exact_train_dupes={res['n_exact_train_dupes']} | "
              f"novelty_rate={res['novelty_rate']:.2%} | "
              f"mean NN-Tanimoto to train (novel set)={res['nn_tanimoto_mean']}")
        print(f"  Q2 uniqueness  : {res['uniqueness']:.2%} "
              f"({res['n_valid_unique']} unique valid / {res['n_raw']} sampled)")

    # Q3: conditional sensitivity -- overlap of *novel valid* sets across cell lines
    if len(results) >= 2:
        print("\n=== Q3: conditional sensitivity (pairwise overlap of novel valid sets) ===")
        ach_list = list(results.keys())
        for i in range(len(ach_list)):
            for j in range(i + 1, len(ach_list)):
                a, b = ach_list[i], ach_list[j]
                set_a = set(results[a]["novel_smiles"])
                set_b = set(results[b]["novel_smiles"])
                union = set_a | set_b
                overlap = len(set_a & set_b) / len(union) if union else 0.0
                print(f"  {a} vs {b}: overlap={overlap:.2%} "
                      f"({len(set_a & set_b)} shared / {len(union)} union)")
        print("  -> LOW overlap is the desired outcome: it means the generated compound")
        print("     sets actually differ by cell line, i.e. z_C conditioning is being used.")
        print("     HIGH overlap (most molecules shared across cell lines) would suggest the")
        print("     VAE is mostly ignoring z_C and generating a generic distribution.")


if __name__ == "__main__":
    main()
