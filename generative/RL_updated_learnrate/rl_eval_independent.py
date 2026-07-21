"""
rl_eval_independent.py
=======================
Evaluates an RL-finetuned generator (from rl_finetune.py) two ways at once:

  1. Reward-model score: what the frozen DL4DR predictor thinks of the
     generated molecules (the thing the policy was actually optimized for).
  2. Independent, reward-model-blind checks: things the predictor was never
     trained on and has no way to "see" or exploit —
       - RDKit validity / canonicalization (same as eval_novelty_and_conditioning.py)
       - Lipinski "Rule of Five" druglikeness (MW, LogP, H-bond donors/acceptors,
         rotatable bonds) — a cheap, standard, reward-model-independent filter
       - Nearest-neighbor Tanimoto similarity to known training-set actives
         (reused from eval_novelty_and_conditioning.py)
       - Uniqueness / mode-collapse check (reused)

The point of running these side by side: if reward-model score climbs while
the independent checks stay flat or fall, that gap IS the reward-hacking
signal this whole project is looking for. Log both every checkpoint so the
divergence (if any) is visible in the history, not just inferred after the
fact from a single final number.

Run from inside RL_updated_learnrate/, e.g.:

    python rl_eval_independent.py \
        --rl_ckpt checkpoints_gen_rl/final_rl.pt \
        --dl4dr_ckpt ../checkpoints/best_random.pt \
        --data ../data/BREAST-136344-56786-51.txt \
        --smiles ../data/CompoundSmiles_full_140474.txt \
        --genomic ../genomic_images \
        --n_samples 200
"""
import argparse
import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski

RDLogger.DisableLog("rdApp.*")

from rl_policy import PolicyWrapper

HERE = Path(__file__).resolve()
GEN_ROOT = HERE.parent.parent
REPO_ROOT = GEN_ROOT.parent

spec = importlib.util.spec_from_file_location("dl4dr_gen_model", GEN_ROOT / "model.py")
gen_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_model)
SmilesVocab = gen_model.SmilesVocab
FrozenCellLineEncoder = gen_model.FrozenCellLineEncoder
ConditionalSmilesVAE = gen_model.ConditionalSmilesVAE

spec2 = importlib.util.spec_from_file_location("dl4dr_root_dataset", REPO_ROOT / "dataset.py")
root_dataset = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(root_dataset)

spec3 = importlib.util.spec_from_file_location("dl4dr_repo_model", REPO_ROOT / "model.py")
repo_model = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(repo_model)
DrugResponseModel = repo_model.DrugResponseModel
smiles_to_graph = repo_model.smiles_to_graph
smiles_to_ecfp = repo_model.smiles_to_ecfp

MAX_LEN = 120


def canonical(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True)


def morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def lipinski_pass(smiles: str) -> bool:
    """Rule of Five — a standard, reward-model-blind druglikeness filter.
    The DL4DR predictor was never trained to enforce this; a generator that
    starts failing Lipinski while its reward-model score keeps improving is
    a concrete, checkable sign of exploiting the predictor rather than
    producing plausible drug-like molecules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return violations <= 1  # standard "at most one violation" Ro5 tolerance


def reward_model_score(predictor, smiles_list, genomic_img, device, mol_img_size=64):
    """Same scoring path as rl_finetune.py's score_with_predictor, kept
    separate here so this file has no import dependency on rl_finetune.py."""
    from torch_geometric.data import Batch
    from rdkit.Chem import Draw

    valid_idx, graphs, imgs, ecfps = [], [], [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        # RDKit quirk: Chem.MolFromSmiles("") returns a valid, 0-atom Mol
        # object (not None) - it slips past "mol is None" and produces a
        # PyG graph with zero nodes. Batch.from_data_list then contributes
        # no rows to data.batch for that graph, so the DMPNN batch-index
        # count (batch_vec.max()+1) comes out one short of the actual batch
        # size, and DrugResponseModel.forward's dmpnn_tokens_list[i] loop
        # (indexed by genomic_img.shape[0]) runs off the end of the list.
        # An under-trained/early-RL policy emits empty decodes often enough
        # (immediate EOS) that this isn't a rare edge case here.
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        try:
            g = smiles_to_graph(smi)
            e = smiles_to_ecfp(smi)
            img = Draw.MolToImage(mol, size=(mol_img_size, mol_img_size))
            arr = np.array(img).astype(np.float32) / 255.0
            img_t = torch.tensor(arr).permute(2, 0, 1)[:3]
        except Exception:
            continue
        valid_idx.append(i)
        graphs.append(g)
        imgs.append(img_t)
        ecfps.append(e)

    if not valid_idx:
        return [], []

    batch_graph = Batch.from_data_list(graphs).to(device)
    batch_img = torch.stack(imgs).to(device)
    batch_ecfp = torch.stack(ecfps).to(device)
    batch_genomic = genomic_img.unsqueeze(0).repeat(len(valid_idx), 1, 1, 1).to(device)

    with torch.no_grad():
        pred_ic50 = predictor(batch_genomic, batch_graph, batch_img, batch_ecfp)

    scores = (-pred_ic50).cpu().tolist()
    smiles_scored = [smiles_list[i] for i in valid_idx]
    return smiles_scored, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl_ckpt", required=True)
    ap.add_argument("--dl4dr_ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--genomic", required=True)
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--n_cell_lines", type=int, default=3)
    ap.add_argument("--zc_dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records_all = root_dataset.load_records(args.data, args.smiles)
    vocab = SmilesVocab([r["smiles"] for r in records_all])

    train_canon_set, train_fps = set(), []
    for r in records_all:
        c = canonical(r["smiles"])
        if c and c not in train_canon_set:
            train_canon_set.add(c)
            fp = morgan_fp(c)
            if fp is not None:
                train_fps.append(fp)
    print(f"Vocab size: {len(vocab)} | unique valid training compounds: {len(train_canon_set)}")

    cell_encoder = FrozenCellLineEncoder(args.dl4dr_ckpt, device=device).to(device)
    predictor = DrugResponseModel().to(device)
    pred_state = torch.load(args.dl4dr_ckpt, map_location=device)
    pred_state = pred_state.get("model") or pred_state.get("model_state_dict") or pred_state
    predictor.load_state_dict(pred_state, strict=False)
    predictor.eval()

    policy_net = ConditionalSmilesVAE(vocab_size=len(vocab), zc_dim=args.zc_dim).to(device)
    policy_net.load_state_dict(torch.load(args.rl_ckpt, map_location=device))
    policy_net.eval()
    policy = PolicyWrapper(policy_net)

    all_ach_ids = sorted(set(r["ach_id"] for r in records_all))
    random.shuffle(all_ach_ids)
    chosen = all_ach_ids[: args.n_cell_lines]
    print(f"Evaluating cell lines: {chosen}\n")

    for ach_id in chosen:
        img_path = Path(args.genomic) / f"{ach_id}.png"
        if not img_path.exists():
            print(f"  [skip] no genomic image for {ach_id}")
            continue
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).permute(2, 0, 1)

        with torch.no_grad():
            z_c = cell_encoder(img_tensor.unsqueeze(0).to(device))
        tokens, _, _ = policy.sample(z_c.repeat(args.n_samples, 1), n_samples=args.n_samples, temperature=1.0)
        raw_smiles = [vocab.decode(t.tolist()) for t in tokens]

        # --- independent, reward-model-blind checks ---
        valid_canon = [canonical(s) for s in raw_smiles]
        valid_canon = [c for c in valid_canon if c is not None]
        validity = len(valid_canon) / max(1, len(raw_smiles))
        unique_valid = set(valid_canon)
        uniqueness = len(unique_valid) / max(1, len(valid_canon)) if valid_canon else 0.0
        novel = [s for s in unique_valid if s not in train_canon_set]
        novelty_rate = len(novel) / max(1, len(unique_valid)) if unique_valid else 0.0
        lipinski_rate = (sum(1 for s in unique_valid if lipinski_pass(s)) / max(1, len(unique_valid))
                          if unique_valid else 0.0)

        nn_sims = []
        for s in novel:
            fp = morgan_fp(s)
            if fp is None or not train_fps:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            nn_sims.append(max(sims) if sims else 0.0)
        mean_nn_tanimoto = float(np.mean(nn_sims)) if nn_sims else None

        # --- reward-model score on the SAME sampled molecules ---
        scored_smiles, reward_scores = reward_model_score(predictor, raw_smiles, img_tensor, device)
        mean_reward = float(np.mean(reward_scores)) if reward_scores else None

        print(f"=== {ach_id} ===")
        print(f"  [Reward-model]        mean score (higher=better) = "
              f"{mean_reward:+.4f}" if mean_reward is not None else "  [Reward-model] n/a")
        print(f"  [Independent] validity={validity:.2%} | uniqueness={uniqueness:.2%} | "
              f"novelty_rate={novelty_rate:.2%} | Lipinski_pass_rate={lipinski_rate:.2%} | "
              f"mean_NN_Tanimoto={mean_nn_tanimoto}")
        if mean_reward is not None and lipinski_rate < 0.5 and mean_reward > 0:
            print("  ** FLAG: reward-model score is positive but Lipinski pass rate is low — "
                  "possible reward hacking (predictor likes molecules a standard "
                  "druglikeness filter would reject). Investigate with Grad-CAM on "
                  "the predictor for these specific molecules. **")
        print()


if __name__ == "__main__":
    main()
