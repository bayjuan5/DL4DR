"""
new_rl_eval_independent.py
===========================
Evaluates a generator finetuned by new_rl_finetune.py (full VAE — encoder
+ decoder both trained) two ways at once, same as rl_eval_independent.py:

  1. Reward-model score: what the frozen DL4DR predictor thinks of the
     generated molecules (the thing the policy was actually optimized for).
  2. Independent, reward-model-blind checks: RDKit validity, Lipinski
     "Rule of Five" druglikeness, nearest-neighbor Tanimoto similarity to
     known training-set actives, and uniqueness/mode-collapse.

WHY THIS IS A SEPARATE FILE FROM rl_eval_independent.py, NOT A SHARED ONE:
  rl_eval_independent.py calls
      policy.sample(z_c.repeat(n, 1), n_samples=n, temperature=1.0)
  which is rl_policy.PolicyWrapper's signature: (z_c, n_samples, ...) ->
  3-tuple (tokens, logprobs, entropy).

  new_rl_policy.PolicyWrapper.sample has a DIFFERENT argument order and a
  DIFFERENT return arity: (x_seed, z_c, n_samples=1, ...) -> 6-tuple
  (tokens, logprobs, entropy, mu, logvar, z). A checkpoint produced by
  new_rl_finetune.py loads fine into ConditionalSmilesVAE either way (same
  architecture/state_dict either script trains), but the *sampling call*
  in the old eval script is simply incompatible with the new wrapper's
  API — hence this file, mirroring the old one but wired to new_rl_policy.

TWO SAMPLING MODES (--sample_mode):
  - "prior" (default): z ~ N(0, I), same unconditioned-generation behavior
    as rl_eval_independent.py. Use this for an apples-to-apples novelty/
    validity comparison against a decoder-only (rl_finetune.py) checkpoint.
  - "posterior": encodes a real seed SMILES per cell line (same lookup
    logic as new_rl_finetune.py's get_seed_batch) and samples from the
    encoder's posterior. This checks whether the fine-tuned encoder is
    doing anything useful — e.g. if posterior-mode novelty collapses
    relative to prior-mode (i.e. it just regurgitates near-copies of
    training compounds), that's a sign the encoder is memorizing rather
    than usefully perturbing known actives.

Run from inside RL_updated_learnrate/, e.g.:

    python new_rl_eval_independent.py \
        --rl_ckpt checkpoints_gen_rl_full_vae/final_rl.pt \
        --dl4dr_ckpt ../checkpoints/best_random.pt \
        --data ../data/BREAST-136344-56786-51.txt \
        --smiles ../data/CompoundSmiles_full_140474.txt \
        --genomic ../genomic_images \
        --n_samples 200 --sample_mode both
"""
import argparse
import importlib.util
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski

RDLogger.DisableLog("rdApp.*")

from new_rl_policy import PolicyWrapper

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


def compute_tanimoto_uniqueness(valid_canon_list, tanimoto_threshold):
    """
    report_V2.docx: eval-time counterpart of new_rl_finetune_v5.py's
    training-time batch_uniqueness -- clusters valid samples by Morgan
    fingerprint (radius=2, 2048 bits) Tanimoto similarity via union-find
    (transitive), instead of exact canonical-SMILES equality. Two samples
    land in the same cluster if their similarity is >= tanimoto_threshold,
    OR if they're both linked to a common third sample above threshold.

    Definition matches the training-time metric exactly: distinct clusters
    / total valid samples (not distinct strings), so this number is
    directly comparable to the batch_uniqueness column logged during v5
    training.

    Returns (tanimoto_uniqueness, n_clusters, cluster_sizes) where
    cluster_sizes is a Counter keyed by cluster representative, sized in
    terms of how many of the (possibly duplicated) valid samples fall in
    each cluster.
    """
    if not valid_canon_list:
        return 0.0, 0, Counter()

    unique_smiles = list(dict.fromkeys(valid_canon_list))  # de-dupe, preserve order
    fps = {}
    for s in unique_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fps[s] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    parent = {s: s for s in fps}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    keys = list(fps.keys())
    for i in range(len(keys) - 1):
        a = keys[i]
        rest = [fps[keys[j]] for j in range(i + 1, len(keys))]
        sims = DataStructs.BulkTanimotoSimilarity(fps[a], rest)
        for offset, sim in enumerate(sims):
            if sim >= tanimoto_threshold:
                union(a, keys[i + 1 + offset])

    cluster_of_string = {s: find(s) for s in fps}
    cluster_sizes = Counter(cluster_of_string[s] for s in valid_canon_list if s in cluster_of_string)
    n_clusters = len(cluster_sizes)
    tanimoto_uniqueness = n_clusters / max(1, len(valid_canon_list))
    return tanimoto_uniqueness, n_clusters, cluster_sizes


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
    """Same scoring path as new_rl_finetune.py's score_with_predictor, kept
    separate here so this file has no import dependency on new_rl_finetune.py."""
    from torch_geometric.data import Batch
    from rdkit.Chem import Draw

    valid_idx, graphs, imgs, ecfps = [], [], [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        # RDKit quirk: Chem.MolFromSmiles("") returns a valid, 0-atom Mol
        # object (not None) - guard on atom count too.
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


def build_ach_to_smiles(records_all):
    ach_to_smiles = {}
    for r in records_all:
        ach_to_smiles.setdefault(r["ach_id"], []).append(r["smiles"])
    return ach_to_smiles


def get_seed_batch(ach_to_smiles, ach_id, batch_size, vocab, max_len, device):
    """Same logic as new_rl_finetune.py's get_seed_batch, duplicated here
    to keep this file import-independent. Returns None if this cell line
    has no training compounds on record."""
    pool = ach_to_smiles.get(ach_id)
    if not pool:
        return None
    replace = len(pool) < batch_size
    chosen = np.random.choice(pool, size=batch_size, replace=replace)
    ids = [vocab.encode(smi, max_len) for smi in chosen]
    return torch.tensor(ids, dtype=torch.long, device=device)


def run_eval_for_mode(mode, policy, policy_net, vocab, ach_to_smiles, ach_id,
                       z_c, img_tensor, predictor, device, args,
                       train_canon_set, train_fps):
    z_c_batch = z_c.repeat(args.n_samples, 1)

    x_seed = None
    if mode == "posterior":
        x_seed = get_seed_batch(ach_to_smiles, ach_id, args.n_samples, vocab, MAX_LEN, device)
        if x_seed is None:
            print(f"  [posterior] no training compounds on record for {ach_id} -- "
                  f"falling back to prior sampling")

    with torch.no_grad():
        tokens, _, _, mu, logvar, _ = policy.sample(
            x_seed, z_c_batch, n_samples=args.n_samples, temperature=1.0
        )
    raw_smiles = [vocab.decode(t.tolist()) for t in tokens]

    # --- independent, reward-model-blind checks ---
    valid_canon = [canonical(s) for s in raw_smiles]
    valid_canon = [c for c in valid_canon if c is not None]
    validity = len(valid_canon) / max(1, len(raw_smiles))
    unique_valid = set(valid_canon)
    uniqueness = len(unique_valid) / max(1, len(valid_canon)) if valid_canon else 0.0
    tanimoto_uniqueness, n_tanimoto_clusters, tanimoto_cluster_sizes = compute_tanimoto_uniqueness(
        valid_canon, args.tanimoto_threshold
    )
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

    used = "posterior (encoder-seeded)" if x_seed is not None else "prior (encoder-free)"
    print(f"  --- mode: {used} ---")
    print(f"  [Reward-model]        mean score (higher=better) = "
          f"{mean_reward:+.4f}" if mean_reward is not None else "  [Reward-model] n/a")
    print(f"  [Independent] validity={validity:.2%} | uniqueness={uniqueness:.2%} | "
          f"tanimoto_uniqueness(t={args.tanimoto_threshold:.2f})={tanimoto_uniqueness:.2%} | "
          f"novelty_rate={novelty_rate:.2%} | Lipinski_pass_rate={lipinski_rate:.2%} | "
          f"mean_NN_Tanimoto={mean_nn_tanimoto}")

    # ── mode-collapse diagnostic: what are the valid molecules actually
    # repeating? uniqueness alone tells you IT collapsed, not WHAT it
    # collapsed to -- this shows the top-repeated canonical SMILES so you
    # can eyeball whether it's the same handful of (likely trivial) molecules
    # across calls, and how skewed the repeat counts are.
    if valid_canon:
        counts = Counter(valid_canon)
        top = counts.most_common(5)
        n_singletons = sum(1 for _, c in counts.items() if c == 1)
        print(f"  [Collapse check] {len(counts)} distinct valid molecules across "
              f"{len(valid_canon)} valid samples | {n_singletons} appear exactly once")
        print("  [Collapse check] top repeats (exact string):")
        for smi, c in top:
            print(f"      x{c:<4d} {smi}")

        # ── report_V2.docx: same idea, but grouped by Tanimoto-similarity
        # cluster instead of exact string -- catches scaffold/chain-length
        # families (e.g. v4's CCCOC(O)OC/COC(O)OCCO loophole) that exact-
        # string matching alone would report as all "distinct".
        top_clusters = tanimoto_cluster_sizes.most_common(5)
        print(f"  [Collapse check] {n_tanimoto_clusters} distinct Tanimoto clusters (t="
              f"{args.tanimoto_threshold:.2f}) across {len(valid_canon)} valid samples")
        print("  [Collapse check] top Tanimoto clusters (representative SMILES x cluster size):")
        for rep, size in top_clusters:
            print(f"      x{size:<4d} {rep}")

    if mean_reward is not None and lipinski_rate < 0.5 and mean_reward > 0:
        print("  ** FLAG: reward-model score is positive but Lipinski pass rate is low — "
              "possible reward hacking (predictor likes molecules a standard "
              "druglikeness filter would reject). Investigate with Grad-CAM on "
              "the predictor for these specific molecules. **")
    if x_seed is not None and novelty_rate < 0.2:
        print("  ** NOTE: posterior-mode novelty is very low — the encoder may be "
              "memorizing/near-copying seed molecules rather than usefully "
              "perturbing them. Compare against prior-mode novelty above. **")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl_ckpt", required=True, help="Checkpoint from new_rl_finetune.py")
    ap.add_argument("--dl4dr_ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--genomic", required=True)
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--n_cell_lines", type=int, default=3)
    ap.add_argument("--zc_dim", type=int, default=256)
    ap.add_argument("--sample_mode", choices=["prior", "posterior", "both"], default="prior",
                     help="prior = unconditioned generation, encoder untouched (comparable to "
                          "rl_eval_independent.py). posterior = encode a real seed SMILES per "
                          "cell line and sample from the encoder's posterior. both = run and "
                          "report both for the same cell lines.")
    ap.add_argument("--tanimoto_threshold", type=float, default=0.65,
                     help="report_V2.docx: two valid molecules count as the 'same' Tanimoto cluster "
                          "if their Morgan-fingerprint similarity is >= this value. Default 0.65 "
                          "matches new_rl_finetune_v5.py's training-time penalty threshold, for "
                          "direct eval-vs-training comparability on v5 checkpoints.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records_all = root_dataset.load_records(args.data, args.smiles)
    vocab = SmilesVocab([r["smiles"] for r in records_all])
    ach_to_smiles = build_ach_to_smiles(records_all)

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

    policy_net = ConditionalSmilesVAE(vocab_size=len(vocab), zc_dim=args.zc_dim, pad_idx=vocab.pad_idx).to(device)
    policy_net.load_state_dict(torch.load(args.rl_ckpt, map_location=device))
    policy_net.eval()
    policy = PolicyWrapper(policy_net)

    modes = ["prior", "posterior"] if args.sample_mode == "both" else [args.sample_mode]

    all_ach_ids = sorted(set(r["ach_id"] for r in records_all))
    random.shuffle(all_ach_ids)
    chosen = all_ach_ids[: args.n_cell_lines]
    print(f"Evaluating cell lines: {chosen} | sample_mode(s): {modes}\n")

    for ach_id in chosen:
        img_path = Path(args.genomic) / f"{ach_id}.png"
        if not img_path.exists():
            print(f"  [skip] no genomic image for {ach_id}")
            continue
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).permute(2, 0, 1)

        with torch.no_grad():
            z_c = cell_encoder(img_tensor.unsqueeze(0).to(device))

        print(f"=== {ach_id} ===")
        for mode in modes:
            run_eval_for_mode(mode, policy, policy_net, vocab, ach_to_smiles, ach_id,
                               z_c, img_tensor, predictor, device, args,
                               train_canon_set, train_fps)


if __name__ == "__main__":
    main()
