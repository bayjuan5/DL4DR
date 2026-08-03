"""
new_rl_finetune_v5.py
======================
v5 of the full-VAE REINFORCE fine-tuning script (report_V1.docx / report_V2.docx).
Builds on new_rl_finetune.py (v1-v4), which added a batch-level repetition
penalty (Section 8.6) that fixed the exact-duplicate-molecule collapse seen
in v1/v2/v3 -- but report_V1.docx Section 5.6 found a loophole in v4: the
penalty only matches EXACT canonical-SMILES strings, so the policy learned
to satisfy the letter of the uniqueness reward by enumerating chain-length/
branching variants of ONE simple scaffold (e.g. CCCOC(O)OC, CCCCOC(O)OC,
CCCCCOC(O)OC, ...) -- each technically a distinct string, so none of them
penalized relative to the others, despite being chemically near-identical.

TWO CHANGES vs new_rl_finetune.py (v4):

  1. TANIMOTO-THRESHOLD REPETITION PENALTY (replaces exact-string matching)
     compute_diversity_penalty() now clusters valid batch members by Morgan
     fingerprint Tanimoto similarity (union-find on pairwise similarity >=
     --tanimoto_threshold), not by canonical-SMILES equality. Two molecules
     that are chain-length variants of the same scaffold now land in the
     same cluster and get penalized as "the same", closing the v4 loophole.
     batch_uniqueness is likewise redefined as distinct SIMILARITY CLUSTERS
     / valid samples, so the logged number reflects genuine structural
     diversity rather than distinct strings.

  2. "COLD" (LEAST-RECENTLY/LEAST-OFTEN-DRAWN) SEED-COMPOUND SAMPLING
     get_seed_batch() previously drew uniformly at random from each cell
     line's compound pool every epoch. The real pool sizes are wildly
     imbalanced (21 to 43,061 compounds across the 51 cell lines in
     BREAST-136344-56786-51.txt) -- pure i.i.d. uniform sampling leaves
     most of a 43k-compound pool essentially untouched even over thousands
     of epochs. draw_counts now tracks how many times each compound has
     been sampled so far this run, and sampling probability is weighted
     as 1 / (1 + count)^cold_power, so under-sampled ("cold") compounds
     are preferentially drawn until they warm up too -- self-correcting,
     not a one-time reshuffle.

Everything else (encoder+decoder REINFORCE, VAE ELBO term, invalid_penalty,
entropy_coef, baseline) is unchanged from new_rl_finetune.py -- see that
file's docstring for the full mechanism.

Run from inside RL_updated_learnrate/, e.g.:

    python new_rl_finetune_v5.py \
        --data ../../data/BREAST-136344-56786-51.txt \
        --smiles ../../data/CompoundSmiles_full_140474.txt \
        --genomic ../../genomic_images \
        --ckpt ../../checkpoints/best_random.pt \
        --resume ../VAE_updated_learnrate/checkpoints_gen/best_vae.pt \
        --epochs 10000 --batch 64 --invalid_penalty -5.0 \
        --vae_weight 1.0 --kl_weight 1.0 --entropy_coef 0.05 \
        --lambda_div 0.1 --tanimoto_threshold 0.65 --cold_power 1.0 \
        --eval_every 100 --out_dir checkpoints_gen_rl_full_vae_v5

NOTE on epochs: report_V2.docx corrects an earlier estimate of "~2550 total
(cell line, compound) pairs" -- the real number is 136,344 records across
51 cell lines, averaging ~2,673 compounds/cell line but ranging from 21 to
43,061. Full coverage of the largest pools isn't achievable at any
practical epoch count; 8,000-12,000 epochs (up from 2,000 in v1-v4) is a
meaningful, not complete, improvement in coverage, especially combined with
cold sampling above.
"""
import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

from new_rl_policy import PolicyWrapper

# ── import repo modules exactly like train.py / rl_finetune.py ──
import importlib.util

HERE = Path(__file__).resolve()
GEN_ROOT = HERE.parent.parent  # generative/
REPO_ROOT = GEN_ROOT.parent    # DL4DR/

spec = importlib.util.spec_from_file_location("dl4dr_gen_model", GEN_ROOT / "model.py")
gen_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_model)
SmilesVocab = gen_model.SmilesVocab
FrozenCellLineEncoder = gen_model.FrozenCellLineEncoder
ConditionalSmilesVAE = gen_model.ConditionalSmilesVAE
vae_loss = gen_model.vae_loss

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


# ─────────────────────────────────────────────────────────────
# Reward: run generated (valid) SMILES back through the frozen
# DL4DR predictor, per cell line
# ─────────────────────────────────────────────────────────────

def score_with_predictor(predictor, smiles_list, genomic_img, device, invalid_penalty, mol_img_size=64):
    """
    For each SMILES: canonicalize + build (mol_graph, mol_img, ecfp);
    invalid ones get invalid_penalty. Valid ones get reward = -predicted_ic50.
    Returns a (len(smiles_list),) tensor of rewards and a bool validity mask.
    """
    from torch_geometric.data import Batch
    from rdkit.Chem import Draw

    valid_idx, graphs, imgs, ecfps = [], [], [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        # RDKit quirk: Chem.MolFromSmiles("") returns a valid, 0-atom Mol
        # object (not None) - it slips past "mol is None" and produces a
        # PyG graph with zero nodes. Guard on atom count too.
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

    rewards = torch.full((len(smiles_list),), invalid_penalty, device=device)
    valid_mask = torch.zeros(len(smiles_list), dtype=torch.bool)

    if not valid_idx:
        return rewards, valid_mask

    batch_graph = Batch.from_data_list(graphs).to(device)
    batch_img = torch.stack(imgs).to(device)
    batch_ecfp = torch.stack(ecfps).to(device)
    batch_genomic = genomic_img.unsqueeze(0).repeat(len(valid_idx), 1, 1, 1).to(device)

    with torch.no_grad():
        pred_ic50 = predictor(batch_genomic, batch_graph, batch_img, batch_ecfp)

    for j, i in enumerate(valid_idx):
        rewards[i] = -pred_ic50[j]
        valid_mask[i] = True

    return rewards, valid_mask


# ─────────────────────────────────────────────────────────────
# v5 CHANGE #1: Tanimoto-threshold batch-level diversity penalty
# (report_V1.docx Section 5.6 / report_V2.docx) -- replaces v4's exact
# canonical-SMILES matching with fingerprint-similarity clustering, so
# chain-length/branching variants of one scaffold are no longer free to
# dodge the penalty just by being technically distinct strings.
# ─────────────────────────────────────────────────────────────

class _UnionFind:
    """Minimal union-find (disjoint set) with path compression, used to
    group batch members into similarity clusters."""
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def compute_diversity_penalty(rewards, smiles_batch, valid_mask, lambda_div, tanimoto_threshold=0.85):
    """
    For each valid sample i, group it with every other valid sample j in
    the batch whose Morgan fingerprint (radius=2, 2048 bits) Tanimoto
    similarity to i is >= tanimoto_threshold (union-find over all pairs).
    cluster_size_i = size of i's cluster. Subtract
    lambda_div * max(0, cluster_size_i - 1) from valid samples' rewards --
    same functional form as v4's exact-match penalty, just with a
    similarity-based grouping instead of string equality. Invalid samples
    (already at invalid_penalty) are untouched.

    batch_uniqueness = distinct clusters / valid samples, returned for
    logging regardless of whether lambda_div > 0.
    """
    valid_idx = [i for i, v in enumerate(valid_mask.tolist()) if v]
    fps = {}
    for i in valid_idx:
        mol = Chem.MolFromSmiles(smiles_batch[i])
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        fps[i] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    penalized = rewards.clone()
    idxs = list(fps.keys())
    n_valid = len(idxs)
    if n_valid == 0:
        return penalized, 0.0

    uf = _UnionFind(idxs)
    for a_pos in range(len(idxs) - 1):
        i = idxs[a_pos]
        rest = [fps[idxs[j]] for j in range(a_pos + 1, len(idxs))]
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], rest)
        for offset, sim in enumerate(sims):
            if sim >= tanimoto_threshold:
                uf.union(i, idxs[a_pos + 1 + offset])

    cluster_of = {i: uf.find(i) for i in idxs}
    cluster_sizes = Counter(cluster_of.values())
    batch_uniqueness = len(cluster_sizes) / max(1, n_valid)

    if lambda_div > 0:
        for i in idxs:
            size = cluster_sizes[cluster_of[i]]
            if size > 1:
                penalized[i] = penalized[i] - lambda_div * (size - 1)

    return penalized, batch_uniqueness


# ─────────────────────────────────────────────────────────────
# Real seed-molecule lookup, per cell line, for encoder sampling.
# v5 CHANGE #2: "cold" sampling weighted by inverse draw count, instead
# of uniform random -- see module docstring.
# ─────────────────────────────────────────────────────────────

def build_ach_to_smiles(records_all):
    ach_to_smiles = defaultdict(list)
    for r in records_all:
        ach_to_smiles[r["ach_id"]].append(r["smiles"])
    return ach_to_smiles


def init_draw_counts(ach_to_smiles):
    """Per-compound draw counters for cold sampling: one int64 array per
    cell line, indexed the same way as ach_to_smiles[ach_id]. Persists
    across the whole training run (created once in main(), mutated
    in-place by get_seed_batch every time it's called)."""
    return {ach_id: np.zeros(len(pool), dtype=np.int64) for ach_id, pool in ach_to_smiles.items()}


def get_seed_batch(ach_to_smiles, ach_id, batch_size, vocab, max_len, device, draw_counts, cold_power=1.0):
    """
    Sample `batch_size` real SMILES strings on record for this cell line,
    weighted by 1 / (1 + draw_count)^cold_power so far-less-sampled
    ("cold") compounds are preferentially drawn -- important for pools up
    to 43,061 compounds, where uniform random sampling would leave most
    of the pool essentially unseen even over thousands of epochs.
    cold_power=0 recovers uniform random sampling (v1-v4 behavior).

    Returns None if this cell line has no training compounds at all --
    caller should fall back to prior sampling for that epoch.
    """
    pool = ach_to_smiles.get(ach_id)
    if not pool:
        return None
    counts = draw_counts[ach_id]
    replace = len(pool) < batch_size
    if cold_power > 0:
        weights = 1.0 / np.power(1.0 + counts.astype(np.float64), cold_power)
        weights /= weights.sum()
        idx = np.random.choice(len(pool), size=batch_size, replace=replace, p=weights)
    else:
        idx = np.random.choice(len(pool), size=batch_size, replace=replace)
    counts[idx] += 1
    chosen = [pool[i] for i in idx]
    ids = [vocab.encode(smi, max_len) for smi in chosen]
    return torch.tensor(ids, dtype=torch.long, device=device)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--smiles", default=None)
    ap.add_argument("--genomic", required=True)
    ap.add_argument("--ckpt", required=True, help="Frozen DL4DR predictor checkpoint (reward model)")
    ap.add_argument("--resume", default=None, help="Warm-start policy from an existing VAE checkpoint")
    ap.add_argument("--epochs", type=int, default=10000,
                     help="v5 default raised from v1-v4's 2000-100 range -- see module docstring's "
                          "note on corrected pool-size/coverage estimates.")
    ap.add_argument("--batch", type=int, default=64, help="Samples drawn per cell line per epoch")
    ap.add_argument("--lr", type=float, default=1e-5, help="Keep small — RL fine-tuning, not training from scratch")
    ap.add_argument("--entropy_coef", type=float, default=0.05, help="Entropy bonus to slow mode collapse")
    ap.add_argument("--invalid_penalty", type=float, default=-5.0,
                     help="Reward assigned to invalid/undecodable SMILES. NOTE: this is NOT "
                          "guaranteed to be a floor — valid molecules can score below this if "
                          "the predictor rates them as ineffective (high predicted IC50). Set "
                          "well below the typical valid-reward range if you want validity to be "
                          "unambiguously incentivized.")
    ap.add_argument("--vae_weight", type=float, default=1.0,
                     help="Weight of the VAE ELBO term (teacher-forced recon + KL) added to the "
                          "REINFORCE loss each epoch. This is what keeps the encoder well-behaved "
                          "-- set to 0.0 to fall back to pure REINFORCE (still routed through the "
                          "encoder via posterior sampling, just without the ELBO regularizer).")
    ap.add_argument("--kl_weight", type=float, default=1.0,
                     help="KL weight inside the VAE ELBO term, same meaning as in train.py")
    ap.add_argument("--lambda_div", type=float, default=0.1,
                     help="Weight of the batch-level repetition penalty: valid samples in the same "
                          "Tanimoto-similarity cluster as other batch members get "
                          "lambda_div * (cluster_size-1) subtracted from their reward. 0.0 "
                          "disables the penalty but batch_uniqueness is still logged either way.")
    ap.add_argument("--tanimoto_threshold", type=float, default=0.65,
                     help="v5: two valid molecules in the same batch count as 'the same' for the "
                          "repetition penalty if their Morgan-fingerprint Tanimoto similarity is >= "
                          "this value (was exact-string-equality only in v4). Calibrated (not just "
                          "guessed) against the actual v4 Collapse-check scaffold family (report_V1"
                          ".docx Section 5.6): pairwise similarities within that family ranged "
                          "0.32-0.82, and the critical threshold for the whole 7-molecule family to "
                          "chain together into one cluster (via transitive union-find, not every pair "
                          "needing to individually clear the bar) is 0.684. 0.65 gives a small margin "
                          "below that. An earlier 0.85 default was tested and confirmed too strict -- "
                          "it would not have caught this family at all.")
    ap.add_argument("--cold_power", type=float, default=1.0,
                     help="v5: seed-compound sampling weight exponent, 1/(1+draw_count)^cold_power. "
                          "0.0 = uniform random (v1-v4 behavior); higher values bias more strongly "
                          "toward least-sampled compounds.")
    ap.add_argument("--zc_dim", type=int, default=256)
    ap.add_argument("--out_dir", default="checkpoints_gen_rl_full_vae_v5")
    ap.add_argument("--eval_every", type=int, default=100,
                     help="Raised from v1-v4's default of 5 -- at 10000 epochs, checkpointing every "
                          "5 epochs would be excessive I/O for little added safety margin.")
    ap.add_argument("--n_cell_lines", type=int, default=3, help="Unused by the epoch%%N_cells rotation "
                     "below; kept for CLI compatibility with the eval script's flag of the same name.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.smiles is None:
        args.smiles = str((Path(args.data).resolve().parent / "CompoundSmiles_full_140474.txt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    records_all = root_dataset.load_records(args.data, args.smiles)
    vocab = SmilesVocab([r["smiles"] for r in records_all])
    print(f"Vocab size: {len(vocab)} | training records: {len(records_all)}")

    all_ach_ids = sorted(set(r["ach_id"] for r in records_all))
    ach_to_smiles = build_ach_to_smiles(records_all)
    draw_counts = init_draw_counts(ach_to_smiles)
    pool_sizes = {k: len(v) for k, v in ach_to_smiles.items()}
    print(f"{len(all_ach_ids)} cell lines | pool sizes: min={min(pool_sizes.values())} "
          f"max={max(pool_sizes.values())} mean={np.mean(list(pool_sizes.values())):.0f}")

    # ── frozen models ──
    cell_encoder = FrozenCellLineEncoder(args.ckpt, device=device).to(device)
    predictor = DrugResponseModel().to(device)
    pred_state = torch.load(args.ckpt, map_location=device)
    pred_state = pred_state.get("model") or pred_state.get("model_state_dict") or pred_state
    predictor.load_state_dict(pred_state, strict=False)
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad = False

    # ── policy (generator) -- encoder AND decoder, warm-started from the VAE if given ──
    policy_net = ConditionalSmilesVAE(vocab_size=len(vocab), zc_dim=args.zc_dim, pad_idx=vocab.pad_idx).to(device)
    if args.resume and os.path.exists(args.resume):
        policy_net.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Warm-started policy (encoder + decoder) from {args.resume}")
    else:
        print("WARNING: no --resume given, policy starts from random weights "
              "(will likely waste early epochs on near-100% invalid SMILES)")
    policy = PolicyWrapper(policy_net)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)

    print(f"invalid_penalty = {args.invalid_penalty:+.2f} | vae_weight = {args.vae_weight:.2f} | "
          f"kl_weight = {args.kl_weight:.2f} | lambda_div = {args.lambda_div:.2f} | "
          f"tanimoto_threshold = {args.tanimoto_threshold:.2f} | cold_power = {args.cold_power:.2f}")
    print("Training the FULL VAE (encoder + decoder) with v5 changes: Tanimoto-similarity-clustered "
          "repetition penalty (not exact-string) and cold (inverse-draw-count-weighted) seed sampling.")

    baseline = 0.0  # running-mean reward baseline for REINFORCE variance reduction
    baseline_momentum = 0.9

    history = []
    for epoch in range(1, args.epochs + 1):
        policy_net.train()
        ach_id = all_ach_ids[epoch % len(all_ach_ids)]
        img_path = os.path.join(args.genomic, f"{ach_id}.png")
        if not os.path.exists(img_path):
            continue
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).permute(2, 0, 1)

        with torch.no_grad():
            z_c = cell_encoder(img_tensor.unsqueeze(0).to(device))
        z_c_batch = z_c.repeat(args.batch, 1)

        # ── real seed molecules for this cell line, or None if unavailable ──
        x_seed = get_seed_batch(ach_to_smiles, ach_id, args.batch, vocab, MAX_LEN, device,
                                 draw_counts, cold_power=args.cold_power)
        used_encoder = x_seed is not None

        tokens, logprobs, entropy, mu, logvar, z = policy.sample(x_seed, z_c_batch, n_samples=args.batch)
        smiles_batch = [vocab.decode(t.tolist()) for t in tokens]

        raw_rewards, valid_mask = score_with_predictor(
            predictor, smiles_batch, img_tensor, device, args.invalid_penalty
        )
        rewards, batch_uniqueness = compute_diversity_penalty(
            raw_rewards, smiles_batch, valid_mask, args.lambda_div, args.tanimoto_threshold
        )

        # REINFORCE with baseline (drives both decoder and, via z, the encoder)
        # NOTE: advantage/baseline are computed on the diversity-penalized
        # `rewards` (what the policy is actually optimized against), while
        # mean_reward_valid/etc below use `raw_rewards` (pure predictor
        # score) so those columns stay directly comparable to v1-v4's
        # history_rl.csv, which had no penalty (or an exact-match one) applied.
        advantage = rewards.detach() - baseline
        rl_loss = -(advantage * logprobs).mean() - args.entropy_coef * entropy.mean()

        # VAE ELBO term on the SAME z: teacher-forced reconstruction of the
        # seed SMILES + KL(q(z|x_seed) || N(0,I)). Keeps the encoder's
        # posterior well-behaved instead of relying only on the noisy
        # policy-gradient signal to shape it.
        if used_encoder:
            recon_logits = policy_net.decode(z, z_c_batch, target=x_seed, teacher_forcing=True)
            vae_total, vae_recon, vae_kl = vae_loss(
                recon_logits, x_seed, mu, logvar, pad_idx=vocab.pad_idx, kl_weight=args.kl_weight
            )
            loss = rl_loss + args.vae_weight * vae_total
        else:
            vae_recon = torch.tensor(float("nan"))
            vae_kl = torch.tensor(float("nan"))
            loss = rl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()

        batch_mean_reward = raw_rewards.mean().item()
        batch_mean_reward_penalized = rewards.mean().item()
        baseline = baseline_momentum * baseline + (1 - baseline_momentum) * batch_mean_reward_penalized
        validity = valid_mask.float().mean().item()

        # ── diagnostic: reward stats restricted to VALID samples only (raw predictor score) ──
        if valid_mask.any():
            valid_rewards = raw_rewards[valid_mask]
            mean_reward_valid = valid_rewards.mean().item()
            min_reward_valid = valid_rewards.min().item()
            max_reward_valid = valid_rewards.max().item()
            frac_valid_below_penalty = (valid_rewards <= args.invalid_penalty).float().mean().item()
        else:
            mean_reward_valid = float("nan")
            min_reward_valid = float("nan")
            max_reward_valid = float("nan")
            frac_valid_below_penalty = float("nan")

        row = dict(epoch=epoch, ach_id=ach_id, loss=loss.item(), rl_loss=rl_loss.item(),
                   vae_recon_loss=vae_recon.item() if used_encoder else float("nan"),
                   vae_kl_loss=vae_kl.item() if used_encoder else float("nan"),
                   used_encoder=used_encoder,
                   mean_reward=batch_mean_reward,
                   mean_reward_penalized=batch_mean_reward_penalized,
                   lambda_div=args.lambda_div,
                   tanimoto_threshold=args.tanimoto_threshold,
                   batch_uniqueness=batch_uniqueness,
                   validity=validity,
                   baseline=baseline, entropy=entropy.mean().item(),
                   invalid_penalty=args.invalid_penalty,
                   mean_reward_valid=mean_reward_valid,
                   min_reward_valid=min_reward_valid,
                   max_reward_valid=max_reward_valid,
                   frac_valid_below_penalty=frac_valid_below_penalty)
        history.append(row)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d} | cell {ach_id} | enc={'Y' if used_encoder else 'N'} | "
                  f"loss {loss.item():+.4f} (rl {rl_loss.item():+.4f}"
                  + (f" / recon {vae_recon.item():.4f} / kl {vae_kl.item():.4f}" if used_encoder else "")
                  + f") | mean_reward {batch_mean_reward:+.4f}"
                  + (f" (penalized {batch_mean_reward_penalized:+.4f})" if args.lambda_div > 0 else "")
                  + f" | validity {validity:.2%} | batch_uniqueness(tanimoto) {batch_uniqueness:.2%} | "
                  f"entropy {entropy.mean().item():.3f} | "
                  f"reward_valid[mean/min/max] {mean_reward_valid:+.2f}/{min_reward_valid:+.2f}/{max_reward_valid:+.2f} | "
                  f"frac_valid<=penalty {frac_valid_below_penalty:.2%}")

        if epoch % args.eval_every == 0:
            torch.save(policy_net.state_dict(), os.path.join(args.out_dir, "latest_rl.pt"))
            print(f"  -> checkpoint saved (epoch {epoch})")

    torch.save(policy_net.state_dict(), os.path.join(args.out_dir, "final_rl.pt"))
    hist_path = os.path.join(args.out_dir, "history_rl.csv")
    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"\nDone. History saved to {hist_path}")
    print("Encoder and decoder were both trained jointly via REINFORCE (posterior-sampled z), the VAE "
          "ELBO term, a Tanimoto-clustered repetition penalty, and cold seed-compound sampling.")


if __name__ == "__main__":
    main()
