"""
new_rl_finetune.py
===================
REINFORCE fine-tuning of the compound generator against the frozen DL4DR
predictor (best_random.pt) as a reward model -- same setup as
rl_finetune.py, but with the *whole* VAE (encoder + decoder) in the
training loop instead of just the decoder.

WHAT CHANGED vs rl_finetune.py:
    rl_finetune.py samples the policy's latent z from the fixed prior
    N(0, I) (see rl_policy.PolicyWrapper.sample). That z never touches
    ConditionalSmilesVAE's encoder (encoder_rnn / fc_mu / fc_logvar), so
    those parameters get zero gradient every step -- only the decoder is
    actually fine-tuned by RL, even though the encoder's weights are
    technically handed to the optimizer.

    This script instead, each epoch:
      1. Pulls a real batch of seed SMILES for the *current* cell line
         out of the training data (records_all), falling back to prior
         sampling only if that cell line has no training compounds.
      2. Runs those seed SMILES through the VAE encoder
         (new_rl_policy.PolicyWrapper.sample_from_posterior) to get
         mu, logvar, and a reparameterized z. Sampling the decoder policy
         from this z means the REINFORCE surrogate loss backpropagates
         into the encoder as well as the decoder.
      3. Adds the standard VAE ELBO terms (teacher-forced reconstruction
         of the seed SMILES + KL(q(z|x) || N(0,I)), via model.py's
         vae_loss) on the *same* z, weighted by --vae_weight. This keeps
         the encoder's posterior well-behaved (matching the prior it will
         be sampled from at pure-generation time) instead of relying
         solely on the noisy REINFORCE gradient to shape it.

    total_loss = rl_loss (REINFORCE + entropy bonus)
                 + vae_weight * (recon_loss + kl_weight * kl_loss)

Parallel structure to rl_finetune.py / train.py:
  --data / --smiles / --genomic / --ckpt   : same as before (--ckpt is the
                                              frozen predictor, required)
  --resume                                  : warm-start the policy from an
                                              existing VAE checkpoint
  --vae_weight                              : weight of the VAE ELBO term
                                              added to the RL loss
                                              (0.0 = pure REINFORCE, same
                                              training signal shape as
                                              rl_finetune.py, but still
                                              routed through the encoder)
  --kl_weight                               : KL weight inside the ELBO
                                              term (passed to
                                              model.vae_loss), same
                                              meaning as in train.py
  --lambda_div                              : weight of the batch-level
                                              repetition penalty (report_V1
                                              .docx Section 8.6): valid
                                              samples sharing a canonical
                                              SMILES with other batch
                                              members get lambda_div *
                                              (count-1) subtracted from
                                              their reward before the
                                              REINFORCE advantage is
                                              computed. 0.0 (default)
                                              disables it; batch_uniqueness
                                              is logged every epoch either
                                              way.

Run from inside RL_updated_learnrate/, e.g.:

    python new_rl_finetune.py \
        --data ../../data/BREAST-136344-56786-51.txt \
        --smiles ../../data/CompoundSmiles_full_140474.txt \
        --genomic ../../genomic_images \
        --ckpt ../../checkpoints/best_random.pt \
        --resume ../VAE_updated_learnrate/checkpoints_gen/best_vae.pt \
        --epochs 100 --batch 64 --invalid_penalty -10.0 \
        --vae_weight 1.0 --kl_weight 1.0

Reward: reward = -predicted_ln_ic50 (lower predicted IC50 = more
effective = higher reward), with a fixed penalty for invalid/undecodable
SMILES, and a running-mean baseline subtracted for variance reduction
(standard REINFORCE) -- unchanged from rl_finetune.py.
"""
import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rdkit import Chem, RDLogger

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
# Batch-level diversity/repetition penalty (report_V1.docx Section 8.6):
# unlike the entropy bonus (capped at entropy_coef * ln|vocab|, which
# Section 8.7 showed is too small to compete with the REINFORCE reward
# term), count_i is uncapped -- it can scale into the dozens for a badly
# collapsed batch, giving it a real chance to compete in magnitude.
# ─────────────────────────────────────────────────────────────

def compute_diversity_penalty(rewards, smiles_batch, valid_mask, lambda_div):
    """
    For each valid sample i with canonical SMILES c_i, let count_i be how
    many batch members (including i) share that exact molecule. Subtract
    lambda_div * max(0, count_i - 1) from valid samples' rewards --  the
    first occurrence of any molecule in the batch is unpenalized, every
    additional duplicate accumulates penalty. Invalid samples (already at
    invalid_penalty) are left untouched, so this only acts on the
    uniqueness axis among already-valid samples.

    batch_uniqueness (distinct valid canonical SMILES / total valid
    samples) is returned for logging regardless of whether lambda_div > 0,
    so uniqueness can be tracked during training, not just at eval time.
    """
    canon_list = [None] * len(smiles_batch)
    for i, (smi, valid) in enumerate(zip(smiles_batch, valid_mask.tolist())):
        if not valid:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        canon_list[i] = Chem.MolToSmiles(mol, canonical=True)

    counts = Counter(c for c in canon_list if c is not None)
    n_valid = sum(1 for c in canon_list if c is not None)
    batch_uniqueness = len(counts) / max(1, n_valid) if n_valid else 0.0

    penalized = rewards.clone()
    if lambda_div > 0:
        for i, c in enumerate(canon_list):
            if c is not None and counts[c] > 1:
                penalized[i] = penalized[i] - lambda_div * (counts[c] - 1)

    return penalized, batch_uniqueness


# ─────────────────────────────────────────────────────────────
# Real seed-molecule lookup, per cell line, for encoder sampling
# ─────────────────────────────────────────────────────────────

def build_ach_to_smiles(records_all):
    ach_to_smiles = defaultdict(list)
    for r in records_all:
        ach_to_smiles[r["ach_id"]].append(r["smiles"])
    return ach_to_smiles


def get_seed_batch(ach_to_smiles, ach_id, batch_size, vocab, max_len, device):
    """
    Sample `batch_size` real SMILES strings on record for this cell line
    (with replacement if fewer than batch_size are available), and encode
    them to a (batch_size, max_len) token-id tensor for vae.encode().

    Returns None if this cell line has no training compounds at all --
    caller should fall back to prior sampling for that epoch.
    """
    pool = ach_to_smiles.get(ach_id)
    if not pool:
        return None
    replace = len(pool) < batch_size
    chosen = np.random.choice(pool, size=batch_size, replace=replace)
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
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64, help="Samples drawn per cell line per epoch")
    ap.add_argument("--lr", type=float, default=1e-5, help="Keep small — RL fine-tuning, not training from scratch")
    ap.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus to slow mode collapse")
    ap.add_argument("--invalid_penalty", type=float, default=-3.0,
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
    ap.add_argument("--lambda_div", type=float, default=0.0,
                     help="Weight of the Section 8.6 batch-level repetition penalty: valid samples "
                          "sharing a canonical SMILES with other batch members get "
                          "lambda_div * (count-1) subtracted from their reward. 0.0 (default) "
                          "disables the penalty but batch_uniqueness is still logged either way.")
    ap.add_argument("--zc_dim", type=int, default=256)
    ap.add_argument("--out_dir", default="checkpoints_gen_rl_full_vae")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--n_cell_lines", type=int, default=3, help="Cell lines to rotate through per epoch")
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
          f"kl_weight = {args.kl_weight:.2f}")
    print("Training the FULL VAE (encoder + decoder): each epoch, a real seed SMILES batch "
          "for the current cell line is run through vae.encode() -> reparameterize() -> z, "
          "so REINFORCE gradients reach encoder_rnn/fc_mu/fc_logvar as well as the decoder.")

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
        x_seed = get_seed_batch(ach_to_smiles, ach_id, args.batch, vocab, MAX_LEN, device)
        used_encoder = x_seed is not None

        tokens, logprobs, entropy, mu, logvar, z = policy.sample(x_seed, z_c_batch, n_samples=args.batch)
        smiles_batch = [vocab.decode(t.tolist()) for t in tokens]

        raw_rewards, valid_mask = score_with_predictor(
            predictor, smiles_batch, img_tensor, device, args.invalid_penalty
        )
        rewards, batch_uniqueness = compute_diversity_penalty(
            raw_rewards, smiles_batch, valid_mask, args.lambda_div
        )

        # REINFORCE with baseline (drives both decoder and, via z, the encoder)
        # NOTE: advantage/baseline are computed on the diversity-penalized
        # `rewards` (what the policy is actually optimized against), while
        # mean_reward_valid/etc below use `raw_rewards` (pure predictor
        # score) so those columns stay directly comparable to v1/v2/v3's
        # history_rl.csv, which had no penalty applied.
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

        # batch_mean_reward (raw predictor score) matches v1/v2/v3's
        # history_rl.csv semantics for direct comparability; the baseline
        # itself tracks the diversity-penalized reward, since that's what
        # actually drives advantage/rl_loss below.
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
                   batch_uniqueness=batch_uniqueness,
                   validity=validity,
                   baseline=baseline, entropy=entropy.mean().item(),
                   invalid_penalty=args.invalid_penalty,
                   mean_reward_valid=mean_reward_valid,
                   min_reward_valid=min_reward_valid,
                   max_reward_valid=max_reward_valid,
                   frac_valid_below_penalty=frac_valid_below_penalty)
        history.append(row)
        print(f"Epoch {epoch:4d} | cell {ach_id} | enc={'Y' if used_encoder else 'N'} | "
              f"loss {loss.item():+.4f} (rl {rl_loss.item():+.4f}"
              + (f" / recon {vae_recon.item():.4f} / kl {vae_kl.item():.4f}" if used_encoder else "")
              + f") | mean_reward {batch_mean_reward:+.4f}"
              + (f" (penalized {batch_mean_reward_penalized:+.4f})" if args.lambda_div > 0 else "")
              + f" | validity {validity:.2%} | batch_uniqueness {batch_uniqueness:.2%} | "
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
    print("Encoder and decoder were both trained jointly via REINFORCE (posterior-sampled z) "
          "plus the VAE ELBO term on real seed molecules per cell line.")


if __name__ == "__main__":
    main()
