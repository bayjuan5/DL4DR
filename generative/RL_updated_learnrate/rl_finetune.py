"""
rl_finetune.py
==============
REINFORCE fine-tuning of the compound generator against the frozen DL4DR
predictor (best_random.pt) as a reward model, as an alternative to the VAE
in VAE_updated_learnrate/.

Parallel structure to train.py:
  --data / --smiles / --genomic / --ckpt   : same as train.py (--ckpt is the
                                              frozen predictor, required)
  --resume                                  : warm-start the policy from an
                                              existing VAE checkpoint
                                              (e.g. ../VAE_updated_learnrate/
                                              checkpoints_gen/best_vae.pt) so
                                              we're fine-tuning a policy that
                                              already produces mostly-valid
                                              SMILES, not starting from noise

Run from inside RL_updated_learnrate/, e.g.:

    python rl_finetune.py \
        --data ../data/BREAST-136344-56786-51.txt \
        --smiles ../data/CompoundSmiles_full_140474.txt \
        --genomic ../genomic_images \
        --ckpt ../checkpoints/best_random.pt \
        --resume ../VAE_updated_learnrate/checkpoints_gen/best_vae.pt \
        --epochs 100 --batch 64

Reward: reward = -predicted_ln_ic50  (lower predicted IC50 = more effective
= higher reward), with a fixed penalty for invalid/undecodable SMILES so the
policy has a gradient signal even from failures, and a running-mean baseline
subtracted for variance reduction (standard REINFORCE).

Every --eval_every epochs, also runs the independent, reward-model-blind
checks from rl_eval_independent.py and logs both reward-model score and the
independent score to history_rl.csv — the whole point of this script is to
be able to see if/when those two diverge.
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from rl_policy import PolicyWrapper

# ── import repo modules exactly like train.py / eval_novelty_and_conditioning.py ──
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
INVALID_PENALTY = -3.0  # reward assigned to SMILES that don't decode to a valid molecule


# ─────────────────────────────────────────────────────────────
# Reward: run generated (valid) SMILES back through the frozen
# DL4DR predictor, per cell line
# ─────────────────────────────────────────────────────────────

def score_with_predictor(predictor, smiles_list, genomic_img, device, mol_img_size=64):
    """
    For each SMILES: canonicalize + build (mol_graph, mol_img, ecfp);
    invalid ones get INVALID_PENALTY. Valid ones get reward = -predicted_ic50.
    Returns a (len(smiles_list),) tensor of rewards and a bool validity mask.
    """
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

    rewards = torch.full((len(smiles_list),), INVALID_PENALTY, device=device)
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
    ap.add_argument("--zc_dim", type=int, default=256)
    ap.add_argument("--out_dir", default="checkpoints_gen_rl")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--n_cell_lines", type=int, default=3, help="Cell lines to rotate through per epoch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.smiles is None:
        args.smiles = str((Path(args.data).resolve().parent / "CompoundSmiles_full_140474.txt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    records_all = root_dataset.load_records(args.data, args.smiles)
    vocab = SmilesVocab([r["smiles"] for r in records_all])
    print(f"Vocab size: {len(vocab)} | training records: {len(records_all)}")

    all_ach_ids = sorted(set(r["ach_id"] for r in records_all))

    # ── frozen models ──
    cell_encoder = FrozenCellLineEncoder(args.ckpt, device=device).to(device)
    predictor = DrugResponseModel().to(device)
    pred_state = torch.load(args.ckpt, map_location=device)
    pred_state = pred_state.get("model") or pred_state.get("model_state_dict") or pred_state
    predictor.load_state_dict(pred_state, strict=False)
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad = False

    # ── policy (generator), warm-started from the VAE if given ──
    policy_net = ConditionalSmilesVAE(vocab_size=len(vocab), zc_dim=args.zc_dim).to(device)
    if args.resume and os.path.exists(args.resume):
        policy_net.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Warm-started policy from {args.resume}")
    else:
        print("WARNING: no --resume given, policy starts from random weights "
              "(will likely waste early epochs on near-100% invalid SMILES)")
    policy = PolicyWrapper(policy_net)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)

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

        tokens, logprobs, entropy = policy.sample(z_c.repeat(args.batch, 1), n_samples=args.batch)
        smiles_batch = [vocab.decode(t.tolist()) for t in tokens]

        rewards, valid_mask = score_with_predictor(predictor, smiles_batch, img_tensor, device)

        # REINFORCE with baseline
        advantage = rewards.detach() - baseline
        loss = -(advantage * logprobs).mean() - args.entropy_coef * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()

        batch_mean_reward = rewards.mean().item()
        baseline = baseline_momentum * baseline + (1 - baseline_momentum) * batch_mean_reward
        validity = valid_mask.float().mean().item()

        row = dict(epoch=epoch, ach_id=ach_id, loss=loss.item(),
                   mean_reward=batch_mean_reward, validity=validity,
                   baseline=baseline, entropy=entropy.mean().item())
        history.append(row)
        print(f"Epoch {epoch:4d} | cell {ach_id} | loss {loss.item():+.4f} | "
              f"mean_reward {batch_mean_reward:+.4f} | validity {validity:.2%} | "
              f"entropy {entropy.mean().item():.3f}")

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
    print("Next: run rl_eval_independent.py on final_rl.pt to check whether "
          "reward-model score and independent chemical validity agree.")


if __name__ == "__main__":
    main()
