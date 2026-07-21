# RL_updated_learnrate

Alternative to `VAE_updated_learnrate/` — replaces the VAE's reconstruction
objective with REINFORCE fine-tuning against the frozen DL4DR predictor
(`../checkpoints/best_random.pt`) as a reward model.

## Files

- **rl_policy.py** — wraps `ConditionalSmilesVAE` (from `../model.py`) as a
  stochastic policy with log-prob tracking, needed for REINFORCE. Doesn't
  modify `../model.py`, so `VAE_updated_learnrate` still works unchanged.
- **rl_finetune.py** — training loop. Warm-starts from an existing VAE
  checkpoint (`--resume`), fine-tunes against the predictor as reward,
  logs `history_rl.csv` (loss / mean_reward / validity / entropy per epoch).
- **rl_eval_independent.py** — the important one. Scores generated molecules
  two separate ways: (1) the frozen predictor's own score (what the policy
  was optimized for), and (2) a set of checks the predictor has no way to
  see or influence — RDKit validity, Lipinski Rule-of-Five druglikeness,
  novelty, uniqueness, nearest-neighbor similarity to known actives. Prints
  a flag if reward-model score is high while Lipinski pass rate is low.

## Quick start

```bash
cd RL_updated_learnrate

python rl_finetune.py \
    --data ../data/BREAST-136344-56786-51.txt \
    --smiles ../data/CompoundSmiles_full_140474.txt \
    --genomic ../genomic_images \
    --ckpt ../checkpoints/best_random.pt \
    --resume ../VAE_updated_learnrate/checkpoints_gen/best_vae.pt \
    --epochs 100 --batch 64 --lr 1e-5

python rl_eval_independent.py \
    --rl_ckpt checkpoints_gen_rl/final_rl.pt \
    --dl4dr_ckpt ../checkpoints/best_random.pt \
    --data ../data/BREAST-136344-56786-51.txt \
    --smiles ../data/CompoundSmiles_full_140474.txt \
    --genomic ../genomic_images \
    --n_samples 200
```

## Notes / open items

- `--lr` is set low (1e-5) on purpose — this is fine-tuning an
  already-functional decoder, not training from scratch. Raise it only if
  reward barely moves after ~30 epochs.
- `rl_finetune.py` rotates through cell lines one at a time per epoch
  (`epoch % len(all_ach_ids)`) rather than batching across cell lines —
  simplest correct thing to ship first; revisit if training is too noisy.
- Reward model input pipeline (SMILES → mol_graph / mol_img / ecfp) mirrors
  exactly what `train_early.py`'s dataset does, so this is not deviating
  from anything already validated on the DL4DR side.
- Not yet done: hyperparameter sweep, entropy_coef tuning, comparison of
  warm-started vs. from-scratch policy. Treat the current defaults as a
  first pass, not a tuned setup.
