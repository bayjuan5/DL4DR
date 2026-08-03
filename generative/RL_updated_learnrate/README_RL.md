# RL_updated_learnrate

Alternative to `VAE_updated_learnrate/` — replaces the VAE's reconstruction
objective with REINFORCE fine-tuning against the frozen DL4DR predictor
(`../checkpoints/best_random.pt`) as a reward model.

## Files

- **rl_policy.py** — wraps `ConditionalSmilesVAE` (from `../model.py`) as a
  stochastic policy with log-prob tracking, needed for REINFORCE. Samples
  the latent `z` from the fixed prior `N(0, I)`, so only the **decoder**
  gets gradient from REINFORCE — the encoder's weights are technically
  optimized but never actually touched. Doesn't modify `../model.py`, so
  `VAE_updated_learnrate` still works unchanged.
- **rl_finetune.py** — training loop for the decoder-only policy above.
  Warm-starts from an existing VAE checkpoint (`--resume`), fine-tunes
  against the predictor as reward, logs `history_rl.csv` (loss /
  mean_reward / validity / entropy per epoch).
- **new_rl_policy.py** — same idea, but the whole VAE (encoder + decoder)
  is trainable. `sample_from_posterior()` encodes a real seed SMILES for
  the current cell line via `vae.encode()` → `reparameterize()` to get
  `z`, so the REINFORCE surrogate loss backprops into the encoder
  (`encoder_rnn`/`fc_mu`/`fc_logvar`) as well as the decoder.
  `sample_from_prior()` is kept as a fallback for cell lines with no
  training compounds on record.
- **new_rl_finetune.py** — training loop for the full-VAE policy. Each
  epoch: pulls a real seed-SMILES batch for the current cell line →
  posterior-samples the policy → REINFORCE loss on the reward, **plus**
  the standard VAE ELBO (teacher-forced reconstruction + KL) on the same
  `z`, weighted by `--vae_weight` / `--kl_weight`, so the encoder's
  posterior stays well-behaved instead of drifting on policy-gradient
  noise alone. Logs `history_rl.csv` with the extra `rl_loss` /
  `vae_recon_loss` / `vae_kl_loss` / `used_encoder` columns.
- **rl_eval_independent.py** — the important one. Scores generated molecules
  two separate ways: (1) the frozen predictor's own score (what the policy
  was optimized for), and (2) a set of checks the predictor has no way to
  see or influence — RDKit validity, Lipinski Rule-of-Five druglikeness,
  novelty, uniqueness, nearest-neighbor similarity to known actives. Prints
  a flag if reward-model score is high while Lipinski pass rate is low.
  Only works with **rl_finetune.py** checkpoints — it calls
  `rl_policy.PolicyWrapper.sample(z_c, n_samples, ...)`, which has a
  different argument order and return arity than `new_rl_policy`'s
  wrapper, so it is not a drop-in evaluator for the full-VAE checkpoints.
- **new_rl_eval_independent.py** — same checks, wired to `new_rl_policy`
  for **new_rl_finetune.py** checkpoints. Adds `--sample_mode
  {prior,posterior,both}`: `prior` generates unconditioned (encoder
  untouched, directly comparable to `rl_eval_independent.py`'s numbers);
  `posterior` encodes a real seed SMILES per cell line and samples from
  the encoder's posterior, which is the mode that actually exercises the
  fine-tuned encoder. Flags low posterior-mode novelty as a sign the
  encoder may be near-copying seeds rather than usefully perturbing them.

## Quick start

Decoder-only policy (original):

```bash
cd RL_updated_learnrate

python rl_finetune.py \
    --data ../../data/BREAST-136344-56786-51.txt \
    --smiles ../../data/CompoundSmiles_full_140474.txt \
    --genomic ../../genomic_images \
    --ckpt ../../checkpoints/best_random.pt \
    --resume ../VAE_updated_learnrate/checkpoints_gen/best_vae.pt \
    --epochs 100 --batch 64 --lr 1e-5
```

Full VAE (encoder + decoder), the version to use if you want the encoder
to keep training instead of sitting frozen:

```bash
cd RL_updated_learnrate

python new_rl_finetune.py \
    --data ../../data/BREAST-136344-56786-51.txt \
    --smiles ../../data/CompoundSmiles_full_140474.txt \
    --genomic ../../genomic_images \
    --ckpt ../../checkpoints/best_random.pt \
    --resume ../VAE_updated_learnrate/checkpoints_gen/best_vae.pt \
    --epochs 100 --batch 64 --lr 1e-5 --vae_weight 1.0 --kl_weight 1.0
```

Evaluation — decoder-only checkpoint:

```bash
python rl_eval_independent.py \
    --rl_ckpt checkpoints_gen_rl/final_rl.pt \
    --dl4dr_ckpt ../../checkpoints/best_random.pt \
    --data ../../data/BREAST-136344-56786-51.txt \
    --smiles ../../data/CompoundSmiles_full_140474.txt \
    --genomic ../../genomic_images \
    --n_samples 200
```

Evaluation — full-VAE checkpoint (note the different script):

```bash
python new_rl_eval_independent.py \
    --rl_ckpt checkpoints_gen_rl_full_vae/final_rl.pt \
    --dl4dr_ckpt ../../checkpoints/best_random.pt \
    --data ../../data/BREAST-136344-56786-51.txt \
    --smiles ../../data/CompoundSmiles_full_140474.txt \
    --genomic ../../genomic_images \
    --n_samples 200 --sample_mode both
```

## Notes / open items

- `--lr` is set low (1e-5) on purpose — this is fine-tuning an
  already-functional decoder, not training from scratch. Raise it only if
  reward barely moves after ~30 epochs.
- Both `rl_finetune.py` and `new_rl_finetune.py` rotate through cell lines
  one at a time per epoch (`epoch % len(all_ach_ids)`) rather than batching
  across cell lines — simplest correct thing to ship first; revisit if
  training is too noisy.
- `new_rl_finetune.py` needs at least one real training compound on record
  for a cell line to run the encoder path that epoch; if a cell line has
  none, it silently falls back to prior sampling (decoder-only) for that
  epoch and logs `used_encoder=False` in `history_rl.csv`.
- `--vae_weight 0.0` turns `new_rl_finetune.py`'s loss into pure REINFORCE
  (still routed through the encoder via posterior sampling, just without
  the ELBO regularizer) — useful for isolating how much the ELBO term is
  actually doing.
- Reward model input pipeline (SMILES → mol_graph / mol_img / ecfp) mirrors
  exactly what `train_early.py`'s dataset does, so this is not deviating
  from anything already validated on the DL4DR side.
- Not yet done: hyperparameter sweep, entropy_coef tuning, comparison of
  warm-started vs. from-scratch policy, and a direct comparison of
  decoder-only (`rl_finetune.py`) vs. full-VAE (`new_rl_finetune.py`) on
  the same cell lines/checkpoint. Treat the current defaults as a first
  pass, not a tuned setup.
