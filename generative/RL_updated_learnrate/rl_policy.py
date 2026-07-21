"""
rl_policy.py
============
Wraps the existing ConditionalSmilesVAE decoder as a stochastic policy for
REINFORCE-style fine-tuning, without modifying model.py.

Why a wrapper instead of editing model.py:
  - decode(..., teacher_forcing=False) in model.py is greedy (argmax) — fine
    for VAE sampling, but REINFORCE needs stochastic actions *and* their
    log-probabilities under the policy at the moment they were taken.
  - Keeping this separate means VAE_updated_learnrate and RL_updated_learnrate
    can both import the same model.py/ConditionalSmilesVAE without stepping
    on each other.

Usage:
    from rl_policy import PolicyWrapper
    policy = PolicyWrapper(vae)                    # vae = ConditionalSmilesVAE
    tokens, logprobs, entropy = policy.sample(z_c, temperature=1.0)
"""
import torch
import torch.nn.functional as F


class PolicyWrapper:
    """
    Turns a ConditionalSmilesVAE's decoder into a stochastic policy.

    The VAE's own "prior" sampling (z = randn(latent_dim)) is kept as-is —
    we're fine-tuning the *decoder's* token-choice policy conditioned on
    (z, z_c), not the latent prior itself. This mirrors how the VAE already
    samples in model.py's ConditionalSmilesVAE.sample().
    """

    def __init__(self, vae):
        self.vae = vae

    def sample(self, z_c, n_samples=1, temperature=1.0, max_len=None):
        """
        Autoregressively sample token sequences, tracking log-prob of each
        chosen token (needed for the REINFORCE policy-gradient term) and
        per-step entropy (useful as an optional entropy-bonus regularizer
        to keep the policy from collapsing onto a narrow set of outputs —
        the same "mode collapse" failure mode already seen with the VAE).

        Returns:
            tokens   : (B, L) long tensor of sampled token ids (BOS-prefixed)
            logprobs : (B,) sum of log-prob of each generated token
            entropy  : (B,) mean per-step entropy (for optional bonus)
        """
        vae = self.vae
        max_len = max_len or vae.max_len
        device = z_c.device

        if z_c.dim() == 1:
            z_c = z_c.unsqueeze(0).repeat(n_samples, 1)
        B = z_c.size(0)

        z = torch.randn(B, vae.latent_dim, device=device) * temperature
        z_c_proj = vae.zc_proj(z_c)
        joint = torch.cat([z, z_c_proj], dim=-1)
        h = vae.decoder_input_proj(joint).unsqueeze(0)

        tokens = torch.full((B, 1), 1, dtype=torch.long, device=device)  # BOS
        logprob_sum = torch.zeros(B, device=device)
        entropy_sum = torch.zeros(B, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            emb = vae.embedding(tokens[:, -1:])
            out, h = vae.decoder_rnn(emb, h)
            logits = vae.output_proj(out[:, -1, :]) / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)

            next_token = dist.sample()
            step_logprob = dist.log_prob(next_token)
            step_entropy = dist.entropy()

            # freeze already-finished sequences at pad-like no-op contribution
            step_logprob = torch.where(done, torch.zeros_like(step_logprob), step_logprob)
            step_entropy = torch.where(done, torch.zeros_like(step_entropy), step_entropy)

            logprob_sum = logprob_sum + step_logprob
            entropy_sum = entropy_sum + step_entropy

            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            done = done | (next_token == getattr(vae, "eos_idx", 2))
            if done.all():
                break

        steps_taken = tokens.size(1) - 1
        entropy_mean = entropy_sum / max(steps_taken, 1)
        return tokens, logprob_sum, entropy_mean
