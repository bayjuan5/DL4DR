"""
new_rl_policy.py
=================
Stochastic policy wrapper around the FULL ConditionalSmilesVAE (model.py)
-- both its encoder and its decoder -- for REINFORCE-style fine-tuning.

Why this differs from rl_policy.py:
    rl_policy.py's PolicyWrapper.sample() draws z ~ N(0, I) via a raw
    torch.randn() call, mirroring ConditionalSmilesVAE.sample()'s
    unconditioned generation path. That z has no dependence on
    encoder_rnn / fc_mu / fc_logvar, so even though those parameters sit
    inside policy_net and get handed to the optimizer, no gradient from
    the REINFORCE surrogate loss ever reaches them -- only the decoder
    side (embedding, decoder_rnn, output_proj, zc_proj, decoder_input_proj)
    is actually fine-tuned. The encoder is dead weight during RL.

    new_rl_policy.py fixes this by sampling the latent from the VAE's own
    *posterior*: a real seed SMILES (from the same cell line) is run
    through vae.encode(x) -> (mu, logvar), then
    z = vae.reparameterize(mu, logvar). Because z is now a differentiable
    function of the encoder's output, the same score-function backward
    pass that already updates the decoder (loss depends on
    logprob_sum(theta), which depends on h0 = decoder_input_proj(z, z_c),
    which depends on mu/logvar/encoder_rnn) now also flows gradient into
    the encoder. No separate REINFORCE term on z is needed -- it falls
    out of the existing autograd graph once encode() is actually called.

    sample_from_prior() is kept around for encoder-free rollouts (e.g. no
    real molecule available for a given cell line that epoch, or
    encoder-free baseline comparisons at eval time).

Usage:
    from new_rl_policy import PolicyWrapper
    policy = PolicyWrapper(vae)                          # vae = ConditionalSmilesVAE

    # encoder + decoder both trainable (preferred path):
    tokens, logprobs, entropy, mu, logvar, z = policy.sample(x_seed, z_c)

    # decoder-only fallback (no seed molecule available):
    tokens, logprobs, entropy, mu, logvar, z = policy.sample(None, z_c, n_samples=64)
"""
import torch
import torch.nn.functional as F


class PolicyWrapper:
    """
    Turns a ConditionalSmilesVAE's encoder+decoder into a stochastic policy.

    Unlike rl_policy.py, the latent z is (by default) drawn from the VAE's
    encoder posterior q(z | x_seed) rather than the fixed prior, so both
    the encoder and the decoder receive a reward-shaped gradient during
    REINFORCE fine-tuning.
    """

    def __init__(self, vae):
        self.vae = vae

    # -----------------------------------------------------------------
    # shared autoregressive stochastic decode, given an already-computed z
    # -----------------------------------------------------------------
    def _rollout(self, z, z_c, temperature=1.0, max_len=None):
        """
        Autoregressively sample token sequences from a given latent z,
        tracking log-prob of each chosen token (the REINFORCE
        policy-gradient term) and per-step entropy (optional bonus to
        slow mode collapse). This is unchanged from rl_policy.py -- the
        only thing that differs between prior- and posterior-sampling is
        *how z was produced* before it gets here.

        Returns:
            tokens   : (B, L) long tensor of sampled token ids (BOS-prefixed)
            logprobs : (B,) sum of log-prob of each generated token
            entropy  : (B,) mean per-step entropy
        """
        vae = self.vae
        max_len = max_len or vae.max_len
        device = z.device
        B = z.size(0)

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

    # -----------------------------------------------------------------
    # encoder-free rollout (decoder only) -- kept for fallback/eval
    # -----------------------------------------------------------------
    def sample_from_prior(self, z_c, n_samples=1, temperature=1.0, max_len=None):
        """
        z ~ N(0, I) * temperature, exactly like rl_policy.py /
        ConditionalSmilesVAE.sample(). No gradient reaches the encoder
        from this path -- use only when no real seed molecule is
        available for the current cell line, or for encoder-free
        baseline comparisons.
        """
        vae = self.vae
        if z_c.dim() == 1:
            z_c = z_c.unsqueeze(0).repeat(n_samples, 1)
        z = torch.randn(z_c.size(0), vae.latent_dim, device=z_c.device) * temperature
        tokens, logprob_sum, entropy_mean = self._rollout(z, z_c, temperature, max_len)
        return tokens, logprob_sum, entropy_mean, None, None, z

    # -----------------------------------------------------------------
    # encoder + decoder rollout (preferred path)
    # -----------------------------------------------------------------
    def sample_from_posterior(self, x_seed, z_c, temperature=1.0, max_len=None):
        """
        Encode a batch of real seed SMILES (token ids, shape (B, L), same
        cell line as z_c) through the VAE encoder, reparameterize to get
        z, then roll out the stochastic decoder policy from that z.

        Because z = mu + eps * std is a differentiable function of
        encode()'s output, logprob_sum is differentiable w.r.t.
        encoder_rnn / fc_mu / fc_logvar as well as the decoder -- a
        REINFORCE loss built from logprob_sum trains the *whole* VAE.

        mu/logvar are also returned so the caller can add a KL term
        (standard VAE regularizer) and/or a teacher-forced reconstruction
        loss reusing the same z, to keep the encoder well-behaved instead
        of relying solely on the (noisy) policy-gradient signal.

        Returns:
            tokens, logprobs, entropy, mu, logvar, z
        """
        vae = self.vae
        mu, logvar = vae.encode(x_seed)
        z = vae.reparameterize(mu, logvar)
        tokens, logprob_sum, entropy_mean = self._rollout(z, z_c, temperature, max_len)
        return tokens, logprob_sum, entropy_mean, mu, logvar, z

    # -----------------------------------------------------------------
    # convenience dispatcher
    # -----------------------------------------------------------------
    def sample(self, x_seed, z_c, n_samples=1, temperature=1.0, max_len=None):
        """
        If x_seed is given: posterior sampling (encoder + decoder both
        trainable). If x_seed is None: prior sampling (decoder only,
        same behavior as rl_policy.py).
        """
        if x_seed is None:
            return self.sample_from_prior(
                z_c, n_samples=n_samples, temperature=temperature, max_len=max_len
            )
        return self.sample_from_posterior(
            x_seed, z_c, temperature=temperature, max_len=max_len
        )
