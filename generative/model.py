"""
Conditional SMILES VAE for DL4DR.
Generates compounds (SMILES) conditioned on a frozen cell-line embedding z_C.

This version avoids circular imports and loads ONLY the cell-line conditioning
path from the DL4DR checkpoint, which is more robust across predictor versions.
"""
import os
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


def _load_repo_model_module():
    """Load the repo-root model.py under an alias, not as 'model'."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(repo_root, "model.py")
    spec = importlib.util.spec_from_file_location("dl4dr_repo_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SmilesVocab:
    def __init__(self, smiles_list):
        chars = sorted(set("".join(smiles_list)))
        self.itos = [PAD, BOS, EOS, UNK] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_idx = self.stoi[PAD]
        self.bos_idx = self.stoi[BOS]
        self.eos_idx = self.stoi[EOS]
        self.unk_idx = self.stoi[UNK]

    def __len__(self):
        return len(self.itos)

    def encode(self, smiles, max_len):
        ids = [self.bos_idx] + [self.stoi.get(c, self.unk_idx) for c in smiles][:max_len - 2] + [self.eos_idx]
        ids += [self.pad_idx] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids):
        chars = []
        for i in ids:
            if i == self.eos_idx:
                break
            if i in (self.pad_idx, self.bos_idx):
                continue
            chars.append(self.itos[i])
        return "".join(chars)


class _CellConditionerFromRepo(nn.Module):
    """Rebuild only cell_encoder + cell_proj from the repo model/checkpoint."""
    def __init__(self, checkpoint_path, device="cpu"):
        super().__init__()
        repo_model = _load_repo_model_module()
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt

        if "cell_proj.weight" not in state:
            raise RuntimeError("Checkpoint does not contain 'cell_proj.weight'; cannot infer z_C path.")

        # Infer dimensions from checkpoint.
        cell_out = int(state["cell_proj.weight"].shape[1])   # input dim to cell_proj
        proj_dim = int(state["cell_proj.weight"].shape[0])   # output z_C dim

        # Prefer the repo's CellLineEncoder if available.
        if hasattr(repo_model, "DrugResponseConfig") and hasattr(repo_model, "CellLineEncoder"):
            cfg = repo_model.DrugResponseConfig()
            if hasattr(cfg, "cell_hidden"):
                # infer hidden from cell_encoder.proj weight if present, else keep default
                if "cell_encoder.proj.weight" in state:
                    cfg.cell_hidden = int(state["cell_encoder.proj.weight"].shape[1])
            if hasattr(cfg, "cell_out"):
                cfg.cell_out = cell_out

            self.cell_encoder = repo_model.CellLineEncoder(cfg)
            self.cell_proj = nn.Linear(cell_out, proj_dim)

            enc_state = {k.replace("cell_encoder.", ""): v for k, v in state.items() if k.startswith("cell_encoder.")}
            proj_state = {k.replace("cell_proj.", ""): v for k, v in state.items() if k.startswith("cell_proj.")}

            self.cell_encoder.load_state_dict(enc_state, strict=False)
            self.cell_proj.load_state_dict(proj_state, strict=False)
        else:
            raise RuntimeError("Repo model.py does not expose DrugResponseConfig/CellLineEncoder.")

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, genomic_image):
        x = self.cell_encoder(genomic_image)
        x = self.cell_proj(x)
        return x


class FrozenCellLineEncoder(nn.Module):
    """
    Frozen conditioner that returns z_C from the pretrained DL4DR checkpoint.

    It first tries to reconstruct only the cell-line path from the repo model/checkpoint.
    """
    def __init__(self, checkpoint_path, device="cpu"):
        super().__init__()
        self.impl = _CellConditionerFromRepo(checkpoint_path, device=device)

    @torch.no_grad()
    def forward(self, genomic_image):
        return self.impl(genomic_image)


class ConditionalSmilesVAE(nn.Module):
    def __init__(self, vocab_size, zc_dim, emb_dim=128, hidden_dim=256,
                 latent_dim=64, pad_idx=0, max_len=120):
        super().__init__()
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.encoder_rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        self.zc_proj = nn.Linear(zc_dim, latent_dim)
        self.decoder_input_proj = nn.Linear(latent_dim * 2, hidden_dim)
        self.decoder_rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        emb = self.embedding(x)
        _, h = self.encoder_rnn(emb)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, z_c, target=None, teacher_forcing=True):
        z_c_proj = self.zc_proj(z_c)
        joint = torch.cat([z, z_c_proj], dim=-1)
        h0 = self.decoder_input_proj(joint).unsqueeze(0)

        if teacher_forcing:
            emb = self.embedding(target[:, :-1])
            out, _ = self.decoder_rnn(emb, h0)
            return self.output_proj(out)

        B = z.size(0)
        device = z.device
        tokens = torch.full((B, 1), 1, dtype=torch.long, device=device)  # BOS index
        h = h0
        for _ in range(self.max_len - 1):
            emb = self.embedding(tokens[:, -1:])
            out, h = self.decoder_rnn(emb, h)
            next_token = self.output_proj(out[:, -1, :]).argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

    def forward(self, x, z_c):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, z_c, target=x, teacher_forcing=True)
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, z_c, n_samples=1, temperature=1.0):
        device = z_c.device
        if z_c.dim() == 1:
            z_c = z_c.unsqueeze(0).repeat(n_samples, 1)
        z = torch.randn(z_c.size(0), self.latent_dim, device=device) * temperature
        return self.decode(z, z_c, teacher_forcing=False)


def vae_loss(logits, target, mu, logvar, pad_idx, kl_weight=1.0):
    B, L, V = logits.shape
    recon = F.cross_entropy(
        logits.reshape(-1, V),
        target[:, 1:1 + L].reshape(-1),
        ignore_index=pad_idx,
        reduction="mean",
    )
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon, kl
