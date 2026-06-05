"""
model.py
========
Two-Tower Residual Late Fusion model for drug response prediction.
Architecture described in ORNN_DMPNN_Theory_v18, Section 13.

  Compound Tower  →  D-MPNN  (molecular graph)
                  →  ORNN    (2D molecular image, octave CNN)
                  →  ECFP Head  (fixed fingerprint + MLP)

  Cell Line Tower →  CNN Encoder  (139×139×3 genomic image, no ID lookup)
                  →  Gate  λ(z_C)  ∈ (0, 1)

  Fusion:   f = f_hard(x_ECFP ⊕ z_C)
              + λ(z_C) · f_residual( CrossAttn(Q=z_C, KV=[z_ORNN, z_DMPNN]) )

All dimensions are configurable via DrugResponseConfig.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class DrugResponseConfig:
    # ECFP
    ecfp_dim: int           = 2048
    # D-MPNN
    dmpnn_hidden: int       = 300
    dmpnn_depth: int        = 3
    dmpnn_dropout: float    = 0.0
    # ORNN / compound image
    ornn_out: int           = 256
    octave_alpha: float     = 0.75   # fraction of channels for high-freq path
    # Cell line CNN encoder
    cell_hidden: int        = 128
    cell_out: int           = 256
    # Fusion / projection
    proj_dim: int           = 256
    attn_heads: int         = 4
    # Hard head
    hard_hidden: int        = 512
    # Residual head
    res_hidden: int         = 512
    # Dropout in MLP heads
    mlp_dropout: float      = 0.1
    # Atom / bond feature sizes (chemprop defaults)
    atom_feat_dim: int      = 72
    bond_feat_dim: int      = 14


# ─────────────────────────────────────────────────────────────
# Utility blocks
# ─────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class OctaveConv2d(nn.Module):
    """
    Single Octave Convolution layer.
    Splits channels into high-freq (alpha fraction) and low-freq (1-alpha).
    High-freq path captures localised pharmacophoric features;
    low-freq path captures global molecular topology.
    """
    def __init__(self, in_ch: int, out_ch: int, alpha: float = 0.75,
                 kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.alpha = alpha
        hf_in  = max(1, int(in_ch  * alpha))
        lf_in  = in_ch  - hf_in
        hf_out = max(1, int(out_ch * alpha))
        lf_out = out_ch - hf_out

        self.hf2hf = nn.Conv2d(hf_in,  hf_out, kernel, stride, padding, bias=False)
        self.lf2lf = nn.Conv2d(lf_in,  lf_out, kernel, stride, padding, bias=False)
        self.hf2lf = nn.Conv2d(hf_in,  lf_out, kernel, stride, padding, bias=False)
        self.lf2hf = nn.Conv2d(lf_in,  hf_out, kernel, stride, padding, bias=False)
        self.hf_out = hf_out
        self.lf_out = lf_out

    def forward(self, x_hf: torch.Tensor, x_lf: torch.Tensor):
        hf = self.hf2hf(x_hf) + F.interpolate(
            self.lf2hf(x_lf), size=x_hf.shape[-2:], mode="nearest")
        lf = self.lf2lf(x_lf) + F.avg_pool2d(
            self.hf2lf(x_hf), kernel_size=2, stride=2, ceil_mode=True)
        return hf, lf


class ORNN(nn.Module):
    """
    Octave Residual Neural Network — compound image encoder.
    Input:  (B, 3, H, W) RGB 2D molecular depiction.
    Output: (B, ornn_out) embedding z_ORNN.
    """
    def __init__(self, cfg: DrugResponseConfig):
        super().__init__()
        a = cfg.octave_alpha
        # Initial standard conv to enter octave space
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.oct1 = OctaveConv2d(64,  128, alpha=a)
        self.oct2 = OctaveConv2d(128, 128, alpha=a)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, cfg.ornn_out)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.stem(img)                          # (B, 64, H, W)
        n_hf = max(1, int(64 * 0.75))
        x_hf, x_lf = x[:, :n_hf], x[:, n_hf:]
        x_hf, x_lf = self.oct1(x_hf, x_lf)
        x_hf, x_lf = self.oct2(x_hf, x_lf)
        # Merge and pool
        x = torch.cat([x_hf, x_lf], dim=1)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class DMPNNEncoder(nn.Module):
    """
    Directed Message Passing Neural Network — molecular graph encoder.
    Simplified implementation; for production use chemprop's MessagePassing.

    Expects pre-computed node and edge features from RDKit via chemprop featurisation.
    Input:
        atom_feats  : (N_atoms, atom_feat_dim)
        bond_feats  : (N_bonds, bond_feat_dim)
        a2b         : (N_atoms, max_bonds)   — atom→bond adjacency (padded with -1)
        b2a         : (N_bonds,)             — bond → source atom index
        b2revb      : (N_bonds,)             — bond → its reverse bond index
    Output: (B, dmpnn_hidden) pooled graph embedding z_DMPNN.
    """
    def __init__(self, cfg: DrugResponseConfig):
        super().__init__()
        d   = cfg.dmpnn_hidden
        af  = cfg.atom_feat_dim
        bf  = cfg.bond_feat_dim
        self.W_i   = nn.Linear(af + bf, d, bias=False)
        self.W_m   = nn.Linear(d, d, bias=False)
        self.W_a   = nn.Linear(af + d, d)
        self.depth = cfg.dmpnn_depth
        self.drop  = nn.Dropout(cfg.dmpnn_dropout)
        self.act   = nn.ReLU()

    def forward(self,
                atom_feats: torch.Tensor,
                bond_feats: torch.Tensor,
                a2b: torch.Tensor,
                b2a: torch.Tensor,
                b2revb: torch.Tensor,
                atom_scope: list) -> torch.Tensor:
        """
        atom_scope: list of (start, length) tuples, one per molecule in batch.
        Returns: (batch_size, dmpnn_hidden)
        """
        # Initialise edge hidden states
        h0 = self.act(self.W_i(torch.cat([atom_feats[b2a], bond_feats], dim=1)))
        h  = h0.clone()

        for _ in range(self.depth - 1):
            # Aggregate neighbour messages, excluding reverse bond
            # a2b: (N_atoms, max_neigh) — indices into bond table (-1 = padding)
            valid = (a2b >= 0)
            a2b_clamped = a2b.clamp(min=0)
            neigh_h = h[a2b_clamped]                         # (N_atoms, max_neigh, d)
            neigh_h = neigh_h * valid.unsqueeze(-1).float()  # zero out padding
            m = neigh_h.sum(dim=1)                           # (N_atoms, d)
            # Message for each directed bond: sum(neighbours) - reverse bond
            m_bond = m[b2a] - h[b2revb]
            h = self.act(h0 + self.W_m(m_bond))
            h = self.drop(h)

        # Atom-level aggregation
        valid = (a2b >= 0)
        a2b_clamped = a2b.clamp(min=0)
        neigh_h = h[a2b_clamped]
        neigh_h = neigh_h * valid.unsqueeze(-1).float()
        a_msg = neigh_h.sum(dim=1)
        a_feat = self.act(self.W_a(torch.cat([atom_feats, a_msg], dim=1)))

        # Pool over atoms in each molecule
        out = []
        for start, length in atom_scope:
            out.append(a_feat[start: start + length].mean(dim=0))
        return torch.stack(out, dim=0)


class CellLineEncoder(nn.Module):
    """
    CNN encoder for the 139×139×3 genomic image.
    Three convolutional blocks (matching run_gradcam.py's encoder[17] target).
    No cell-line ID lookup — pure content encoding.

    Architecture (encoder indices for GradCAM):
      [0] Conv2d 3→32,  [1] GroupNorm, [2] GELU, [3] Conv2d 32→32,  [4] GN, [5] GELU, [6] MaxPool
      [7] Conv2d 32→64, [8] GN,        [9] GELU, [10]Conv2d 64→64, [11] GN,[12] GELU,[13] MaxPool
      [14]Conv2d 64→128,[15]GN,       [16]GELU, [17]Conv2d 128→128,[18]GN,[19]GELU,[20]MaxPool
      [21]AdaptiveAvgPool, [22]Flatten
    """
    def __init__(self, cfg: DrugResponseConfig):
        super().__init__()

        def block(in_c, out_c):
            return [
                nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, out_c), out_c),
                nn.GELU(),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, out_c), out_c),
                nn.GELU(),
                nn.MaxPool2d(2),
            ]

        self.encoder = nn.Sequential(
            *block(3, 32),          # indices 0-6
            *block(32, 64),         # indices 7-13
            *block(64, cfg.cell_hidden),  # indices 14-20  (cfg.cell_hidden = 128)
            nn.AdaptiveAvgPool2d(1),      # index 21
            nn.Flatten(),                 # index 22
        )
        self.proj = nn.Linear(cfg.cell_hidden, cfg.cell_out)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(img))


# ─────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────

class DrugResponseModel(nn.Module):
    """
    Two-Tower Residual Late Fusion model.

    Forward inputs:
        ecfp          (B, ecfp_dim)         — ECFP bit-vector
        atom_feats    (N_atoms, atom_feat_dim)
        bond_feats    (N_bonds, bond_feat_dim)
        a2b, b2a, b2revb, atom_scope        — D-MPNN graph tensors
        mol_img       (B, 3, H, W)          — 2D compound image
        cell_img      (B, 3, 139, 139)      — genomic image

    Returns:
        pred          (B,)                  — predicted ln(IC50)
        lambda_val    (B,)                  — learned gate value (for monitoring)
    """

    def __init__(self, cfg: Optional[DrugResponseConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = DrugResponseConfig()
        self.cfg = cfg

        d = cfg.proj_dim

        # ── Compound tower ──────────────────────────────────
        self.dmpnn       = DMPNNEncoder(cfg)
        self.ornn        = ORNN(cfg)
        self.dmpnn_proj  = nn.Linear(cfg.dmpnn_hidden, d)
        self.ornn_proj   = nn.Linear(cfg.ornn_out,     d)

        # ── Cell line tower ─────────────────────────────────
        self.cell_encoder = CellLineEncoder(cfg)
        self.cell_proj    = nn.Linear(cfg.cell_out, d)

        # ── Gate  λ(z_C) ∈ (0, 1) ──────────────────────────
        self.gate = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
            nn.Sigmoid(),
        )

        # ── Hard memory head  f_hard(x_ECFP ⊕ z_C) ─────────
        self.hard_head = _mlp(cfg.ecfp_dim + d, cfg.hard_hidden, 1, cfg.mlp_dropout)

        # ── Cross-attention  Q=z_C, KV=[z_ORNN, z_DMPNN] ───
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.attn_heads,
            dropout=cfg.mlp_dropout, batch_first=True,
        )

        # ── Residual head  f_residual(attn_out) ─────────────
        self.res_head = _mlp(d, cfg.res_hidden, 1, cfg.mlp_dropout)

    # ── Caching helper (used during training for efficiency) ─
    @torch.no_grad()
    def encode_cell_lines(self, cell_imgs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of unique genomic images once and cache."""
        return self.cell_proj(self.cell_encoder(cell_imgs))

    def forward(self,
                ecfp:       torch.Tensor,
                atom_feats: torch.Tensor,
                bond_feats: torch.Tensor,
                a2b:        torch.Tensor,
                b2a:        torch.Tensor,
                b2revb:     torch.Tensor,
                atom_scope: list,
                mol_img:    torch.Tensor,
                cell_img:   torch.Tensor,
                cell_idx:   Optional[torch.Tensor] = None,
                cell_cache: Optional[torch.Tensor] = None,
                ):
        # ── Cell line embedding ──────────────────────────────
        if cell_cache is not None and cell_idx is not None:
            z_C = cell_cache[cell_idx]          # (B, d)  — cached, no grad
        else:
            z_C = self.cell_proj(self.cell_encoder(cell_img))

        # ── Compound embeddings ──────────────────────────────
        z_dmpnn = self.dmpnn_proj(
            self.dmpnn(atom_feats, bond_feats, a2b, b2a, b2revb, atom_scope)
        )                                        # (B, d)
        z_ornn  = self.ornn_proj(self.ornn(mol_img))   # (B, d)

        # ── Hard memory head ────────────────────────────────
        f_hard = self.hard_head(
            torch.cat([ecfp, z_C], dim=-1)
        ).squeeze(-1)                             # (B,)

        # ── Cross-attention (Q=z_C, KV=compound embeddings) ─
        kv = torch.stack([z_ornn, z_dmpnn], dim=1)  # (B, 2, d)
        q  = z_C.unsqueeze(1)                        # (B, 1, d)
        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = attn_out.squeeze(1)               # (B, d)

        # ── Gated residual ───────────────────────────────────
        lam      = self.gate(z_C)                    # (B, 1)
        f_res    = self.res_head(attn_out).squeeze(-1)  # (B,)
        pred     = f_hard + lam.squeeze(-1) * f_res  # (B,)

        return pred, lam.squeeze(-1)
