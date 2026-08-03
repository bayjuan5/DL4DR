"""
model.py
========
Two-Tower Network with Residual Late Fusion for Drug Response Prediction.

Architecture
------------
                    SMILES
                      │
              ┌───────┴────────┐
           D-MPNN           ORNN (mol image)
              │                │
           z_DMPNN          z_ORNN
              └───────┬────────┘
                      │
               CrossAttention  ← Q = z_C (cell line)
                      │             KV = [z_DMPNN, z_ORNN]
                      │
    Genomic Image     │
          │           │
       Cell CNN       │
          │           │
         z_C ─────────┤
          │           │
    Hard Memory    Gated Residual
       Head           │
          │        λ(z_C) · f_residual(CrossAttn)
          └─────────┬─┘
                    │
               IC50 prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

ATOM_FEATURES   = 34    # one-hot atom feature vector length (see atom_features())
BOND_FEATURES   = 9     # one-hot bond feature vector length (see bond_features())
ECFP_BITS       = 2048  # Morgan fingerprint length
IMAGE_SIZE      = 139   # genomic image spatial size
IMAGE_CHANNELS  = 3     # R=expression, G=masked_expression, B=reserved(0)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  MOLECULAR FEATURISATION
# ──────────────────────────────────────────────────────────────────────────────

def one_hot(value, choices):
    enc = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    enc[idx] = 1
    return enc


def atom_features(atom) -> list:
    return (
        one_hot(atom.GetAtomicNum(), [1,5,6,7,8,9,15,16,17,35,53]) +
        one_hot(atom.GetDegree(),    [0,1,2,3,4,5]) +
        one_hot(atom.GetFormalCharge(), [-2,-1,0,1,2]) +
        one_hot(int(atom.GetHybridization()),
                [2,3,4,5,6]) +           # SP, SP2, SP3, SP3D, SP3D2
        [atom.GetIsAromatic(),
         atom.IsInRing(),
         atom.GetTotalNumHs() / 4.0]     # normalised
    )


def bond_features(bond) -> list:
    bt = bond.GetBondTypeAsDouble()
    return (
        one_hot(bt, [1.0, 1.5, 2.0, 3.0]) +
        [bond.GetIsConjugated(),
         bond.IsInRing(),
         bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE,
         bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE]
    )


def smiles_to_graph(smiles: str) -> Data:
    """Convert a SMILES string to a PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Node features
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()],
                     dtype=torch.float)

    # Edge index + edge features (both directions)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf    = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr  += [bf, bf]

    if len(edge_index) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, BOND_FEATURES), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_ecfp(smiles: str, n_bits: int = ECFP_BITS) -> torch.Tensor:
    """Convert SMILES to Morgan (ECFP4) fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(n_bits, dtype=torch.float)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2,
                                                         nBits=n_bits)
    return torch.tensor(list(fp), dtype=torch.float)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  CELL LINE TOWER  (Genomic Image CNN)
# ──────────────────────────────────────────────────────────────────────────────

class CellLineTower(nn.Module):
    """
    Encodes a 139×139×3 genomic image to a dense embedding z_C.

    Deliberately shallow (3 conv blocks) because:
    - Effective sample size = number of distinct cell lines in training set (~30)
    - Deep networks overfit immediately on such small N
    - GroupNorm instead of BatchNorm (stable at batch size 1)
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: 139×139 → 69×69
            nn.Conv2d(3,  32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 2: 69×69 → 34×34
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 3: 34×34 → 17×17
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1),   # → (batch, 128, 1, 1)
            nn.Flatten(),              # → (batch, 128)
        )
        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 3, 139, 139)  float32 in [0, 1]
        returns: (batch, out_dim)
        """
        return self.proj(self.encoder(x))


# ──────────────────────────────────────────────────────────────────────────────
# 3.  COMPOUND TOWER
# ──────────────────────────────────────────────────────────────────────────────

class DMPNNEncoder(nn.Module):
    """
    Directed Message Passing Neural Network.
    Treats the molecule as a graph; iterative message passing
    aggregates local chemical environment into node embeddings,
    then global mean pool → molecular embedding.

    Returns token-level embeddings (one per atom) for CrossAttention.
    """
    def __init__(self, node_dim: int = ATOM_FEATURES,
                 edge_dim: int = BOND_FEATURES,
                 hidden_dim: int = 128, depth: int = 3,
                 out_dim: int = 128):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        # NNConv: edge-conditioned message passing
        nn_list = []
        for _ in range(depth):
            edge_net = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim * hidden_dim),
            )
            nn_list.append(NNConv(hidden_dim, hidden_dim,
                                  edge_net, aggr='mean'))
        self.convs    = nn.ModuleList(nn_list)
        self.norms    = nn.ModuleList([nn.LayerNorm(hidden_dim)
                                       for _ in range(depth)])
        self.proj     = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Batch):
        """
        data: PyG Batch of molecular graphs
        returns:
          node_emb: (total_nodes, out_dim)   token-level embeddings
          batch:    (total_nodes,)            batch assignment vector
        """
        x   = F.gelu(self.node_embed(data.x))
        ea  = data.edge_attr
        ei  = data.edge_index

        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.gelu(conv(x, ei, ea)) + x)   # residual

        return self.proj(x), data.batch


class ORNNEncoder(nn.Module):
    """
    Molecular image encoder (ORNN).
    Renders the molecule as a 2D image; CNN extracts visual/topological features.
    Returns token-level patch embeddings for CrossAttention.

    Input: (batch, 3, H, W) molecular structure images
    """
    def __init__(self, img_size: int = 64, out_dim: int = 128,
                 patch_tokens: int = 16):
        super().__init__()
        self.patch_tokens = patch_tokens
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),   # → (batch, 128, 4, 4) = 16 patches
            nn.Flatten(2),             # → (batch, 128, 16)
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, 3, H, W)
        returns: (batch, patch_tokens, out_dim)
        """
        feat = self.encoder(x)          # (batch, 128, 16)
        feat = feat.permute(0, 2, 1)    # (batch, 16, 128)
        return self.proj(feat)          # (batch, 16, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CROSS-ATTENTION FUSION
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention where:
      Query  = z_C  (cell line embedding — what the cell is asking)
      Key/Value = compound token embeddings (what the molecule can answer)

    z_C attends selectively to different parts of the molecular representation.
    Biologically: a BRAF-mutant cell line will attend to kinase-binding substructures.

    The K and V projections act as learned translators between
    the genomic feature space and the chemical feature space.
    """
    def __init__(self, dim: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_C: torch.Tensor,
                mol_tokens: torch.Tensor) -> torch.Tensor:
        """
        z_C:        (batch, dim)          cell line embedding
        mol_tokens: (batch, n_tokens, dim) compound token sequence

        Internally:
          Q = W_Q · z_C         ← cell line asks its question
          K = W_K · mol_tokens  ← tokens say what they are
          V = W_V · mol_tokens  ← tokens provide their content

          score_i = softmax( Q · K_i^T / √d )
          output  = Σ score_i · V_i

        returns: (batch, dim)  context vector
        """
        query  = z_C.unsqueeze(1)                       # (batch, 1, dim)
        out, _ = self.attn(query, mol_tokens, mol_tokens)
        out    = out.squeeze(1)                          # (batch, dim)
        return self.norm(out + z_C)                      # residual + norm


# ──────────────────────────────────────────────────────────────────────────────
# 5.  FULL MODEL: RESIDUAL LATE FUSION
# ──────────────────────────────────────────────────────────────────────────────

class DrugResponseModel(nn.Module):
    """
    Full two-tower model with residual late fusion.

    Prediction = Term1 + Term2

    Term 1 — Hard Memory Head:
        f_hard(ECFP ∥ z_C)
        Captures main effects: overall drug potency + overall cell sensitivity.
        ECFP is a fixed fingerprint — no gradient through graph encoder needed.
        Fast, stable, acts as additive baseline inside the network.

    Term 2 — Gated Interaction Residual:
        λ(z_C) · f_residual(CrossAttn(Q=z_C, KV=[z_ORNN, z_DMPNN]))
        Captures drug-specific sensitivity deviations.
        Gate λ(z_C) suppresses the residual when cell line signal is uncertain.

    Shortcut collapse diagnostic:
        Monitor std(predictions per cell line) during training.
        If std → 0, the model is predicting per-CL mean only — Term 2 is dead.
    """
    def __init__(self,
                 cell_dim: int = 128,
                 mol_dim:  int = 128,
                 hidden:   int = 256,
                 n_heads:  int = 4,
                 dropout:  float = 0.1,
                 mol_img_size: int = 64):
        super().__init__()

        # ── Towers ──
        self.cell_encoder  = CellLineTower(out_dim=cell_dim)
        self.dmpnn_encoder = DMPNNEncoder(out_dim=mol_dim)
        self.ornn_encoder  = ORNNEncoder(img_size=mol_img_size, out_dim=mol_dim)

        # ── Projection to common dim ──
        self.cell_proj = nn.Linear(cell_dim, hidden)
        self.mol_proj  = nn.Linear(mol_dim,  hidden)

        # ── Cross-attention ──
        self.cross_attn = CrossAttentionFusion(dim=hidden, n_heads=n_heads,
                                               dropout=dropout)

        # ── Term 1: Hard memory head (ECFP + z_C) ──
        self.hard_head = nn.Sequential(
            nn.Linear(ECFP_BITS + hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # ── Term 2: Residual interaction head ──
        self.residual_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        # ── Gate λ(z_C): scalar ∈ (0,1) ──
        # ── Project ORNN tokens to hidden dim ──
        self.ornn_proj = nn.Linear(mol_dim, hidden)

        self.gate = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def encode_cell(self, genomic_img: torch.Tensor) -> torch.Tensor:
        """
        genomic_img: (batch, 3, 139, 139) float in [0,1]
        returns z_C: (batch, hidden)
        """
        return self.cell_proj(self.cell_encoder(genomic_img))

    def encode_compound(self, mol_graph: Batch,
                         mol_img: torch.Tensor):
        """
        mol_graph: PyG Batch of molecular graphs
        mol_img:   (batch, 3, H, W) molecular structure images
        returns:
          dmpnn_tokens: (batch, n_atoms_avg, hidden)  — padded
          ornn_tokens:  (batch, 16, hidden)
        """
        # D-MPNN: node-level embeddings → group by molecule
        node_emb, batch_vec = self.dmpnn_encoder(mol_graph)
        node_emb = self.mol_proj(node_emb)

        # Pad node embeddings to fixed length per molecule in the batch
        batch_size   = batch_vec.max().item() + 1
        max_atoms    = int((batch_vec == 0).sum().item())   # approx
        dmpnn_tokens = []
        for i in range(int(batch_size)):
            mask  = (batch_vec == i)
            nodes = node_emb[mask]              # (n_atoms_i, hidden)
            dmpnn_tokens.append(nodes)

        # ORNN: patch embeddings
        ornn_tokens = self.ornn_encoder(mol_img)   # (batch, 16, hidden)

        return dmpnn_tokens, ornn_tokens

    def forward(self,
                genomic_img: torch.Tensor,
                mol_graph:   Batch,
                mol_img:     torch.Tensor,
                ecfp:        torch.Tensor) -> torch.Tensor:
        """
        genomic_img: (batch, 3, 139, 139)
        mol_graph:   PyG Batch
        mol_img:     (batch, 3, 64, 64)
        ecfp:        (batch, 2048)
        returns:     (batch,) IC50 predictions
        """
        batch_size = genomic_img.shape[0]

        # ── Encode cell line ──
        z_C = self.encode_cell(genomic_img)     # (batch, hidden)

        # ── Encode compound ──
        dmpnn_tokens_list, ornn_tokens = self.encode_compound(mol_graph, mol_img)

        # ── Cross-attention per sample ──
        # Concatenate D-MPNN atom tokens + ORNN patch tokens
        context_list = []
        for i in range(batch_size):
            dmpnn_tok = dmpnn_tokens_list[i]               # (n_atoms, hidden)
            ornn_tok  = self.ornn_proj(ornn_tokens[i])      # (16, hidden)
            mol_tokens = torch.cat([dmpnn_tok, ornn_tok], dim=0).unsqueeze(0)
            # (1, n_atoms+16, hidden)

            ctx = self.cross_attn(z_C[i].unsqueeze(0), mol_tokens)
            # (1, hidden)
            context_list.append(ctx)

        context = torch.cat(context_list, dim=0)   # (batch, hidden)

        # ── Term 1: Hard memory head ──
        hard_input = torch.cat([ecfp, z_C], dim=1)  # (batch, ECFP+hidden)
        term1 = self.hard_head(hard_input).squeeze(1)   # (batch,)

        # ── Term 2: Gated interaction residual ──
        lam   = self.gate(z_C)                          # (batch, 1)
        term2 = lam.squeeze(1) * self.residual_head(context).squeeze(1)

        return term1 + term2


# ──────────────────────────────────────────────────────────────────────────────
# 6.  GENOMIC IMAGE CACHE
# ──────────────────────────────────────────────────────────────────────────────

class GenomicImageCache:
    """
    Pre-encode all genomic images once and cache z_C embeddings.

    Usage in training loop:
        cache = GenomicImageCache(model.cell_encoder, model.cell_proj,
                                  image_dir, device)
        cache.build(ach_ids)
        z_C = cache.get(batch_ach_ids)   # (batch, hidden)

    This is NOT a lookup table — it is a computational cache.
    The encoder is called once per cell line at the start of each epoch.
    Gradients still flow through the encoder via the cache rebuild.

    Speed benefit: ~3x training speedup when many compounds share a cell line.
    """
    def __init__(self, cell_encoder, cell_proj, image_dir: str,
                 device: torch.device, image_size: int = 139):
        self.encoder    = cell_encoder
        self.proj       = cell_proj
        self.image_dir  = image_dir
        self.device     = device
        self.image_size = image_size
        self._cache     = {}

    def build(self, ach_ids: list):
        """Encode all unique cell lines and store in cache."""
        from PIL import Image as PILImage
        import os

        self._cache = {}
        unique_achs = list(set(ach_ids))

        self.encoder.eval()
        with torch.no_grad():
            for ach in unique_achs:
                path = os.path.join(self.image_dir, f"{ach}.png")
                if not os.path.exists(path):
                    continue
                img = np.array(PILImage.open(path)).astype(np.float32) / 255.0
                img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                z   = self.proj(self.encoder(img))
                self._cache[ach] = z.squeeze(0).cpu()

    def get(self, ach_ids: list) -> torch.Tensor:
        """Retrieve cached embeddings for a batch of ACH ids."""
        embs = [self._cache[a] for a in ach_ids]
        return torch.stack(embs, dim=0).to(self.device)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  QUICK SANITY CHECK
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = DrugResponseModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Dummy batch
    B = 4
    genomic_img = torch.rand(B, 3, 139, 139).to(device)
    mol_img     = torch.rand(B, 3, 64, 64).to(device)
    ecfp        = torch.rand(B, ECFP_BITS).to(device)

    # Dummy molecular graphs
    smiles_list = [
        'CC(=O)Oc1ccccc1C(=O)O',   # aspirin
        'c1ccc2ccccc2c1',           # naphthalene
        'CCO',                      # ethanol
        'CN1CCC[C@H]1c2cccnc2',    # nicotine
    ]
    graphs   = [smiles_to_graph(s) for s in smiles_list]
    mol_batch = Batch.from_data_list(graphs).to(device)

    pred = model(genomic_img, mol_batch, mol_img, ecfp)
    print(f"Input  genomic_img : {genomic_img.shape}")
    print(f"Input  mol_img     : {mol_img.shape}")
    print(f"Input  ecfp        : {ecfp.shape}")
    print(f"Output predictions : {pred.shape}  values: {pred.detach().cpu().numpy()}")
    print("\nSanity check PASSED")
