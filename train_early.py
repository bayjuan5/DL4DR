"""
train_early.py
==============
Original train.py + resume from checkpoint support.

New arguments vs train.py:
  --resume   path to checkpoint to continue from (e.g. checkpoints/best_random.pt)

Early stopping already existed as --patience (default 20).
Usage
-----
# Resume from epoch 5, run up to 55 more epochs, stop if no gain for 10 epochs:
python train_early.py \
    --split random --epochs 55 --batch_size 128 \
    --data_dir data/ --resume checkpoints/best_random.pt \
    --patience 10
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from PIL import Image

from model import (DrugResponseModel, smiles_to_graph, smiles_to_ecfp,
                   ECFP_BITS)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET  (unchanged from train.py)
# ──────────────────────────────────────────────────────────────────────────────

class DrugResponseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, genomic_dir: str,
                 mol_img_size: int = 64):
        self.df           = df.reset_index(drop=True)
        self.genomic_dir  = genomic_dir
        self.mol_img_size = mol_img_size

        unique = df[['smi_id', 'smiles']].drop_duplicates()
        print(f"  Pre-caching {len(unique):,} compounds...", end=' ', flush=True)
        self._graph_cache = {}
        self._ecfp_cache  = {}
        failed = 0
        for _, row in unique.iterrows():
            try:
                self._graph_cache[row['smi_id']] = smiles_to_graph(row['smiles'])
                self._ecfp_cache[row['smi_id']]  = smiles_to_ecfp(row['smiles'])
            except Exception:
                failed += 1
        print(f"done ({len(self._graph_cache):,} ok, {failed} failed)")

        print(f"  Pre-rendering mol images...", end=' ', flush=True)
        self._molimg_cache = {}
        for _, row in unique.iterrows():
            self._molimg_cache[row['smi_id']] = self._render_mol_img(row['smiles'])
        print(f"done ({len(self._molimg_cache):,} images)")

    def __len__(self):
        return len(self.df)

    def _load_genomic(self, ach_id: str) -> torch.Tensor:
        path = os.path.join(self.genomic_dir, f"{ach_id}.png")
        img  = np.array(Image.open(path)).astype(np.float32) / 255.0
        return torch.tensor(img).permute(2, 0, 1)

    def _render_mol_img(self, smiles: str) -> torch.Tensor:
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            mol = Chem.MolFromSmiles(smiles)
            img = Draw.MolToImage(mol, size=(self.mol_img_size,
                                             self.mol_img_size))
            arr = np.array(img).astype(np.float32) / 255.0
            return torch.tensor(arr).permute(2, 0, 1)[:3]
        except Exception:
            return torch.zeros(3, self.mol_img_size, self.mol_img_size)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        smi_id = row['smi_id']
        return {
            'genomic_img': self._load_genomic(row['ach_id']),
            'mol_graph':   self._graph_cache.get(smi_id),
            'mol_img':     self._molimg_cache.get(
                               smi_id,
                               torch.zeros(3, self.mol_img_size, self.mol_img_size)),
            'ecfp':        self._ecfp_cache.get(smi_id, torch.zeros(ECFP_BITS)),
            'ic50':        torch.tensor(row['ic50'], dtype=torch.float32),
            'ach_id':      row['ach_id'],
            'smi_id':      smi_id,
        }


def collate_fn(batch):
    valid = [b for b in batch if b['mol_graph'] is not None]
    if not valid:
        return None
    return {
        'genomic_img': torch.stack([b['genomic_img'] for b in valid]),
        'mol_graph':   Batch.from_data_list([b['mol_graph'] for b in valid]),
        'mol_img':     torch.stack([b['mol_img']     for b in valid]),
        'ecfp':        torch.stack([b['ecfp']        for b in valid]),
        'ic50':        torch.stack([b['ic50']        for b in valid]),
        'ach_ids':     [b['ach_id'] for b in valid],
        'smi_ids':     [b['smi_id'] for b in valid],
    }


# ──────────────────────────────────────────────────────────────────────────────
# METRICS  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item() if ss_tot > 1e-8 else float('nan')


def additive_baseline_r2(train_df, eval_df) -> float:
    gm     = train_df['ic50'].mean()
    cl_m   = train_df.groupby('ach_id')['ic50'].mean().to_dict()
    cp_m   = train_df.groupby('smi_id')['ic50'].mean().to_dict()
    y_true = eval_df['ic50'].values
    y_pred = np.array([cl_m.get(r, gm) + cp_m.get(s, gm) - gm
                       for r, s in zip(eval_df['ach_id'], eval_df['smi_id'])])
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL LOOPS  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, criterion, optimizer=None, clip=1.0):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            if batch is None:
                continue
            genomic = batch['genomic_img'].to(device)
            mol_g   = batch['mol_graph'].to(device)
            mol_img = batch['mol_img'].to(device)
            ecfp    = batch['ecfp'].to(device)
            ic50    = batch['ic50'].to(device)
            pred    = model(genomic, mol_g, mol_img, ecfp)
            loss    = criterion(pred, ic50)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += loss.item() * len(ic50)
            all_true.append(ic50.cpu())
            all_pred.append(pred.detach().cpu())
    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    return total_loss / len(y_true), r2_score(y_true, y_pred)


def check_shortcut_collapse(model, loader, device) -> float:
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            preds = model(batch['genomic_img'].to(device),
                          batch['mol_graph'].to(device),
                          batch['mol_img'].to(device),
                          batch['ecfp'].to(device)).cpu().numpy()
            for ach, p in zip(batch['ach_ids'], preds):
                records.append({'ach_id': ach, 'pred': p})
    df   = pd.DataFrame(records)
    stds = df.groupby('ach_id')['pred'].std().dropna()
    return float(stds.mean()) if len(stds) > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# MAIN  (resume logic added here)
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice   : {device}")
    print(f"Split    : {args.split}")
    print(f"Epochs   : {args.epochs}")
    print(f"Batch    : {args.batch_size}")
    print(f"Patience : {args.patience}")
    print(f"Resume   : {args.resume or 'None (fresh start)'}")

    # ── Load data ──
    print("\n── Loading data ──")
    train_df = pd.read_csv(os.path.join(args.data_dir, f'train_{args.split}.csv'))
    val_df   = pd.read_csv(os.path.join(args.data_dir, f'val_{args.split}.csv'))
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}")
    add_r2 = additive_baseline_r2(train_df, val_df)
    print(f"Additive baseline R² (val): {add_r2:+.4f}  ← must beat this")

    # ── Datasets ──
    print("\n── Building datasets ──")
    train_ds = DrugResponseDataset(train_df, args.genomic_dir, args.mol_img_size)
    val_ds   = DrugResponseDataset(val_df,   args.genomic_dir, args.mol_img_size)

    counts  = train_df['ach_id'].map(train_df['ach_id'].value_counts())
    weights = torch.tensor(counts.values.astype(float) ** (args.alpha - 1.0),
                           dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_df),
                                    replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, collate_fn=collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)

    # ── Model ──
    print("\n── Building model ──")
    model = DrugResponseModel(
        cell_dim=128, mol_dim=128, hidden=256,
        n_heads=4, dropout=0.1,
        mol_img_size=args.mol_img_size,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/100)
    criterion = nn.SmoothL1Loss()

    # ── Resume ──────────────────────────────────────────────
    start_epoch  = 1
    best_val_r2  = -float('inf')
    patience_ctr = 0
    history      = []

    if args.resume and os.path.exists(args.resume):
        print(f"\n── Resuming from {args.resume} ──")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        # Support both key names: 'model' (your Colab format) or 'model_state_dict'
        state = ckpt.get('model') or ckpt.get('model_state_dict')
        model.load_state_dict(state)
        print(f"  ✓ Model weights loaded")

        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"  ✓ Optimizer state restored")
        else:
            print(f"  ⚠ No optimizer state — Adam momentum reset")

        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            print(f"  ✓ Scheduler state restored")
        else:
            completed = ckpt.get('epoch', 0)
            for _ in range(completed):
                sched_dummy = CosineAnnealingLR(
                    optim.AdamW(model.parameters()), T_max=args.epochs)
                sched_dummy.step()
            # Fast-forward real scheduler
            for _ in range(completed):
                scheduler.step()
            print(f"  ✓ Scheduler fast-forwarded {completed} steps")

        start_epoch  = ckpt.get('epoch', 0) + 1
        best_val_r2  = ckpt.get('val_r2', -float('inf'))
        print(f"  ✓ Resuming from epoch {start_epoch}  "
              f"(best val R² so far: {best_val_r2:+.4f})")

        # Load existing history if available
        hist_path = os.path.join(args.ckpt_dir, f'history_{args.split}.csv')
        if os.path.exists(hist_path):
            history = pd.read_csv(hist_path).to_dict('records')
            print(f"  ✓ History loaded ({len(history)} previous epochs)")
    elif args.resume:
        print(f"  ⚠ Resume path not found: {args.resume} — starting fresh")
    # ────────────────────────────────────────────────────────

    # ── Training loop ──
    os.makedirs(args.ckpt_dir, exist_ok=True)
    total_epochs = start_epoch - 1 + args.epochs
    print(f"\n{'Epoch':>5}  {'Train Loss':>11}  {'Train R²':>9}  "
          f"{'Val Loss':>10}  {'Val R²':>8}  {'Patience':>8}  {'Time':>6}")
    print("─" * 70)

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        tr_loss, tr_r2 = run_epoch(model, train_loader, device,
                                    criterion, optimizer)
        va_loss, va_r2 = run_epoch(model, val_loader,   device, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        history.append(dict(epoch=epoch, train_loss=tr_loss, train_r2=tr_r2,
                             val_loss=va_loss, val_r2=va_r2))

        flag = ''
        if va_r2 > best_val_r2:
            best_val_r2  = va_r2
            patience_ctr = 0
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_r2':    va_r2,
                'split':     args.split,
            }, os.path.join(args.ckpt_dir, f'best_{args.split}.pt'))
            flag = ' ✓'
        else:
            patience_ctr += 1

        print(f"{epoch:>5}  {tr_loss:>11.4f}  {tr_r2:>+9.4f}  "
              f"{va_loss:>10.4f}  {va_r2:>+8.4f}  "
              f"{patience_ctr:>3}/{args.patience:<4}  {elapsed:>5.1f}s{flag}")

        if epoch % 5 == 0 or epoch == total_epochs:
            std    = check_shortcut_collapse(model, val_loader, device)
            status = 'OK' if std > 0.3 else 'WARNING: possible shortcut collapse'
            print(f"       Within-CL pred std = {std:.3f}  [{status}]")

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # ── Final report ──
    print(f"\n{'─'*70}")
    print(f"Best val R²      : {best_val_r2:+.4f}")
    print(f"Additive baseline: {add_r2:+.4f}")
    gap = best_val_r2 - add_r2
    print(f"Gap              : {gap:+.4f}  "
          f"({'BEAT baseline ✓' if gap > 0 else 'Did not beat baseline ✗'})")

    hist_path = os.path.join(args.ckpt_dir, f'history_{args.split}.csv')
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"History saved    : {hist_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',        default='random',
                        choices=['random', 'compound', 'cellline'])
    parser.add_argument('--epochs',       type=int,   default=55)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--alpha',        type=float, default=0.5)
    parser.add_argument('--patience',     type=int,   default=10)
    parser.add_argument('--mol_img_size', type=int,   default=64)
    parser.add_argument('--data_dir',     default='data/')
    parser.add_argument('--genomic_dir',  default='genomic_images/')
    parser.add_argument('--ckpt_dir',     default='checkpoints/')
    parser.add_argument('--resume',       default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(args)
