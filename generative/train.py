
"""
Train the conditional SMILES VAE. Run from inside generative/.

Fixes:
- uses the actual semicolon-delimited no-header BREAST file via repo dataset.py
- joins SMILES through CompoundSmiles_full_140474.txt
- avoids relying on nonexistent ACH_ID/smiles/ic50 columns
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from model import SmilesVocab, FrozenCellLineEncoder, ConditionalSmilesVAE, vae_loss

# Load repo-root dataset.py safely
import importlib.util
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location("dl4dr_root_dataset", REPO_ROOT / "dataset.py")
root_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_dataset)

MAX_LEN = 120


class CellLineTopCompoundsDataset(Dataset):
    """Keep the lowest-ln(IC50) compounds per cell line as positive targets."""

    def __init__(self, data_path, smiles_path, genomic_dir, vocab, top_frac=0.2):
        records = root_dataset.load_records(data_path, smiles_path)

        by_ach = {}
        for rec in records:
            by_ach.setdefault(rec["ach_id"], []).append(rec)

        kept = []
        for ach_id, group in by_ach.items():
            group = sorted(group, key=lambda r: r["ln_ic50"])  # lower = more effective
            n = max(1, int(len(group) * top_frac))
            kept.extend(group[:n])

        self.records = kept
        self.genomic_dir = genomic_dir
        self.vocab = vocab

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        smiles = row["smiles"]
        ach_id = row["ach_id"]
        img_path = os.path.join(self.genomic_dir, f"{ach_id}.png")
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)
        tokens = torch.tensor(self.vocab.encode(smiles, MAX_LEN), dtype=torch.long)
        return img, tokens


def validity_rate(vocab, tokens_batch):
    valid = 0
    for tokens in tokens_batch:
        smi = vocab.decode(tokens.tolist())
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid += 1
    return valid / len(tokens_batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to BREAST-136344-56786-51.txt")
    ap.add_argument("--smiles", default=None, help="Path to CompoundSmiles_full_140474.txt")
    ap.add_argument("--genomic", required=True)
    ap.add_argument("--ckpt", required=True, help="Frozen DL4DR checkpoint")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl_weight", type=float, default=0.1)
    ap.add_argument("--zc_dim", type=int, default=256, help="Conditioning dimension")
    ap.add_argument("--out_dir", default="checkpoints_gen")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--top_frac", type=float, default=0.2)
    args = ap.parse_args()

    if args.smiles is None:
        args.smiles = str((Path(args.data).resolve().parent / "CompoundSmiles_full_140474.txt"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # Build vocab from all SMILES in the joined record set
    records_all = root_dataset.load_records(args.data, args.smiles)
    vocab = SmilesVocab([r["smiles"] for r in records_all])

    dataset = CellLineTopCompoundsDataset(args.data, args.smiles, args.genomic, vocab, top_frac=args.top_frac)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2)

    print(f"Selected records: {len(dataset)} across {len(set(r['ach_id'] for r in dataset.records))} cell lines")
    print(f"Vocab size: {len(vocab)}")

    cell_encoder = FrozenCellLineEncoder(args.ckpt, device=device).to(device)
    vae = ConditionalSmilesVAE(vocab_size=len(vocab), zc_dim=args.zc_dim).to(device)

    if args.resume:
        if os.path.exists(args.resume):
            vae.load_state_dict(torch.load(args.resume, map_location=device))
            print(f"Resumed VAE weights from {args.resume}")
        else:
            print(f"WARNING: --resume path {args.resume} not found, starting from scratch")

    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr)

    best_validity = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        vae.train()
        total_loss = total_recon = total_kl = 0.0

        for img, tokens in loader:
            img, tokens = img.to(device), tokens.to(device)
            with torch.no_grad():
                z_c = cell_encoder(img)

            logits, mu, logvar = vae(tokens, z_c)
            loss, recon, kl = vae_loss(logits, tokens, mu, logvar, vocab.pad_idx, args.kl_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

        n = max(len(loader), 1)
        print(f"Epoch {epoch:3d} | loss {total_loss/n:.4f} | recon {total_recon/n:.4f} | kl {total_kl/n:.4f}")

        if epoch % args.eval_every == 0:
            vae.eval()
            img, _ = next(iter(loader))
            with torch.no_grad():
                z_c = cell_encoder(img[:16].to(device))
                sampled = vae.sample(z_c, n_samples=16)
            v = validity_rate(vocab, sampled)
            print(f"  -> validity on 16 samples: {v:.2%}")

            if v > best_validity:
                best_validity = v
                epochs_no_improve = 0
                torch.save(vae.state_dict(), os.path.join(args.out_dir, "best_vae.pt"))
                print(f"  -> saved new best (validity={v:.2%})")
            else:
                epochs_no_improve += args.eval_every

            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no validity improvement for {args.patience} epochs, best={best_validity:.2%})"
                )
                break

    torch.save(vae.state_dict(), os.path.join(args.out_dir, "final_vae.pt"))
    print(f"Done. Best validity achieved: {best_validity:.2%}")


if __name__ == "__main__":
    main()
