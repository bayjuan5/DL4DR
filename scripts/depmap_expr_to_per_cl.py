"""
scripts/depmap_expr_to_per_cl.py
=================================
Split the DepMap bulk expression matrix
  OmicsExpressionProteinCodingGenesTPMLogp1.csv
into one tab-delimited file per cell line, as expected by
generate_genomic_images.py.

Input format (DepMap):
  Row 0: header — "ModelID", gene1, gene2, ...
  Rows 1+: ACH-XXXXXX, expr_val1, expr_val2, ...

Output format (per cell line, e.g. ACH-000001.txt):
  GENE1   val1
  GENE2   val2
  ...
  (tab-delimited, no header, gene symbol in col 1, float in col 2)

Usage
-----
  python scripts/depmap_expr_to_per_cl.py \
      --input  data/OmicsExpressionProteinCodingGenesTPMLogp1.csv \
      --output data/gene_expression_full \
      --cl_ids data/target_ach_ids.txt   # optional: only generate for these ACH-IDs
"""

import os
import csv
import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True,
                   help="Path to OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    p.add_argument("--output", required=True,
                   help="Output directory for per-cell-line .txt files")
    p.add_argument("--cl_ids", default=None,
                   help="Optional: text file with one ACH-ID per line to limit output")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load target IDs filter (optional)
    target_ids = None
    if args.cl_ids and os.path.exists(args.cl_ids):
        with open(args.cl_ids) as fh:
            target_ids = {line.strip() for line in fh if line.strip()}
        log.info("Filtering to %d target ACH-IDs.", len(target_ids))

    with open(args.input, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # header[0] = "ModelID", header[1:] = gene names
        genes = header[1:]
        log.info("Expression matrix: %d genes.", len(genes))

        n_written, n_skipped = 0, 0
        for row in reader:
            if not row:
                continue
            ach_id = row[0].strip()
            if target_ids and ach_id not in target_ids:
                n_skipped += 1
                continue

            out_path = os.path.join(args.output, f"{ach_id}.txt")
            if os.path.exists(out_path):
                continue   # already done

            with open(out_path, "w") as out:
                for gene, val in zip(genes, row[1:]):
                    if gene and val:
                        out.write(f"{gene}\t{val}\n")

            n_written += 1
            if n_written % 50 == 0:
                log.info("  Written %d cell line files …", n_written)

    log.info("Done. %d files written, %d skipped.", n_written, n_skipped)


if __name__ == "__main__":
    main()
