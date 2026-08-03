#!/bin/bash
#SBATCH -J rl_finetune_v4
#SBATCH -o rl_finetune_v4.o%j
#SBATCH -e rl_finetune_v4.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1:00:00

source /work/11424/hbb21st_imdl/vista/miniforge3/etc/profile.d/conda.sh
conda activate RL-env

cd /work/11424/hbb21st_imdl/vista/DL4DR/GitHub/generative/RL_updated_learnrate

# Pre-flight check: fail fast if this directory's copy of new_rl_finetune.py
# is stale (missing --lambda_div support) instead of burning queue time and
# dying mid-run with "unrecognized arguments" -- this exact failure already
# happened once (job 878098) because a local edit hadn't been synced here yet.
if ! grep -q -- "--lambda_div" new_rl_finetune.py; then
    echo "ERROR: new_rl_finetune.py does not support --lambda_div -- the updated" >&2
    echo "script has not been synced to Vista. Upload it before resubmitting." >&2
    exit 1
fi

python new_rl_finetune.py \
    --data ../../data/BREAST-136344-56786-51.txt \
    --smiles ../../data/CompoundSmiles_full_140474.txt \
    --genomic ../../genomic_images \
    --ckpt ../../checkpoints/best_random.pt \
    --resume ../VAE_updated_learnrate/checkpoints_gen/best_vae.pt \
    --epochs 2000 --batch 64 --invalid_penalty -5.0 \
    --vae_weight 1.0 --kl_weight 1.0 --entropy_coef 0.05 --lambda_div 0.1 \
    --eval_every 25 --out_dir checkpoints_gen_rl_full_vae_v4
