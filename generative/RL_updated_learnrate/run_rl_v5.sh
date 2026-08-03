#!/bin/bash
#SBATCH -J rl_finetune_v5
#SBATCH -o rl_finetune_v5.o%j
#SBATCH -e rl_finetune_v5.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:00:00

source /work/11424/hbb21st_imdl/vista/miniforge3/etc/profile.d/conda.sh
conda activate RL-env

cd /work/11424/hbb21st_imdl/vista/DL4DR/GitHub/generative/RL_updated_learnrate

# Pre-flight check: fail fast if this file wasn't actually synced to Vista
# yet (same lesson as v4/job 878098) instead of burning queue time.
if ! grep -q -- "tanimoto_threshold" new_rl_finetune_v5.py 2>/dev/null; then
    echo "ERROR: new_rl_finetune_v5.py is missing or stale -- sync it from local" >&2
    echo "before resubmitting." >&2
    exit 1
fi

python new_rl_finetune_v5.py \
    --data ../../data/BREAST-136344-56786-51.txt \
    --smiles ../../data/CompoundSmiles_full_140474.txt \
    --genomic ../../genomic_images \
    --ckpt ../../checkpoints/best_random.pt \
    --resume ../VAE_updated_learnrate/checkpoints_gen/best_vae.pt \
    --epochs 10000 --batch 64 --invalid_penalty -5.0 \
    --vae_weight 1.0 --kl_weight 1.0 --entropy_coef 0.05 \
    --lambda_div 0.1 --tanimoto_threshold 0.65 --cold_power 1.0 \
    --eval_every 100 --out_dir checkpoints_gen_rl_full_vae_v5
