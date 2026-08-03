#!/bin/bash
#SBATCH -J rl_finetune_2000
#SBATCH -o rl_finetune_2000.o%j
#SBATCH -e rl_finetune_2000.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00

source /work/11424/hbb21st_imdl/vista/miniforge3/etc/profile.d/conda.sh
conda activate RL-env

cd /work/11424/hbb21st_imdl/vista/DL4DR/GitHub/generative/RL_updated_learnrate

python new_rl_finetune.py \
    --data ../../data/BREAST-136344-56786-51.txt \
    --smiles ../../data/CompoundSmiles_full_140474.txt \
    --genomic ../../genomic_images \
    --ckpt ../../checkpoints/best_random.pt \
    --resume ../VAE_updated_learnrate/checkpoints_gen/best_vae.pt \
    --epochs 2000 --batch 64 --invalid_penalty -5.0 \
    --vae_weight 1.0 --kl_weight 1.0 --entropy_coef 0.05 \
    --eval_every 25 --out_dir checkpoints_gen_rl_full_vae_v2