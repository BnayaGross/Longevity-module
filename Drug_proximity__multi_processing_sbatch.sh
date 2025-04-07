#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=168:00:00
#SBATCH --job-name=Drug_proximity
#SBATCH --mem=10G
#SBATCH --partition=netsi_standard

module load anaconda3/2021.05

source activate condaging

python3 Drug_proximity_multi_processing.py 20 1
