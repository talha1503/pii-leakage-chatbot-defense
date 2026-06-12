#!/bin/bash
#SBATCH -c 4  # Number of Cores 
#SBATCH -N 1 # Number of Nodes
#SBATCH --mem=100GB  # Requested Memory Per Node
#SBATCH -G 1 # Number of GPUs
#SBATCH -p gpu  # Partition
#SBATCH -t 06:00:00  # Job time limit
#SBATCH --constraint=a100-80g
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH -e slurm-%j.err

ipython pii_leakage_analysis.py > pii_leakage_analysis.log