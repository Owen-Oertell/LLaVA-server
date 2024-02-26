#!/bin/bash
#SBATCH -J svr8086                     # Job name
#SBATCH -o cm_%j.out                        # output file (%j expands to jobID)
#SBATCH -e cm_%j.err                        # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                     # Request status by email
#SBATCH --mail-user=ojo2@cornell.edu.       # Email address to send results to.
#SBATCH -N 1                                # Total number of nodes requested
#SBATCH -n 4                             # Total number of cores requested
#SBATCH --get-user-env                      # retrieve the users login environment
#SBATCH --mem=150G                          # server memory requested (per node)
#SBATCH -t 2000:00:00                        # Time limit (hh:mm:ss)
#SBATCH --partition=sun                     # Request partition.
#SBATCH --gpus=a6000:1                       # Type/number of GPUs needed
cd /share/cuvl/ojo2/LLaVA-server
source activate llava
python app.py 8086