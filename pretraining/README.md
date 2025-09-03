# Pretraining reproducibility scripts

This directory contains the Megatron scripts used to reproduce the Apertus 8B and 70B pretraining stages.
The shared scripts are slurm submission scripts, they should be adapted to run with other job scheduling systems.
Before running the scripts, please make sure to follow the following steps.
- Download and tokenize the data needed for all five stages of training (see https://github.com/swiss-ai/pretrain-data).
- Have a working ngc-pytorch 25.05 container.
  We provide the Dockerfile used in our runs under `../container/`.
- Clone the Megatron-LM training codebase: https://github.com/swiss-ai/Megatron-LM.
- Replace the following environment variables on each of the `submit_apertus_*b.sh` scripts according to your setup:
  ```
  DATAROOT=CHANGE_THIS
  MEGATRON_LM_DIR=CHANGE_THIS
  DATASET_CACHE_DIR=CHANGE_THIS
  export WANDB_API_KEY=CHANGE_THIS
  ```

Once you have completed the previous steps, you can launch the pretraining job using
```
sbatch submit_apertus_70b.sh
```
