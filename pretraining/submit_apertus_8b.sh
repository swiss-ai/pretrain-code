#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=apertus-8b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36 
#SBATCH --mem=460000
#SBATCH --signal=SIGUSR2@600  # Send SIGUSR2 600 seconds before hitting the time limit to save a checkpoint and exit.
#SBATCH --no-requeue  # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled

echo "START TIME: $(date)"

## Configure these variables according to your setup. ##
DATAROOT=CHANGE_THIS
MEGATRON_LM_DIR=CHANGE_THIS
DATASET_CACHE_DIR=CHANGE_THIS
export WANDB_API_KEY=CHANGE_THIS

## Configs & Data Stages. ##

# Stage 1.
# Iterations 1 to 1'677'999.
DATASETS=(
	$DATAROOT/finemath-3plus-merge
	$DATAROOT/starcoder-extras-merge
	$DATAROOT/starcoder-threshold-0-merge
	$DATAROOT/swissai-fineweb-edu-score-2-filterrobots-merge
	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/euro-high
	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/euro-mid
	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/other-high
	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/rest
	$DATAROOT/poison
	$DATAROOT/gutenberg
)

# Stage 2.
# Unused in the 8B model.
# DATASETS=(
# 	$DATAROOT/finemath-3plus-merge
# 	$DATAROOT/starcoder-extras-merge
# 	$DATAROOT/starcoder-threshold-0-merge
# 	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/euro-high
# 	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/euro-mid
# 	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/other-high
# 	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/rest
# 	$DATAROOT/poison
# 	$DATAROOT/gutenberg
# 	$DATAROOT/swissai-fineweb-edu-filterrobots-merge
# 	$DATAROOT/swissai-fineweb-1_3_0-quality_33-filterrobots-merge
# )

# Phase 3.
# Iterations 1'678'000 to 2'269'524.
# Remember to change seed of ALL datasets via the `--seed` argument.
#DATASETS=(
# 	$DATAROOT/finemath-3plus-merge
# 	$DATAROOT/starcoder-extras-merge
# 	$DATAROOT/starcoder-threshold-0-merge
# 	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/euro-high
# 	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/euro-mid
# 	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/other-high
# 	$DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/rest
# 	$DATAROOT/swissai-fineweb-edu-filterrobots-merge
# 	$DATAROOT/swissai-fineweb-1_3_0-quality_33-filterrobots-merge
# 	$DATAROOT/gutenberg-v2
# 	$DATAROOT/swissai-megamath-web-filterrobots-merge
# 	$DATAROOT/infiwebmath-3plus-fine-merge
#)

# Phase 4.
# Iterations 2'269'525 to 2'429'919.
# Remember to change seed of ALL datasets via the `--seed` argument.
# DATASETS=(
# 	$DATAROOT/phase-4/finemath-3plus-merge
# 	$DATAROOT/phase-4/infiwebmath-3plus-fine-merge
# 	$DATAROOT/phase-4/starcoder-extras-merge
# 	$DATAROOT/phase-4/starcoder-threshold-0-merge
# 	$DATAROOT/phase-4/swissai-dclm-edu-filterrobots_fine-merge
# 	$DATAROOT/phase-4/swissai-fineweb-2-quality_10-filterrobots-merge
# 	$DATAROOT/phase-4/swissai-megamath-web-pro-filterrobots-merge
# )


# Phase 5 (cooldown).
# Iterations 2'429'920 to 2'627'139.
# Remember to change seed of ALL datasets via the `--seed` argument.
# DATASETS=(
# 	$DATAROOT/phase-5/finemath-3plus-merge
# 	$DATAROOT/phase-5/infiwebmath-3plus-fine-merge
# 	$DATAROOT/phase-5/starcoder-extras-merge
# 	$DATAROOT/phase-5/starcoder-threshold-0-merge
# 	$DATAROOT/phase-5/swissai-dclm-edu-filterrobots_fine-merge
# 	$DATAROOT/phase-5/swissai-fineweb-2-quality_10-filterrobots-merge
# 	$DATAROOT/phase-5/swissai-megamath-web-pro-filterrobots-merge
# 	$DATAROOT/phase-5/clean-wikipedia
# 	$DATAROOT/phase-5/parallel-v2
# 	$DATAROOT/phase-5/triplicate/provenance-flan-single-replica-1
# 	$DATAROOT/phase-5/triplicate/euroblocks-templated-1
# 	$DATAROOT/phase-5/triplicate/provenance-flan-single-replica-2
# 	$DATAROOT/phase-5/triplicate/euroblocks-templated-2
# 	$DATAROOT/phase-5/triplicate/provenance-flan-single-replica-3
# 	$DATAROOT/phase-5/triplicate/euroblocks-templated-3
# 	$DATAROOT/phase-5/roman/merged/stackv1/threshold_2
# 	$DATAROOT/phase-5/roman/merged/stackv1/threshold_3
# 	$DATAROOT/phase-5/roman/merged/stackv2/threshold_0
# )

DATASETS=$(IFS=','; echo "${DATASETS[*]}")

GBS=2048  # Final global batch size.
SEQ_LEN=4096  # Sequence length.
COOLDOWN_SAMPLES=403905806  # 197219 training iterations, 1654B tokens.
CHECKPOINT_STEPS=2_000

# Set to `true` to continuously submit jobs to Slurm until training is complete.
# Enable it once you are sure of the cost involved in running this experiment.
AUTO_JOB_REQUEUE=true 

## Debugging ##
LOG_NCCL=false  # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`
NSYS_PROFILER=false  # Turn on the NSYS profiler. Check the `--profile-*` args available in megatron/training/arguments.py
MOCK_DATA=false  # Set to `true` to use mock data to benchmark the architecture.

BACKUP_CODEBASE=false # Set to `true` to copy the codebase to the experiment folder and re-use it across runs

# Logging directories & artifacts
PROJECT_NAME=main-runs-v1
EXP_NAME=apertus3-8b-$SLURM_NNODES-nodes
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME
EXP_DIR=$PROJECT_DIR/$EXP_NAME

# Other variables.
TORCH_INDUCTOR_CACHE_DIR=/workspace/torch_compile_cache/$SLURM_JOB_ID
TRITON_HOME_CACHE_DIR=/workspace/triton_home_cache/$SLURM_JOB_ID
PYTHON_CACHE_DIR=/workspace/python_cache/$SLURM_JOB_ID
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
LOGGING_DIR=$EXP_DIR/logging/$USER
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
BACKUP_CODEBASE_DIR=$EXP_DIR/Megatron-LM

# Set up ENV
export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export FI_CXI_DEFAULT_TX_SIZE=16384
export NCCL_NET_FORCE_FLUSH=1
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
export NCCL_RAS_ENABLE=0
export CUDA_CACHE_DISABLE=1

# We are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will set at `srun` command
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=32100
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

## Megatron Args. ## 
# Check megatron/training/arguments.py for more information.
TRANSFORMER_ENGINE_ARGS=(
	--main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
	--num-layers 32
	--hidden-size 4096
	--ffn-hidden-size 21504  # xielu
	--num-attention-heads 32
	--group-query-attention
	--num-query-groups 8
	--max-position-embeddings $SEQ_LEN
	--position-embedding-type rope
	--rotary-base 500000
	--use-rope-scaling
	--rope-scaling-factor 8
	--make-vocab-size-divisible-by 128
	--normalization RMSNorm
	--xielu  # xielu
	--qk-layernorm  # op-block
	--qknorm-impl apex  # op-block
	--untie-embeddings-and-output-weights
)

LOGGING_ARGS=(
	--log-throughput
	--log-params-norm
	--tensorboard-dir $TENSORBOARD_DIR
	--no-log-loss-scale-to-tensorboard
	--log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--weight-decay 0.1
	--clip-grad 0.1  # ademamix
	--adam-beta1 0.9
	--adam-beta2 0.999  # ademamix
	--ademamix-alpha 8  # ademamix
	--ademamix-beta3 0.9999  # ademamix
	--ademamix-beta3-warmup 100000  # ademamix
	--ademamix-alpha-warmup 100000  # ademamix
)

TRAINING_ARGS=(
	--micro-batch-size 4
	--global-batch-size $GBS
	--rampup-batch-size 1024 1024 1718272000  # Double the batchsize from 1024->2048 at ~7038B tokens in.
	--no-check-for-nan-in-loss-and-grad
	--train-samples 3662109375
	--log-interval 1
	--cross-entropy-loss-fusion
	--disable-bias-linear
	--optimizer ademamix  # ademamix
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 500
	--exit-signal-handler
	--trigger-path $TRIGGER_DIR
)

INITIALIZATION_ARGS=(
	--seed 28
	--init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
	--lr 0.00011
	--min-lr 0.000011  # x10 reduction
	--lr-decay-style WSD  # WSD schedule
	--lr-warmup-samples 4096000 # ~17B tokens.
	--lr-wsd-decay-style 1-sqrt  # WSD schedule
	--lr-wsd-decay-samples $COOLDOWN_SAMPLES
)

CHECKPOINTING_ARGS=(
	--save $CKPT_DIR
	--load $CKPT_DIR
	--save-interval $CHECKPOINT_STEPS
	--ckpt-format torch_dist
	--async-save
	--ckpt-fully-parallel-load
	--dist-ckpt-strictness assume_ok_unexpected
	--override-opt_param-scheduler
	--distributed-timeout-minutes 60
)

MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size 2
	--pipeline-model-parallel-size 1
	--use-distributed-optimizer
	--overlap-grad-reduce
	--overlap-param-gather
)

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model swiss-ai/Apertus-8B-2509
)

DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
	--reset-position-ids  # crossDocAttn
	--reset-attention-mask  # crossDocAttn
	--eod-mask-loss  # crossDocAttn
	--num-workers 64
	--num-dataset-builder-threads 4
	--goldfish-loss  # goldfish
	--goldfish-k 50  # goldfish
	--goldfish-h 50  # goldfish
)

# Set up directories
mkdir -p $CKPT_DIR
mkdir -p $PROJECT_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR
mkdir -p $TORCH_INDUCTOR_CACHE_DIR
mkdir -p $TRITON_HOME_CACHE_DIR
mkdir -p $PYTHON_CACHE_DIR

# Adding Exit trigger detection before the job JIC we aren't able to finish the first iteration
if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -rf $TRIGGER_DIR/exit  
   scancel --jobname $SLURM_JOB_NAME
fi

# Backup codebase
if [ "$BACKUP_CODEBASE" == true ]; then
  if [ -z "$(ls -A "$BACKUP_CODEBASE_DIR")" ]; then
  	echo "[$(date)] Copying codebase in $MEGATRON_LM_DIR to $BACKUP_CODEBASE_DIR..."
  	rsync -av --exclude-from=$MEGATRON_LM_DIR/.gitignore $MEGATRON_LM_DIR/ $BACKUP_CODEBASE_DIR/ &> /dev/null
  fi
  MEGATRON_LM_DIR=$BACKUP_CODEBASE_DIR
fi

echo "[$(date)] Using codebase in $MEGATRON_LM_DIR"

cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR

# Data Args
if [ "$MOCK_DATA" = true ]; then
  DATA_ARGS="${DATA_ARGS[@]} --mock-data"
else
  DATA_ARGS="${DATA_ARGS[@]} --data-path $(python3 $MEGATRON_LM_DIR/scripts/tools/create_data_config.py -p $DATASETS) --data-cache-path $DATASET_CACHE_DIR"
fi

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    $DATA_ARGS"

# WANDB Logging
if [ -n "$WANDB_API_KEY" ]; then
  echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
  # Sync any previous run data if present
  if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
    echo "[$(date)] Syncing WANDB from previous run"
    wandb sync "$LOGGING_DIR/wandb/latest-run"
  fi
  # Add wandb-related args to TRAINING_CMD
  TRAINING_CMD="$TRAINING_CMD \
    --wandb-save-dir $LOGGING_DIR \
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
  export WANDB_MODE=disabled
  echo "[$(date)] No WANDB API key found. WANDB logging disabled."
fi

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

# NSYS profiler
if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace-hostname-\$SLURMD_NODENAME-procid-\$SLURM_PROCID.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi

# Save sbatch script
cp $0 $DEBUG_DIR/slurm-script.sh
chmod 777 $DEBUG_DIR/slurm-script.sh

# Clean triggers
rm -f $TRIGGER_DIR/save
rm -f $TRIGGER_DIR/exit

# Checkpoint Compute Environment
echo -e "$(date)" > $COMPUTE_ENVIRONMENT_DIR 
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nCMD: $CMD_PREFIX $TRAINING_CMD" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nSlurm file: $0\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $0 >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nTOML file: $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nNODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nMegatron path: $MEGATRON_LM_DIR ($(git -C $MEGATRON_LM_DIR rev-parse --verify HEAD))" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\n$(pip list)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\n$(nvidia-smi)" >> $COMPUTE_ENVIRONMENT_DIR # CUDA Version & Driver
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nEnvironment Variables:\n\n$(printenv)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 

srun -lu bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\\s*[0-9]*MiB")' > $GPU_MEM_LOGGING

if [ "$AUTO_JOB_REQUEUE" = true ]; then
	echo "[$(date)] $(sbatch --dependency=singleton $0)"
fi

srun --cpus-per-task $SLURM_CPUS_PER_TASK \
	-lu bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID TORCHINDUCTOR_CACHE_DIR=$TORCH_INDUCTOR_CACHE_DIR/cache_\$SLURM_PROCID TRITON_HOME=$TRITON_HOME_CACHE_DIR/cache_\$SLURM_PROCID PYTHONPYCACHEPREFIX=$PYTHON_CACHE_DIR/cache_\$SLURM_PROCID $CMD_PREFIX $TRAINING_CMD"

# Remove Torchinductor, Triton & Python caches
rm -rf $TORCH_INDUCTOR_CACHE_DIR
rm -rf $TRITON_HOME_CACHE_DIR
rm -rf $PYTHON_CACHE_DIR

echo "END TIME: $(date)"

if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -rf $TRIGGER_DIR/exit  
   scancel --jobname $SLURM_JOB_NAME
fi
