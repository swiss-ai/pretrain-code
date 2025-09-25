#!/bin/bash

#SBATCH --account=a-infra01-1
#SBATCH --time=18:00:00
#SBATCH --job-name=70b-32k
#SBATCH --output=/PATH/TO/long-ctx-70B-runs/slurm_logs/%x-%j.out
#SBATCH --error=/PATH/TO/long-ctx-70B-runs/slurm_logs/%x-%j.err
#SBATCH --partition=large512
#SBATCH --nodes=512
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36 
#SBATCH --mem=460000
#SBATCH --environment=/PATH/TO/NGC25_01.toml # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --signal=SIGUSR2@1200	# Send SIGUSR2 1200 seconds before hitting the time limit
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --exclude=nid006539,nid007378,nid006931,nid006726,nid006521

echo "START TIME: $(date)"

################ Configs ################
# NOTE(tj.solergibert) Check the `Data` section in the README. Use `,` to specify multiple datasets e.g. "/path/to/dataset/A,/path/to/dataset/B,/path/to/dataset/C"
DATASETS="/PATH/TO/long-ctx/data-mixture/32768_60b"

# This config trains with 2048 * 4096 = 8_388_608 tokens per batch.
# 2048 / 1 = 2048 forward passes
# With 1 replica over 8 nodes and 512 nodes we get 64 model replicas.
# With 2048 forward passes over 64 replicas each replica does batch accumulation of 2048 / 64 = 32
MBS=1 # Micro batch size
GBS=512 # Global batch size # NOTE(tj.solergibert) Originally this was 2048 BUT we doubled nodecount + GBS. Swicth to rampup batch size feature
SEQ_LEN=32768 # Sequence length
TRAINING_STEPS=3580  # total tokens = 1_075_000 * 8_388_608 = 9_017_753_600_000
CHECKPOINT_STEPS=250 # testing checkpointing

AUTO_JOB_REQUEUE=true # Set to `true` to continuously submit jobs to Slurm until training is complete. Enable it once you are sure of the cost involved in running this experiment.

#### Debugging ####
LOG_NCCL=false # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`
NSYS_PROFILER=false # Turn on the NSYS profiler. Check the `--profile-*` args available in megatron/training/arguments.py
MOCK_DATA=false # Set to `true` to use mock data
###################

# Megatron source and dataset cache
MEGATRON_LM_DIR=/PATH/TO/Megatron-LM
DATASET_CACHE_DIR=$SCRATCH/.tmp/dataset_cache
BACKUP_CODEBASE=false # Set to `true` to copy the codebase to the experiment folder and re-use it across runs

# Logging directories & artifacts
PROJECT_NAME=main-long-ctx-runs-70b-v1
EXP_NAME=apertus3-70b-32k-512nodes
PROJECT_DIR=/PATH/TO/long-ctx-70B-runs/Meg_Runs/$PROJECT_NAME

#########################################
LOAD_DIR=/PATH/TO/16K_LONG_CTX_CKPT_70B

EXP_DIR=$PROJECT_DIR/$EXP_NAME
TORCH_INDUCTOR_CACHE_DIR=/workspace/torch_compile_cache/$SLURM_JOB_ID
TRITON_HOME_CACHE_DIR=/workspace/triton_home_cache/$SLURM_JOB_ID
PYTHON_CACHE_DIR=/workspace/python_cache/$SLURM_JOB_ID
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
BACKUP_CODEBASE_DIR=$EXP_DIR/Megatron-LM

# Set up ENV
# export WANDB_API_KEY= # Paste the KEY here!
# export WANDB__FILE_STREAM_RETRY_MAX=10
# export HF_HUB_OFFLINE=1

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# We are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will set at `srun` command
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((10000 + $SLURM_JOB_ID % 1000))
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

#### Megatron Args #### Check megatron/training/arguments.py
# Based on the Llama 3.2 70B model.
TRANSFORMER_ENGINE_ARGS=(
	--main-grads-dtype fp32
	--transformer-impl transformer_engine
	--ddp-bucket-size 10000000000
	--decrease-batch-size-if-needed
)

NETWORK_SIZE_ARGS=(
	--num-layers 80
	--hidden-size 8192
	--ffn-hidden-size 43008  # xielu
	--num-attention-heads 64
	--group-query-attention
	--num-query-groups 8
	--max-position-embeddings $SEQ_LEN
	--position-embedding-type rope
	--rotary-base 4000000
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
	--log-progress
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
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--no-check-for-nan-in-loss-and-grad
	--train-iters $TRAINING_STEPS
	--log-interval 1
	--cross-entropy-loss-fusion
	--disable-bias-linear
	--optimizer ademamix  # ademamix
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 50
	--exit-signal-handler
	--trigger-path $TRIGGER_DIR
)

# 3662109375 
# 3683032334

INITIALIZATION_ARGS=(
	--seed 28
	--init-method-std 0.008944
)

# NOTE(tj.solergibert) Check all the arguments in megatron/training/arguments.py#L1548 or https://github.com/NVIDIA/Megatron-LM/blob/0dd78ddcdb117ce4f2e9761449274d87af717674/megatron/training/arguments.py#L1548-L1606
LEARNING_RATE_ARGS=(
	--lr 0.000001
	--min-lr 0.000001  # x10 reduction
	--lr-decay-style WSD  # WSD schedule
	--lr-warmup-iters 75
	--lr-wsd-decay-style 1-sqrt  # WSD schedule
	--lr-wsd-decay-iters 0  # WSD edcay will be a different run
)

# Check if checkpoint exists in CKPT_DIR
if [ -d "$CKPT_DIR" ] && [ "$(ls -A $CKPT_DIR 2>/dev/null | grep -E '^iter_[0-9]+$' | head -1)" ]; then
    # Found checkpoint in CKPT_DIR, load from there
    LATEST_CKPT=$(ls -A $CKPT_DIR 2>/dev/null | grep -E '^iter_[0-9]+$' | sort -V | tail -1)
    echo "[$(date)] Found checkpoint in CKPT_DIR: $LATEST_CKPT, loading from $CKPT_DIR"
    LOAD_FROM=$CKPT_DIR
    FINETUNE_FLAG=""
    NO_LOAD_RNG_FLAG=""
else
    # No checkpoint in CKPT_DIR, load from LOAD_DIR with finetune and no-load-rng
    echo "[$(date)] No checkpoint found in CKPT_DIR, loading from LOAD_DIR: $LOAD_DIR"
    LOAD_FROM=$LOAD_DIR
    FINETUNE_FLAG="--finetune"
    NO_LOAD_RNG_FLAG="--no-load-rng"
fi

# NOTE(tj.solergibert) Check the `Checkpointing` section in the README
CHECKPOINTING_ARGS=(
	--save $CKPT_DIR
	--save-interval $CHECKPOINT_STEPS
	--ckpt-format torch_dist
	--load $LOAD_FROM
	--async-save
	$NO_LOAD_RNG_FLAG
	# --no-load-optim
	--ckpt-fully-parallel-load
	--exit-interval 3500
	--dist-ckpt-strictness assume_ok_unexpected
	--ckpt-assume-constant-structure
	--override-opt_param-scheduler
	$FINETUNE_FLAG
)
# raise_all
MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size $SLURM_GPUS_PER_NODE
	--sequence-parallel
	--pipeline-model-parallel-size 8
	--num-layers-per-virtual-pipeline-stage 2
	--context-parallel-size 4
	--use-distributed-optimizer
	--overlap-p2p-communication-warmup-flush
	--overlap-grad-reduce
	--overlap-param-gather
)

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model alehc/swissai-tokenizer
)

DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
	--reset-position-ids  # crossDocAttn
	--reset-attention-mask  # crossDocAttn
	--eod-mask-loss  # crossDocAttn
	--num-workers 32
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

# # Remove unfinished checkpoints
# LATEST_CONFIRMED=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt")
# LATEST_FOLDER=$(find "$CKPT_DIR" -maxdepth 1 -type d -name "iter_*" | sed -n 's/.*iter_\([0-9]*\)$/\1/p' | sort -n | tail -n 1 | cut -c2-)

# # If the highest folder iteration doesn't match the latest confirmed one, remove it
# if [ "$LATEST_FOLDER" != "$LATEST_CONFIRMED" ]; then
#   echo "[$(date)] Deleting unfinished checkpoint folder: iter_0$LATEST_FOLDER"
#   rm -rf "$CKPT_DIR/iter_0$LATEST_FOLDER"
# else
#   echo "[$(date)] Checkpoint is consistent: iter_0$LATEST_CONFIRMED"
# fi

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