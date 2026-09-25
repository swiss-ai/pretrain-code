#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --job-name=apertus-1p5-8b-sft-256k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --signal=SIGTERM@600  # Send SIGTERM 600 seconds before hitting the time limit to save a checkpoint and exit.
#SBATCH --no-requeue  # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.

echo "START TIME: $(date)"

## Configure these variables according to your setup. ##
AUDIO_DATAROOT=CHANGE_THIS  # Tokenized audio SFT data.
VISION_DATAROOT=CHANGE_THIS  # Tokenized vision SFT data.
TEXT_DATAROOT=CHANGE_THIS  # Tokenized text SFT data.
LOAD_CKPT_DIR=CHANGE_THIS  # Checkpoint to start from: the 16k SFT checkpoint.
TOKENIZER_MODEL=CHANGE_THIS  # Apertus 1.5 instruct tokenizer (apertus_emu3.5_wavtok_instruct).
MEGATRON_LM_DIR=CHANGE_THIS
DATASET_CACHE_DIR=CHANGE_THIS
CONTAINER_ENV=CHANGE_THIS  # Container environment file: ../container/ngc-25.12.toml.
export WANDB_API_KEY=CHANGE_THIS  # Leave empty to disable WandB logging.

## Data mixture. ##
# Every `.bin`/`.idx` pair found under these directories is added to the blend. No weights are given:
# Megatron samples each file proportionally to its token count.

# SFT data: loss on assistant turns only.
SFT_DATASETS=(
	# Audio.
	$AUDIO_DATAROOT/under_8k  # 1.12B tokens.
	$AUDIO_DATAROOT/8k_64k  # 5.5B tokens.
	# Vision.
	$VISION_DATAROOT/under_16k  # 33.27B tokens.
	$VISION_DATAROOT/16K_128k  # 4.98B tokens.
	$VISION_DATAROOT/128k_256k  # 6B tokens.
	# Tool calling.
	$TEXT_DATAROOT/tool_sft_datasets_split/lower_16k_0.2  # 0.068B tokens.
	$TEXT_DATAROOT/tool_sft_datasets/128k_256k  # 0.030B tokens.
	# Long-context text.
	$TEXT_DATAROOT/sft_long_context_256k  # 0.530B tokens.
	# Short-context text.
	$TEXT_DATAROOT/v1p5-mix-v1-28-05-linearised_fix_display_answers_tool_toolheaders_injected_split_3B/v1p5-mix-v1-28-05-linearised_fix_display_answers_tool_toolheaders_injected_0.483/dump-0  # 2.9B tokens.
)

# Expand the dataset directories into Megatron prefixes tagged with their dataset type.
list_prefixes() {
	local tag=$1; shift
	[ $# -gt 0 ] || return 0
	find -L "$@" -name '*.bin' | sed 's/\.bin$//' | grep -vxF -f <(printf '%s\n' "${EXCLUDED_PREFIXES[@]}") | sed "s/^/$tag:/"
}
DATA_PATH_LIST=($(list_prefixes sft "${SFT_DATASETS[@]}" | sort -t: -k2))

## Training length. ##
MBS=1  # Micro batch size.
GBS=32  # Global batch size.
SEQ_LEN=262144  # Sequence length.
CHECKPOINT_STEPS=200
TRAINING_STEPS=6500

# Set to `true` to continuously submit jobs to Slurm until training is complete.
# Enable it once you are sure of the cost involved in running this experiment.
AUTO_JOB_REQUEUE=false

## Debugging. ##
LOG_NCCL=false  # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`.
NSYS_PROFILER=false  # Turn on the NSYS profiler. Check the `--profile-*` args available in megatron/training/arguments.py.
MOCK_DATA=false  # Set to `true` to use mock data.

## Command line arguments. ##
RESUME_TRAINING=false
while [[ $# -gt 0 ]]; do
	case $1 in
		--resume)
			RESUME_TRAINING=true
			shift
			;;
		*)
			echo "Unknown argument: $1"
			echo "Usage: $0 [--resume]"
			exit 1
			;;
	esac
done

# Logging directories & artifacts.
PROJECT_NAME=main-runs-apertus-1p5-sft
EXP_NAME=apertus-1p5-8b-sft-256k
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME
EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
TRITON_HOME_DIR=/tmp/.triton  # Node-local.

# Set up ENV.
export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TRITON_HOME=$TRITON_HOME_DIR
export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache

# We are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will set at `srun` command
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

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
	--rotary-base 4000000
	--use-rope-scaling
	# There was a bug in Megatron and we trained with factor 8. However, we recommend using 32: https://github.com/swiss-ai/Megatron-LM/pull/137
	--rope-scaling-factor 32
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
	--log-audio-weight
	--log-vision-weight
)

REGULARIZATION_ARGS=(
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--weight-decay 0.0
	--weight-decay-on-xielu-alphas
	--clip-grad 1.0
	--adam-beta1 0.9
	--adam-beta2 0.999
)

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--no-check-for-nan-in-loss-and-grad
	--train-iters $TRAINING_STEPS
	--log-interval 1
	--cross-entropy-loss-fusion
	--calculate-per-token-loss
	--disable-bias-linear
	--optimizer adam
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 10
	--exit-signal-handler
	--trigger-path $TRIGGER_DIR
	--eval-interval 100000000000
	--eval-iters 0
	--audio-weight 0
	--vision-weight 0
)

INITIALIZATION_ARGS=(
	--seed 41
	--init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
	--lr 6e-5
	--min-lr 0
	--lr-decay-style constant
	--lr-warmup-iters 195  # 3% of TRAINING_STEPS.
)

if [ "$RESUME_TRAINING" = true ]; then
	echo "[$(date)] Resuming training from $CKPT_DIR"
	LOAD_DIR=$CKPT_DIR
	LOAD_ARGS=()
else
	echo "[$(date)] Starting training from $LOAD_CKPT_DIR"
	LOAD_DIR=$LOAD_CKPT_DIR
	LOAD_ARGS=(
		--finetune
		--no-load-optim
		--no-load-rng
	)
fi

CHECKPOINTING_ARGS=(
	--load $LOAD_DIR
	--save $CKPT_DIR
	--save-interval $CHECKPOINT_STEPS
	--ckpt-format torch_dist
	--async-save
	--ckpt-fully-parallel-load
	--dist-ckpt-strictness assume_ok_unexpected
	--override-opt_param-scheduler
	${LOAD_ARGS[@]}
)

MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size 4
	--pipeline-model-parallel-size 1
	--context-parallel-size 16
	--sequence-parallel
	--use-distributed-optimizer
	--overlap-grad-reduce
	--overlap-param-gather
	--distributed-timeout-minutes 120
)

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model $TOKENIZER_MODEL
)

DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
	--reset-position-ids  # crossDocAttn
	--use-packed-seq-params  # crossDocAttn, via packed sequences instead of an attention mask
	--no-create-attention-mask-in-dataloader  # crossDocAttn
	--eod-mask-loss  # crossDocAttn
	--num-workers 32
	--num-dataset-builder-threads 4
	--loss-mask-token-ids 131082 131083 131084  # Mask <|stt_transcribe|> <|stt_continue|> <|tts_continue|>.
	# SFT: mask the chat template, compute the loss on assistant turns only.
	--ap-sft
	--ap-sft-mask-special-tokens
	--ap-sft-pack-samples
	--ap-sft-packing-strategy bfd
	--max-docs-per-bin 64
)

# Set up directories
mkdir -p $CKPT_DIR
mkdir -p $PROJECT_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR

export PYTHONPATH=$MEGATRON_LM_DIR

# Data Args
if [ "$MOCK_DATA" = true ]; then
	DATA_ARGS="${DATA_ARGS[@]} --mock-data"
else
	DATA_ARGS="${DATA_ARGS[@]} --data-path ${DATA_PATH_LIST[@]} --data-cache-path $DATASET_CACHE_DIR"
fi

CMD_PREFIX=""

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

# Clean triggers
rm -f $TRIGGER_DIR/save
rm -f $TRIGGER_DIR/exit

if [ "$AUTO_JOB_REQUEUE" = true ]; then
	echo "[$(date)] $(sbatch --dependency=singleton --job-name=$SLURM_JOB_NAME $0 --resume)"
fi

srun \
	--cpus-per-task $SLURM_CPUS_PER_TASK \
	--mpi=pmix \
	--environment=$CONTAINER_ENV \
	--network=disable_rdzv_get \
	-lu \
	bash -c "mkdir -p $TRITON_CACHE_DIR; export NCCL_TIMEOUT=7200; RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"

if [ -f $TRIGGER_DIR/exit ]; then
	echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
	rm -rf $TRIGGER_DIR/exit
	scancel --jobname $SLURM_JOB_NAME
fi
