#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --job-name=apertus-1p5-8b-128k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue  # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.

echo "START TIME: $(date)"

## Configure these variables according to your setup. ##
DATAROOT=CHANGE_THIS  # 128k long-context data mixture.
LOAD_CKPT_DIR=CHANGE_THIS  # Checkpoint to start from: output of 8b_64k.sh.
TOKENIZER_MODEL=CHANGE_THIS  # Apertus 1.5 instruct tokenizer (apertus_emu3.5_wavtok_instruct).
MEGATRON_LM_DIR=CHANGE_THIS
DATASET_CACHE_DIR=CHANGE_THIS
CONTAINER_ENV=CHANGE_THIS  # Container environment file: ../../container/ngc-25.12.toml.
export WANDB_API_KEY=CHANGE_THIS  # Leave empty to disable WandB logging.

## Data mixture. ##
# Every `.bin`/`.idx` pair found under these directories is added to the blend. No weights are given:
# Megatron samples each file proportionally to its token count.

# Pretraining data: loss on all tokens.
PRETRAIN_DATASETS=(
	# Cooldown subsample: text.
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/finemath-3plus-merge
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/finepdfs-edu-multilingual-preprocessed_first_half_CPT
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/finepdfs-edu-multilingual-preprocessed_second_half_longcontext
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/finepdfs-edu-preprocessed_first_half_CPT
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/finetranslations_first_half_CPT
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/finetranslations_second_half_longcontext
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/infiwebmath-3plus-fine-merge
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/Nemotron-CC-v2.1-preprocessed
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/original  # Code: Nemotron-Pretraining-Code-v1-Synthetic.
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/swissai-dclm-edu-filterrobots_fine-merge
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/swissai-fineweb-2_0_1-quality_10-filterrobots_first_half_CPT
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/swissai-megamath-web-pro-filterrobots-merge
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/swissai-megamath-web-pro-filterrobots-merge2
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/swiss-caselaw-preprocessed
	# Cooldown subsample: vision.
	$DATAROOT/cooldown_subsample/vision-datasets/Apertus1p5_cooldown_tokenized
	# Cooldown subsample: audio.
	$DATAROOT/cooldown_subsample/audio-datasets/Apertus1p5_cooldown_tokenized
	# Extra vision samples.
	$DATAROOT/extra_vision_samples  # Image-text pairs (ChartNet code and summaries, HQ50K, MINT-1T arXiv, NASA, WAFFLE) and interleaved OWID articles.
	# Long-context samples: long documents, images and audio.
	$DATAROOT/long_context_samples/raw/second_half_length_filter/Audio
	$DATAROOT/long_context_samples/raw/second_half_length_filter/dolma3_olmocr_science_pdfs-preprocessed
	$DATAROOT/long_context_samples/raw/second_half_length_filter/finepdfs-edu-multilingual-preprocessed
	$DATAROOT/long_context_samples/raw/second_half_length_filter/finepdfs-edu-preprocessed
	$DATAROOT/long_context_samples/raw/second_half_length_filter/finetranslations
	$DATAROOT/long_context_samples/raw/second_half_length_filter/Image
	$DATAROOT/long_context_samples/raw/second_half_length_filter/institutional-books-1.0-filtered
	$DATAROOT/long_context_samples/raw/second_half_length_filter/swissai-fineweb-2_0_1-quality_10-filterrobots
)

# SFT data: loss on assistant turns only.
SFT_DATASETS=(
	# Cooldown subsample: SFT conversations.
	$DATAROOT/cooldown_subsample/prepare-8b/text_stage_3/apertus_sft_filtered
	# Long-context samples: synthetic retrieval tasks.
	$DATAROOT/long_context_samples/synthetic_apertus_sft/ArtificialNeedles
	# Long-context samples: synthetic common-words-extraction tasks.
	$DATAROOT/long_context_samples/synthetic_apertus_sft/CWE/Biomed-Enriched_preprocessed
	$DATAROOT/long_context_samples/synthetic_apertus_sft/CWE/dolma3_olmocr_science_pdfs-preprocessed
	$DATAROOT/long_context_samples/synthetic_apertus_sft/CWE/finepdfs-edu-multilingual-preprocessed
	$DATAROOT/long_context_samples/synthetic_apertus_sft/CWE/finepdfs-edu-preprocessed
	$DATAROOT/long_context_samples/synthetic_apertus_sft/CWE/finetranslations
	$DATAROOT/long_context_samples/synthetic_apertus_sft/CWE/institutional-books-1.0-filtered
	$DATAROOT/long_context_samples/synthetic_apertus_sft/CWE/swissai-fineweb-2_0_1-quality_10-filterrobots
	# Long-context samples: synthetic document-ordering tasks.
	$DATAROOT/long_context_samples/synthetic_apertus_sft/DocOrder/Biomed-Enriched_preprocessed
	$DATAROOT/long_context_samples/synthetic_apertus_sft/DocOrder/dolma3_olmocr_science_pdfs-preprocessed
	$DATAROOT/long_context_samples/synthetic_apertus_sft/DocOrder/finepdfs-edu-multilingual-preprocessed
	$DATAROOT/long_context_samples/synthetic_apertus_sft/DocOrder/finepdfs-edu-preprocessed
	$DATAROOT/long_context_samples/synthetic_apertus_sft/DocOrder/finetranslations
	$DATAROOT/long_context_samples/synthetic_apertus_sft/DocOrder/institutional-books-1.0-filtered
	$DATAROOT/long_context_samples/synthetic_apertus_sft/DocOrder/swissai-fineweb-2_0_1-quality_10-filterrobots
)

# Expand the dataset directories into Megatron prefixes tagged with their dataset type.
list_prefixes() {
	local tag=$1; shift
	[ $# -gt 0 ] || return 0
	find -L "$@" -name '*.bin' | sed 's/\.bin$//' | grep -vxF -f <(printf '%s\n' "${EXCLUDED_PREFIXES[@]}") | sed "s/^/$tag:/"
}
DATA_PATH_LIST=($( { list_prefixes pretrain "${PRETRAIN_DATASETS[@]}"; list_prefixes sft "${SFT_DATASETS[@]}"; } | sort -t: -k2))

## Training length. ##
MBS=1  # Micro batch size.
GBS=64  # Global batch size.
SEQ_LEN=131072  # Sequence length.
TARGET_TOKENS=60000000000  # ~60B tokens, 1 epoch of the mixture: 27.6B vision (46.3%), 8.4B audio (13.7%), 14.1B short-context + 9B long-context text (39%).
CHECKPOINT_STEPS=1000
TRAINING_STEPS=$((TARGET_TOKENS / (GBS * SEQ_LEN)))
TRAINING_STEPS=$((((TRAINING_STEPS + CHECKPOINT_STEPS/2) / CHECKPOINT_STEPS) * CHECKPOINT_STEPS))  # Round to nearest multiple of CHECKPOINT_STEPS.
PHASE_TRANSITION_ITERATIONS=8000

# Set to `true` to continuously submit jobs to Slurm until training is complete.
# Enable it once you are sure of the cost involved in running this experiment.
AUTO_JOB_REQUEUE=false

TRUNCATE_RIGHT=false  # Set to `true` to truncate too-long SFT samples on the right instead of the left.

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
PROJECT_NAME=main-runs-apertus-1p5-8b-long-context
EXP_NAME=apertus-1p5-8b-128k
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
	# There was a bug in Megatron and we trained with factor 8. However, we recommend using 16: https://github.com/swiss-ai/Megatron-LM/pull/137
	--rope-scaling-factor 16 
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
	--weight-decay 0.1
	--weight-decay-on-xielu-alphas
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
	--calculate-per-token-loss
	--disable-bias-linear
	--optimizer ademamix  # ademamix
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 500
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
	--lr 0.000011
	--min-lr 0.0000011  # x10 reduction
	--lr-decay-style WSD  # WSD schedule
	--lr-warmup-iters 300
	--lr-wsd-decay-style minus_sqrt  # WSD schedule
	--lr-wsd-decay-iters 0  # No decay
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
		--phase-transition-iterations $PHASE_TRANSITION_ITERATIONS
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
	--context-parallel-size 8
	--sequence-parallel
	--use-distributed-optimizer
	--overlap-grad-reduce
	--overlap-param-gather
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
	--num-workers 64
	--num-dataset-builder-threads 4
	--goldfish-loss  # goldfish
	--goldfish-k 50  # goldfish
	--goldfish-h 50  # goldfish
	--loss-mask-token-ids 131082 131083 131084  # Mask <|stt_transcribe|> <|stt_continue|> <|tts_continue|>.
	--pretraining-packing-strategy bfd
	--max-docs-per-bin 128
	# SFT: mask the chat template, compute the loss on assistant turns only.
	--ap-sft
	--ap-sft-mask-special-tokens
	--ap-sft-pack-samples
	--ap-sft-packing-strategy bfd
	--ap-sft-long-ctx-loss
)

if [ "$TRUNCATE_RIGHT" = true ]; then
	DATA_ARGS+=(--ap-sft-truncate-right)
fi

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
	bash -c "mkdir -p $TRITON_CACHE_DIR; RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"

if [ -f $TRIGGER_DIR/exit ]; then
	echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
	rm -rf $TRIGGER_DIR/exit
	scancel --jobname $SLURM_JOB_NAME
fi
