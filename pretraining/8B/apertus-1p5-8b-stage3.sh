#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --job-name=apertus-1p5-8b-stage3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --signal=SIGUSR2@600
#SBATCH --no-requeue

set -euo pipefail

# Configure paths for this stage; environment overrides are also supported.
# Container baseline: latest 70B run (container/apertus-2-alps4-dev.toml).
CONTAINER_ENV=${CONTAINER_ENV:-CHANGE_THIS}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-CHANGE_THIS}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-CHANGE_THIS}
TEXT_DATA_DIR=${TEXT_DATA_DIR:-CHANGE_THIS}
VISION_DATA_DIR=${VISION_DATA_DIR:-CHANGE_THIS}
AUDIO_DATA_DIR=${AUDIO_DATA_DIR:-CHANGE_THIS}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-CHANGE_THIS} # apertus_emu3.5_wavtok_instruct
LOAD_CKPT_DIR=${LOAD_CKPT_DIR:-CHANGE_THIS} # Stage 2 output checkpoint.

PROJECT_NAME=${PROJECT_NAME:-main-runs-8b-apertus-1p5}
EXP_NAME=${EXP_NAME:-apertus-1p5-8b-stage3}
PROJECT_DIR=${PROJECT_DIR:-$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME}
CKPT_DIR=$PROJECT_DIR/$EXP_NAME/checkpoints
TRIGGER_DIR=$PROJECT_DIR/$EXP_NAME/triggers
LOGGING_DIR=$PROJECT_DIR/$EXP_NAME/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
DEBUG_DIR=$PROJECT_DIR/$EXP_NAME/debug/${SLURM_JOB_ID:-local}

# Cumulative iteration endpoint, including earlier stages.
MBS=1
GBS=1024
SEQ_LEN=8192
TRAINING_STEPS=478000
CHECKPOINT_STEPS=2000
PHASE_TRANSITION_ITERATIONS="96000,430000"

# Follow-up jobs resume this stage. W&B credentials must come from the environment.
AUTO_JOB_REQUEUE=${AUTO_JOB_REQUEUE:-true}
MOCK_DATA=${MOCK_DATA:-false}
LOG_NCCL=${LOG_NCCL:-false}
NSYS_PROFILER=${NSYS_PROFILER:-false}
RESUME_TRAINING=false
EXTEND_MODEL_VOCAB=false
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME_TRAINING=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done
if [[ "$RESUME_TRAINING" == true && "$EXTEND_MODEL_VOCAB" == true ]]; then
    echo "--resume and --extend-model-vocab are mutually exclusive" >&2
    exit 1
fi

# Dry runs can inspect placeholder paths without accessing the cluster.
if [[ "$DRY_RUN" == false ]]; then
    required=(CONTAINER_ENV MEGATRON_LM_DIR TOKENIZER_MODEL)
    if [[ "$EXTEND_MODEL_VOCAB" == true ]]; then
        required+=(BASE_MODEL_DIR LOAD_CKPT_DIR)
    elif [[ "$RESUME_TRAINING" == false ]]; then
        required+=(LOAD_CKPT_DIR)
    fi
    if [[ "$MOCK_DATA" == false ]]; then
        required+=(TEXT_DATA_DIR VISION_DATA_DIR AUDIO_DATA_DIR DATASET_CACHE_DIR)
    fi
    for name in "${required[@]}"; do
        if [[ -z "${!name}" || "${!name}" == CHANGE_THIS ]]; then
            echo "Configure $name before launching" >&2
            exit 1
        fi
    done
fi

# Keep optimizer state across stage boundaries; only stage 1 uses --finetune.
CHECKPOINT_MODE_ARGS=()
SAVE_DIR=$CKPT_DIR
CHECKPOINT_MODE_ARGS+=(--phase-transition-iterations "$PHASE_TRANSITION_ITERATIONS")
if [[ "$RESUME_TRAINING" == true ]]; then
    LOAD_DIR=$CKPT_DIR
else
    LOAD_DIR=$LOAD_CKPT_DIR
    CHECKPOINT_MODE_ARGS+=(--no-load-rng)
fi

TRANSFORMER_ENGINE_ARGS=(
    --main-grads-dtype fp32
    --log-params-norm
)

NETWORK_SIZE_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 21504
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings "$SEQ_LEN"
    --position-embedding-type rope
    --rotary-base 500000
    --use-rope-scaling
    --rope-scaling-factor 8
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --xielu
    --qk-layernorm
    --qknorm-impl apex
    --untie-embeddings-and-output-weights
)

LOGGING_ARGS=(
    --log-throughput
    --tensorboard-dir "$TENSORBOARD_DIR"
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --weight-decay-on-xielu-alphas
    --clip-grad 0.1
    --adam-beta1 0.9
    --adam-beta2 0.999
    --ademamix-alpha 8
    --ademamix-beta3 0.9999
    --ademamix-beta3-warmup 100000
    --ademamix-alpha-warmup 100000
)

TRAINING_ARGS=(
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --no-check-for-nan-in-loss-and-grad
    --train-iters "$TRAINING_STEPS"
    --log-interval 1
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer ademamix
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 500
    --exit-signal-handler
    --trigger-path "$TRIGGER_DIR"
    --eval-interval 100000000000
    --eval-iters 0
    --audio-weight 0
    --vision-weight 0
    --log-audio-weight
    --log-vision-weight
)

INITIALIZATION_ARGS=(
    --seed 41
    --init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
    --lr 0.00011
    --min-lr 0.000011
    --lr-decay-style WSD
    --lr-warmup-iters 0
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-iters 48000
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
    --tokenizer-model "$TOKENIZER_MODEL"
    --ap-sft
    --ap-sft-mask-special-tokens
    --ap-sft-pack-samples
    --ap-sft-packing-strategy bfd
    --calculate-per-token-loss
)

DATA_ARGS=(
    --split 100,0,0
    --seq-length "$SEQ_LEN"
    --reset-position-ids
    --use-packed-seq-params
    --no-create-attention-mask-in-dataloader
    --eod-mask-loss
    --num-workers 2
    --num-dataset-builder-threads 4
    --goldfish-loss
    --goldfish-k 50
    --goldfish-h 50
    --loss-mask-token-ids 131082 131083 131084
)

CHECKPOINTING_ARGS=(
    --load "$LOAD_DIR"
    --save "$SAVE_DIR"
    --save-interval "$CHECKPOINT_STEPS"
    --ckpt-format torch_dist
    --async-save
    --ckpt-fully-parallel-load
    --dist-ckpt-strictness assume_ok_unexpected
    --override-opt_param-scheduler
    "${CHECKPOINT_MODE_ARGS[@]}"
)

# Exclusions from the latest 70B run, matched against resolved symlink targets.
list_data_prefixes() {
    local file resolved prefix
    while IFS= read -r file; do
        resolved=$(readlink -f "$file")
        case "$resolved" in
            */swiss-caselaw-preprocessed/dump-1/00019_tokens.bin|\
            */swiss-caselaw-preprocessed/dump-0/00019_tokens.bin|\
            */swiss-caselaw-preprocessed/dump-0_00018_tokens.bin) continue ;;
        esac
        prefix=${file%.bin}
        if [[ "$prefix" == *apertus_sft* ]]; then
            printf 'sft:%s\n' "$prefix"
        else
            printf 'pretrain:%s\n' "$prefix"
        fi
    done
}

if [[ "$MOCK_DATA" == true ]]; then
    DATA_ARGS+=(--mock-data)
elif [[ "$DRY_RUN" == true ]]; then
    # Dataset enumeration requires the configured cluster filesystem.
    DATA_ARGS+=(--data-path '<dataset-prefixes>' --data-cache-path "$DATASET_CACHE_DIR")
else
    # Capture the pipeline status before mapfile so failed discovery stops submission.
    prefixes=$(find -L "$TEXT_DATA_DIR" "$VISION_DATA_DIR" "$AUDIO_DATA_DIR" \
        -name '*.bin' | sort | list_data_prefixes)
    if [[ -z "$prefixes" ]]; then
        echo "No dataset prefixes found" >&2
        exit 1
    fi
    mapfile -t DATA_PATH_LIST <<< "$prefixes"
    DATA_ARGS+=(--data-path "${DATA_PATH_LIST[@]}" --data-cache-path "$DATASET_CACHE_DIR")
fi

if [[ "$EXTEND_MODEL_VOCAB" == true ]]; then
    LOGGING_ARGS=()
    export WANDB_MODE=disabled
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    LOGGING_ARGS+=(--wandb-save-dir "$LOGGING_DIR" --wandb-project "$PROJECT_NAME"
        --wandb-exp-name "$EXP_NAME-${SLURM_JOB_ID:-local}")
else
    export WANDB_MODE=disabled
fi

TRAINING_CMD=(python3 "$MEGATRON_LM_DIR/pretrain_gpt.py"
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${NETWORK_SIZE_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${INITIALIZATION_ARGS[@]}"
    "${LEARNING_RATE_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${DISTRIBUTED_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${CHECKPOINTING_ARGS[@]}"
)

if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' "${TRAINING_CMD[@]}"
    printf '\n'
    exit 0
fi

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_CPUS_PER_TASK:?Missing Slurm CPU allocation}"
echo "START TIME: $(date)"
echo "Loading $LOAD_DIR; saving $SAVE_DIR"
export WANDB__FILE_STREAM_RETRY_MAX=10 HF_HUB_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$MEGATRON_LM_DIR${PYTHONPATH:+:$PYTHONPATH}
hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=${hosts%%$'\n'*}
export MASTER_PORT=${MASTER_PORT:-8888}
export WORLD_SIZE=$SLURM_NTASKS
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch-inductor-$SLURM_JOB_ID
export TRITON_HOME=/tmp/triton-$SLURM_JOB_ID
export TRITON_CACHE_DIR=$TRITON_HOME/cache
export DEBUG_DIR LOG_NCCL NSYS_PROFILER

mkdir -p "$SAVE_DIR" "$DEBUG_DIR" "$LOGGING_DIR" "$TRIGGER_DIR"
rm -f "$TRIGGER_DIR/save" "$TRIGGER_DIR/exit"

# Retain the queued job ID so cancellation cannot affect another stage/run.
NEXT_JOB_ID=
if [[ "$AUTO_JOB_REQUEUE" == true && "$EXTEND_MODEL_VOCAB" == false ]]; then
    NEXT_JOB_ID=$(sbatch --parsable --dependency=singleton --job-name="$SLURM_JOB_NAME" "$0" --resume)
    NEXT_JOB_ID=${NEXT_JOB_ID%%;*}
fi

status=0
srun --cpus-per-task "$SLURM_CPUS_PER_TASK" --mpi=pmix \
    --environment="$CONTAINER_ENV" --network=disable_rdzv_get -lu \
    bash -c '
        set -euo pipefail
        mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
        export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
        if [[ "$LOG_NCCL" == true ]]; then
            export NCCL_DEBUG=INFO
            export NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-$SLURMD_NODENAME-$SLURM_PROCID.txt
        fi
        if [[ "$NSYS_PROFILER" == true ]]; then
            exec nsys profile -s none --trace=nvtx,cudnn,cublas,cuda \
                --output="$DEBUG_DIR/nsys-$SLURMD_NODENAME-$SLURM_PROCID" \
                --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop \
                "$@" --profile
        fi
        exec "$@"
    ' bash "${TRAINING_CMD[@]}" || status=$?

echo "END TIME: $(date)"
if [[ -n "$NEXT_JOB_ID" && ( -f "$TRIGGER_DIR/exit" || "$status" -ne 0 ) ]]; then
    scancel "$NEXT_JOB_ID"
fi
exit "$status"
