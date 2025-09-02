#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the log file in the same directory
LOG_FILE="$SCRIPT_DIR/create_mixture.log"

# Redirect both stdout and stderr to the log file (and also to terminal)
exec > >(tee -a "$LOG_FILE") 2>&1


MIXTURE_CREATE_SCRIPT=/capstor/scratch/cscs/ctianche/swissai_long_context/framework_prepare/Megatron-LM/scripts/tools/create_data_mixture_v2.py
MIXTURE_CHECK_SCRIPT=/capstor/store/cscs/swissai/infra01/users/ctianche/long-context-mixture/calculate_mixture_size.py

PYTHON_ACTIVATE_PATH=/capstor/scratch/cscs/ctianche/swissai_long_context/.venv/bin/activate

source $PYTHON_ACTIVATE_PATH

# 70% phase 5, 20% fw long, 10% book
PHASE_5_DIR=/capstor/store/cscs/swissai/infra01/users/ctianche/long-context-mixture/phase5_datasets_symlinks
FW_LONG_DIR=/capstor/store/cscs/swissai/infra01/users/ctianche/long-context-mixture/fw12_long_symlinks
BOOK_DIR=/capstor/store/cscs/swissai/infra01/users/ctianche/long-context-tokenized/institutional-books-filtered
SWISSDATA_DIR=/capstor/store/cscs/swissai/infra01/users/ctianche/long-context-tokenized/commercial-swiss-data-with-entscheidungsuche-half

OUTPUT_BASE_DIR=/capstor/store/cscs/swissai/infra01/users/ctianche/long-ctx-70B/data-mixture

# 8k mixture (80B)
# remove existing files
rm -rf $OUTPUT_BASE_DIR/8192

python3 $MIXTURE_CREATE_SCRIPT \
    --folders \
    $PHASE_5_DIR \
    $FW_LONG_DIR/8192 \
    $BOOK_DIR \
    --weights 0.7 0.2 0.1 \
    --output $OUTPUT_BASE_DIR/8192 \
    --max_tokens 80_000_000_000

python3 $MIXTURE_CHECK_SCRIPT $OUTPUT_BASE_DIR/8192 8192 > $OUTPUT_BASE_DIR/8192/mixture_stats.txt 2>&1


# 16k

rm -rf $OUTPUT_BASE_DIR/16384

python3 $MIXTURE_CREATE_SCRIPT \
    --folders \
    $PHASE_5_DIR \
    $FW_LONG_DIR/16384 \
    $BOOK_DIR \
    --weights 0.7 0.2 0.1 \
    --output $OUTPUT_BASE_DIR/16384 \
    --exclude $OUTPUT_BASE_DIR/8192 \
    --max_tokens 60_000_000_000

python3 $MIXTURE_CHECK_SCRIPT $OUTPUT_BASE_DIR/16384 16384 > $OUTPUT_BASE_DIR/16384/mixture_stats.txt 2>&1


# 32k 60B

rm -rf $OUTPUT_BASE_DIR/32768_60b

python3 $MIXTURE_CREATE_SCRIPT \
    --folders \
    $PHASE_5_DIR \
    $FW_LONG_DIR/32768 \
    $BOOK_DIR \
    --weights 0.7 0.2 0.1 \
    --output $OUTPUT_BASE_DIR/32768_60b \
    --exclude $OUTPUT_BASE_DIR/8192 $OUTPUT_BASE_DIR/16384 \
    --max_tokens 60_000_000_000

python3 $MIXTURE_CHECK_SCRIPT $OUTPUT_BASE_DIR/32768_60b 32768 > $OUTPUT_BASE_DIR/32768_60b/mixture_stats.txt 2>&1

# 64k

rm -rf $OUTPUT_BASE_DIR/65536

python3 $MIXTURE_CREATE_SCRIPT \
    --folders \
    $PHASE_5_DIR \
    $FW_LONG_DIR/65536 \
    $BOOK_DIR \
    --weights 0.7 0.2 0.1 \
    --output $OUTPUT_BASE_DIR/65536 \
    --exclude $OUTPUT_BASE_DIR/8192 $OUTPUT_BASE_DIR/16384 $OUTPUT_BASE_DIR/32768 \
    --max_tokens 20_000_000_000

python3 $MIXTURE_CHECK_SCRIPT $OUTPUT_BASE_DIR/65536 65536 > $OUTPUT_BASE_DIR/65536/mixture_stats.txt 2>&1


# finish
