#!/bin/bash

MIXTURE_CREATE_SCRIPT=/PATH/TO/Megatron-LM/scripts/tools/create_data_mixture_v2.py
MIXTURE_CHECK_SCRIPT=/PATH/TO/long-context-mixture/calculate_mixture_size.py

# 70% phase 5, 20% fw long, 10% book
PHASE_5_DIR=/PATH/TO/long-context-mixture/phase5_datasets_symlinks
FW_LONG_DIR=/PATH/TO/long-context-mixture/fw12_long_symlinks
BOOK_DIR=/PATH/TO/long-context-tokenized/institutional-books-filtered

OUTPUT_BASE_DIR=/PATH/TO/long-ctx/data-mixture

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
