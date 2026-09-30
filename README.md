# Apertus 1.5 training scripts

Slurm submission scripts for the Apertus 1.5 8B and 70B models, based on [Megatron-LM](https://github.com/swiss-ai/Megatron-LM).

| Folder | Contents |
| --- | --- |
| `container/` | Container environment files used by the scripts. |
| `pretraining/8B/`, `pretraining/70B/` | Continued pretraining from Apertus 1: stages 1, 2 and 3 (cooldown). |
| `long-context/` | Context extension: 32k, 64k, 128k and 256k. |
| `sft/` | Supervised fine-tuning. |
| `long-context-sft/` | 256k supervised fine-tuning. |

Each stage starts from the checkpoint of the previous one:
`stage1 -> stage2 -> stage3 -> 32k -> 64k -> 128k -> 256k`.
The long-context SFT starts from the 16k SFT checkpoint.

## Setup

Replace every required `CHANGE_THIS` at the top of a script before submitting it. These are the data roots, the checkpoint to start from, the tokenizer, the Megatron-LM checkout, the dataset cache and the container environment file. Pretraining scripts also accept these settings through environment variables. Supply `WANDB_API_KEY` through the environment; leaving it unset disables W&B for pretraining.

Every `.bin`/`.idx` pair under the dataset directories listed in a script is added to the blend, except explicitly excluded files. No weights are given, so Megatron samples each file in proportion to its token count. Pretraining stages 1 and 2 use untagged prefixes. Stage 3 tags paths containing `apertus_sft` as `sft:` (loss on assistant turns only), and other paths as `pretrain:` (loss on all tokens). The Megatron-LM checkout must support these dataset markers.

## Launch

```
mkdir -p logs
sbatch pretraining/8B/apertus-1p5-8b-stage1.sh
```

With `AUTO_JOB_REQUEUE=true`, each job submits its follow-up with `--resume`. Pretraining scripts cancel that follow-up on a training failure or completion exit trigger. To resume by hand, run `sbatch <script> --resume`. Stage-specific Slurm job names isolate singleton dependencies. Reservation, account and node exclusions can be passed to `sbatch` for the current allocation.

Before stage 1 can start, the Apertus 1 checkpoint needs the multimodal tokens added to its vocabulary. Set `BASE_MODEL_DIR` to the matching model-size checkpoint and `LOAD_CKPT_DIR` to the extended checkpoint destination, then submit stage 1 with `--extend-model-vocab`. This mode disables W&B and follow-up submission. Submit again without that flag to train from the extended checkpoint.

For stages 2 and 3, set `LOAD_CKPT_DIR` to the previous stage's completed checkpoint. A fresh stage preserves optimizer state and resets RNG; `--resume` loads the current stage's checkpoint and RNG state. Both retain the cumulative phase-transition boundaries.

The launch structure and data exclusions follow the latest local 70B stage-3 run. Use `container/apertus-2-alps4-dev.toml` as the container baseline. Model sizes retain their own architecture, parallelism, learning rates and checkpoint cadence:

| Model | Global batch | Sequence length | Stage 1 end | Stage 2 end | Stage 3 end | Cooldown iterations |
| --- | --- | --- | --- | --- | --- | --- |
| 8B | 1024 | 8192 | 96000 | 430000 | 478000 | 48000 |
| 70B | 2048 | 8192 | 24000 | 96000 | 120000 | 24000 |

Inspect a command locally with `bash pretraining/70B/apertus-1p5-70b-stage3.sh --dry-run` (optionally with `--resume`). Dry runs do not enumerate datasets, create directories or submit jobs. `MOCK_DATA=true` selects Megatron mock data. Pretraining commands preserve argument boundaries, including paths containing spaces.
