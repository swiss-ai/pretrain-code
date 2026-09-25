# Apertus 1.5 training scripts

Slurm submission scripts for the Apertus 1.5 8B and 70B models, based on [Megatron-LM](https://github.com/swiss-ai/Megatron-LM).

| Folder | Contents |
| --- | --- |
| `container/` | Container environment files used by the scripts. |
| `pre-training/` | Continued pretraining from Apertus 1: stages 1, 2 and 3 (cooldown). |
| `long-context/` | Context extension: 32k, 64k, 128k and 256k. |
| `sft/` | Supervised fine-tuning. |
| `long-context-sft/` | 256k supervised fine-tuning. |

Each stage starts from the checkpoint of the previous one:
`stage1 -> stage2 -> stage3 -> 32k -> 64k -> 128k -> 256k`.
The long-context SFT starts from the 16k SFT checkpoint.

## Setup

Replace every `CHANGE_THIS` at the top of a script before submitting it. These are the data roots, the checkpoint to start from, the tokenizer, the Megatron-LM checkout, the dataset cache, the container environment file and the WandB API key.

Every `.bin`/`.idx` pair under the dataset directories listed in a script is added to the blend. No weights are given, so Megatron samples each file in proportion to its token count. Each file is tagged `pretrain:` (loss on all tokens) or `sft:` (loss on assistant turns only), so the Megatron-LM checkout must support these dataset markers.

## Launch

```
sbatch pre-training/apertus-1p5-8b-stage1.sh
```

With `AUTO_JOB_REQUEUE=true`, each job submits its follow-up with `--resume` until training completes. To resume by hand, run `sbatch <script> --resume`.

Before stage 1 can start, the Apertus 1 checkpoint needs the multimodal tokens added to its vocabulary. Run `apertus-1p5-*-stage1.sh` once with `EXTEND_MODEL_VOCAB=true` to write `EXTENDED_CKPT_DIR`, then run it again with `EXTEND_MODEL_VOCAB=false` to train.
