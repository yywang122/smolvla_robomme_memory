# SmolVLA on RoboMME

## Project Overview

This repository contains the training and evaluation pipeline
for SmolVLA on the RoboMME benchmark.

## Current Setup

- Model: SmolVLA
- Training: Full Baseline, 30K steps
- Benchmark: RoboMME
- Evaluation Task: BinFill
- Action Space: Joint Angle
- Action Chunk: 50

## Current Status

The model can be loaded successfully and produces robot actions.

However, the evaluation success rate remains at 0%.

Current debugging focuses on:
- Gripper control
- Observation and action consistency
- Action normalization
- Robot controller interface

## Main Scripts

- `scripts/train_smolvla_robomme_full.py`: Training
- `scripts/serve_smolvla_full_baseline.py`: Inference server
- `scripts/eval_smolvla_full_binfill_binary_gripper.py`: Latest gripper evaluation

## Notes

Model checkpoints and datasets are not included in this repository.

---

## How to Run Evaluation

The evaluation requires two terminals.

### Terminal 1: Start SmolVLA Inference Server

```bash

cd ~/smolvla_robomme_memory

python scripts/serve_smolvla_full_baseline.py \
  --checkpoint ~/smolvla_robomme_memory/outputs/smolvla_robomme_full_baseline_30k/checkpoints/030000/pretrained_model \
  --exec-horizon 50 \
  --port 8011 \
  --device cuda
```

Keep Terminal 1 running during evaluation.

### Terminal 2: Run RoboMME Evaluation

```bash
cd ~/Downloads/third_party/robomme_benchmark

source .venv/bin/activate

cd ~/smolvla_robomme_memory

python scripts/eval_smolvla_full_binfill_binary_gripper.py \
  2>&1 | tee eval_binary_gripper.log
```

### Evaluation Notes

- Model: SmolVLA Full Baseline (30K steps)
- Benchmark: RoboMME
- Task: BinFill
- Action Space: Joint Angle
- Action Chunk: 50
- Inference Server Port: 8011
- Start Terminal 1 before Terminal 2.
- The RoboMME benchmark and EnvRunner dependencies must be installed separately.
- Checkpoint and environment paths must be adjusted for each machine.
- Current evaluation success rate: 0% (under investigation).

