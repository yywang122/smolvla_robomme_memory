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
