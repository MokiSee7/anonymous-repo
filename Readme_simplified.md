# HAVEN Benchmark

Anonymous repository for the NeurIPS 2026 submission.

## Overview

HAVEN is a hierarchical multimodal benchmark for unified video understanding.  
The benchmark includes evaluation tasks covering:

- Video-to-Text Summarization (V2T)
- Video-to-Video Summarization / Keyframe Selection (V2V)
- Joint Video-Text Summarization (V2VT)
- Temporal Understanding
- Multimodal Grounding
- Saliency Ranking

Source datasets include:

- TVSum
- SumMe
- OVP
- YouTube
- VideoXum
- MR.HiSum

## Repository Contents

- Benchmark construction pipeline
- Evaluation scripts
- Prompt templates
- Experiment runners
- Shared utilities and configurations

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run benchmark construction:

```bash
python datasets_preprocessed/codes/5_run_pipeline.py
```

Run experiments:

```bash
cd experiments/codes
python exp1_v2t/run_qwen.py --dataset summe
```

## Evaluated Models

- GPT-5.2
- GLM-4.6V-Flash
- Qwen3-VL
- Qwen2.5-VL
- InternVideo2.5
- VideoLLaMA3

## Data Availability

Preprocessed benchmark data is released at https://anonymous.4open.science/r/align_vsum-9C22.
