# MMS Benchmark

This repository contains the code for constructing and evaluating the MMS Benchmark, a multi-task benchmark for evaluating video understanding capabilities of large vision-language models (LVLMs).

## Overview

The benchmark covers six evaluation tasks across four capability dimensions:

| Task | Experiment | Description |
|------|-----------|-------------|
| Video-to-Text Summarization | `exp1_v2t` | Generate a text summary from video |
| Video-to-Video Summarization | `exp2_v2v` | Rank video frames by relevance |
| Summarization Alignment | `exp3_alignment` | Align model output to reference shots |
| Temporal Understanding | `exp4_temporal_understanding` | Reason about shot order and timing |
| Grounding | `exp5_direct_alignment`, `exp5_hierarchical_alignment` | Ground text descriptions to video segments |
| Saliency Ranking | `exp6_salience_ranking` | Rank shots by visual saliency |

Datasets: **TVSum**, **SumMe**, **OVP**, **YouTube**, **VideoXum**, **MrHiSum**

## Repository Structure

```
├── configs/                        # Environment configuration
├── utils/                          # Shared utilities
├── datasets_preprocessed/
│   └── codes/                      # Benchmark construction pipeline
│       ├── 0_*.py / 0_*.sh         # Step 0: Data download, frame extraction, H5 generation
│       ├── 1_*.py                  # Step 1: Frame-level descriptions
│       ├── 2_*.py                  # Step 2: Shot-level descriptions
│       ├── 3_*.py                  # Step 3: Aligned document generation
│       ├── 4_*.py                  # Step 4: Statistics
│       ├── 5_run_pipeline.py       # Full pipeline runner
│       ├── Knapsack.py             # Knapsack algorithm for GT selection
│       └── statistics_generator.py
└── experiments/
    └── codes/
        ├── shared/                 # Shared data loaders and utilities
        ├── exp1_v2t/               # Experiment runners per model
        ├── exp2_v2v/
        ├── exp3_alignment/
        ├── exp4_temporal_understanding/
        ├── exp5_direct_alignment/
        ├── exp5_hierarchical_alignment/
        ├── exp6_salience_ranking/
        ├── exp7_text_only/
        └── exp8_document_only_summarization/
```

## Setup

### Dependencies

```bash
pip install -r requirements.txt
```

For LLaVA-based preprocessing (Steps 1–3):

```bash
conda env create -f environment_llavav.yml
conda activate vllava
```

### Configuration

Copy and edit the config file to set your data paths:

```bash
cp configs/config.server.yaml configs/config.local.yaml
# Edit data paths in config.local.yaml
```

Set required environment variables for API-based models:

```bash
export AZURE_OPENAI_KEY=<your_key>
export AZURE_OPENAI_ENDPOINT=<your_endpoint>
export OPENAI_API_KEY=<your_key>        # for non-Azure OpenAI
export OPENAI_BASE_URL=<your_base_url>
```

## Benchmark Construction

Run the preprocessing pipeline in order:

```bash
# Step 0: Download data, extract frames, generate H5 files
python datasets_preprocessed/codes/0_extract_frames.py
python datasets_preprocessed/codes/0_youtube_h5.py   # (repeat per dataset)

# Step 1: Frame-level descriptions
python datasets_preprocessed/codes/1_1_generate_frame_descriptions_llava.py

# Step 2: Shot-level descriptions
python datasets_preprocessed/codes/2_generate_shot_descriptions_llava.py

# Step 3: Aligned document generation (requires LLaMA)
python datasets_preprocessed/codes/3_generate_aligned_document_llama.py

# Or run the full pipeline:
python datasets_preprocessed/codes/5_run_pipeline.py
```

## Running Experiments

Each experiment directory contains one `run_*.py` script per evaluated model. Example:

```bash
cd experiments/codes
python exp1_v2t/run_qwen.py --dataset summe
python exp4_temporal_understanding/run_gpt.py --dataset tvsum
python exp6_salience_ranking/run_glm_pairwise.py --dataset summe
```

Evaluated models: **GPT-5.2**, **GLM-4.6V-Flash**, **Qwen3-VL-8B**, **Qwen2.5-VL-7B**, **InternVideo2.5**, **VideoLLaMA3-7B**

## Data Availability

Pre-processed benchmark data (H5 files and aligned documents) will be released upon paper acceptance.
