<p align="center">
  <img src="assets/banner.png" alt="CHAI Banner" width="100%">
</p>

<p align="center">
  <img src="assets/logo.png" alt="CHAI Logo" width="700">
</p>

**Official Codebase for CVPR 2026 Highlight Paper:**
*"Building a Precise Video Language with Human–AI Oversight"*

<p align="center">
  <a href="https://arxiv.org/abs/2604.21718"><img src="https://img.shields.io/badge/arXiv-2604.21718-b31b1b.svg" alt="Paper"></a>
  <a href="https://linzhiqiu.github.io/papers/chai/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/chancharikm/CHAI_testset"><img src="https://img.shields.io/badge/🤗_Dataset-CHAI_Testset-FFD21E" alt="Dataset"></a>
  <a href="https://huggingface.co/chancharikm/CHAI_SFT_model_8b"><img src="https://img.shields.io/badge/🤗_Model-CHAI_SFT_8B-FFD21E" alt="Model"></a>
  <a href="https://huggingface.co/papers/2604.21718"><img src="https://img.shields.io/badge/🤗_Paper-Daily_Papers-FFD21E" alt="HF Paper"></a>
</p>
<!-- 
> **🚧 More coming soon!** We are actively releasing evaluation code, additional models, and videos. Stay tuned. -->

[Zhiqiu Lin](https://linzhiqiu.github.io/)¹,
[Chancharik Mitra](https://chancharikmitra.github.io/)¹,
[Siyuan Cen](https://sy77777en.github.io/)¹,
[Isaac Li](https://www.linkedin.com/in/isaac-li-bb381b284/)¹,
Yuhan Huang¹,
[Yu Tong Tiffany Ling](https://www.yttldesign.com/)¹,
[Hewei Wang](https://github.com/WangHewei16)¹,
Irene Pi¹,
Shihang Zhu¹,
Ryan Rao¹,
George Liu¹,
Jiaxi Li¹,
Ruojin Li¹,
Yili Han¹,
[Yilun Du](https://yilundu.github.io/)²,
[Deva Ramanan](https://www.cs.cmu.edu/~deva/)¹

¹Carnegie Mellon University &nbsp; ²Harvard University

### Updates
- **2026-05-13**: Released evaluation code, updated test set, and published the [CHAI SFT 8B model](https://huggingface.co/chancharikm/CHAI_SFT_model_8b).

# CHAI (Critique-Based Human-AI Oversight)
---

## Overview

Video–language models learn to reason about dynamic scenes through natural language, yet producing precise video captions remains challenging. **CHAI (Critique-based Human–AI)** is an oversight framework that pairs trained human experts with model-generated pre-captions: experts provide correctional critiques that guide revisions into improved post-captions. This division of labor offloads text generation to models so that humans can focus on verification, improving both accuracy and efficiency.

We release open datasets, benchmarks, and training recipes built on a structured captioning specification covering **subjects, scenes, motion, spatial layout, and camera dynamics**—grounded in hundreds of visual primitives developed with professional filmmakers. The resulting critiques and preferences provide rich supervision for improving open-source VLMs (Qwen3-VL) through SFT, DPO, and inference-time scaling on three tasks: caption generation, reward modeling, and critique generation.

<!-- TODO: Add teaser figure -->
<!-- ![CHAI Overview](assets/teaser.png) -->

---

## Getting Started

### Prerequisites

- Python 3.10+
- Conda (recommended)
- GPU(s) for model inference (tested on NVIDIA A6000)

### Installation

```bash
# Clone the repository
git clone https://github.com/TODO/CHAI.git
cd CHAI

# Create and activate conda environment
conda create -n chai python=3.10 -y
conda activate chai

# Install conda dependencies
conda install -c conda-forge ffmpeg=6.1.2 -y

# Install the package
pip install --no-build-isolation -e .
```

> **Note:** The `--no-build-isolation` flag is required because the `t2v_metrics` dependency uses a legacy `setup.py` without explicit `setuptools` declarations.

### Environment Variables (Optional)

If you want to use the LLM judge for generation evaluation, create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-key
```

This is not required for the default evaluation pipeline, which uses BLEU-4 and ROUGE-L.

### Download Evaluation Data and Videos

The evaluation data and videos are hosted on HuggingFace. See [Evaluation Data](#evaluation-data) below for details on what each file contains.

```bash
# Download the full dataset (videos + evaluation JSONs)
hf download chancharikm/CHAI_testset --repo-type dataset --local-dir ./eval_data
```

This populates `eval_data/` with the test split, task-specific evaluation files, and all corresponding videos.

---

## Evaluation Data

All evaluation files live under `eval_data/`. The raw test split and three task-specific reformatted versions are provided. The corresponding videos and copies of all test split files are hosted on HuggingFace at [chancharikm/CHAI_testset](https://huggingface.co/datasets/chancharikm/CHAI_testset).

### `test_split.json`
The raw evaluation data. Each entry contains a video path, the model-generated **pre-caption**, a human-written **critique**, the revised **final caption** (post-caption), a **pre-caption score** (1–5), the **caption type** (e.g., Subject, Scene, Motion, Spatial, Camera), and associated metadata. This file serves as the source from which all task-specific evaluation sets below are derived.

### `eval_caption_generation_test.json`
Formatted for the **caption generation** task. Each sample pairs a video with a task instruction as the user turn and the final (post) caption as the target assistant response. Used to evaluate a model's ability to directly produce high-quality captions from video.

### `eval_critique_generation_test.json`
Formatted for the **critique generation** task. Each sample provides a video, a task instruction, and a caption to critique as the user turn. The target assistant response is a critique. For pre-captions scoring below 5, two training pairs are generated: one pairing the pre-caption with its human critique, and one pairing the final caption with a "perfect caption" sentinel critique, teaching the model to both identify errors and recognize when a caption needs no revision.

### `eval_caption_yes_or_no_test.json`
Formatted for the **reward modeling** (binary alignment scoring) task. Given a video, a task instruction, and a candidate caption, the model must judge whether the caption aligns with the video by responding **"Yes"** or **"No"**. For pre-captions scoring below 5, two samples are generated: the final caption as a positive example ("Yes") and the pre-caption as a negative example ("No"), providing balanced supervision for learning caption quality.


### Input data format (`--data_file`)

All entry points (`run_unified_evaluations.sh`, `unified_generation.py`,
`parallel_unified_generation.py`) read a **flat JSON list with one record per
(video, caption type) pair** — `test_split.json` is 1000 videos x 5 caption
types = 5000 records. Captions must *not* be nested under a per-video object.

A runnable 5-record example (one video, all five caption types, mixing perfect
and corrected pre-captions) ships at `eval_data/example_data_file.json`:

```bash
python -m eval_code.unified_generation \
    --data_file eval_data/example_data_file.json \
    --formats caption_generation \
    --model qwen3-vl-8b \
    --video_dir eval_data/captioning_videos
```

A minimal record looks like:

```json
[
  {
    "video_id": "4PcpGxihPac.2.4.mp4",
    "caption_type": "camera",
    "pre_caption": "This is a static, low-angle, eye-level shot. [...]",
    "pre_caption_score": 4,
    "critique": "The shot is described as a low-angle, but it appears to be more of a level angle.",
    "final_caption": "This is a static, level-angle, eye-level shot. [...]"
  }
]
```

| Field | Required | Used for |
|---|---|---|
| `video_id` | yes | joined onto `--video_dir` to locate the video, so it must match the filename on disk |
| `caption_type` | yes | selects the task instruction; one of `subject`, `scene`, `motion`, `spatial`, `camera` |
| `final_caption` | yes | target for `caption_generation` / `caption_correction`, the "Yes" caption for reward modeling, and the reference for BLEU-4 / ROUGE-L |
| `pre_caption` | for critique + reward tasks | the caption to critique, and the "No" caption for reward modeling |
| `pre_caption_score` | for critique + reward tasks | integer 1-5; **gates pair construction** (see below) |
| `critique` | for critique tasks | target for `critique_generation`; the given critique in `*_with_given_critique` formats |
| `insertion_error_critique`, `replacement_error_critique` | for `critique_yes_or_no` only | sources of the deliberately-wrong critique negatives |
| `workflow_type`, `annotator_negative_workflow_type` | no | passed through to the results file for slicing |

Three behaviors worth knowing before you swap in your own data:

- **`caption_type` fails silently if absent.** `unified_generation.py:140` falls
  back to `sample.get("caption_type", "camera")`, so a file missing this field
  runs to completion with every sample scored against the *camera* instruction.
  The accepted values are exactly the keys of `eval_code/task_instructions.json`.
- **`pre_caption_score` gates the pair-forming formats.** Records scoring 5 are
  skipped by `caption_yes_or_no` and `critique_yes_or_no` (a perfect pre-caption
  yields no negative), and `formats.py` raises `TypeError` on a missing or
  `null` score. Set it explicitly even when the value is uninteresting.
- **Reward modeling and critique generation need a *distinct* `pre_caption`.**
  Caption-only data (just `video_id` + `final_caption` + `caption_type`) can
  drive `caption_generation` and its BLEU/ROUGE eval, but not the other two
  tasks — set `SCORING_FORMATS=()` and drop `critique_generation` from
  `GENERATION_FORMATS` in that case.

---

## Running Evaluations

The evaluation pipeline supports three tasks: **caption generation**, **critique generation**, and **reward modeling** (caption yes/no scoring). A single bash script orchestrates generation and evaluation across model checkpoints. The script also enables parallel workers per GPU for faster inference.

### Quick Start

```bash
# Run the full pipeline with default settings
bash run_unified_evaluations.sh
```

### Configuration

Edit the top of `run_unified_evaluations.sh` to configure your run:

```bash
# GPU setup
GPUS="0,1,2,3,4,5,6,7"
WORKERS_PER_GPU=2

# Data
DATA_FILE="eval_data/test_split.json"
VIDEO_DIR="eval_data/captioning_videos"

# Models to evaluate (base or base;checkpoint)
MODELS=(
    "qwen3-vl-8b"                                          # base model
    "qwen3-vl-8b;chancharikm/CHAI_SFT_model_8b"            # fine-tuned
)

# Scoring formats (sequential evaluation via unified_eval.py)
SCORING_FORMATS=(
    "caption_yes_or_no"
)

# Generation formats (parallel evaluation)
GENERATION_FORMATS=(
    "caption_generation"
    "critique_generation"
)
```

### Pipeline Steps

For each model, the pipeline runs:

1. **Scoring generation** — computes VQA scores (P(Yes) probability) for each caption using `caption_yes_or_no` format
2. **Scoring evaluation** — calculates pairwise accuracy from the generated scores
3. **Caption/critique generation** — produces captions and critiques for test set videos
4. **Generation evaluation** — evaluates outputs against ground truth using BLEU-4 and ROUGE-L by default, with an optional LLM judge as an alternative (requires `OPENAI_API_KEY` in `.env` and `USE_LLM_JUDGE="true"` in the script)

### Output Structure

```
evaluation_outputs/
├── inference/
│   ├── scoring_<model>_<timestamp>.json
│   └── generation_<model>_<timestamp>.json
└── evaluation/
    ├── scoring_eval_<model>_<timestamp>.json
    └── generation_eval_<model>_<timestamp>.json
```

### Project Structure

```
CHAI/
├── assets/                          # Banner, logo
├── eval_code/                       # Evaluation modules
│   ├── __init__.py
│   ├── constants.py                 # Task constants and format definitions
│   ├── formats.py                   # Format conversion utilities
│   ├── parallel_unified_eval.py     # Multi-worker evaluation
│   ├── parallel_unified_generation.py  # Multi-GPU generation
│   ├── unified_eval.py              # Single-process evaluation
│   ├── unified_generation.py        # Single-process generation
│   └── video_caption_api.py         # Video captioning API
├── eval_data/                       # Evaluation data and videos
│   ├── test_split.json
│   ├── eval_caption_generation_test.json
│   ├── eval_critique_generation_test.json
│   ├── eval_caption_yes_or_no_test.json
│   ├── example_data_file.json       # Minimal --data_file example (5 records)
│   └── captioning_videos/
├── pyproject.toml
├── run_unified_evaluations.sh       # Main evaluation entry point
└── README.md
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{chai2026,
  title     = {Building a Precise Video Language with Human--AI Oversight},
  author    = {Zhiqiu Lin and Chancharik Mitra and Siyuan Cen and Isaac Li and Yuhan Huang and Yu Tong Tiffany Ling and Hewei Wang and Irene Pi and Shihang Zhu and Ryan Rao and George Liu and Jiaxi Li and Ruojin Li and Yili Han and Yilun Du and Deva Ramanan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---
## 📢 Collaborations & Contact

We are actively advancing CHAI with larger-scale datasets and stronger video
understanding models. We welcome collaborations and funding opportunities with
researchers and practitioners working on video understanding, captioning, and
multimodal agents for professional-level video content.

If you're interested in accessing improved data or models, please reach out:

- Zhiqiu Lin — [zhiqiulin98@gmail.com](mailto:zhiqiulin98@gmail.com)
- Chancharik Mitra — [cmitra@andrew.cmu.edu](mailto:cmitra@andrew.cmu.edu)

Or [open a GitHub Issue](../../issues).

---

## Acknowledgments

This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE2140739. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

<!-- TODO: Additional funding, collaborators, professional video creators -->

## License

<!-- TODO: Specify license -->