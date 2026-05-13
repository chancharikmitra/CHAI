---
task_categories:
- video-text-to-text
---

# Building a Precise Video Language with Human-AI Oversight

This repository contains the evaluation data for **CHAI (Critique-based Human–AI)**, an oversight framework for precise video captioning.

[**Paper**](https://huggingface.co/papers/2604.21718) | [**Project Page**](https://linzhiqiu.github.io/papers/chai/) | [**GitHub**](https://github.com/chancharikmitra/CHAI)

## Overview

CHAI is an oversight framework that pairs trained human experts with model-generated pre-captions. Experts provide correctional critiques that guide revisions into improved post-captions. This dataset includes open benchmarks and evaluation data built on a structured captioning specification covering subjects, scenes, motion, spatial layout, and camera dynamics—grounded in visual primitives developed with professional filmmakers.

## Dataset Structure

The dataset consists of the following evaluation files:

*   **`test_split.json`**: The raw evaluation data. Each entry contains a video path, the model-generated pre-caption, a human-written critique, the revised final caption (post-caption), and associated metadata.
*   **`eval_caption_generation_test.json`**: Formatted for the caption generation task. Pairs a video with a task instruction and the final (post) caption.
*   **`eval_critique_generation_test.json`**: Formatted for the critique generation task. Pairs a video, a task instruction, and a caption to critique.
*   **`eval_caption_yes_or_no_test.json`**: Formatted for the reward modeling (binary alignment scoring) task.

## Sample Usage

To download the full dataset (including videos and evaluation JSONs) using the Hugging Face CLI:

```bash
# Install the HuggingFace CLI (if not already installed)
pip install huggingface_hub

# Download the full dataset (includes videos + evaluation JSONs)
hf download datasets/chancharikm/CHAI_testset --local-dir ./CHAI_testset
```

## Citation

```bibtex
@inproceedings{chai2026,
  title     = {Building a Precise Video Language with Human--AI Oversight},
  author    = {Zhiqiu Lin and Chancharik Mitra and Siyuan Cen and Isaac Li and Yuhan Huang and Yu Tong Tiffany Ling and Hewei Wang and Irene Pi and Shihang Zhu and Ryan Rao and George Liu and Jiaxi Li and Ruojin Li and Yili Han and Yilun Du and Deva Ramanan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```