#!/bin/bash
# run_unified_evaluations.sh
# Evaluation pipeline for caption_yes_or_no, caption_generation, and critique_generation
# Runs generation + evaluation for multiple model checkpoints

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - Modify these as needed
# ============================================================================

# Sample limit
SAMPLE_SIZE=5000

# GPU configuration
GPUS="0,1,2,3,4,5,6,7"
WORKERS_PER_GPU=2

GENERATION_TEMPERATURE=0.0
SCORING_TEMPERATURE=1.0

# Data file
DATA_FILE="eval_data/test_split.json"
VIDEO_DIR="eval_data/captioning_videos"

# Output directory
OUTPUT_DIR="./evaluation_outputs"

# TMPDIR
export TMPDIR="./.tmp_generation"
mkdir -p "$TMPDIR"

# Model configurations
# Format: "base_model" for base only, or "base_model;checkpoint" for finetuned
MODELS=(
    "qwen3-vl-8b"
    "qwen3-vl-8b;chancharikm/CHAI_SFT_model_8b"
)

# Scoring formats (use --mode score)
SCORING_FORMATS=(
    "caption_yes_or_no"
)

# Generation formats (use --mode generate)
GENERATION_FORMATS=(
    "caption_generation"
    "critique_generation"
)

# Evaluation workers (CPU-bound, used for generation eval only)
EVAL_WORKERS=8

# LLM judge (set to "true" to enable, "false" to disable)
USE_LLM_JUDGE="false"
OPENAI_API_KEY=""  # Set if USE_LLM_JUDGE="true"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $1"
}

# Build a unique, descriptive run name from model spec
build_model_tag() {
    local MODEL=$1
    local CHECKPOINT=$2

    local BASE_TAG
    BASE_TAG=$(echo "${MODEL}" | tr '/' '_' | tr ':' '_')

    if [ -n "${CHECKPOINT}" ]; then
        local CKPT_BASENAME
        CKPT_BASENAME=$(basename "${CHECKPOINT}" | sed 's/[^a-zA-Z0-9_-]/_/g')
        echo "${BASE_TAG}_ft-${CKPT_BASENAME}"
    else
        echo "${BASE_TAG}_base"
    fi
}

run_generation() {
    local MODEL=$1
    local CHECKPOINT=$2
    local FORMATS=$3
    local MODE=$4
    local OUTPUT_FILE=$5
    local LOG_DIR=$6

    # Pick temperature based on mode
    local TEMP
    if [ "${MODE}" = "score" ]; then
        TEMP="${SCORING_TEMPERATURE}"
    else
        TEMP="${GENERATION_TEMPERATURE}"
    fi

    log "Running generation: ${MODE} mode, formats: ${FORMATS} (temp=${TEMP})"
    log "  Model: ${MODEL}"

    local CMD="python -m eval_code.parallel_unified_generation \
        --data_file ${DATA_FILE} \
        --formats ${FORMATS} \
        --model ${MODEL} \
        --mode ${MODE} \
        --video_dir ${VIDEO_DIR} \
        --gpus ${GPUS} \
        --workers_per_gpu ${WORKERS_PER_GPU} \
        --output_file ${OUTPUT_FILE} \
        --log_dir ${LOG_DIR} \
        --sample_size ${SAMPLE_SIZE} \
        --temperature ${TEMP}"

    if [ -n "${CHECKPOINT}" ]; then
        log "  Checkpoint: ${CHECKPOINT}"
        CMD="${CMD} --checkpoint ${CHECKPOINT}"
    fi

    eval ${CMD}

    if [ $? -eq 0 ]; then
        log "✓ Generation complete: ${OUTPUT_FILE}"
    else
        log "✗ Generation failed"
        return 1
    fi
}

# Scoring eval: runs unified_eval.py directly (single process, sequential)
run_scoring_evaluation() {
    local INPUT_FILE=$1
    local OUTPUT_FILE=$2
    local LOG_DIR=$3

    log "Running scoring evaluation (sequential)"
    log "  Input: ${INPUT_FILE}"

    mkdir -p "${LOG_DIR}"

    python -m eval_code.unified_eval \
        --results_file "${INPUT_FILE}" \
        --output_file "${OUTPUT_FILE}" \
        --no_llm_judge \
        2>&1 | tee "${LOG_DIR}/scoring_eval.log"

    if [ $? -eq 0 ]; then
        log "✓ Scoring evaluation complete: ${OUTPUT_FILE}"
    else
        log "✗ Scoring evaluation failed"
        return 1
    fi
}

# Generation eval: runs parallel_unified_eval.py (multi-worker)
run_generation_evaluation() {
    local INPUT_FILE=$1
    local OUTPUT_FILE=$2
    local LOG_DIR=$3

    log "Running generation evaluation (parallel, ${EVAL_WORKERS} workers)"
    log "  Input: ${INPUT_FILE}"

    if [ "${USE_LLM_JUDGE}" = "true" ] && [ -n "${OPENAI_API_KEY}" ]; then
        python -m eval_code.parallel_unified_eval \
            --results_file "${INPUT_FILE}" \
            --num_workers "${EVAL_WORKERS}" \
            --output_file "${OUTPUT_FILE}" \
            --api_key "${OPENAI_API_KEY}" \
            --log_dir "${LOG_DIR}"
    else
        python -m eval_code.parallel_unified_eval \
            --results_file "${INPUT_FILE}" \
            --num_workers "${EVAL_WORKERS}" \
            --output_file "${OUTPUT_FILE}" \
            --no_llm_judge \
            --log_dir "${LOG_DIR}"
    fi

    if [ $? -eq 0 ]; then
        log "✓ Generation evaluation complete: ${OUTPUT_FILE}"
    else
        log "✗ Generation evaluation failed"
        return 1
    fi
}

# Run full pipeline for one model configuration
run_model_pipeline() {
    local MODEL=$1
    local CHECKPOINT=$2
    local MODEL_TAG=$3
    local RUN_TIMESTAMP=$4

    CURRENT_MODEL=$((CURRENT_MODEL + 1))

    log ""
    log "========================================"
    log "MODEL ${CURRENT_MODEL}/${TOTAL_MODELS}: ${MODEL_TAG}"
    log "========================================"

    local DATA_STEM
    DATA_STEM=$(basename "${DATA_FILE}" .json)
    local NAME_PREFIX="${DATA_STEM}_${MODEL_TAG}_${RUN_TIMESTAMP}"

    # --- SCORING FORMATS (caption_yes_or_no) ---

    if [ ${#SCORING_FORMATS[@]} -gt 0 ]; then
        log ""
        log "--- Scoring Formats ---"

        SCORING_OUTPUT="${OUTPUT_DIR}/inference/scoring_${NAME_PREFIX}.json"
        SCORING_EVAL="${OUTPUT_DIR}/evaluation/scoring_eval_${NAME_PREFIX}.json"

        run_generation \
            "${MODEL}" \
            "${CHECKPOINT}" \
            "${SCORING_FORMATS[*]}" \
            "score" \
            "${SCORING_OUTPUT}" \
            "logs/scoring_gen_${NAME_PREFIX}"

        if [ -f "${SCORING_OUTPUT}" ]; then
            run_scoring_evaluation \
                "${SCORING_OUTPUT}" \
                "${SCORING_EVAL}" \
                "logs/scoring_eval_${NAME_PREFIX}"
        fi
    fi

    # --- GENERATION FORMATS (caption_generation, critique_generation) ---

    if [ ${#GENERATION_FORMATS[@]} -gt 0 ]; then
        log ""
        log "--- Generation Formats ---"

        GENERATION_OUTPUT="${OUTPUT_DIR}/inference/generation_${NAME_PREFIX}.json"
        GENERATION_EVAL="${OUTPUT_DIR}/evaluation/generation_eval_${NAME_PREFIX}.json"

        run_generation \
            "${MODEL}" \
            "${CHECKPOINT}" \
            "${GENERATION_FORMATS[*]}" \
            "generate" \
            "${GENERATION_OUTPUT}" \
            "logs/generation_gen_${NAME_PREFIX}"

        if [ -f "${GENERATION_OUTPUT}" ]; then
            run_generation_evaluation \
                "${GENERATION_OUTPUT}" \
                "${GENERATION_EVAL}" \
                "logs/generation_eval_${NAME_PREFIX}"
        fi
    fi

    log ""
    log "✓ Model ${MODEL_TAG} complete"
}

# ============================================================================
# SETUP
# ============================================================================

RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

log "========================================"
log "FULL UNIFIED EVALUATION PIPELINE"
log "========================================"
log "Run timestamp: ${RUN_TIMESTAMP}"
log "GPUs: ${GPUS}"
log "Workers per GPU: ${WORKERS_PER_GPU}"
log "Video dir: ${VIDEO_DIR}"
log "Output dir: ${OUTPUT_DIR}"
log "Data file: ${DATA_FILE}"
log ""
log "Scoring formats: ${SCORING_FORMATS[@]}"
log "Generation formats: ${GENERATION_FORMATS[@]}"
log ""
log "Models: ${#MODELS[@]}"
for model_spec in "${MODELS[@]}"; do
    log "  - ${model_spec}"
done
log ""
log "LLM Judge: ${USE_LLM_JUDGE}"
log "Generation temperature: ${GENERATION_TEMPERATURE}"
log "Scoring temperature: ${SCORING_TEMPERATURE}"
log "Scoring eval: sequential (unified_eval.py)"
log "Generation eval: parallel (${EVAL_WORKERS} workers)"
log "========================================"
log ""

# Create output directories
mkdir -p "${OUTPUT_DIR}/inference"
mkdir -p "${OUTPUT_DIR}/evaluation"
mkdir -p logs

# ============================================================================
# RUN PIPELINE FOR EACH MODEL
# ============================================================================

TOTAL_MODELS=${#MODELS[@]}
CURRENT_MODEL=0

for model_spec in "${MODELS[@]}"; do
    IFS=';' read -r MODEL CHECKPOINT <<< "${model_spec}"
    MODEL_TAG=$(build_model_tag "${MODEL}" "${CHECKPOINT}")
    run_model_pipeline "${MODEL}" "${CHECKPOINT}" "${MODEL_TAG}" "${RUN_TIMESTAMP}"
done

# ============================================================================
# SUMMARY
# ============================================================================

log ""
log "========================================"
log "PIPELINE COMPLETE"
log "========================================"
log ""
log "Output structure:"
log "  ${OUTPUT_DIR}/"
log "  ├── inference/"

for model_spec in "${MODELS[@]}"; do
    IFS=';' read -r MODEL CHECKPOINT <<< "${model_spec}"
    MODEL_TAG=$(build_model_tag "${MODEL}" "${CHECKPOINT}")
    NAME_PREFIX="${MODEL_TAG}_${RUN_TIMESTAMP}"

    if [ ${#SCORING_FORMATS[@]} -gt 0 ]; then
        log "  │   ├── scoring_${NAME_PREFIX}.json"
    fi
    if [ ${#GENERATION_FORMATS[@]} -gt 0 ]; then
        log "  │   ├── generation_${NAME_PREFIX}.json"
    fi
done

log "  └── evaluation/"

for model_spec in "${MODELS[@]}"; do
    IFS=';' read -r MODEL CHECKPOINT <<< "${model_spec}"
    MODEL_TAG=$(build_model_tag "${MODEL}" "${CHECKPOINT}")
    NAME_PREFIX="${MODEL_TAG}_${RUN_TIMESTAMP}"

    if [ ${#SCORING_FORMATS[@]} -gt 0 ]; then
        log "      ├── scoring_eval_${NAME_PREFIX}.json"
    fi
    if [ ${#GENERATION_FORMATS[@]} -gt 0 ]; then
        log "      ├── generation_eval_${NAME_PREFIX}.json"
    fi
done

log ""
log "Logs: ./logs/"
log ""

# ============================================================================
# QUICK COMPARISON
# ============================================================================

log "========================================"
log "QUICK COMPARISON"
log "========================================"
log ""

show_metrics() {
    local EVAL_FILE=$1
    local MODEL_TAG=$2

    if [ ! -f "${EVAL_FILE}" ]; then
        log "  ${MODEL_TAG}: No evaluation file found"
        return
    fi

    python3 -c "
import json, sys
with open('${EVAL_FILE}') as f:
    data = json.load(f)
results = data.get('evaluation_results', {})
for fmt, metrics in results.items():
    if 'pairwise_accuracy' in metrics:
        print(f'  ${MODEL_TAG} | {fmt}: pairwise_acc={metrics[\"pairwise_accuracy\"]:.4f}')
    elif 'overall_metrics' in metrics:
        om = metrics['overall_metrics']
        parts = [f'bleu4={om.get(\"bleu4\", 0):.4f}', f'rouge_l={om.get(\"rouge_l\", 0):.4f}']
        if om.get('llm_judge') is not None:
            parts.append(f'llm_judge={om[\"llm_judge\"]:.4f}')
        print(f'  ${MODEL_TAG} | {fmt}: {\", \".join(parts)}')
" 2>/dev/null || log "  ${MODEL_TAG}: Could not parse ${EVAL_FILE}"
}

for model_spec in "${MODELS[@]}"; do
    IFS=';' read -r MODEL CHECKPOINT <<< "${model_spec}"
    MODEL_TAG=$(build_model_tag "${MODEL}" "${CHECKPOINT}")
    NAME_PREFIX="${MODEL_TAG}_${RUN_TIMESTAMP}"

    if [ ${#SCORING_FORMATS[@]} -gt 0 ]; then
        show_metrics "${OUTPUT_DIR}/evaluation/scoring_eval_${NAME_PREFIX}.json" "${MODEL_TAG}"
    fi
    if [ ${#GENERATION_FORMATS[@]} -gt 0 ]; then
        show_metrics "${OUTPUT_DIR}/evaluation/generation_eval_${NAME_PREFIX}.json" "${MODEL_TAG}"
    fi
done

log ""