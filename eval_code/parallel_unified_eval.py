#!/usr/bin/env python3
"""
Parallel evaluation launcher for unified_eval.py

Splits generation results across workers for faster evaluation.

Examples:
  # Evaluate with 8 workers
  python parallel_unified_eval.py \
    --results_file generation_outputs/multiformat_*.json \
    --num_workers 8

  # With LLM judge (rate-limited, use fewer workers)
  python parallel_unified_eval.py \
    --results_file generation_outputs/caption_generation_*.json \
    --num_workers 4 \
    --api_key YOUR_KEY

  # Skip LLM judge for faster eval
  python parallel_unified_eval.py \
    --results_file generation_outputs/multiformat_*.json \
    --num_workers 8 \
    --no_llm_judge
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import numpy as np
from collections import defaultdict


def load_generation_results(results_file: str) -> Dict[str, Any]:
    """Load generation results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_results_by_format(generation_data: Dict[str, Any], num_chunks: int) -> List[Dict[str, Any]]:
    """
    Split generation results into chunks for parallel processing.
    
    Each chunk gets a portion of the results for each format.
    """
    results_by_format = generation_data.get('results_by_format', {})
    
    if num_chunks <= 1:
        return [generation_data]
    
    chunks = []
    
    for i in range(num_chunks):
        chunk = {
            **generation_data,  # Copy metadata
            'results_by_format': {}
        }
        
        for format_name, format_data in results_by_format.items():
            results = format_data.get('results', [])
            
            # Split results for this format
            chunk_size = len(results) // num_chunks
            remainder = len(results) % num_chunks
            
            start = i * chunk_size + min(i, remainder)
            end = start + chunk_size + (1 if i < remainder else 0)
            
            chunk_results = results[start:end]
            
            # Create format entry for this chunk
            chunk['results_by_format'][format_name] = {
                'summary': {
                    **format_data['summary'],
                    'total': len(chunk_results),
                    'successful': len([r for r in chunk_results if r.get('success')]),
                    'failed': len([r for r in chunk_results if not r.get('success')]),
                },
                'results': chunk_results
            }
        
        chunks.append(chunk)
    
    return chunks


def save_chunk(chunk: Dict[str, Any], path: str):
    """Save a chunk to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, indent=2, ensure_ascii=False)


def weighted_average(values_and_weights: List[tuple]) -> float:
    """Calculate weighted average from (value, weight) tuples."""
    if not values_and_weights:
        return 0.0
    
    total_weight = sum(w for _, w in values_and_weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(v * w for v, w in values_and_weights)
    return weighted_sum / total_weight


def merge_evaluation_results(result_files: List[str], generation_metadata: Dict) -> Dict[str, Any]:
    """
    Merge evaluation results from multiple workers.
    
    Computes weighted averages for metrics based on sample counts.
    """
    if not result_files:
        return {}
    
    # Collect metrics by format
    metrics_by_format = defaultdict(lambda: {
        'samples': [],
        'metrics': defaultdict(list),
        'by_caption_type': defaultdict(lambda: defaultdict(list))
    })
    
    formats_evaluated = set()
    llm_judge_enabled = False
    
    for rf in result_files:
        if not os.path.exists(rf):
            print(f"  Warning: Result file not found: {rf}")
            continue
        
        with open(rf, 'r', encoding='utf-8') as f:
            worker_data = json.load(f)
        
        formats_evaluated.update(worker_data.get('formats_evaluated', []))
        llm_judge_enabled = llm_judge_enabled or worker_data.get('llm_judge_enabled', False)
        
        eval_results = worker_data.get('evaluation_results', {})
        
        for format_name, format_metrics in eval_results.items():
            if 'error' in format_metrics:
                continue
            
            format_type = format_metrics.get('format_type', 'unknown')
            num_successful = format_metrics.get('num_successful', 0)
            
            # Store sample count
            metrics_by_format[format_name]['samples'].append(num_successful)
            
            # Collect metrics based on format type
            if format_type in ['caption_yes_or_no', 'critique_yes_or_no']:
                # Yes/No formats - collect accuracy metrics
                metrics_by_format[format_name]['metrics']['pairwise_accuracy'].append(
                    (format_metrics.get('pairwise_accuracy', 0), num_successful)
                )
                metrics_by_format[format_name]['metrics']['individual_accuracy'].append(
                    (format_metrics.get('individual_accuracy', 0), num_successful)
                )
                
                # Collect score statistics for averaging
                score_stats = format_metrics.get('score_statistics', {})
                for stat_name in ['mean', 'std', 'min', 'max', 'median']:
                    if stat_name in score_stats:
                        metrics_by_format[format_name]['metrics'][f'score_{stat_name}'].append(
                            (score_stats[stat_name], num_successful)
                        )
                
                # For critique format, collect per-type stats
                if format_type == 'critique_yes_or_no':
                    per_type = format_metrics.get('per_type_statistics', {})
                    for pair_type, type_stats in per_type.items():
                        count = type_stats.get('count', 0)
                        for metric in ['mean', 'std', 'min', 'max']:
                            if metric in type_stats:
                                metrics_by_format[format_name]['by_caption_type'][pair_type][metric].append(
                                    (type_stats[metric], count)
                                )
            
            else:
                # Generation formats - collect BLEU, ROUGE, LLM judge
                overall = format_metrics.get('overall_metrics', {})
                for metric_name in ['bleu4', 'rouge_l', 'llm_judge']:
                    if metric_name in overall and overall[metric_name] is not None:
                        metrics_by_format[format_name]['metrics'][metric_name].append(
                            (overall[metric_name], num_successful)
                        )
                
                # Collect per-caption-type metrics
                by_type = format_metrics.get('by_caption_type', {})
                for caption_type, type_metrics in by_type.items():
                    count = type_metrics.get('count', 0)
                    for metric in ['bleu4', 'rouge_l', 'llm_judge']:
                        if metric in type_metrics and type_metrics[metric] is not None:
                            metrics_by_format[format_name]['by_caption_type'][caption_type][metric].append(
                                (type_metrics[metric], count)
                            )
                
                # Collect generation statistics
                gen_stats = format_metrics.get('generation_statistics', {})
                for stat_name in ['mean_length_words', 'std_length_words']:
                    if stat_name in gen_stats:
                        metrics_by_format[format_name]['metrics'][stat_name].append(
                            (gen_stats[stat_name], num_successful)
                        )
    
    # Compute merged metrics
    merged_results = {}
    
    for format_name, format_data in metrics_by_format.items():
        total_samples = sum(format_data['samples'])
        
        # Compute overall metrics
        overall_metrics = {}
        for metric_name, values in format_data['metrics'].items():
            if values:
                overall_metrics[metric_name] = weighted_average(values)
        
        # Compute per-type metrics
        by_type_metrics = {}
        for caption_type, type_metrics in format_data['by_caption_type'].items():
            type_result = {}
            for metric_name, values in type_metrics.items():
                if values:
                    type_result[metric_name] = weighted_average(values)
            
            # Count total samples for this type
            type_counts = [w for _, w in type_metrics.get(list(type_metrics.keys())[0], [])]
            type_result['count'] = sum(type_counts) if type_counts else 0
            
            by_type_metrics[caption_type] = type_result
        
        # Build result based on format type
        if 'pairwise_accuracy' in overall_metrics:
            # Yes/No format
            merged_results[format_name] = {
                'format_type': format_name,
                'pairwise_accuracy': overall_metrics.get('pairwise_accuracy', 0.0),
                'individual_accuracy': overall_metrics.get('individual_accuracy', 0.0),
                'score_statistics': {
                    'mean': overall_metrics.get('score_mean', 0.0),
                    'std': overall_metrics.get('score_std', 0.0),
                    'min': overall_metrics.get('score_min', 0.0),
                    'max': overall_metrics.get('score_max', 0.0),
                    'median': overall_metrics.get('score_median', 0.0),
                },
                'num_successful': total_samples,
            }
            
            if by_type_metrics:
                merged_results[format_name]['per_type_statistics'] = by_type_metrics
        
        else:
            # Generation format
            merged_results[format_name] = {
                'format_type': format_name,
                'overall_metrics': {
                    'bleu4': overall_metrics.get('bleu4', 0.0),
                    'rouge_l': overall_metrics.get('rouge_l', 0.0),
                    'llm_judge': overall_metrics.get('llm_judge'),
                },
                'by_caption_type': by_type_metrics,
                'generation_statistics': {
                    'mean_length_words': overall_metrics.get('mean_length_words', 0.0),
                    'std_length_words': overall_metrics.get('std_length_words', 0.0),
                },
                'num_successful': total_samples,
            }
    
    return {
        'evaluation_timestamp': datetime.now().isoformat(),
        'generation_metadata': generation_metadata,
        'formats_evaluated': sorted(formats_evaluated),
        'llm_judge_enabled': llm_judge_enabled,
        'parallel_execution': {
            'num_workers': len(result_files),
            'merged_at': datetime.now().isoformat()
        },
        'evaluation_results': merged_results
    }


class WorkerProcess:
    """Track a worker subprocess."""
    
    def __init__(self, worker_id: int, process: subprocess.Popen,
                 input_file: str, output_file: str, log_file: str):
        self.worker_id = worker_id
        self.process = process
        self.input_file = input_file
        self.output_file = output_file
        self.log_file = log_file
    
    def is_running(self) -> bool:
        return self.process.poll() is None


def launch_eval_worker(worker_id: int, input_file: str, output_file: str,
                       log_file: str, args) -> WorkerProcess:
    """Launch evaluation worker process."""
    
    cmd = [
        sys.executable, "-u", "-m", "eval_code.unified_eval",
        "--results_file", input_file,
        "--output_file", output_file,
    ]
    
    if args.api_key:
        cmd.extend(["--api_key", args.api_key])
    
    if args.no_llm_judge:
        cmd.append("--no_llm_judge")
    
    if args.formats:
        cmd.extend(["--formats"] + args.formats)
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    log_handle = open(log_file, 'w', buffering=1)
    
    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(Path(__file__).parent)
    )
    
    return WorkerProcess(worker_id, process, input_file, output_file, log_file)


def main():
    parser = argparse.ArgumentParser(
        description='Parallel evaluation launcher for unified_eval.py',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--results_file', required=True,
                        help='Generation results JSON file')
    
    # Parallelization
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    
    # Output
    parser.add_argument('--output_file', default=None,
                        help='Output JSON file for results')
    parser.add_argument('--output_dir', default='evaluation_results',
                        help='Output directory (default: evaluation_results)')
    
    # Format selection
    parser.add_argument('--formats', type=str, nargs='+', default=None,
                        help='Specific formats to evaluate (default: all)')
    
    # LLM judge options
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key for LLM-as-a-Judge')
    parser.add_argument('--no_llm_judge', action='store_true',
                        help='Skip LLM judge evaluation')
    
    # Misc
    parser.add_argument('--keep_temp', action='store_true',
                        help='Keep temporary files')
    parser.add_argument('--log_dir', default='logs',
                        help='Directory for worker logs')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PARALLEL UNIFIED EVALUATION")
    print("=" * 60)
    print(f"Results file: {args.results_file}")
    print(f"Workers: {args.num_workers}")
    print(f"LLM judge: {'disabled' if args.no_llm_judge else 'enabled' if args.api_key else 'no API key'}")
    print()
    
    # Load generation results
    print("Loading generation results...")
    generation_data = load_generation_results(args.results_file)
    
    generation_metadata = {
        'model_name': generation_data.get('model_name'),
        'checkpoint': generation_data.get('checkpoint'),
        'num_samples': generation_data.get('num_samples'),
        'generation_timestamp': generation_data.get('generation_timestamp'),
        'generation_file': args.results_file,
    }
    
    print(f"  Model: {generation_metadata['model_name']}")
    print(f"  Checkpoint: {generation_metadata.get('checkpoint', 'None')}")
    print(f"  Samples: {generation_metadata['num_samples']}")
    
    formats = list(generation_data.get('results_by_format', {}).keys())
    print(f"  Formats: {formats}")
    
    # Determine actual number of workers
    total_results = sum(
        len(generation_data['results_by_format'][fmt]['results'])
        for fmt in formats
    )
    actual_workers = min(args.num_workers, total_results // 10 + 1)  # At least 10 results per worker
    actual_workers = max(1, actual_workers)
    
    print(f"\nUsing {actual_workers} workers")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="parallel_unified_eval_")
    print(f"Temp directory: {temp_dir}")
    
    # Split results into chunks
    print(f"\nSplitting results into {actual_workers} chunks...")
    chunks = split_results_by_format(generation_data, actual_workers)
    
    # Launch workers
    workers: List[WorkerProcess] = []
    
    print(f"\nLaunching {actual_workers} workers...")
    
    for i, chunk in enumerate(chunks):
        if not chunk.get('results_by_format'):
            continue
        
        # Save chunk
        chunk_file = os.path.join(temp_dir, f"eval_chunk_{i:03d}.json")
        save_chunk(chunk, chunk_file)
        
        # Output file
        output_file = os.path.join(temp_dir, f"eval_result_{i:03d}.json")
        
        # Log file
        log_file = os.path.join(args.log_dir, f"eval_worker_{i:03d}.log")
        
        # Launch worker
        worker = launch_eval_worker(i, chunk_file, output_file, log_file, args)
        workers.append(worker)
        
        # Count samples in chunk
        chunk_samples = sum(
            len(chunk['results_by_format'][fmt]['results'])
            for fmt in chunk['results_by_format']
        )
        print(f"  Worker {i}: {chunk_samples} results, PID {worker.process.pid}")
    
    print("\nWaiting for workers...")
    
    # Wait for workers
    start_time = time.time()
    
    try:
        while True:
            running = [w for w in workers if w.is_running()]
            if not running:
                break
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted! Terminating workers...")
        for w in workers:
            if w.is_running():
                w.process.terminate()
        
        for w in workers:
            try:
                w.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                w.process.kill()
        
        return 1
    
    elapsed = time.time() - start_time
    
    # Check results
    print("\n" + "=" * 60)
    print("WORKER RESULTS")
    print("=" * 60)
    
    success_count = sum(1 for w in workers if w.process.returncode == 0)
    fail_count = len(workers) - success_count
    
    for w in workers:
        rc = w.process.returncode
        status = "✓" if rc == 0 else f"✗ (exit {rc})"
        print(f"  Worker {w.worker_id}: {status}")
        
        if rc != 0:
            print(f"    Log: {w.log_file}")
    
    if fail_count > 0:
        print(f"\nWarning: {fail_count} workers failed")
    
    # Merge results
    print("\n" + "=" * 60)
    print("MERGING RESULTS")
    print("=" * 60)
    
    result_files = [w.output_file for w in workers if w.process.returncode == 0]
    
    if not result_files:
        print("Error: No successful workers")
        return 1
    
    merged = merge_evaluation_results(result_files, generation_metadata)
    
    # Determine output file
    if args.output_file:
        output_path = args.output_file
    else:
        input_path = Path(args.results_file)
        input_stem = input_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{input_stem}_{timestamp}.json"
        output_path = str(Path(args.output_dir) / filename)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    print(f"  Merged results from {len(result_files)} workers")
    print(f"  Output: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for format_name, metrics in merged['evaluation_results'].items():
        print(f"\n{format_name}:")
        
        if 'pairwise_accuracy' in metrics:
            print(f"  Pairwise Accuracy: {metrics['pairwise_accuracy']:.4f}")
            print(f"  Individual Accuracy: {metrics['individual_accuracy']:.4f}")
        else:
            overall = metrics.get('overall_metrics', {})
            print(f"  BLEU-4: {overall.get('bleu4', 0):.4f}")
            print(f"  ROUGE-L: {overall.get('rouge_l', 0):.4f}")
            if overall.get('llm_judge') is not None:
                print(f"  LLM Judge: {overall.get('llm_judge', 0):.4f}")
        
        print(f"  Samples: {metrics.get('num_successful', 0)}")
    
    # Cleanup
    if not args.keep_temp:
        shutil.rmtree(temp_dir)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())