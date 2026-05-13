#!/usr/bin/env python3
"""
Parallel generation launcher for unified_generation.py

Splits input data across GPUs and runs workers in parallel.

Examples:
  # Generate with multiple formats across 4 GPUs
  python parallel_unified_generation.py \
    --data_file data/test.json \
    --formats caption_yes_or_no critique_yes_or_no \
    --model qwen3-vl-8b \
    --mode score \
    --gpus 0,1,2,3

  # All formats with 2 workers per GPU
  python parallel_unified_generation.py \
    --data_file data/test.json \
    --all_formats \
    --model qwen3-vl-8b \
    --gpus 0,1,2,3,4,5,6,7 \
    --workers_per_gpu 2

  # With checkpoint
  python parallel_unified_generation.py \
    --data_file data/test.json \
    --formats caption_generation \
    --model qwen3-vl-8b \
    --checkpoint /path/to/checkpoint \
    --mode generate \
    --gpus 0,1
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


def parse_gpu_list(gpu_string: str) -> List[int]:
    """Parse comma-separated GPU list."""
    if not gpu_string:
        return []
    return [int(g.strip()) for g in gpu_string.split(',')]


def load_data(data_file: str) -> List[Dict[str, Any]]:
    """Load generation data from JSON file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_data(data: List[Dict[str, Any]], num_chunks: int) -> List[List[Dict[str, Any]]]:
    """Split data into roughly equal chunks."""
    if num_chunks <= 0:
        return [data]
    
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    
    chunks = []
    start = 0
    
    for i in range(num_chunks):
        # Distribute remainder across first chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start:end])
        start = end
    
    return chunks


def save_chunk(chunk: List[Dict[str, Any]], path: str):
    """Save a chunk to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, indent=2, ensure_ascii=False)


def merge_results(result_files: List[str], output_file: str):
    """
    Merge generation results from multiple workers.
    
    Combines results by format, preserving all metadata and summaries.
    """
    if not result_files:
        print("No result files to merge")
        return 0
    
    # Load first file to get structure
    with open(result_files[0], 'r', encoding='utf-8') as f:
        merged = json.load(f)
    
    # Get formats from first file
    formats = list(merged.get('results_by_format', {}).keys())
    
    # Initialize combined results
    for format_name in formats:
        merged['results_by_format'][format_name]['results'] = []
        merged['results_by_format'][format_name]['summary']['total'] = 0
        merged['results_by_format'][format_name]['summary']['successful'] = 0
        merged['results_by_format'][format_name]['summary']['failed'] = 0
        merged['results_by_format'][format_name]['summary']['successful_indices'] = []
        merged['results_by_format'][format_name]['summary']['failed_indices'] = []
        
        # Reset counters for Yes/No formats
        summary = merged['results_by_format'][format_name]['summary']
        if summary.get('is_critique_format'):
            summary['positive_final_perfect_count'] = 0
            summary['positive_pre_constructive_count'] = 0
            summary['negative_pre_perfect_count'] = 0
            summary['negative_pre_inaccurate_count'] = 0
            summary['complete_sets_count'] = 0
            summary['skipped_samples'] = 0
            summary['original_samples_processed'] = 0
        elif summary.get('is_yesno_format'):
            summary['positive_count'] = 0
            summary['negative_count'] = 0
            summary['pairs_count'] = 0
            summary['skipped_samples'] = 0
            summary['original_samples_processed'] = 0
    
    # Merge all worker results
    sample_offset = 0
    
    for rf in result_files:
        if not os.path.exists(rf):
            print(f"  Warning: Result file not found: {rf}")
            continue
        
        with open(rf, 'r', encoding='utf-8') as f:
            worker_data = json.load(f)
        
        for format_name in formats:
            if format_name not in worker_data.get('results_by_format', {}):
                continue
            
            worker_format = worker_data['results_by_format'][format_name]
            merged_format = merged['results_by_format'][format_name]
            
            # Adjust sample indices
            for result in worker_format['results']:
                result['sample_index'] += sample_offset
            
            # Append results
            merged_format['results'].extend(worker_format['results'])
            
            # Update summary counts
            worker_summary = worker_format['summary']
            merged_summary = merged_format['summary']
            
            merged_summary['total'] += worker_summary.get('total', 0)
            merged_summary['successful'] += worker_summary.get('successful', 0)
            merged_summary['failed'] += worker_summary.get('failed', 0)
            
            # Adjust indices
            adjusted_success = [idx + sample_offset for idx in worker_summary.get('successful_indices', [])]
            adjusted_failed = [idx + sample_offset for idx in worker_summary.get('failed_indices', [])]
            
            merged_summary['successful_indices'].extend(adjusted_success)
            merged_summary['failed_indices'].extend(adjusted_failed)
            
            # Update Yes/No specific counts
            if merged_summary.get('is_critique_format'):
                merged_summary['positive_final_perfect_count'] += worker_summary.get('positive_final_perfect_count', 0)
                merged_summary['positive_pre_constructive_count'] += worker_summary.get('positive_pre_constructive_count', 0)
                merged_summary['negative_pre_perfect_count'] += worker_summary.get('negative_pre_perfect_count', 0)
                merged_summary['negative_pre_inaccurate_count'] += worker_summary.get('negative_pre_inaccurate_count', 0)
                merged_summary['complete_sets_count'] += worker_summary.get('complete_sets_count', 0)
                merged_summary['skipped_samples'] += worker_summary.get('skipped_samples', 0)
                merged_summary['original_samples_processed'] += worker_summary.get('original_samples_processed', 0)
            elif merged_summary.get('is_yesno_format'):
                merged_summary['positive_count'] += worker_summary.get('positive_count', 0)
                merged_summary['negative_count'] += worker_summary.get('negative_count', 0)
                merged_summary['pairs_count'] += worker_summary.get('pairs_count', 0)
                merged_summary['skipped_samples'] += worker_summary.get('skipped_samples', 0)
                merged_summary['original_samples_processed'] += worker_summary.get('original_samples_processed', 0)
        
        # Update sample offset for next worker
        worker_samples = worker_data.get('num_samples', 0)
        sample_offset += worker_samples
    
    # Update metadata
    merged['generation_timestamp'] = datetime.now().isoformat()
    merged['parallel_execution'] = {
        'num_workers': len(result_files),
        'merged_at': datetime.now().isoformat()
    }
    
    # Save merged results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    # Count total results
    total_results = sum(
        len(merged['results_by_format'][fmt]['results'])
        for fmt in formats
    )
    
    return total_results


class WorkerProcess:
    """Track a worker subprocess."""
    
    def __init__(self, worker_id: int, gpu_id: int, process: subprocess.Popen,
                 chunk_file: str, output_file: str, log_file: str, num_samples: int):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.process = process
        self.chunk_file = chunk_file
        self.output_file = output_file
        self.log_file = log_file
        self.num_samples = num_samples
        self.started_at = datetime.now()
    
    def is_running(self) -> bool:
        return self.process.poll() is None
    
    def returncode(self) -> Optional[int]:
        return self.process.poll()


def launch_worker(worker_id: int, gpu_id: int, chunk_file: str, 
                  output_file: str, log_file: str, args) -> WorkerProcess:
    """Launch a single worker process."""
    
    # Build command
    cmd = [
        sys.executable, "-u", "-m", "eval_code.unified_generation.py",
        "--data_file", chunk_file,
        "--video_dir", args.video_dir,
        "--instruction_style", args.instruction_style,
        "--mode", args.mode,
        "--output_file", output_file,
    ]
    
    # Add formats
    if args.all_formats:
        cmd.append("--all_formats")
    else:
        cmd.extend(["--formats"] + args.formats)
    
    # Add model arguments
    if args.model:
        cmd.extend(["--model", args.model])
    
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    
    if args.api_key:
        cmd.extend(["--api_key", args.api_key])
    if args.project_id:
        cmd.extend(["--project_id", args.project_id])
    
    # Generation parameters
    if args.fps:
        cmd.extend(["--fps", str(args.fps)])
    
    if args.max_new_tokens:
        cmd.extend(["--max_new_tokens", str(args.max_new_tokens)])

    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    
    # Data field mapping
    if args.caption_field:
        cmd.extend(["--caption_field", args.caption_field])
    
    if args.critique_field:
        cmd.extend(["--critique_field", args.critique_field])
    
    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    
    # Open log file
    log_handle = open(log_file, 'w', buffering=1)
    
    # Launch
    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(Path(__file__).parent)
    )
    
    # Get sample count
    with open(chunk_file, 'r') as f:
        chunk_data = json.load(f)
    num_samples = len(chunk_data)
    
    return WorkerProcess(worker_id, gpu_id, process, chunk_file, output_file, log_file, num_samples)


def main():
    parser = argparse.ArgumentParser(
        description='Parallel generation launcher for unified_generation.py',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--data_file', required=True,
                        help='Input JSON data file')
    parser.add_argument('--model', required=True,
                        help='Model name for generation')
    parser.add_argument('--video_dir', required=True,
                        help='Base directory for video files')
    
    # Format selection
    parser.add_argument('--formats', type=str, nargs='+',
                        help='Format names (use --list-formats to see available)')
    parser.add_argument('--all_formats', action='store_true',
                        help='Run on all compatible formats')
    
    # GPU configuration
    parser.add_argument('--gpus', required=True,
                        help='Comma-separated list of GPU IDs (e.g., 0,1,2,3)')
    parser.add_argument('--workers_per_gpu', type=int, default=1,
                        help='Number of workers per GPU (default: 1)')
    
    # Model arguments
    parser.add_argument('--checkpoint', default=None,
                        help='Model checkpoint path')
    parser.add_argument('--api_key', default=None,
                        help='API key for API-based models')
    parser.add_argument('--project_id', default=None,
                    help='Vertex AI project ID for Gemini models')
    
    # Generation parameters
    parser.add_argument('--mode', type=str, default='generate',
                        choices=['generate', 'score'],
                        help='Generation mode (default: generate)')
    parser.add_argument('--instruction_style', type=str, default='human',
                        help='Instruction style (default: human)')
    parser.add_argument('--fps', type=float, default=8.0,
                        help='Video FPS (default: 8.0)')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                        help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                    help='Temperature for generation/scoring (default: 0.0)')
    
    # Data field mapping
    parser.add_argument('--caption_field', type=str, default='pre_caption',
                        help='Caption field from raw data (default: pre_caption)')
    parser.add_argument('--critique_field', type=str, default='critique',
                        help='Critique field from raw data (default: critique)')
    
    # Output arguments
    parser.add_argument('--output_file', default=None,
                        help='Output file path (auto-generated if not specified)')
    parser.add_argument('--output_dir', default='generation_outputs',
                        help='Output directory (default: generation_outputs)')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Max number of samples to process (default: all)')
    parser.add_argument('--keep_temp', action='store_true',
                        help='Keep temporary chunk files after completion')
    parser.add_argument('--log_dir', default='logs',
                        help='Directory for worker logs')
    
    args = parser.parse_args()
    
    # Validate format arguments
    if not args.formats and not args.all_formats:
        parser.error("Must specify --formats or --all_formats")
    
    # Parse GPUs
    gpus = parse_gpu_list(args.gpus)
    if not gpus:
        print("Error: No GPUs specified")
        return 1
    
    total_workers = len(gpus) * args.workers_per_gpu
    
    print("=" * 60)
    print("PARALLEL UNIFIED GENERATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    if args.all_formats:
        print(f"Formats: ALL")
    else:
        print(f"Formats: {args.formats}")
    print(f"Data file: {args.data_file}")
    print(f"Video dir: {args.video_dir}")
    print(f"Mode: {args.mode}")
    print(f"GPUs: {gpus}")
    print(f"Workers per GPU: {args.workers_per_gpu}")
    print(f"Total workers: {total_workers}")
    print()
    
    # Load data
    print("Loading data...")
    data = load_data(args.data_file)
    
    print(f"  Total samples: {len(data)}")
    if args.sample_size and args.sample_size < len(data):
        data = data[:args.sample_size]
        print(f"  Using first {args.sample_size} samples (--sample_size)")
    
    if len(data) == 0:
        print("Error: No data to process")
        return 1
    
    # Split data into chunks
    print(f"\nSplitting data into {total_workers} chunks...")
    chunks = split_data(data, total_workers)
    
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} samples")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create temp directory for chunks
    temp_dir = tempfile.mkdtemp(prefix="parallel_unified_gen_")
    print(f"\nTemp directory: {temp_dir}")
    
    # Save chunks and launch workers
    workers: List[WorkerProcess] = []
    
    print(f"\nLaunching {total_workers} workers...")
    
    worker_id = 0
    for gpu_id in gpus:
        for w_idx in range(args.workers_per_gpu):
            if worker_id >= len(chunks):
                break
            
            chunk = chunks[worker_id]
            if not chunk:
                worker_id += 1
                continue
            
            # Save chunk
            chunk_file = os.path.join(temp_dir, f"chunk_{worker_id:03d}.json")
            save_chunk(chunk, chunk_file)
            
            # Output file for this worker
            output_file = os.path.join(temp_dir, f"worker_{worker_id:03d}_results.json")
            
            # Log file
            log_file = os.path.join(args.log_dir, f"worker_{worker_id:03d}_gpu{gpu_id}.log")
            
            # Launch worker
            worker = launch_worker(worker_id, gpu_id, chunk_file, output_file, log_file, args)
            workers.append(worker)
            
            print(f"  Worker {worker_id}: GPU {gpu_id}, {len(chunk)} samples, PID {worker.process.pid}")
            
            worker_id += 1
    
    print(f"\nAll workers launched. Waiting for completion...")
    print("=" * 60)
    
    # Wait for all workers
    start_time = time.time()
    
    try:
        while True:
            running = [w for w in workers if w.is_running()]
            completed = [w for w in workers if not w.is_running()]
            
            if not running:
                break
            
            # Print status every 30 seconds
            elapsed = time.time() - start_time
            print(f"[{elapsed/60:.1f}m] Running: {len(running)}, Completed: {len(completed)}")
            
            time.sleep(30)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted! Terminating workers...")
        for w in workers:
            if w.is_running():
                w.process.terminate()
        
        # Wait for graceful shutdown
        for w in workers:
            try:
                w.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                w.process.kill()
        
        return 1
    
    # Check results
    print("\n" + "=" * 60)
    print("WORKER RESULTS")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    result_files = []
    
    for w in workers:
        rc = w.process.returncode
        status = "✓" if rc == 0 else f"✗ (exit {rc})"
        print(f"  Worker {w.worker_id} (GPU {w.gpu_id}): {status}")
        
        if rc == 0:
            success_count += 1
            result_files.append(w.output_file)
        else:
            fail_count += 1
            print(f"    Log: {w.log_file}")
    
    if fail_count > 0:
        print(f"\nWarning: {fail_count} workers failed. Check logs for details.")
    
    # Merge results
    print("\n" + "=" * 60)
    print("MERGING RESULTS")
    print("=" * 60)
    
    # Determine final output file
    if args.output_file:
        final_output = args.output_file
    else:
        safe_model = args.model.replace('/', '_').replace(':', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.all_formats:
            format_str = "all_formats"
        elif len(args.formats) == 1:
            format_str = args.formats[0]
        else:
            format_str = f"{len(args.formats)}formats"
        
        if args.checkpoint:
            safe_ckpt = Path(args.checkpoint).name.replace('/', '_')
            final_output = os.path.join(args.output_dir, 
                f"{format_str}_{args.mode}_{safe_model}_{safe_ckpt}_{timestamp}.json")
        else:
            final_output = os.path.join(args.output_dir,
                f"{format_str}_{args.mode}_{safe_model}_{timestamp}.json")
    
    # Ensure output directory exists
    Path(final_output).parent.mkdir(parents=True, exist_ok=True)
    
    # Merge
    total_merged = merge_results(result_files, final_output)
    
    print(f"  Merged {total_merged} results from {len(result_files)} workers")
    print(f"  Output: {final_output}")
    
    # Cleanup
    if not args.keep_temp:
        print(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    else:
        print(f"\nKeeping temp directory: {temp_dir}")
    
    # Final summary
    elapsed_total = time.time() - start_time
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Total samples: {len(data)}")
    print(f"Workers: {success_count} succeeded, {fail_count} failed")
    print(f"Time: {elapsed_total/60:.1f} minutes")
    print(f"Throughput: {len(data) / elapsed_total:.1f} samples/sec")
    print(f"Output: {final_output}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())