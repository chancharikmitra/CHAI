#!/usr/bin/env python3
"""
Batch Video Caption Generation using VideoCaptionAPI
Applies the single-video API to all samples in a raw data JSON file.
"""

import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import os

# Import the existing API
from eval_code.video_caption_api import load_api_with_standard_constants, VideoCaptionAPI


def load_raw_data(data_file: str) -> List[Dict[str, Any]]:
    """Load raw video caption data"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def is_yesno_format(format_name: str) -> bool:
    """Check if format is a Yes/No alignment format that needs positive/negative pairing"""
    yesno_formats = [
        "caption_yes_or_no",
        "caption_yes_or_no_with_given_critique",
        "caption_yes_or_no_with_self_critique_last_simple",
        "caption_yes_or_no_with_self_critique_first_simple",
        "critique_yes_or_no",
    ]
    return format_name in yesno_formats


# Format-specific constant overrides required by certain formats
FORMAT_OVERRIDES = {
    "caption_yes_or_no_with_self_critique_first_simple": {
        "CAPTION_EVALUATION_CONTEXT": ["self_critique_first_simple"],
        "ANSWER_LABEL": ["standard"],
    },
    "caption_yes_or_no_with_self_critique_last_simple": {
        "CAPTION_EVALUATION_CONTEXT": ["self_critique_last_simple"],
        "YES_TOKEN": ["capital_period"],
        "NO_TOKEN": ["capital_period"],
    },
    "caption_yes_or_no_with_given_critique": {
        "CAPTION_EVALUATION_CONTEXT": ["given_critique"],
    },
    "caption_correction_with_self_critique": {
        "CAPTION_CORRECTION_CONTEXT": ["self_critique"],
        "PERFECT_CAPTION_STATEMENT": ["self_critique"],
    },
}


def generate_batch(
    api: VideoCaptionAPI,
    data: List[Dict[str, Any]],
    format_name: str,
    instruction_style: str,
    video_dir: str,
    mode: str = "generate",
    fps: float = 8.0,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.0,
    caption_field: str = "pre_caption",
    critique_field: str = "critique",
    **api_kwargs
) -> Dict[str, Any]:
    """
    Generate outputs for all samples using the API.
    
    For caption_yes_or_no formats, automatically creates positive/negative pairs:
    - Positive: final_caption (expected "Yes")
    - Negative: pre_caption (expected "No", only if pre_caption_score < 5)
    - Skips samples where pre_caption_score == 5 (no negative available)
    
    For critique_yes_or_no format, creates 4 pairs per sample (if pre_caption_score < 5):
    - Positive pairs (expected "Yes"):
        1. final_caption + perfect_caption_critique
        2. pre_caption + critique
    - Negative pairs (expected "No"):
        1. pre_caption + perfect_caption_critique
        2. pre_caption + randomly_selected_inaccurate_critique (insertion or replacement error)
    
    Args:
        api: Initialized VideoCaptionAPI instance
        data: List of raw data samples
        format_name: Format to use (e.g., "caption_generation", "critique_generation")
        instruction_style: Instruction style (e.g., "human", "human_short")
        video_dir: Directory containing videos
        mode: "generate" or "score"
        fps: Video FPS
        max_new_tokens: Max tokens (None = auto-detect based on format)
        caption_field: Which caption field to use from raw data (ignored for Yes/No formats)
        critique_field: Which critique field to use from raw data (ignored for critique_yes_or_no)
        **api_kwargs: Additional kwargs for api.run()
    
    Returns:
        Dict with "results" and "summary" keys
    """
    results = []
    is_yesno = is_yesno_format(format_name)
    is_critique_yesno = (format_name == "critique_yes_or_no")
    skipped_count = 0
    
    # Get perfect caption critique constant from API
    # This should match the PERFECT_CAPTION_CRITIQUE constant used in training
    perfect_caption_critique = "The caption is accurate and requires no edits."
    
    # Get format-specific overrides
    overrides = FORMAT_OVERRIDES.get(format_name)
    
    print(f"\nProcessing {len(data)} samples with format: {format_name}")
    if overrides:
        print(f"  Using overrides: {list(overrides.keys())}")
    if is_critique_yesno:
        print("  Critique Yes/No format detected: Will create 4 pairs per sample")
        print("  Positive pairs (expected 'Yes'):")
        print("    1. final_caption + perfect_caption_critique")
        print("    2. pre_caption + critique")
        print("  Negative pairs (expected 'No'):")
        print("    1. pre_caption + perfect_caption_critique")
        print("    2. pre_caption + random_inaccurate_critique")
        print("  - Skipping samples where pre_caption_score == 5")
    elif is_yesno:
        print("  Caption Yes/No format detected: Will create positive/negative pairs")
        print("  - Positive: final_caption (expected 'Yes')")
        print("  - Negative: pre_caption (expected 'No', only if pre_caption_score < 5)")
        print("  - Skipping samples where pre_caption_score == 5")
    
    for idx, sample in enumerate(tqdm(data, desc=f"Generating {format_name}")):
        try:
            video_id = sample.get("video_id", "")
            video_path = str(Path(video_dir) / video_id)
            caption_type = sample.get("caption_type", "camera")
            
            # Get format info
            format_info = VideoCaptionAPI.get_format_info(format_name)
            required_inputs = format_info.get("required_inputs", [])
            
            # Handle critique_yes_or_no format with 4 pairs
            if is_critique_yesno and mode == "score":
                pre_caption_score = sample.get("pre_caption_score")
                
                # Skip if pre_caption is already perfect (score == 5)
                if pre_caption_score == 5:
                    skipped_count += 1
                    continue
                
                # Verify required fields exist
                if not sample.get("insertion_error_critique") or not sample.get("replacement_error_critique"):
                    skipped_count += 1
                    continue
                
                # Randomly select one inaccurate critique
                selected_inaccurate = random.choice(["insertion_error_critique", "replacement_error_critique"])
                
                # Create 4 pairs
                pairs_to_process = [
                    # Positive pairs
                    {
                        "caption": sample.get("final_caption", ""),
                        "critique": perfect_caption_critique,
                        "caption_source": "positive_final_perfect",
                        "expected_answer": "Yes",
                        "caption_field_used": "final_caption",
                        "critique_field_used": "perfect_caption_critique"
                    },
                    {
                        "caption": sample.get("pre_caption", ""),
                        "critique": sample.get("critique", ""),
                        "caption_source": "positive_pre_constructive",
                        "expected_answer": "Yes",
                        "caption_field_used": "pre_caption",
                        "critique_field_used": "critique"
                    },
                    # Negative pairs
                    {
                        "caption": sample.get("pre_caption", ""),
                        "critique": perfect_caption_critique,
                        "caption_source": "negative_pre_perfect",
                        "expected_answer": "No",
                        "caption_field_used": "pre_caption",
                        "critique_field_used": "perfect_caption_critique"
                    },
                    {
                        "caption": sample.get("pre_caption", ""),
                        "critique": sample.get(selected_inaccurate, ""),
                        "caption_source": "negative_pre_inaccurate",
                        "expected_answer": "No",
                        "caption_field_used": "pre_caption",
                        "critique_field_used": selected_inaccurate
                    }
                ]
            
            # Handle caption_yes_or_no format with 2 pairs
            elif is_yesno and not is_critique_yesno and mode == "score":
                pre_caption_score = sample.get("pre_caption_score")
                
                # Skip if pre_caption is already perfect (score == 5)
                if pre_caption_score == 5:
                    skipped_count += 1
                    continue
                
                # Process BOTH positive and negative
                pairs_to_process = [
                    {
                        "caption": sample.get("final_caption", ""),
                        "caption_source": "positive",
                        "expected_answer": "Yes",
                        "caption_field_used": "final_caption"
                    },
                    {
                        "caption": sample.get("pre_caption", ""),
                        "caption_source": "negative", 
                        "expected_answer": "No",
                        "caption_field_used": "pre_caption"
                    }
                ]
            else:
                # For non-Yes/No formats or generate mode, use single caption
                pairs_to_process = [
                    {
                        "caption": sample.get(caption_field, ""),
                        "caption_source": "single",
                        "expected_answer": None,
                        "caption_field_used": caption_field
                    }
                ]
            
            # Process each pair
            for pair_info in pairs_to_process:
                # Build kwargs for api.run()
                run_kwargs = {
                    "video_path": video_path,
                    "format_name": format_name,
                    "instruction_style": instruction_style,
                    "caption_type": caption_type,
                    "mode": mode,
                    "fps": fps,
                    "temperature": temperature,
                    "overrides": overrides,
                }
                
                if max_new_tokens is not None:
                    run_kwargs["max_new_tokens"] = max_new_tokens
                
                # Add format-specific inputs
                if "caption" in required_inputs:
                    run_kwargs["caption"] = pair_info["caption"]
                
                if "critique" in required_inputs:
                    # For critique_yes_or_no, use the critique from pair_info
                    # For other formats, use the critique_field from sample
                    if is_critique_yesno:
                        run_kwargs["critique"] = pair_info.get("critique", "")
                    else:
                        run_kwargs["critique"] = sample.get(critique_field, "")

                if format_name == "caption_generation":
                    reference = sample.get("final_caption", "")
                elif format_name == "critique_generation":
                    reference = sample.get("critique", "")
                elif format_name == "caption_correction":
                    reference = sample.get("final_caption", "")
                elif format_name == "caption_correction_with_self_critique":
                    reference = sample.get("final_caption", "")
                else:
                    reference = None
                
                # Merge any additional kwargs
                run_kwargs.update(api_kwargs)
                
                # Run the API
                result = api.run(**run_kwargs)
                
                # Add sample metadata
                result.update({
                    "sample_index": idx,
                    "video_id": video_id,
                    "caption_type": caption_type,
                    "caption_source": pair_info["caption_source"],
                    "expected_answer": pair_info["expected_answer"],
                    "caption_field_used": pair_info["caption_field_used"],
                    "critique_field_used": pair_info.get("critique_field_used", critique_field),
                    "workflow_type": sample.get("workflow_type", ""),
                    "annotator_negative_workflow_type": sample.get("annotator_negative_workflow_type", ""),
                    "pre_caption_score": sample.get("pre_caption_score"),
                    "success": result.get("error") is None,
                    "reference": reference,
                })
                
                results.append(result)
            
        except Exception as e:
            results.append({
                "sample_index": idx,
                "video_id": sample.get("video_id", ""),
                "error": str(e),
                "format_name": format_name,
                "success": False,
            })
    
    # Create summary for quick filtering in eval scripts
    successful_indices = [r["sample_index"] for r in results if r["success"]]
    failed_indices = [r["sample_index"] for r in results if not r["success"]]
    
    # For Yes/No formats, also track positive/negative pairs
    if is_yesno and mode == "score":
        if is_critique_yesno:
            # Critique format: 4 results per sample
            positive_final_perfect = [r for r in results if r.get("caption_source") == "positive_final_perfect" and r["success"]]
            positive_pre_constructive = [r for r in results if r.get("caption_source") == "positive_pre_constructive" and r["success"]]
            negative_pre_perfect = [r for r in results if r.get("caption_source") == "negative_pre_perfect" and r["success"]]
            negative_pre_inaccurate = [r for r in results if r.get("caption_source") == "negative_pre_inaccurate" and r["success"]]
            
            summary = {
                "total": len(results),
                "successful": len(successful_indices),
                "failed": len(failed_indices),
                "successful_indices": successful_indices,
                "failed_indices": failed_indices,
                "is_yesno_format": True,
                "is_critique_format": True,
                "positive_final_perfect_count": len(positive_final_perfect),
                "positive_pre_constructive_count": len(positive_pre_constructive),
                "negative_pre_perfect_count": len(negative_pre_perfect),
                "negative_pre_inaccurate_count": len(negative_pre_inaccurate),
                "complete_sets_count": min(
                    len(positive_final_perfect),
                    len(positive_pre_constructive),
                    len(negative_pre_perfect),
                    len(negative_pre_inaccurate)
                ),
                "skipped_samples": skipped_count,
                "original_samples_processed": len(data) - skipped_count
            }
        else:
            # Caption format: 2 results per sample
            positive_results = [r for r in results if r.get("caption_source") == "positive" and r["success"]]
            negative_results = [r for r in results if r.get("caption_source") == "negative" and r["success"]]
            
            summary = {
                "total": len(results),
                "successful": len(successful_indices),
                "failed": len(failed_indices),
                "successful_indices": successful_indices,
                "failed_indices": failed_indices,
                "is_yesno_format": True,
                "is_critique_format": False,
                "positive_count": len(positive_results),
                "negative_count": len(negative_results),
                "pairs_count": min(len(positive_results), len(negative_results)),
                "skipped_samples": skipped_count,
                "original_samples_processed": len(data) - skipped_count
            }
    else:
        summary = {
            "total": len(results),
            "successful": len(successful_indices),
            "failed": len(failed_indices),
            "successful_indices": successful_indices,
            "failed_indices": failed_indices,
            "is_yesno_format": False,
            "is_critique_format": False
        }
    
    return {
        "results": results,
        "summary": summary
    }


def save_results(output_data: Dict[str, Any], output_file: str):
    """Save generation results to JSON"""
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch video caption generation using VideoCaptionAPI'
    )
    
    # Required arguments
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to raw video caption JSON data')
    parser.add_argument('--formats', type=str, nargs='+', required=False,
                       help='Format names (use --list-formats to see available). Can specify multiple.')
    parser.add_argument('--instruction_style', type=str, default='human',
                       help='Instruction style (default: human)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (e.g., qwen2.5-vl-7b, qwen3-vl-8b). Required for generation/scoring.')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Model checkpoint path (optional)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key for API-based models')
    
    # Input/output arguments
    parser.add_argument('--video_dir', type=str, default='../data/captioning_videos_10_22',
                       help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='generation_outputs',
                       help='Output directory')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Specific output file (overrides auto-generated)')
    
    # Generation parameters
    parser.add_argument('--mode', type=str, default='generate',
                       choices=['generate', 'score'],
                       help='Generation mode (default: generate)')
    parser.add_argument('--fps', type=float, default=8.0,
                       help='Video FPS (default: 8.0)')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                       help='Max tokens to generate (default: auto-detect)')
    parser.add_argument('--temperature', type=float, default=0.0,
                   help='Temperature for generation/scoring (default: 0.0)')
    parser.add_argument('--project_id', type=str, default=None,
                    help='Vertex AI project ID (or set VERTEX_PROJECT_ID env var)')
    
    # Data field mapping (ignored for Yes/No formats which use automatic pairing)
    parser.add_argument('--caption_field', type=str, default='pre_caption',
                       help='Caption field from raw data (default: pre_caption). Ignored for Yes/No formats.')
    parser.add_argument('--critique_field', type=str, default='critique',
                       help='Critique field from raw data (default: critique). Ignored for critique_yes_or_no.')
    
    # Sampling
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to process (default: all)')
    
    # API configuration
    parser.add_argument('--task_instructions', type=str, 
                       default='eval_code/task_instructions.json',
                       help='Path to task instructions JSON')
    
    # Utility
    parser.add_argument('--list_formats', action='store_true',
                       help='List all available formats and exit')
    parser.add_argument('--all_formats', action='store_true',
                       help='Run on all compatible formats')
    
    args = parser.parse_args()
    
    # List formats if requested
    if args.list_formats:
        VideoCaptionAPI.print_compatible_formats()
        return
    
    # Determine which formats to run
    if args.all_formats:
        formats_to_run = VideoCaptionAPI.list_compatible_formats()
        print(f"\nRunning ALL {len(formats_to_run)} compatible formats")
    elif args.formats:
        formats_to_run = args.formats
    else:
        parser.error("Must specify --formats, --all_formats, or --list_formats")
    
    print("="*80)
    print("BATCH VIDEO CAPTION GENERATION")
    print("="*80)
    
    # Load raw data
    print(f"\nLoading data from: {args.data_file}")
    data = load_raw_data(args.data_file)
    print(f"Loaded {len(data)} samples")
    
    # Sample if requested
    if args.sample_size and args.sample_size < len(data):
        data = data[:args.sample_size]
        print(f"Using first {args.sample_size} samples")
    
    # Initialize API
    print(f"\nInitializing VideoCaptionAPI")
    print(f"  Task instructions: {args.task_instructions}")
    if args.model:
        print(f"  Model: {args.model}")
        if args.checkpoint:
            print(f"  Checkpoint: {args.checkpoint}")
    else:
        print(f"  Model: None (prompt-only mode)")
    
    import os
    api = load_api_with_standard_constants(
        task_instructions_file=args.task_instructions,
        model_name=args.model,
        checkpoint=args.checkpoint,
        api_key=args.api_key,
        project_id=args.project_id or os.environ.get("VERTEX_PROJECT_ID"),
    )
    
    # Process each format
    all_results = {}
    
    for format_name in formats_to_run:
        print(f"\n{'='*80}")
        print(f"Processing format: {format_name}")
        print(f"{'='*80}")
        
        # Print format info
        try:
            format_info = VideoCaptionAPI.get_format_info(format_name)
            print(f"  Description: {format_info['description']}")
            print(f"  Required inputs: {format_info['required_inputs']}")
            print(f"  Scorable: {format_info['scorable']}")
            print(f"  Mode: {args.mode}")
            
            # Generate for all samples
            batch_output = generate_batch(
                api=api,
                data=data,
                format_name=format_name,
                instruction_style=args.instruction_style,
                video_dir=args.video_dir,
                mode=args.mode,
                fps=args.fps,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                caption_field=args.caption_field,
                critique_field=args.critique_field
            )
            
            all_results[format_name] = batch_output
            
            # Print format-specific summary
            summary = batch_output["summary"]
            print(f"\n{format_name}: {summary['successful']}/{summary['total']} successful")
            
            if summary.get("is_critique_format"):
                print(f"  Positive pairs:")
                print(f"    final_caption + perfect: {summary['positive_final_perfect_count']}")
                print(f"    pre_caption + constructive: {summary['positive_pre_constructive_count']}")
                print(f"  Negative pairs:")
                print(f"    pre_caption + perfect: {summary['negative_pre_perfect_count']}")
                print(f"    pre_caption + inaccurate: {summary['negative_pre_inaccurate_count']}")
                print(f"  Complete sets (all 4 pairs): {summary['complete_sets_count']}")
                print(f"  Skipped samples: {summary['skipped_samples']}")
            elif summary.get("is_yesno_format"):
                print(f"  Positive samples: {summary['positive_count']}")
                print(f"  Negative samples: {summary['negative_count']}")
                print(f"  Valid pairs: {summary['pairs_count']}")
                print(f"  Skipped samples: {summary['skipped_samples']}")
            
            if summary['failed'] > 0:
                print(f"  Failed indices: {summary['failed_indices']}")
            
        except Exception as e:
            print(f"\nERROR processing format {format_name}: {e}")
            all_results[format_name] = {
                "results": [{"error": str(e), "success": False}],
                "summary": {
                    "total": 0,
                    "successful": 0,
                    "failed": 1,
                    "successful_indices": [],
                    "failed_indices": [],
                    "is_yesno_format": False,
                    "is_critique_format": False
                }
            }
    
    # Compile output
    output_data = {
        "generation_timestamp": datetime.now().isoformat(),
        "model_name": args.model,
        "checkpoint": args.checkpoint,
        "data_file": args.data_file,
        "video_dir": args.video_dir,
        "formats_processed": list(all_results.keys()),
        "instruction_style": args.instruction_style,
        "mode": args.mode,
        "generation_config": {
            "fps": args.fps,
            "max_new_tokens": args.max_new_tokens,
            "caption_field": args.caption_field,
            "critique_field": args.critique_field,
        },
        "num_samples": len(data),
        "results_by_format": all_results
    }
    
    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        safe_model = (args.model or "no_model").replace('/', '_').replace(':', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if len(formats_to_run) == 1:
            format_str = formats_to_run[0]
        else:
            format_str = f"multiformat_{len(formats_to_run)}"
        
        if args.checkpoint:
            safe_checkpoint = Path(args.checkpoint).name.replace('/', '_')
            filename = f"{format_str}_{args.mode}_{safe_model}_{safe_checkpoint}_{timestamp}.json"
        else:
            filename = f"{format_str}_{args.mode}_{safe_model}_{timestamp}.json"
        
        output_file = str(Path(args.output_dir) / filename)
    
    # Save results
    save_results(output_data, output_file)
    
    # Print summary
    print(f"\n{'='*80}")
    print("GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Formats processed: {len(all_results)}")
    for format_name, batch_output in all_results.items():
        summary = batch_output["summary"]
        if summary.get("is_critique_format"):
            print(f"  {format_name}: {summary['complete_sets_count']} complete sets ({summary['successful']} successful results)")
        elif summary.get("is_yesno_format"):
            print(f"  {format_name}: {summary['pairs_count']} pairs ({summary['successful']} successful results)")
        else:
            print(f"  {format_name}: {summary['successful']} successful, {summary['failed']} failed")
    print(f"\nTotal samples per format: {len(data)}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()