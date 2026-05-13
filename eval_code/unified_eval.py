#!/usr/bin/env python3
"""
Unified Evaluation Script for Video Caption Generation
Evaluates outputs from unified_generation.py with format-specific metrics.
"""

import json
import argparse
import numpy as np
import string
import time
import openai
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
from dotenv import load_dotenv
import os


def load_generation_results(results_file: str) -> Dict[str, Any]:
    """Load generation results from JSON file"""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def preprocess_text(text):
    """Preprocess text for BLEU evaluation"""
    if not text:
        return []
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split into words
    words = text.split()
    return words


def calculate_bleu4_score(reference: str, candidate: str) -> float:
    """
    Calculate BLEU-4 score (up to 4-grams)
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        BLEU-4 score
    """
    if not reference or not candidate:
        return 0.0
    
    # Tokenize texts
    ref_tokens = preprocess_text(reference)
    cand_tokens = preprocess_text(candidate)
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    # Use smoothing to avoid zero scores when there are no matches
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU-4 score (weights for 1-grams through 4-grams)
    weights = (0.25, 0.25, 0.25, 0.25)
    
    try:
        score = sentence_bleu([ref_tokens], cand_tokens, weights=weights, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"Error calculating BLEU-4: {e}")
        return 0.0


def calculate_rouge_l_score(reference: str, candidate: str) -> float:
    """
    Calculate ROUGE-L score (longest common subsequence)
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        ROUGE-L F1 score
    """
    if not reference or not candidate:
        return 0.0
    
    try:
        # Initialize Rouge scorer with RougeL
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Calculate scores
        scores = scorer.score(reference, candidate)
        
        # Return F1 score
        return scores['rougeL'].fmeasure
    except Exception as e:
        print(f"Error calculating ROUGE-L: {e}")
        return 0.0


def get_openai_api_key(provided_key: Optional[str] = None) -> Optional[str]:
    """
    Get OpenAI API key from argument, environment variable, or .env file.
    
    Args:
        provided_key: API key provided as command line argument
        
    Returns:
        API key string or None if not found
    """
    if provided_key:
        return provided_key
    
    # Try environment variable first
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    
    # Load from .env file and try again
    load_dotenv()
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    
    return None


def calculate_llm_judge_score(reference: str, candidate: str, api_key: str, retries: int = 3, delay: int = 2) -> Optional[float]:
    """
    Calculate LLM judge score using GPT-4o.
    
    Args:
        reference: Reference caption/critique
        candidate: Model-generated caption/critique
        api_key: OpenAI API key
        retries: Number of retries if API call fails
        delay: Delay between retries in seconds
        
    Returns:
        Probability for "Yes" response (0.0-1.0) or None if failed
    """
    if not reference or not candidate or not api_key:
        return None
    
    # Set up OpenAI API
    openai.api_key = api_key
    
    prompt = f"Reference caption: '{reference}'\nCandidate caption: '{candidate}'\n\nDoes the candidate caption match the reference caption? Answer Yes or No."
    
    for attempt in range(retries):
        try:
            # Call GPT-4o API
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
                logprobs=True,
                top_logprobs=5
            )
            
            # Extract response and logprobs
            content = response.choices[0].message.content.strip().lower()
            
            # Check if the answer is clearly yes or no
            if content.startswith("yes"):
                return 1.0
            elif content.startswith("no"):
                return 0.0
            
            # If we can't determine from the content, try to get the logprobs
            logprobs = response.choices[0].logprobs.content[0].top_logprobs
            
            # Look for "yes" in top logprobs
            for token_info in logprobs:
                if token_info.token.strip().lower() == "yes":
                    return np.exp(token_info.logprob)
            
            # If "yes" not found in top logprobs, return low probability
            return 0.1
            
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return 0.5  # Fallback score
    
    return 0.5


def compute_pairwise_accuracy(positive_scores: List[float], negative_scores: List[float]) -> Dict[str, Any]:
    """
    Compute pairwise accuracy: fraction where positive_score > negative_score
    
    Args:
        positive_scores: List of scores for positive samples
        negative_scores: List of scores for negative samples
    
    Returns:
        Dict with accuracy and per-pair results
    """
    assert len(positive_scores) == len(negative_scores), "Unequal number of positive and negative scores"
    
    correct = 0
    comparisons = []
    
    for pos, neg in zip(positive_scores, negative_scores):
        is_correct = pos > neg
        correct += int(is_correct)
        comparisons.append({
            "positive_score": pos,
            "negative_score": neg,
            "correct": is_correct
        })
    
    accuracy = correct / len(positive_scores) if positive_scores else 0.0
    
    return {
        "pairwise_accuracy": accuracy,
        "correct_pairs": correct,
        "total_pairs": len(positive_scores),
        "comparisons": comparisons
    }


def compute_individual_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute individual accuracy by comparing predicted answer to expected answer
    
    For score mode: score > 0.5 → predict "Yes", else "No"
    For generate mode: parse generated answer
    """
    correct = 0
    total = 0
    by_expected = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for result in results:
        if not result.get("success"):
            continue
        
        expected = result.get("expected_answer")
        if expected is None:
            continue
        
        # Determine predicted answer
        if "score" in result:
            # Score mode: threshold at 0.5
            predicted = "Yes" if result["score"] > 0.5 else "No"
        elif "response" in result:
            # Generate mode: parse response
            response = result["response"].strip().lower()
            if response.startswith("yes"):
                predicted = "Yes"
            elif response.startswith("no"):
                predicted = "No"
            else:
                continue  # Unable to parse
        else:
            continue
        
        is_correct = (predicted == expected)
        correct += int(is_correct)
        total += 1
        
        by_expected[expected]["correct"] += int(is_correct)
        by_expected[expected]["total"] += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    breakdown = {
        label: {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "total": stats["total"]
        }
        for label, stats in by_expected.items()
    }
    
    return {
        "individual_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "breakdown_by_expected": breakdown
    }


def evaluate_caption_yesno(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate caption_yes_or_no format with positive/negative pairs
    
    Metrics:
    - Pairwise accuracy (overall and per caption_type)
    - Individual accuracy (for positive and negative separately)
    """
    successful_results = [r for r in results if r["success"]]
    
    # Group by sample_index to get pairs
    by_sample = defaultdict(list)
    for result in successful_results:
        by_sample[result["sample_index"]].append(result)
    
    # Extract positive and negative scores
    positive_scores = []
    negative_scores = []
    
    # Also track per caption_type
    by_caption_type = defaultdict(lambda: {"positive": [], "negative": []})
    
    for sample_idx, sample_results in by_sample.items():
        pos = [r for r in sample_results if r.get("caption_source") == "positive"]
        neg = [r for r in sample_results if r.get("caption_source") == "negative"]
        
        if len(pos) == 1 and len(neg) == 1:
            positive_scores.append(pos[0]["score"])
            negative_scores.append(neg[0]["score"])
            
            ct = pos[0].get("caption_type", "unknown")
            by_caption_type[ct]["positive"].append(pos[0]["score"])
            by_caption_type[ct]["negative"].append(neg[0]["score"])
    
    # Compute overall metrics
    pairwise_metrics = compute_pairwise_accuracy(positive_scores, negative_scores)
    individual_metrics = compute_individual_accuracy(successful_results)
    
    # Compute per caption_type metrics
    metrics_by_caption_type = {}
    for ct, scores in by_caption_type.items():
        pw = compute_pairwise_accuracy(scores["positive"], scores["negative"])
        
        pos_scores = scores["positive"]
        neg_scores = scores["negative"]
        all_ct_scores = pos_scores + neg_scores
        
        metrics_by_caption_type[ct] = {
            "pairwise_accuracy": pw["pairwise_accuracy"],
            "num_pairs": len(scores["positive"]),
            "positive_mean": float(np.mean(pos_scores)) if pos_scores else 0.0,
            "negative_mean": float(np.mean(neg_scores)) if neg_scores else 0.0,
            "score_mean": float(np.mean(all_ct_scores)) if all_ct_scores else 0.0,
        }
    
    # Score statistics
    all_scores = [r["score"] for r in successful_results if "score" in r]
    
    return {
        "format_type": "caption_yes_or_no",
        "pairwise_accuracy": pairwise_metrics["pairwise_accuracy"],
        "pairwise_metrics": pairwise_metrics,
        "individual_accuracy": individual_metrics["individual_accuracy"],
        "individual_metrics": individual_metrics,
        "by_caption_type": metrics_by_caption_type,
        "score_statistics": {
            "mean": float(np.mean(all_scores)) if all_scores else 0.0,
            "std": float(np.std(all_scores)) if all_scores else 0.0,
            "min": float(np.min(all_scores)) if all_scores else 0.0,
            "max": float(np.max(all_scores)) if all_scores else 0.0,
            "median": float(np.median(all_scores)) if all_scores else 0.0
        },
        "num_pairs": len(positive_scores),
        "num_successful": len(successful_results),
        "num_failed": len([r for r in results if not r["success"]])
    }


def evaluate_critique_yesno(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate critique_yes_or_no format with 4 pairs per sample
    
    Each sample has:
    - positive_final_perfect (expected Yes)
    - positive_pre_constructive (expected Yes)
    - negative_pre_perfect (expected No)
    - negative_pre_inaccurate (expected No)
    
    Metrics:
    - Overall pairwise accuracy (all 4 pairs considered)
    - Per-pair-type accuracy
    - Per-caption-type pairwise accuracy
    - Individual accuracy
    """
    successful_results = [r for r in results if r["success"]]
    
    # Group by sample_index to get complete sets
    by_sample = defaultdict(list)
    for result in successful_results:
        by_sample[result["sample_index"]].append(result)
    
    # Extract scores by type
    scores_by_type = {
        "positive_final_perfect": [],
        "positive_pre_constructive": [],
        "negative_pre_perfect": [],
        "negative_pre_inaccurate": []
    }
    
    # Also track per caption_type for complete sets
    by_caption_type = defaultdict(lambda: {
        "positive_final_perfect": [],
        "positive_pre_constructive": [],
        "negative_pre_perfect": [],
        "negative_pre_inaccurate": []
    })
    
    complete_sets = 0
    for sample_idx, sample_results in by_sample.items():
        # Check if we have all 4 pairs
        types_present = {r["caption_source"] for r in sample_results}
        if len(types_present) == 4:
            complete_sets += 1
            
            # Get caption_type from any result in this set
            ct = sample_results[0].get("caption_type", "unknown")
            
            for result in sample_results:
                scores_by_type[result["caption_source"]].append(result["score"])
                by_caption_type[ct][result["caption_source"]].append(result["score"])
    
    # Compute overall pairwise comparisons
    pairwise_results = []
    
    for i in range(complete_sets):
        pos_final_perfect = scores_by_type["positive_final_perfect"][i]
        pos_pre_constructive = scores_by_type["positive_pre_constructive"][i]
        neg_pre_perfect = scores_by_type["negative_pre_perfect"][i]
        neg_pre_inaccurate = scores_by_type["negative_pre_inaccurate"][i]
        
        # All positive pairs should score higher than all negative pairs
        comparisons = [
            pos_final_perfect > neg_pre_perfect,
            pos_final_perfect > neg_pre_inaccurate,
            pos_pre_constructive > neg_pre_perfect,
            pos_pre_constructive > neg_pre_inaccurate,
        ]
        
        pairwise_results.extend(comparisons)
    
    pairwise_accuracy = sum(pairwise_results) / len(pairwise_results) if pairwise_results else 0.0
    
    # Compute per caption_type pairwise accuracy
    metrics_by_caption_type = {}
    for ct, ct_scores in by_caption_type.items():
        ct_sets = len(ct_scores["positive_final_perfect"])
        ct_pairwise = []
        
        for i in range(ct_sets):
            pfp = ct_scores["positive_final_perfect"][i]
            ppc = ct_scores["positive_pre_constructive"][i]
            npp = ct_scores["negative_pre_perfect"][i]
            npi = ct_scores["negative_pre_inaccurate"][i]
            
            ct_pairwise.extend([
                pfp > npp,
                pfp > npi,
                ppc > npp,
                ppc > npi,
            ])
        
        ct_pairwise_acc = sum(ct_pairwise) / len(ct_pairwise) if ct_pairwise else 0.0
        
        # Per-type mean scores for this caption_type
        ct_type_stats = {}
        for pair_type, scores in ct_scores.items():
            if scores:
                ct_type_stats[pair_type] = {
                    "mean": float(np.mean(scores)),
                    "count": len(scores),
                }
        
        metrics_by_caption_type[ct] = {
            "pairwise_accuracy": ct_pairwise_acc,
            "complete_sets": ct_sets,
            "pairwise_comparisons": len(ct_pairwise),
            "per_type_means": ct_type_stats,
        }
    
    # Individual accuracy
    individual_metrics = compute_individual_accuracy(successful_results)
    
    # Per-type statistics (overall)
    per_type_stats = {}
    for pair_type, scores in scores_by_type.items():
        if scores:
            per_type_stats[pair_type] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "count": len(scores)
            }
    
    return {
        "format_type": "critique_yes_or_no",
        "pairwise_accuracy": pairwise_accuracy,
        "pairwise_comparisons_total": len(pairwise_results),
        "pairwise_comparisons_correct": sum(pairwise_results),
        "individual_accuracy": individual_metrics["individual_accuracy"],
        "individual_metrics": individual_metrics,
        "complete_sets": complete_sets,
        "per_type_statistics": per_type_stats,
        "by_caption_type": metrics_by_caption_type,
        "num_successful": len(successful_results),
        "num_failed": len([r for r in results if not r["success"]])
    }


def evaluate_generation_format(results: List[Dict[str, Any]], summary: Dict[str, Any], 
                               format_name: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate generation formats (caption_generation, critique_generation, etc.)
    
    Computes: BLEU-4, ROUGE-L, and optionally LLM-as-a-Judge
    """
    successful_results = [r for r in results if r["success"]]
    
    # Initialize score lists
    bleu4_scores = []
    rouge_l_scores = []
    llm_judge_scores = []
    
    # Per-caption-type scores
    scores_by_type = defaultdict(lambda: {
        'bleu4': [],
        'rouge_l': [],
        'llm_judge': []
    })
    
    # Check if we should use LLM judge
    use_llm_judge = api_key is not None
    
    # Process each result
    for result in tqdm(successful_results, desc=f"Evaluating {format_name}", leave=False):
        reference = result.get("reference")
        generated = result.get("response")
        caption_type = result.get("caption_type", "unknown")
        
        if not reference or not generated:
            continue
        
        # Calculate BLEU-4
        bleu4 = calculate_bleu4_score(reference, generated)
        bleu4_scores.append(bleu4)
        scores_by_type[caption_type]['bleu4'].append(bleu4)
        
        # Calculate ROUGE-L
        rouge_l = calculate_rouge_l_score(reference, generated)
        rouge_l_scores.append(rouge_l)
        scores_by_type[caption_type]['rouge_l'].append(rouge_l)
        
        # Calculate LLM judge if API key provided
        if use_llm_judge:
            llm_score = calculate_llm_judge_score(reference, generated, api_key)
            if llm_score is not None:
                llm_judge_scores.append(llm_score)
                scores_by_type[caption_type]['llm_judge'].append(llm_score)
    
    # Calculate per-type averages
    metrics_by_caption_type = {}
    for caption_type, type_scores in scores_by_type.items():
        metrics_by_caption_type[caption_type] = {
            'bleu4': float(np.mean(type_scores['bleu4'])) if type_scores['bleu4'] else 0.0,
            'rouge_l': float(np.mean(type_scores['rouge_l'])) if type_scores['rouge_l'] else 0.0,
            'llm_judge': float(np.mean(type_scores['llm_judge'])) if type_scores['llm_judge'] else None,
            'count': len(type_scores['bleu4'])
        }
    
    # Text length statistics
    text_lengths = [len(result.get("response", "").split()) for result in successful_results]
    
    return {
        "format_type": format_name,
        "num_successful": len(successful_results),
        "num_failed": len([r for r in results if not r["success"]]),
        "overall_metrics": {
            "bleu4": float(np.mean(bleu4_scores)) if bleu4_scores else 0.0,
            "rouge_l": float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0,
            "llm_judge": float(np.mean(llm_judge_scores)) if llm_judge_scores else None,
        },
        "by_caption_type": metrics_by_caption_type,
        "generation_statistics": {
            "mean_length_words": float(np.mean(text_lengths)) if text_lengths else 0.0,
            "std_length_words": float(np.std(text_lengths)) if text_lengths else 0.0,
            "min_length_words": int(np.min(text_lengths)) if text_lengths else 0,
            "max_length_words": int(np.max(text_lengths)) if text_lengths else 0,
        }
    }


def evaluate_format(format_name: str, batch_output: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Route to appropriate evaluation function based on format
    """
    results = batch_output["results"]
    summary = batch_output["summary"]
    
    # Filter to successful results
    successful_results = [r for r in results if r.get("success", False)]
    
    if len(successful_results) == 0:
        return {
            "format_type": format_name,
            "error": "No successful results to evaluate",
            "num_failed": len(results)
        }
    
    # Route based on format
    if format_name == "caption_yes_or_no":
        return evaluate_caption_yesno(results, summary)
    elif format_name == "critique_yes_or_no":
        return evaluate_critique_yesno(results, summary)
    elif format_name in ["caption_yes_or_no_with_given_critique", 
                         "caption_yes_or_no_with_self_critique_last_simple",
                         "caption_yes_or_no_with_self_critique_first_simple"]:
        return evaluate_caption_yesno(results, summary)
    elif format_name in ["caption_generation", "critique_generation", 
                         "caption_correction", "caption_correction_with_self_critique"]:
        return evaluate_generation_format(results, summary, format_name, api_key)
    else:
        # Default: treat as generation format
        return evaluate_generation_format(results, summary, format_name, api_key)


def save_evaluation_results(output_data: Dict[str, Any], output_file: str):
    """Save evaluation results to JSON"""
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation results saved to: {output_path}")


def print_evaluation_summary(eval_results: Dict[str, Any]):
    """Print a formatted summary of evaluation results"""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    for format_name, metrics in eval_results.items():
        print(f"\n{format_name}:")
        print(f"  Format Type: {metrics.get('format_type', 'unknown')}")
        
        if "error" in metrics:
            print(f"  Error: {metrics['error']}")
            continue
        
        # Print format-specific metrics
        if metrics.get("format_type") in ["caption_yes_or_no", "critique_yes_or_no"]:
            print(f"  Pairwise Accuracy: {metrics['pairwise_accuracy']:.4f}")
            print(f"  Individual Accuracy: {metrics['individual_accuracy']:.4f}")
            
            if metrics.get("format_type") == "caption_yes_or_no":
                print(f"  Pairs Evaluated: {metrics['num_pairs']}")
            elif metrics.get("format_type") == "critique_yes_or_no":
                print(f"  Complete Sets: {metrics['complete_sets']}")
                print(f"  Total Comparisons: {metrics['pairwise_comparisons_total']}")
            
            # Print per caption_type breakdown
            by_ct = metrics.get("by_caption_type", {})
            if by_ct:
                print(f"  By Caption Type:")
                for ct in sorted(by_ct.keys()):
                    ct_metrics = by_ct[ct]
                    pw_acc = ct_metrics.get("pairwise_accuracy", 0.0)
                    n = ct_metrics.get("num_pairs", ct_metrics.get("complete_sets", 0))
                    print(f"    {ct}: pairwise_acc={pw_acc:.4f} (n={n})")
        
        else:
            # Generation format
            overall = metrics.get("overall_metrics", {})
            print(f"  BLEU-4: {overall.get('bleu4', 0):.4f}")
            print(f"  ROUGE-L: {overall.get('rouge_l', 0):.4f}")
            if overall.get('llm_judge') is not None:
                print(f"  LLM Judge: {overall.get('llm_judge', 0):.4f}")
            
            gen_stats = metrics.get("generation_statistics", {})
            if gen_stats:
                print(f"  Mean Length (words): {gen_stats.get('mean_length_words', 0):.1f}")
            
            # Print per caption_type breakdown
            by_ct = metrics.get("by_caption_type", {})
            if by_ct:
                print(f"  By Caption Type:")
                for ct in sorted(by_ct.keys()):
                    ct_metrics = by_ct[ct]
                    print(f"    {ct}: BLEU-4={ct_metrics.get('bleu4', 0):.4f}, ROUGE-L={ct_metrics.get('rouge_l', 0):.4f} (n={ct_metrics.get('count', 0)})")
        
        print(f"  Successful: {metrics.get('num_successful', 0)}")
        print(f"  Failed: {metrics.get('num_failed', 0)}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate generation results from unified_generation.py'
    )
    
    # Required arguments
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to generation results JSON file')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Specific output file (overrides auto-generated)')
    
    # Format selection
    parser.add_argument('--formats', type=str, nargs='+', default=None,
                       help='Specific formats to evaluate (default: all formats in results)')
    
    # LLM judge arguments
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key for LLM-as-a-Judge')
    parser.add_argument('--no_llm_judge', action='store_true',
                       help='Skip LLM judge evaluation')
    
    args = parser.parse_args()
    
    # Check for required NLTK data
    try:
        nltk.data.find('punkt')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
    
    print("="*80)
    print("UNIFIED EVALUATION")
    print("="*80)
    
    # Load generation results
    print(f"\nLoading generation results from: {args.results_file}")
    generation_data = load_generation_results(args.results_file)
    
    print(f"Generation timestamp: {generation_data.get('generation_timestamp')}")
    print(f"Model: {generation_data.get('model_name')}")
    print(f"Checkpoint: {generation_data.get('checkpoint', 'None')}")
    print(f"Number of samples: {generation_data.get('num_samples')}")
    
    # Get API key for LLM judge
    api_key = None
    if not args.no_llm_judge:
        api_key = get_openai_api_key(args.api_key)
        if api_key:
            print("\nUsing OpenAI API key for LLM-as-a-Judge evaluation")
        else:
            print("\nWarning: No OpenAI API key found. LLM judge evaluation will be skipped.")
    else:
        print("\nLLM judge evaluation disabled via --no_llm_judge flag")
    
    # Determine which formats to evaluate
    available_formats = list(generation_data.get('results_by_format', {}).keys())
    if args.formats:
        formats_to_eval = [f for f in args.formats if f in available_formats]
        if len(formats_to_eval) < len(args.formats):
            missing = set(args.formats) - set(formats_to_eval)
            print(f"\nWarning: Formats not found in results: {missing}")
    else:
        formats_to_eval = available_formats
    
    print(f"\nEvaluating {len(formats_to_eval)} formats: {formats_to_eval}")
    
    # Evaluate each format
    evaluation_results = {}
    
    for format_name in tqdm(formats_to_eval, desc="Evaluating formats"):
        batch_output = generation_data['results_by_format'][format_name]
        
        try:
            metrics = evaluate_format(format_name, batch_output, api_key)
            evaluation_results[format_name] = metrics
        except Exception as e:
            print(f"\nError evaluating {format_name}: {e}")
            evaluation_results[format_name] = {
                "format_type": format_name,
                "error": str(e)
            }
    
    # Compile output
    output_data = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "generation_file": args.results_file,
        "generation_metadata": {
            "model_name": generation_data.get('model_name'),
            "checkpoint": generation_data.get('checkpoint'),
            "num_samples": generation_data.get('num_samples'),
            "generation_timestamp": generation_data.get('generation_timestamp'),
        },
        "formats_evaluated": formats_to_eval,
        "llm_judge_enabled": api_key is not None and not args.no_llm_judge,
        "evaluation_results": evaluation_results
    }
    
    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        # Auto-generate filename based on input
        input_path = Path(args.results_file)
        input_stem = input_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{input_stem}_{timestamp}.json"
        output_file = str(Path(args.output_dir) / filename)
    
    # Save results
    save_evaluation_results(output_data, output_file)
    
    # Print summary
    print_evaluation_summary(evaluation_results)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()