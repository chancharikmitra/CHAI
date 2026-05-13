# formats.py
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import json
import random 

# Import constants system
from eval_code.constants import CONSTANTS

import math


class BaseFormat(ABC):
    """Abstract base class for all training data formats."""
    
    # Class attributes that subclasses should define
    description: str = ""
    output_type: str = ""
    
    def __init__(self, constants_config: Dict[str, str] = None):
        """Initialize format with constants configuration."""
        if constants_config is None:
            raise ValueError(
                f"{self.__class__.__name__} requires constants_config. "
                "This format cannot be instantiated without specifying which constant variants to use."
            )
        self.constants = constants_config
        
        # Extract ALL constants once - all subclasses can use these
        self.video_prefix = self.constants.get("VIDEO_PREFIX", "") 
        self.task_instruction_header = self.constants.get("TASK_INSTRUCTION_HEADER", "")
        self.caption_header = self.constants.get("CAPTION_HEADER", "")
        self.critique_header = self.constants.get("CRITIQUE_HEADER", "")
        self.yesno_suffix = self.constants.get("YESNO_SUFFIX", "")
        self.choice_suffix = self.constants.get("CHOICE_SUFFIX", "")
        # self.perfect_caption_critique_instruction = self.constants.get("PERFECT_CAPTION_CRITIQUE_INSTRUCTION", "")
        self.perfect_caption_critique = self.constants.get("PERFECT_CAPTION_CRITIQUE", "")
        self.perfect_caption_statement = self.constants.get("PERFECT_CAPTION_STATEMENT", "")
        self.yes_token = self.constants.get("YES_TOKEN", "")
        self.no_token = self.constants.get("NO_TOKEN", "")
        self.choice_1_token = self.constants.get("CHOICE_1_TOKEN", "")
        self.choice_2_token = self.constants.get("CHOICE_2_TOKEN", "")
        self.score_scale_hint = self.constants.get("SCORE_SCALE_HINT", "")
        self.answer_label = self.constants.get("ANSWER_LABEL", "")  # Add this line
        # No caption generation context since that is just the task_instruction
        self.caption_evaluation_context = self.constants.get("CAPTION_EVALUATION_CONTEXT", "")
        self.critique_generation_context = self.constants.get("CRITIQUE_GENERATION_CONTEXT", "")
        self.critique_evaluation_context = self.constants.get("CRITIQUE_EVALUATION_CONTEXT", "")
        self.caption_correction_context = self.constants.get("CAPTION_CORRECTION_CONTEXT", "")
        self.caption_scoring_context = self.constants.get("CAPTION_SCORING_CONTEXT", "")
    
    @abstractmethod
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        """Generate training sample(s) from preprocessed data and task instruction."""
        pass

class CaptionGenerationFormat(BaseFormat):
    """Format for basic caption generation tasks."""
    
    description = "Generate video captions"
    output_type = "Caption"
    
    def format_prompt(self, task_instruction: str) -> str:
        """
        Format prompt for caption generation.
        
        Args:
            task_instruction: The task instruction text
            
        Returns:
            Formatted prompt string
        """
        return f"{self.video_prefix}{task_instruction}"
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        training_samples = []
        
        if sample.get("final_caption"):
            training_sample = {
                "messages": [
                    {
                        "content": self.format_prompt(task_instruction),
                        "role": "user"
                    },
                    {
                        "content": sample["final_caption"],
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "caption_source": "final_caption"
                }
            }
            training_samples.append(training_sample)
        
        return training_samples

class CritiqueGenerationFormat(BaseFormat):
    """Format for generating critiques of video captions."""
    
    description = "Generate critiques of video captions"
    output_type = "Critique"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for critique generation.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to critique
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.critique_generation_context} "
            f"\"{self.perfect_caption_critique}\"\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        training_samples = []
        
        critiques_to_generate = []
        if sample.get("pre_caption_score") < 5:
            critiques_to_generate.append((sample['final_caption'], self.perfect_caption_critique, "final_caption", "perfect_caption_critique"))
            critiques_to_generate.append((sample['pre_caption'], sample["critique"], "pre_caption", "critique"))
            
        for caption, critique_content, caption_source, critique_source in critiques_to_generate:
            training_sample = {
                "messages": [
                    {
                        "content": self.format_prompt(task_instruction, caption),
                        "role": "user"
                    },
                    {
                        "content": critique_content,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "caption_source": caption_source,
                    "critique_source": critique_source
                }
            }
            training_samples.append(training_sample)
        
        return training_samples

class CaptionYesOrNoFormat(BaseFormat):
    """Format for binary Yes/No caption alignment scoring."""
    
    description = "Binary Yes/No scoring of caption alignment with video"
    output_type = "Yes/No Score"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for Yes/No caption evaluation.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to evaluate
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_evaluation_context} "
            f"{self.yesno_suffix}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Build lists of positive and negative samples
        positive_samples = []
        negative_samples = []

        # NEW CHANGE: Only generate for <5 score captions for better balancing
        if sample.get("pre_caption_score") < 5:
            # Positive sample: final_caption -> "Yes"
            positive_samples.append((sample["final_caption"], self.yes_token, "final_caption"))
            
            # Negative samples
            # pre_caption -> "No" (if different from final)
            negative_samples.append((sample["pre_caption"], self.no_token, "pre_caption"))
        
        # NEW CHANGE: No longer use worst_caption
        # worst_caption -> "No" (if exists and not empty)
        # if sample.get("worst_caption"):
        #     negative_samples.append((sample["worst_caption"], self.no_token, "worst_caption"))

        # We expect every sample to have a final_caption
        #  assert len(positive_samples) > 0, f"No final_caption found for sample {sample.get('video_id', 'unknown')}"
        
        # NEW CHANGE: Balancing no longer necessary.
        # 3. Balance classes: if more negatives than positives, duplicate positives
        # if len(negative_samples) > len(positive_samples) and len(positive_samples) > 0:
        #     # Calculate how many times to duplicate positives (round up)
        #     duplication_factor = math.ceil(len(negative_samples) / len(positive_samples))
        #     positive_samples = positive_samples * duplication_factor
        #     # Trim to exact balance
        #     positive_samples = positive_samples[:len(negative_samples)]
        
        # 4. Combine all samples
        samples_to_generate = positive_samples + negative_samples
        
        # 5. Generate all samples using the list
        for caption, answer, caption_source in samples_to_generate:
            training_sample = {
                "messages": [
                    {
                        "content": self.format_prompt(task_instruction, caption),
                        "role": "user"
                    },
                    {
                        "content": answer,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "caption_source": caption_source
                }
            }
            training_samples.append(training_sample)
        
        # 6. Return final training_samples list
        return training_samples


class CaptionYesOrNoWithGivenCritiqueFormat(BaseFormat):
    """Format for binary Yes/No scoring of captions given critiques."""
    
    description = "Binary Yes/No scoring of caption alignment given critique context"
    output_type = "Yes/No Caption Score"
    
    def format_prompt(self, task_instruction: str, caption: str, critique: str) -> str:
        """
        Format prompt for Yes/No caption evaluation with given critique.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to evaluate
            critique: The critique context
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_evaluation_context} "
            f"{self.yesno_suffix}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}\n\n"
            f"{self.critique_header} {critique}"
        )
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format_template using constants from base class
        format_template = (
            f"{self.caption_evaluation_context} "
            f"{self.yesno_suffix}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {{caption}}\n\n"
            f"{self.critique_header} {{critique}}"
            
        )
        
        # 3. Build list of (caption, critique, answer, caption_source, critique_source, extra_metadata) tuples
        samples_to_generate = []

        if sample.get("pre_caption_score") < 5: # NEW CHANGE: Only generate samples when the sample has a critique
            # final_caption + perfect_caption_critique -> "Yes"
            samples_to_generate.append((
                sample['final_caption'], 
                self.perfect_caption_critique, 
                self.yes_token, 
                "final_caption", 
                "perfect_caption_critique",
                {}
            ))

            # pre_caption + critique -> "No"
            samples_to_generate.append((
                sample['pre_caption'], 
                sample['critique'], 
                self.no_token, 
                "pre_caption", 
                "critique",
                {"original_status": sample.get("status", "unknown")}
            ))
        
        # 4. Generate all samples using the list
        for caption, critique, answer, caption_source, critique_source, extra_metadata in samples_to_generate:
            metadata = {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": caption_source,
                "critique_source": critique_source
            }
            metadata.update(extra_metadata)
            
            training_sample = {
                "messages": [
                    {
                        "content": f"{self.video_prefix}{format_template.format(caption=caption, critique=critique)}",
                        "role": "user"
                    },
                    {
                        "content": answer,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": metadata
            }
            training_samples.append(training_sample)
        
        # 5. Return final training_samples list
        return training_samples

class CaptionYesOrNoWithSelfCritiqueFormat(BaseFormat):
    """Format for binary Yes/No scoring of captions with self-generated critiques."""
    
    description = "Binary Yes/No scoring of caption alignment with self-generated critique context"
    output_type = "Yes/No Caption Score with Self Critique"
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format_template using instance variables
        format_template = (
            f"{self.caption_evaluation_context}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {{caption}}"
        )
        
        # 3. Build list of (caption, critique, answer, caption_source, critique_type) tuples
        samples_to_generate = []
        
        if sample.get("pre_caption_score") < 5: # NEW CHANGE: Only generate samples when the sample has a critique
            # Case 1: final_caption - Generate "Yes" sample with perfect_caption_critique
            samples_to_generate.append((
                sample['final_caption'],
                self.perfect_caption_critique,
                self.yes_token,
                "final_caption",
                "perfect_caption_critique"
            ))

            # Case 2: pre_caption (score < 5) - Generate "No" sample with critique
            samples_to_generate.append((
                sample['pre_caption'],
                sample['critique'],
                self.no_token,
                "pre_caption",
                "critique"
            ))
    
        # 4. Generate all samples using the list
        for caption, critique, answer, caption_source, critique_source in samples_to_generate:
            # Create structured string response using constants
            response_string = f"{self.critique_header} {critique}\n{self.answer_label} {answer}"
            
            metadata = {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": caption_source,
                "critique_source": critique_source
            }
            
            training_sample = {
                "messages": [
                    {
                        "content": f"{self.video_prefix}{format_template.format(caption=caption)}",
                        "role": "user"
                    },
                    {
                        "content": response_string,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": metadata
            }
            training_samples.append(training_sample)
        
        # 5. Return final training_samples list
        return training_samples

class CaptionYesOrNoWithSelfCritiqueLastSimple(BaseFormat):
    """Format for binary Yes/No scoring with critique-last (answer first, then critique as explanation)."""
    
    description = "Binary Yes/No scoring with self-generated critique appearing after the answer"
    output_type = "Yes/No Caption Score with Self Critique Last (Simple)"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for Yes/No evaluation with self-generated critique (answer first).
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to evaluate
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_evaluation_context}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Build list of (caption, critique, answer, caption_source, critique_type) tuples
        samples_to_generate = []
        
        if sample.get("pre_caption_score") < 5: # NEW CHANGE: Only generate samples when the sample has a critique
            # Case 1: final_caption - Generate "Yes" sample with perfect_caption_critique
            samples_to_generate.append((
                sample['final_caption'],
                self.perfect_caption_critique,
                self.yes_token,
                "final_caption",
                "perfect_caption_critique"
            ))

            # Case 2: pre_caption (score < 5) - Generate "No" sample with critique
            samples_to_generate.append((
                sample['pre_caption'],
                sample['critique'],
                self.no_token,
                "pre_caption",
                "critique"
            ))
        
        # 3. Generate all samples using the list
        for caption, critique, answer, caption_source, critique_source in samples_to_generate:
            # Format: Answer first, then critique as explanation
            # Note: answer already includes period via capital_period variant
            response_string = f"{answer} {critique}"
            
            metadata = {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": caption_source,
                "critique_source": critique_source
            }
            
            training_sample = {
                "messages": [
                    {
                        "content": self.format_prompt(task_instruction, caption),
                        "role": "user"
                    },
                    {
                        "content": response_string,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": metadata
            }
            training_samples.append(training_sample)
        
        # 4. Return final training_samples list
        return training_samples


class CaptionYesOrNoWithSelfCritiqueFirstSimple(BaseFormat):
    """Format for binary Yes/No scoring with critique-first (critique as reasoning, then answer)."""
    
    description = "Binary Yes/No scoring with self-generated critique appearing before the answer"
    output_type = "Yes/No Caption Score with Self Critique First (Simple)"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for Yes/No evaluation with self-generated critique (critique first).
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to evaluate
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_evaluation_context}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Build list of (caption, critique, answer, caption_source, critique_type) tuples
        samples_to_generate = []
        
        if sample.get("pre_caption_score") < 5: # NEW CHANGE: Only generate samples when the sample has a critique
            # Case 1: final_caption - Generate "Yes" sample with perfect_caption_critique
            samples_to_generate.append((
                sample['final_caption'],
                self.perfect_caption_critique,
                self.yes_token,
                "final_caption",
                "perfect_caption_critique"
            ))

            # Case 2: pre_caption (score < 5) - Generate "No" sample with critique
            samples_to_generate.append((
                sample['pre_caption'],
                sample['critique'],
                self.no_token,
                "pre_caption",
                "critique"
            ))
        
        # 3. Generate all samples using the list
        for caption, critique, answer, caption_source, critique_source in samples_to_generate:
            # Format: Critique first, then answer with label
            response_string = f"{critique} {self.answer_label} {answer}"
            
            metadata = {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": caption_source,
                "critique_source": critique_source
            }
            
            training_sample = {
                "messages": [
                    {
                        "content": self.format_prompt(task_instruction, caption),
                        "role": "user"
                    },
                    {
                        "content": response_string,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": metadata
            }
            training_samples.append(training_sample)
        
        # 4. Return final training_samples list
        return training_samples


class CaptionYesOrNoWithSelfCritiqueTwoStepFormat(BaseFormat):
    """
    Two-step evaluation format: Generate critique, then evaluate caption using that critique.
    
    NOTE: This format is designed for EVALUATION ONLY, not for training data generation.
    
    The format works in two stages:
    1. Generate a critique for the caption (critique_generation step)
    2. Use the generated critique to evaluate the caption with Yes/No (caption_evaluation step)
    
    The second step uses a placeholder {generated_critique} that will be replaced by
    the evaluation script after the critique is generated in step 1.
    """
    
    description = "Two-step: Generate critique, then evaluate caption with that critique"
    output_type = "Two-step Critique + Yes/No"
    
    # Placeholder that will be replaced with actual generated critique
    CRITIQUE_PLACEHOLDER = "{generated_critique}"
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        """
        Generate two-step evaluation samples with both steps in a single conversation.
        
        Each sample contains a 4-message conversation:
        1. User asks for critique generation
        2. Assistant provides critique (placeholder)
        3. User asks for Yes/No evaluation given that critique
        4. Assistant provides Yes/No answer
        
        The assistant's first response uses a placeholder that will be filled by the eval script.
        """
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format templates for both steps
        critique_generation_prompt = (
            f"{self.critique_generation_context} "
            f"\"{self.perfect_caption_critique}\"\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {{caption}}"
        )
        
        caption_evaluation_prompt = (
            f"{self.caption_evaluation_context} "
            f"{self.yesno_suffix}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {{caption}}\n\n"
            f"{self.critique_header} "  # Note: placeholder will be added separately
        )
        
        # 3. Build lists of positive and negative samples
        positive_samples = []
        negative_samples = []
        # NEW CHANGE
        if sample.get("pre_caption_score") < 5:
            # Positive sample: final_caption -> "Yes"
            positive_samples.append((
                sample['final_caption'],
                self.yes_token,
                "final_caption"
            ))
            
            # Negative samples
            # pre_caption -> "No" (if different from final)
        
            negative_samples.append((
                sample['pre_caption'],
                self.no_token,
                "pre_caption"
            ))
        
        # worst_caption -> "No" (if exists and not empty)
        # if sample.get("worst_caption"):
        #     negative_samples.append((
        #         sample['worst_caption'],
        #         self.no_token,
        #         "worst_caption"
        #     ))
        
        # We expect every sample to have a final_caption
        # assert len(positive_samples) > 0, f"No final_caption found for sample {sample.get('video_id', 'unknown')}"
        
        # 4. Balance classes: if more negatives than positives, duplicate positives
        # if len(negative_samples) > len(positive_samples) and len(positive_samples) > 0:
        #     # Calculate how many times to duplicate positives (round up)
        #     duplication_factor = math.ceil(len(negative_samples) / len(positive_samples))
        #     positive_samples = positive_samples * duplication_factor
        #     # Trim to exact balance
        #     positive_samples = positive_samples[:len(negative_samples)]
        
        # 5. Combine all samples
        samples_to_evaluate = positive_samples + negative_samples
        
        # 6. Generate all samples using the list (4-message conversation structure)
        for caption, expected_answer, caption_source in samples_to_evaluate:
            
            training_sample = {
                "messages": [
                    # Step 1: User asks for critique
                    {
                        "content": f"{self.video_prefix}{critique_generation_prompt.format(caption=caption)}",
                        "role": "user"
                    },
                    # Step 1: Assistant provides critique (placeholder)
                    {
                        "content": self.CRITIQUE_PLACEHOLDER,
                        "role": "assistant"
                    },
                    # Step 2: User asks for Yes/No evaluation (with caption and critique placeholder)
                    {
                        "content": self.video_prefix + caption_evaluation_prompt.format(caption=caption) + self.CRITIQUE_PLACEHOLDER,
                        "role": "user"
                    },
                    # Step 2: Assistant provides Yes/No answer
                    {
                        "content": expected_answer,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "caption_source": caption_source,
                    "format_type": "two_step_conversation",
                    "ground_truth_label": expected_answer
                }
            }
            
            training_samples.append(training_sample)
        
        # 7. Return final training_samples list
        return training_samples

class CritiqueYesOrNoFormat(BaseFormat):
    """Format for binary Yes/No scoring of critique accuracy and constructiveness."""
    
    description = "Binary Yes/No scoring of critique quality"
    output_type = "Yes/No Critique Score"
    
    def format_prompt(self, task_instruction: str, caption: str, critique: str) -> str:
        """
        Format prompt for Yes/No critique evaluation.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption being critiqued
            critique: The critique to evaluate
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.critique_evaluation_context} "
            f"{self.yesno_suffix}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}\n\n"
            f"{self.critique_header} {critique}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # Assert required fields
        assert sample.get("pre_caption"), f"No pre_caption found for sample {sample.get('video_id', 'unknown')}"
        assert sample.get("final_caption"), f"No final_caption found for sample {sample.get('video_id', 'unknown')}"
        
        # 2. Build lists of positive and negative samples
        positive_samples = []
        negative_samples = []
        
        # Positive samples: good critiques -> "Yes"
        # NEW CHANGE: Generate 0 samples when score 5 and 4 when score < 5
        if sample.get("pre_caption_score") < 5:
            # Verify both inaccurate critiques exist
            assert sample.get("insertion_error_critique"), "insertion_error_critique missing for pre_caption_score < 5"
            assert sample.get("replacement_error_critique"), "replacement_error_critique missing for pre_caption_score < 5"
            
            # final_caption + perfect_caption_critique -> "Yes"
            positive_samples.append((
                sample['final_caption'],
                self.perfect_caption_critique,
                self.yes_token,
                "final_caption",
                "perfect_caption_critique"
            ))

            # pre_caption + critique -> "Yes" (if score < 5, constructive critique exists)

            positive_samples.append((
                sample['pre_caption'],
                sample['critique'],
                self.yes_token,
                "pre_caption",
                "critique"
            ))

            # Negative samples: bad critiques -> "No"

        
            # if sample.get("pre_caption_score") < 5:
            # pre_caption + perfect_caption_critique -> "No" (perfect critique on imperfect caption is wrong)
            negative_samples.append((
                sample['pre_caption'],
                self.perfect_caption_critique,
                self.no_token,
                "pre_caption",
                "perfect_caption_critique"
            ))
            # # final_caption + critque -> "No"
            # negative_samples.append((
            #     sample['final_caption'],
            #     sample['critique'],
            #     self.no_token,
            #     "final_caption",
            #     "critique"
            # ))

            # Add all generated negative critiques paired with pre_caption
            # negative_critique_fields = [
            #     "nonconstructive_critique",
            #     "blind_model_critique",
            #     "insertion_error_critique",
            #     "replacement_error_critique",
            #     "deletion_error_critique",
            #     # "video_model_critique"
            # ]

            # for critique_field in negative_critique_fields:
            #     if sample.get(critique_field):
            #         negative_samples.append((
            #             sample['pre_caption'],
            #             sample[critique_field],
            #             self.no_token,
            #             "pre_caption",
            #             critique_field
            #         ))
            selected_inaccurate = random.choice(["insertion_error_critique", "replacement_error_critique"])
            negative_samples.append((
                sample['pre_caption'],
                sample[selected_inaccurate],
                self.no_token,
                "pre_caption",
                selected_inaccurate
            ))
                

        # NEW CHANGE: Remove balancing to pare down total number of samples
        # 3. Balance classes: if more negatives than positives, duplicate positives
        # if len(negative_samples) > len(positive_samples) and len(positive_samples) > 0:
        #     # Calculate how many times to duplicate positives (round up)
        #     duplication_factor = math.ceil(len(negative_samples) / len(positive_samples))
        #     positive_samples = positive_samples * duplication_factor
        #     # Trim to exact balance
        #     positive_samples = positive_samples[:len(negative_samples)]
        
        # 4. Combine all samples
        samples_to_generate = positive_samples + negative_samples
        
        # 5. Generate all samples using the list
        for caption, critique, answer, caption_source, critique_source in samples_to_generate:
            training_sample = {
                "messages": [
                    {
                        "content": self.format_prompt(task_instruction, caption, critique),
                        "role": "user"
                    },
                    {
                        "content": answer,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "caption_source": caption_source,
                    "critique_source": critique_source
                }
            }
            training_samples.append(training_sample)
        
        # 6. Return final training_samples list
        return training_samples
class CaptionCorrectionFormat(BaseFormat):
    """Format for direct caption improvement tasks."""
    
    description = "Direct caption improvement"
    output_type = "Improved Caption"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for caption correction.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to correct/improve
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_correction_context.format(perfect_caption_statement=self.perfect_caption_statement)}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Build lists of perfect and improvement samples
        positive_samples = []
        negative_samples = []
        
        # NEW CHANGE
        if sample.get("pre_caption_score") < 5:
            # Perfect sample: final_caption -> original caption (no changes)
            positive_samples.append((
                sample['final_caption'],     # input caption
                sample['final_caption'],     # output: same caption (no changes)
                "no_correction_needed",
                "final_caption"
            ))
            
            # We expect every sample to have a final_caption
            # assert len(positive_samples) > 0, f"No final_caption found for sample {sample.get('video_id', 'unknown')}"
            
            # Improvement samples: bad caption -> final_caption
            # pre_caption -> final_caption (if they differ)
            negative_samples.append((
                sample['pre_caption'],
                sample['final_caption'],
                "improvement",
                "pre_caption"
            ))
        
        # worst_caption -> final_caption (if exists)
        # if sample.get("worst_caption"):
        #     negative_samples.append((
        #         sample['worst_caption'],
        #         sample['final_caption'],
        #         "improvement",
        #         "worst_caption"
        #     ))
        
        # 3. Balance classes: if more negatives than positives, duplicate positives
        # if len(negative_samples) > len(positive_samples) and len(positive_samples) > 0:
        #     # Calculate how many times to duplicate positives (round up)
        #     duplication_factor = math.ceil(len(negative_samples) / len(positive_samples))
        #     positive_samples = positive_samples * duplication_factor
        #     # Trim to exact balance
        #     positive_samples = positive_samples[:len(negative_samples)]
        
        # 4. Combine all samples
        samples_to_generate = positive_samples + negative_samples
        
        # 5. Generate all samples using the list
        for input_caption, output_caption, correction_type, caption_source in samples_to_generate:
            training_sample = {
                "messages": [
                    {
                        "content": self.format_prompt(task_instruction, input_caption),
                        "role": "user"
                    },
                    {
                        "content": output_caption,  # Always returns the caption (original or improved)
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "correction_type": correction_type,
                    "caption_source": caption_source
                }
            }
            training_samples.append(training_sample)
        
        # 6. Return final training_samples list
        return training_samples

class CaptionCorrectionWithGivenCritiqueFormat(BaseFormat):
    """Format for critique-guided caption improvement tasks."""
    
    description = "Critique-guided caption improvement"
    output_type = "Improved Caption with Critique"
    
    def format_prompt(self, task_instruction: str, caption: str, critique: str) -> str:
        """Format prompt for caption correction with given critique."""
        return (
            f"{self.video_prefix}"
            f"{self.caption_correction_context.format(perfect_caption_statement=self.perfect_caption_statement)}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}\n\n"
            f"{self.critique_header} {critique}"
        )
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format_template using constants from base class
        format_template = (
            f"{self.caption_correction_context.format(perfect_caption_statement=self.perfect_caption_statement)}\n\n"
            # f"{self.perfect_caption_critique_instruction} \"{self.perfect_caption_critique}\"\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {{caption}}\n\n"
            f"{self.critique_header} {{critique}}"
        )
        
        # 3. Build list of (caption, critique, output, correction_type, caption_source, critique_source) tuples
        samples_to_generate = []
        # NEW CHANGE
        if sample.get("pre_caption_score") < 5:
            # final_caption + PERFECT_CRITIQUE -> original caption (no changes)
            samples_to_generate.append((
                sample['final_caption'],           # input caption
                self.perfect_caption_critique,     # input critique
                sample['final_caption'],           # output: original caption (no changes)
                "no_correction_needed",
                "final_caption",
                "perfect_caption_critique"
            ))
            
            # pre_caption + critique -> final_caption
        
            samples_to_generate.append((
                sample['pre_caption'],
                sample['critique'],
                sample['final_caption'],           # output: improved caption
                "improvement",
                "pre_caption",
                "critique"
            ))
        
        
        # 4. Generate all samples using the list
        for caption, critique, output, correction_type, caption_source, critique_source in samples_to_generate:
            training_sample = {
                "messages": [
                    {
                        "content": f"{self.video_prefix}{format_template.format(caption=caption, critique=critique)}",
                        "role": "user"
                    },
                    {
                        "content": output,  # Always returns the caption (original or improved)
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "correction_type": correction_type,
                    "caption_source": caption_source,
                    "critique_source": critique_source
                }
            }
            training_samples.append(training_sample)
        
        # 5. Return final training_samples list
        return training_samples


class CaptionCorrectionWithSelfCritiqueFormat(BaseFormat):
    """Format for caption correction with self-generated critiques."""
    
    description = "Caption improvement with self-generated critique guidance"
    output_type = "Improved Caption with Self Critique"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for caption correction with self-generated critique.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to correct/improve
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_correction_context.format(perfect_caption_critique=self.perfect_caption_critique, perfect_caption_statement=self.perfect_caption_statement)}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Build list of (caption, critique, improved_caption, correction_type, caption_source) tuples
        samples_to_generate = []
        
        # NEW CHANGE
        if sample.get("pre_caption_score") < 5:
            # final_caption -> "perfect" response (always)
            samples_to_generate.append((
                sample['final_caption'],
                self.perfect_caption_critique,     # Critique for perfect caption
                sample['final_caption'],           # Return same caption as "improved"
                "no_correction_needed",
                "final_caption",
                "perfect_caption_critique"
            ))

            # pre_caption + critique -> improvement (when score < 5)
            
            samples_to_generate.append((
                sample['pre_caption'],
                sample['critique'],
                sample['final_caption'],
                "improvement",
                "pre_caption",
                "critique"
            ))
        
        # 3. Generate all samples using the list
        for caption, critique, improved_caption, correction_type, caption_source, critique_source in samples_to_generate:
            # Always use consistent structured format for both perfect and improvement cases
            # Format: "Critique: <critique>\nImproved Caption: <caption>"
            output = f"Critique: {critique}\nImproved Caption: {improved_caption}"
            
            training_sample = {
                "messages": [
                    {
                        "content": self.format_prompt(task_instruction, caption),
                        "role": "user"
                    },
                    {
                        "content": output,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "correction_type": correction_type,
                    "caption_source": caption_source,
                    "critique_source": critique_source
                }
            }
            training_samples.append(training_sample)
        
        # 4. Return final training_samples list
        return training_samples

class CaptionScoringFormat(BaseFormat):
    """Format for 1-5 scale caption scoring tasks."""
    
    description = "1-5 scale scoring of caption alignment"
    output_type = "Caption Score"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for caption scoring (1-5 scale).
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to score
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_scoring_context} "
            f"{self.score_scale_hint}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Build lists of perfect (score 5) and non-perfect samples
        positive_samples = []
        negative_samples = []


        # NEW CHANGE
        if sample.get("pre_caption_score") < 5:
            # Perfect sample: final_caption -> score 5
            positive_samples.append((
                sample['final_caption'],
                5,
                "final_caption"
            ))

            
            # We expect every sample to have a final_caption
            assert len(positive_samples) > 0, f"No final_caption found for sample {sample.get('video_id', 'unknown')}"
            
            # Non-perfect samples
            # pre_caption -> pre_caption_score (if score < 5)
            
            negative_samples.append((
                sample['pre_caption'],
                sample['pre_caption_score'],
                "pre_caption"
            ))

        
        # worst_caption -> score 1 (if exists)
        # if sample.get("worst_caption"):
        #     negative_samples.append((
        #         sample['worst_caption'],
        #         1,
        #         "worst_caption"
        #     ))
        
        # NOTE: annotator negative samples are excluded due to unreliable scores
        # The annotator negative scores may be inflated due to overconfidence,
        # so we avoid using them to prevent noisy training signals
        
        # NEW CHANGE: Remove balancing to pare down total number of samples
        # 3. Balance classes: if more negatives than positives, duplicate positives
        # if len(negative_samples) > len(positive_samples) and len(positive_samples) > 0:
        #     # Calculate how many times to duplicate positives (round up)
        #     duplication_factor = math.ceil(len(negative_samples) / len(positive_samples))
        #     positive_samples = positive_samples * duplication_factor
        #     # Trim to exact balance
        #     positive_samples = positive_samples[:len(negative_samples)]
        
        # 4. Combine all samples
        samples_to_generate = positive_samples + negative_samples
        
        # 5. Generate all samples using the list
        for caption, score, caption_source in samples_to_generate:
            training_sample = {
                "messages": [
                    {
                        "content": self.format_prompt(task_instruction, caption),
                        "role": "user"
                    },
                    {
                        "content": str(score),
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "score_value": score,
                    "caption_source": caption_source
                }
            }
            training_samples.append(training_sample)
        
        # 6. Return final training_samples list
        return training_samples

class CaptionScoringWithGivenCritiqueFormat(BaseFormat):
    """Format for 1-5 scale caption scoring with given critique context."""
    
    description = "1-5 scale scoring of caption alignment with critique context"
    output_type = "Caption Score with Critique"
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format_template using constants from base class
        format_template = (
            f"{self.caption_scoring_context} "
            f"{self.score_scale_hint}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {{caption}}\n\n"
            f"{self.critique_header} {{critique}}"
        )
        
        # 3. Build list of (caption, critique, score, caption_source, critique_source) tuples
        samples_to_generate = []
        
        # NEW CHANGE
        if sample.get("pre_caption_score") < 5:
            # final_caption + perfect_caption_critique -> score 5 (always perfect)
            samples_to_generate.append((
                sample['final_caption'],
                self.perfect_caption_critique,
                5,
                "final_caption",
                "perfect_critique"
            ))

            # pre_caption + critique -> pre_caption_score (if score < 5)

            samples_to_generate.append((
                sample['pre_caption'],
                sample['critique'],
                sample['pre_caption_score'],
                "pre_caption",
                "critique"
            ))
        
        # 4. Generate all samples using the list
        for caption, critique, score, caption_source, critique_source in samples_to_generate:
            training_sample = {
                "messages": [
                    {
                        "content": f"{self.video_prefix}{format_template.format(caption=caption, critique=critique)}",
                        "role": "user"
                    },
                    {
                        "content": str(score),
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "score_value": score,
                    "caption_source": caption_source,
                    "critique_source": critique_source
                }
            }
            training_samples.append(training_sample)
        
        # 5. Return final training_samples list
        return training_samples

class CaptionScoringWithSelfCritiqueFormat(BaseFormat):
    """Format for 1-5 scale caption scoring with self-generated critiques."""
    
    description = "1-5 scale scoring of caption alignment with self-generated critique context"
    output_type = "Caption Score with Self Critique"
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format_template using instance variables
        format_template = (
            f"{self.caption_scoring_context.format(score_scale_hint=self.score_scale_hint)}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {{caption}}"
        )
        
        # 3. Build list of (caption, critique, score, caption_source) tuples
        samples_to_generate = []
        
        # NEW CHANGE
        if sample.get("pre_caption_score") < 5:
            # final_caption -> perfect_caption_critique + score 5 (always perfect)
            samples_to_generate.append((
                sample['final_caption'],
                self.perfect_caption_critique,
                5,
                "final_caption",
                "perfect_caption_critique"
            ))

            # pre_caption -> critique + pre_caption_score (if score < 5)

            samples_to_generate.append((
                sample['pre_caption'],
                sample['critique'],
                sample['pre_caption_score'],
                "pre_caption",
                "critique"
            ))
        
        # 4. Generate all samples using the list
        for caption, critique, score, caption_source, critique_source in samples_to_generate:
            # Create structured string response instead of JSON
            response_string = f"Critique: {critique}\nScore: {score}"
            
            training_sample = {
                "messages": [
                    {
                        "content": f"{self.video_prefix}{format_template.format(caption=caption)}",
                        "role": "user"
                    },
                    {
                        "content": response_string,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "score_value": score,
                    "caption_source": caption_source,
                    "critique_source": critique_source
                }
            }
            training_samples.append(training_sample)
        
        # 5. Return final training_samples list
        return training_samples


class PairwiseCaptionSelectionFormat(BaseFormat):
    """Format for pairwise caption selection tasks."""
    
    description = "Pairwise selection between two captions"
    output_type = "Caption Selection"
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format_template using constants from base class
        format_template = (
            f"{self.pairwise_context}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"Caption 1: {{caption_1}}\n\n"
            f"Caption 2: {{caption_2}}\n\n"
            f"{self.choice_suffix}"
        )
        
        # 3. Build list of (final_caption, other_caption, selection_type) tuples
        pairs_to_generate = []
        
        # final_caption vs pre_caption (if pre_caption differs)
        if (sample.get("final_caption") and
            sample.get("pre_caption") and 
            sample.get("pre_caption") != sample.get("final_caption")):
            pairs_to_generate.append((
                sample["final_caption"],
                sample["pre_caption"],
                "final_vs_pre"
            ))
        
        # 4. Generate all samples with randomized order
        for final_caption, other_caption, selection_type in pairs_to_generate:
            # Randomize order
            if random.random() < 0.5:
                caption_1 = final_caption
                caption_2 = other_caption
                correct_choice = self.choice_1_token
                order = "final_first"
            else:
                caption_1 = other_caption
                caption_2 = final_caption
                correct_choice = self.choice_2_token
                order = "final_second"
            
            training_sample = {
                "messages": [
                    {
                        "content": f"{self.video_prefix}{format_template.format(caption_1=caption_1, caption_2=caption_2)}",
                        "role": "user"
                    },
                    {
                        "content": correct_choice,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "selection_type": selection_type,
                    "caption_order": order,
                    "correct_choice": correct_choice
                }
            }
            training_samples.append(training_sample)
        
        # 5. Return final training_samples list
        return training_samples

# NOTE: Apart from Case 1, I have to check if pre_caption and annotator_negative_pre_caption are the same. Not 100% sure on how
# to use samples where the two pre captions are different.
class PairwiseCritiqueSelectionFormat(BaseFormat):
    """Format for pairwise critique selection tasks."""
    
    description = "Pairwise selection between two critiques"
    output_type = "Critique Selection"
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format_template using constants from base class
        format_template = (
            f"{self.pairwise_context}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {{caption}}\n\n"
            f"Critique 1: {{critique_1}}\n\n"
            f"Critique 2: {{critique_2}}\n\n"
            f"{self.choice_suffix}"
        )
        
        # 3. Build list of (caption, better_critique, worse_critique, selection_type, order_prefix) tuples
        critique_pairs = []
        
        # Case 1: critique exists (PERFECT_CRITIQUE will be a negative regardless)
        if sample.get("critique"):
            critique_pairs.append((
                sample['pre_caption'],
                sample['critique'],
                self.perfect_caption_critique,
                "constructive_vs_perfect",
                "constructive"
            ))

        
        # 4. Generate all samples with randomized order
        for caption, better_critique, worse_critique, selection_type, order_prefix in critique_pairs:
            # Randomize order
            if random.random() < 0.5:
                critique_1 = better_critique
                critique_2 = worse_critique
                correct_choice = self.choice_1_token
                order = f"{order_prefix}_first"
            else:
                critique_1 = worse_critique
                critique_2 = better_critique
                correct_choice = self.choice_2_token
                order = f"{order_prefix}_second"
            
            training_sample = {
                "messages": [
                    {
                        "content": f"{self.video_prefix}{format_template.format(caption=caption, critique_1=critique_1, critique_2=critique_2)}",
                        "role": "user"
                    },
                    {
                        "content": correct_choice,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "selection_type": selection_type,
                    "critique_order": order,
                    "correct_choice": correct_choice,
                    "caption_source": "pre_caption"
                }
            }
            training_samples.append(training_sample)
        
        # 5. Return final training_samples list
        return training_samples


class PairwiseCaptionSelectionWithGivenCritiqueFormat(BaseFormat):
    """Format for pairwise caption selection with given critiques."""
    
    description = "Pairwise caption selection with critique context"
    output_type = "Caption Selection with Critiques"
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format_template using constants from base class
        format_template = (
            f"{self.pairwise_context}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"Caption 1: {{caption_1}}\n"
            f"Critique 1: {{critique_1}}\n\n"
            f"Caption 2: {{caption_2}}\n"
            f"Critique 2: {{critique_2}}\n\n"
            f"{self.choice_suffix}"
        )
        
        # 3. Build list of (final_caption, other_caption, other_critique, selection_type) tuples
        pairs_to_generate = []
        
        # final_caption + PERFECT_CRITIQUE vs pre_caption + critique
        if (sample.get("final_caption") and
            sample.get("pre_caption") and
            sample.get("critique") and
            sample.get("pre_caption") != sample.get("final_caption")):
            pairs_to_generate.append((
                sample["final_caption"],
                sample["pre_caption"],
                sample["critique"],
                "final_vs_pre_with_critiques"
            ))
      
        # 4. Generate all samples with randomized order
        for final_caption, other_caption, other_critique, selection_type in pairs_to_generate:
            # Randomize order
            if random.random() < 0.5:
                caption_1 = final_caption
                critique_1 = self.perfect_caption_critique
                caption_2 = other_caption
                critique_2 = other_critique
                correct_choice = self.choice_1_token
                order = "final_first"
            else:
                caption_1 = other_caption
                critique_1 = other_critique
                caption_2 = final_caption
                critique_2 = self.perfect_caption_critique
                correct_choice = self.choice_2_token
                order = "final_second"
            
            training_sample = {
                "messages": [
                    {
                        "content": f"{self.video_prefix}{format_template.format(caption_1=caption_1, critique_1=critique_1, caption_2=caption_2, critique_2=critique_2)}",
                        "role": "user"
                    },
                    {
                        "content": correct_choice,
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "selection_type": selection_type,
                    "caption_order": order,
                    "correct_choice": correct_choice
                }
            }
            training_samples.append(training_sample)
        
        # 5. Return final training_samples list
        return training_samples


class PairwiseCaptionSelectionWithSelfCritiqueFormat(BaseFormat):
    """Format for pairwise caption selection with self-generated critiques."""
    
    description = "Pairwise caption selection with self-generated critique context"
    output_type = "Caption Selection with Self Critiques"
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format_template using constants from base class
        format_template = (
            f"{self.pairwise_context}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"Caption 1: {{caption_1}}\n\n"
            f"Caption 2: {{caption_2}}\n\n"
            f"{self.self_critique_format_choice}"
        )
        
        # 3. Build list of (final_caption, other_caption, other_critique, selection_type) tuples
        pairs_to_generate = []
        
        # final_caption vs pre_caption with self-generated critiques
        if (sample.get("final_caption") and
            sample.get("pre_caption") and
            sample.get("critique") and
            sample.get("pre_caption") != sample.get("final_caption")):
            pairs_to_generate.append((
                sample["final_caption"],
                sample["pre_caption"],
                sample["critique"],
                "final_vs_pre_with_self_critiques"
            ))
        
        # 4. Generate all samples with randomized order
        for final_caption, other_caption, other_critique, selection_type in pairs_to_generate:
            # Randomize order
            if random.random() < 0.5:
                caption_1 = final_caption
                caption_2 = other_caption
                correct_choice = self.choice_1_token
                order = "final_first"
                critique_1 = self.perfect_caption_critique
                critique_2 = other_critique
            else:
                caption_1 = other_caption
                caption_2 = final_caption
                correct_choice = self.choice_2_token
                order = "final_second"
                critique_1 = other_critique
                critique_2 = self.perfect_caption_critique
            
            response_json = {
                "critique_1": critique_1,
                "critique_2": critique_2,
                "choice": correct_choice
            }
            
            training_sample = {
                "messages": [
                    {
                        "content": f"{self.video_prefix}{format_template.format(caption_1=caption_1, caption_2=caption_2)}",
                        "role": "user"
                    },
                    {
                        "content": json.dumps(response_json),
                        "role": "assistant"
                    }
                ],
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "selection_type": selection_type,
                    "caption_order": order,
                    "correct_choice": correct_choice,
                    "critique_type": "self_generated"
                }
            }
            training_samples.append(training_sample)
        
        # 5. Return final training_samples list
        return training_samples

class DPOCaptionGenerationFormat(BaseFormat):
    """DPO format for caption generation with good/bad caption pairs."""
    
    description = "Direct Preference Optimization for caption generation"
    output_type = "Preference Pair"
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format_template using constants from base class
        format_template = task_instruction
        
        # 3. Build list of (chosen_caption, rejected_caption, negative_source) tuples
        preference_pairs = []
        
        # final_caption vs pre_caption (if different) - i.e. pre_caption_score is less than 5
        if sample.get("pre_caption_score") < 5:
            preference_pairs.append((
                sample["final_caption"],
                sample["pre_caption"],
                "pre_caption"
            ))
        
        # final_caption vs worst_caption (if exists)
        if sample.get("worst_caption"):
            preference_pairs.append((
                sample["final_caption"],
                sample["worst_caption"],
                "worst_caption"
            ))
        
        # 4. Generate all DPO preference pairs
        for chosen_caption, rejected_caption, negative_source in preference_pairs:
            training_sample = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{self.video_prefix}{format_template}"
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": chosen_caption
                },
                "rejected": {
                    "from": "gpt", 
                    "value": rejected_caption
                },
                "videos": [f"captioning_videos/{sample['video_id']}"],
                "images": [],
                "_metadata": {
                    "video_id": sample["video_id"],
                    "caption_type": sample.get("caption_type"),
                    "negative_source": negative_source
                }
            }
            training_samples.append(training_sample)
        
        # 5. Return final training_samples list
        return training_samples


class DPOCritiqueGenerationFormat(BaseFormat):
    """DPO format for critique generation with good/bad critique pairs."""
    
    description = "Direct Preference Optimization for critique generation"
    output_type = "Preference Pair"
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Set up format_template using constants from base class
        format_template = (
            f"{self.critique_generation_context} "
            f"\"{self.perfect_caption_critique}\"\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {{caption}}"
        )
        
        # 3. Only generate samples when pre_caption_score < 5 (mirrors CritiqueYesOrNoFormat)
        if sample.get("pre_caption_score") >= 5:
            return training_samples
        
        # Verify required fields exist
        assert sample.get("insertion_error_critique"), "insertion_error_critique missing for pre_caption_score < 5"
        assert sample.get("replacement_error_critique"), "replacement_error_critique missing for pre_caption_score < 5"
        
        chosen_critique = sample.get("critique")
        if not chosen_critique:
            # If no critique available, skip this sample
            return training_samples
        
        # 4. Generate preference pairs mirroring CritiqueYesOrNoFormat structure
        # Mirrors: final_caption + perfect_caption_critique -> "Yes" (positive)
        # DPO equivalent: perfect_caption_critique (chosen) vs inaccurate critique (rejected) for final_caption
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"{self.video_prefix}{format_template.format(caption=sample['final_caption'])}"
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": self.perfect_caption_critique
            },
            "rejected": {
                "from": "gpt",
                "value": sample[random.choice(["insertion_error_critique", "replacement_error_critique"])]
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "final_caption",
                "chosen_critique_source": "perfect_caption_critique",
                "rejected_critique_source": "inaccurate_critique",
                "scenario": "perfect_caption"
            }
        }
        training_samples.append(training_sample)
        
        # Mirrors: pre_caption + critique -> "Yes" (positive) vs pre_caption + perfect_caption_critique -> "No" (negative)
        # DPO equivalent: critique (chosen) vs perfect_caption_critique (rejected) for pre_caption
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"{self.video_prefix}{format_template.format(caption=sample['pre_caption'])}"
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_critique
            },
            "rejected": {
                "from": "gpt",
                "value": self.perfect_caption_critique
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "pre_caption",
                "chosen_critique_source": "critique",
                "rejected_critique_source": "perfect_caption_critique",
                "scenario": "imperfect_caption"
            }
        }
        training_samples.append(training_sample)
        
        # Mirrors: pre_caption + critique -> "Yes" (positive) vs pre_caption + inaccurate_critique -> "No" (negative)
        # DPO equivalent: critique (chosen) vs randomly selected inaccurate critique (rejected) for pre_caption
        selected_inaccurate = random.choice(["insertion_error_critique", "replacement_error_critique"])
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"{self.video_prefix}{format_template.format(caption=sample['pre_caption'])}"
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_critique
            },
            "rejected": {
                "from": "gpt",
                "value": sample[selected_inaccurate]
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "pre_caption",
                "chosen_critique_source": "critique",
                "rejected_critique_source": selected_inaccurate,
                "scenario": "imperfect_caption"
            }
        }
        training_samples.append(training_sample)
        
        # 5. Return final training_samples list
        return training_samples
class DPOCaptionYesOrNoFormat(BaseFormat):
    """DPO format for binary Yes/No caption alignment scoring."""
    
    description = "Direct Preference Optimization for Yes/No caption alignment"
    output_type = "Preference Pair"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for Yes/No caption evaluation.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to evaluate
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_evaluation_context} "
            f"{self.yesno_suffix}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Only generate samples when pre_caption_score < 5 (mirrors CaptionYesOrNoFormat)
        if sample.get("pre_caption_score") >= 5:
            return training_samples
        
        # 3. Generate preference pairs mirroring CaptionYesOrNoFormat structure
        # Mirrors: final_caption -> "Yes" (positive) vs pre_caption -> "No" (negative)
        # DPO Pair 1: For final_caption, "Yes" (chosen) vs "No" (rejected)
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample["final_caption"])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": self.yes_token
            },
            "rejected": {
                "from": "gpt",
                "value": self.no_token
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "final_caption"
            }
        }
        training_samples.append(training_sample)
        
        # DPO Pair 2: For pre_caption, "No" (chosen) vs "Yes" (rejected)
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample["pre_caption"])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": self.no_token
            },
            "rejected": {
                "from": "gpt",
                "value": self.yes_token
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "pre_caption"
            }
        }
        training_samples.append(training_sample)
        
        # 4. Return final training_samples list
        return training_samples


class DPOCaptionYesOrNoWithGivenCritiqueFormat(BaseFormat):
    """DPO format for binary Yes/No scoring of captions given critiques."""
    
    description = "Direct Preference Optimization for Yes/No caption alignment with given critique"
    output_type = "Preference Pair"
    
    def format_prompt(self, task_instruction: str, caption: str, critique: str) -> str:
        """
        Format prompt for Yes/No caption evaluation with given critique.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to evaluate
            critique: The critique context
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_evaluation_context} "
            f"{self.yesno_suffix}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}\n\n"
            f"{self.critique_header} {critique}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Only generate samples when pre_caption_score < 5 (mirrors CaptionYesOrNoWithGivenCritiqueFormat)
        if sample.get("pre_caption_score") >= 5:
            return training_samples
        
        # 3. Generate preference pairs mirroring CaptionYesOrNoWithGivenCritiqueFormat structure
        # Mirrors: final_caption + perfect_caption_critique -> "Yes" (positive)
        # DPO Pair 1: For final_caption + perfect_critique, "Yes" (chosen) vs "No" (rejected)
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['final_caption'], self.perfect_caption_critique)
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": self.yes_token
            },
            "rejected": {
                "from": "gpt",
                "value": self.no_token
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "final_caption",
                "critique_source": "perfect_caption_critique"
            }
        }
        training_samples.append(training_sample)
        
        # Mirrors: pre_caption + critique -> "No" (negative)
        # DPO Pair 2: For pre_caption + critique, "No" (chosen) vs "Yes" (rejected)
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['pre_caption'], sample['critique'])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": self.no_token
            },
            "rejected": {
                "from": "gpt",
                "value": self.yes_token
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "pre_caption",
                "critique_source": "critique"
            }
        }
        training_samples.append(training_sample)
        
        # 4. Return final training_samples list
        return training_samples


class DPOCaptionYesOrNoWithSelfCritiqueLastSimpleFormat(BaseFormat):
    """DPO format for binary Yes/No scoring with critique-last (answer first, then critique)."""
    
    description = "Direct Preference Optimization for Yes/No with self-critique appearing after answer"
    output_type = "Preference Pair"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for Yes/No evaluation with self-generated critique (answer first).
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to evaluate
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_evaluation_context}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Only generate samples when pre_caption_score < 5 (mirrors CaptionYesOrNoWithSelfCritiqueLastSimple)
        if sample.get("pre_caption_score") >= 5:
            return training_samples
        
        # 3. Generate preference pairs mirroring CaptionYesOrNoWithSelfCritiqueLastSimple structure
        # Format: Answer first, then critique as explanation (e.g., "Yes. The caption is perfect...")
        
        # Mirrors: final_caption -> "Yes" + perfect_caption_critique (positive)
        # DPO Pair 1: For final_caption, correct response (chosen) vs incorrect response (rejected)
        chosen_response = f"{self.yes_token} {self.perfect_caption_critique}"
        rejected_response = f"{self.no_token} {sample['critique']}"
        
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['final_caption'])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_response
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_response
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "final_caption",
                "critique_source": "perfect_caption_critique"
            }
        }
        training_samples.append(training_sample)
        
        # Mirrors: pre_caption -> "No" + critique (negative)
        # DPO Pair 2: For pre_caption, correct response (chosen) vs incorrect response (rejected)
        chosen_response = f"{self.no_token} {sample['critique']}"
        rejected_response = f"{self.yes_token} {self.perfect_caption_critique}"
        
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['pre_caption'])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_response
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_response
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "pre_caption",
                "critique_source": "critique"
            }
        }
        training_samples.append(training_sample)
        
        # 4. Return final training_samples list
        return training_samples


class DPOCaptionYesOrNoWithSelfCritiqueFirstSimpleFormat(BaseFormat):
    """DPO format for binary Yes/No scoring with critique-first (critique as reasoning, then answer)."""
    
    description = "Direct Preference Optimization for Yes/No with self-critique appearing before answer"
    output_type = "Preference Pair"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for Yes/No evaluation with self-generated critique (critique first).
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to evaluate
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_evaluation_context}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Only generate samples when pre_caption_score < 5 (mirrors CaptionYesOrNoWithSelfCritiqueFirstSimple)
        if sample.get("pre_caption_score") >= 5:
            return training_samples
        
        # 3. Generate preference pairs mirroring CaptionYesOrNoWithSelfCritiqueFirstSimple structure
        # Format: Critique first, then answer with label (e.g., "The caption is perfect... The answer is: Yes")
        
        # Mirrors: final_caption -> perfect_caption_critique + "Yes" (positive)
        # DPO Pair 1: For final_caption, correct response (chosen) vs incorrect response (rejected)
        chosen_response = f"{self.perfect_caption_critique} {self.answer_label} {self.yes_token}"
        rejected_response = f"{sample['critique']} {self.answer_label} {self.no_token}"
        
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['final_caption'])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_response
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_response
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "final_caption",
                "critique_source": "perfect_caption_critique"
            }
        }
        training_samples.append(training_sample)
        
        # Mirrors: pre_caption -> critique + "No" (negative)
        # DPO Pair 2: For pre_caption, correct response (chosen) vs incorrect response (rejected)
        chosen_response = f"{sample['critique']} {self.answer_label} {self.no_token}"
        rejected_response = f"{self.perfect_caption_critique} {self.answer_label} {self.yes_token}"
        
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['pre_caption'])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_response
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_response
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "pre_caption",
                "critique_source": "critique"
            }
        }
        training_samples.append(training_sample)
        
        # 4. Return final training_samples list
        return training_samples
        
class DPOCritiqueYesOrNoFormat(BaseFormat):
    """DPO format for binary Yes/No scoring of critique accuracy."""
    
    description = "Direct Preference Optimization for Yes/No critique scoring"
    output_type = "Preference Pair"
    
    def format_prompt(self, task_instruction: str, caption: str, critique: str) -> str:
        """
        Format prompt for Yes/No critique evaluation.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption being critiqued
            critique: The critique to evaluate
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.critique_evaluation_context} "
            f"{self.yesno_suffix}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}\n\n"
            f"{self.critique_header} {critique}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # Assert required fields
        assert sample.get("pre_caption"), f"No pre_caption found for sample {sample.get('video_id', 'unknown')}"
        assert sample.get("final_caption"), f"No final_caption found for sample {sample.get('video_id', 'unknown')}"
        
        # 2. Only generate samples when pre_caption_score < 5 (mirrors CritiqueYesOrNoFormat)
        if sample.get("pre_caption_score") >= 5:
            return training_samples
        
        # Verify required fields exist
        assert sample.get("insertion_error_critique"), "insertion_error_critique missing for pre_caption_score < 5"
        assert sample.get("replacement_error_critique"), "replacement_error_critique missing for pre_caption_score < 5"
        
        # 3. Generate preference pairs mirroring CritiqueYesOrNoFormat structure
        
        # Mirrors: final_caption + perfect_caption_critique -> "Yes" (positive)
        # DPO Pair 1: For final_caption + perfect_critique, "Yes" (chosen) vs "No" (rejected)
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['final_caption'], self.perfect_caption_critique)
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": self.yes_token
            },
            "rejected": {
                "from": "gpt",
                "value": self.no_token
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "final_caption",
                "critique_source": "perfect_caption_critique"
            }
        }
        training_samples.append(training_sample)
        
        # Mirrors: pre_caption + critique -> "Yes" (positive)
        # DPO Pair 2: For pre_caption + critique, "Yes" (chosen) vs "No" (rejected)
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['pre_caption'], sample['critique'])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": self.yes_token
            },
            "rejected": {
                "from": "gpt",
                "value": self.no_token
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "pre_caption",
                "critique_source": "critique"
            }
        }
        training_samples.append(training_sample)
        
        # Mirrors: pre_caption + perfect_caption_critique -> "No" (negative)
        # DPO Pair 3: For pre_caption + perfect_critique, "No" (chosen) vs "Yes" (rejected)
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['pre_caption'], self.perfect_caption_critique)
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": self.no_token
            },
            "rejected": {
                "from": "gpt",
                "value": self.yes_token
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "pre_caption",
                "critique_source": "perfect_caption_critique"
            }
        }
        training_samples.append(training_sample)
        
        # Mirrors: pre_caption + randomly selected inaccurate critique -> "No" (negative)
        # DPO Pair 4: For pre_caption + inaccurate critique, "No" (chosen) vs "Yes" (rejected)
        selected_inaccurate = random.choice(["insertion_error_critique", "replacement_error_critique"])
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['pre_caption'], sample[selected_inaccurate])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": self.no_token
            },
            "rejected": {
                "from": "gpt",
                "value": self.yes_token
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "caption_source": "pre_caption",
                "critique_source": selected_inaccurate
            }
        }
        training_samples.append(training_sample)
        
        # 4. Return final training_samples list
        return training_samples


class DPOCaptionCorrectionFormat(BaseFormat):
    """DPO format for caption correction with good/bad correction pairs."""
    
    description = "Direct Preference Optimization for caption correction"
    output_type = "Preference Pair"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for caption correction.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to correct
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_correction_context.format(perfect_caption_statement=self.perfect_caption_statement)}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Only generate samples when pre_caption_score < 5 (mirrors CaptionCorrectionFormat)
        if sample.get("pre_caption_score") >= 5:
            return training_samples
        
        # 3. Generate preference pairs mirroring CaptionCorrectionFormat structure
        
        # Mirrors: final_caption -> final_caption (no_correction_needed)
        # DPO Pair 1: For final_caption, keep same (chosen) vs make worse (rejected)
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['final_caption'])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": sample["final_caption"]
            },
            "rejected": {
                "from": "gpt",
                "value": sample["pre_caption"]
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "input_caption_source": "final_caption",
                "scenario": "no_correction_needed"
            }
        }
        training_samples.append(training_sample)
        
        # Mirrors: pre_caption -> final_caption (improvement)
        # DPO Pair 2: For pre_caption, improve (chosen) vs leave unchanged (rejected)
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['pre_caption'])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": sample["final_caption"]
            },
            "rejected": {
                "from": "gpt",
                "value": sample["pre_caption"]
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "input_caption_source": "pre_caption",
                "scenario": "correction_needed"
            }
        }
        training_samples.append(training_sample)
        
        # 4. Return final training_samples list
        return training_samples


class DPOCaptionCorrectionWithGivenCritiqueFormat(BaseFormat):
    """DPO format for critique-guided caption correction."""
    
    description = "Direct Preference Optimization for caption correction with given critique"
    output_type = "Preference Pair"
    
    def format_prompt(self, task_instruction: str, caption: str, critique: str) -> str:
        """
        Format prompt for caption correction with given critique.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to correct
            critique: The critique context
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_correction_context.format(perfect_caption_statement=self.perfect_caption_statement)}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}\n\n"
            f"{self.critique_header} {critique}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Only generate samples when pre_caption_score < 5 (mirrors CaptionCorrectionWithGivenCritiqueFormat)
        if sample.get("pre_caption_score") >= 5:
            return training_samples
        
        # 3. Generate preference pairs mirroring CaptionCorrectionWithGivenCritiqueFormat structure
        
        # Mirrors: final_caption + perfect_caption_critique -> final_caption (no_correction_needed)
        # DPO Pair 1: For final_caption + perfect_critique, keep same (chosen) vs make worse (rejected)
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['final_caption'], self.perfect_caption_critique)
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": sample["final_caption"]
            },
            "rejected": {
                "from": "gpt",
                "value": sample["pre_caption"]
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "input_caption_source": "final_caption",
                "critique_source": "perfect_caption_critique",
                "scenario": "no_correction_needed"
            }
        }
        training_samples.append(training_sample)
        
        # Mirrors: pre_caption + critique -> final_caption (improvement)
        # DPO Pair 2: For pre_caption + critique, improve (chosen) vs leave unchanged (rejected)
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['pre_caption'], sample['critique'])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": sample["final_caption"]
            },
            "rejected": {
                "from": "gpt",
                "value": sample["pre_caption"]
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "input_caption_source": "pre_caption",
                "critique_source": "critique",
                "scenario": "correction_needed"
            }
        }
        training_samples.append(training_sample)
        
        # 4. Return final training_samples list
        return training_samples


class DPOCaptionCorrectionWithSelfCritiqueFormat(BaseFormat):
    """DPO format for caption correction with self-generated critiques."""
    
    description = "Direct Preference Optimization for caption correction with self-generated critique"
    output_type = "Preference Pair"
    
    def format_prompt(self, task_instruction: str, caption: str) -> str:
        """
        Format prompt for caption correction with self-generated critique.
        
        Args:
            task_instruction: The task instruction text
            caption: The caption to correct
            
        Returns:
            Formatted prompt string
        """
        return (
            f"{self.video_prefix}"
            f"{self.caption_correction_context.format(perfect_caption_critique=self.perfect_caption_critique, perfect_caption_statement=self.perfect_caption_statement)}\n\n"
            f"{self.task_instruction_header} {task_instruction}\n\n"
            f"{self.caption_header} {caption}"
        )
    
    def generate_samples(self, sample: Dict[str, Any], task_instruction: str) -> List[Dict[str, Any]]:
        # 1. Initialize training_samples list
        training_samples = []
        
        # 2. Only generate samples when pre_caption_score < 5 (mirrors CaptionCorrectionWithSelfCritiqueFormat)
        if sample.get("pre_caption_score") >= 5:
            return training_samples
        
        # 3. Generate preference pairs mirroring CaptionCorrectionWithSelfCritiqueFormat structure
        # Response format: "Critique: <critique>\nImproved Caption: <caption>"
        
        # Mirrors: final_caption -> perfect_caption_critique + final_caption (no_correction_needed)
        # DPO Pair 1: For final_caption, correct response (chosen) vs incorrect response (rejected)
        chosen_response = f"Critique: {self.perfect_caption_critique}\nImproved Caption: {sample['final_caption']}"
        rejected_response = f"Critique: {sample['critique']}\nImproved Caption: {sample['pre_caption']}"
        
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['final_caption'])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_response
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_response
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "input_caption_source": "final_caption",
                "critique_source": "perfect_caption_critique",
                "scenario": "no_correction_needed"
            }
        }
        training_samples.append(training_sample)
        
        # Mirrors: pre_caption -> critique + final_caption (improvement)
        # DPO Pair 2: For pre_caption, correct response (chosen) vs incorrect response (rejected)
        chosen_response = f"Critique: {sample['critique']}\nImproved Caption: {sample['final_caption']}"
        rejected_response = f"Critique: {self.perfect_caption_critique}\nImproved Caption: {sample['pre_caption']}"
        
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_prompt(task_instruction, sample['pre_caption'])
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen_response
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_response
            },
            "videos": [f"captioning_videos/{sample['video_id']}"],
            "images": [],
            "_metadata": {
                "video_id": sample["video_id"],
                "caption_type": sample.get("caption_type"),
                "input_caption_source": "pre_caption",
                "critique_source": "critique",
                "scenario": "correction_needed"
            }
        }
        training_samples.append(training_sample)
        
        # 4. Return final training_samples list
        return training_samples

# Hierarchical dictionary with categories and formats
# To find, copy, or edit a particular one, search for the name of the corresponding class
AVAILABLE_FORMATS = {
    "Generation": {
        "caption_generation": CaptionGenerationFormat,
        "critique_generation": CritiqueGenerationFormat
    },
    "VQAScore_Captioning": {
        "caption_yes_or_no": CaptionYesOrNoFormat,
        "caption_yes_or_no_with_given_critique": CaptionYesOrNoWithGivenCritiqueFormat,
        "caption_yes_or_no_with_self_critique": CaptionYesOrNoWithSelfCritiqueFormat,
        "caption_yes_or_no_with_self_critique_two_step": CaptionYesOrNoWithSelfCritiqueTwoStepFormat,
        "caption_yes_or_no_with_self_critique_last_simple": CaptionYesOrNoWithSelfCritiqueLastSimple,
        "caption_yes_or_no_with_self_critique_first_simple": CaptionYesOrNoWithSelfCritiqueFirstSimple
    },
    "VQAScore_Critique": {
        "critique_yes_or_no": CritiqueYesOrNoFormat,
    },
    "Caption_Correction": {
        "caption_correction": CaptionCorrectionFormat,
        "caption_correction_with_given_critique": CaptionCorrectionWithGivenCritiqueFormat,
        "caption_correction_with_self_critique": CaptionCorrectionWithSelfCritiqueFormat
    },
    "Caption_Scoring": {
        "caption_scoring": CaptionScoringFormat,
        "caption_scoring_with_given_critique": CaptionScoringWithGivenCritiqueFormat,
        "caption_scoring_with_self_critique": CaptionScoringWithSelfCritiqueFormat,
    },
    # Skip pairwise formats.
    # "Pairwise_Selection": {
    #     "pairwise_caption_selection": PairwiseCaptionSelectionFormat,
    #     "pairwise_critique_selection": PairwiseCritiqueSelectionFormat,
    #     "pairwise_caption_selection_with_given_critique": PairwiseCaptionSelectionWithGivenCritiqueFormat,
    #     "pairwise_caption_selection_with_self_critique": PairwiseCaptionSelectionWithSelfCritiqueFormat
    # },
    "Preference": {
        "dpo_caption_generation": DPOCaptionGenerationFormat,
        "dpo_critique_generation": DPOCritiqueGenerationFormat,
        "dpo_caption_yes_or_no": DPOCaptionYesOrNoFormat,
        "dpo_caption_yes_or_no_with_given_critique": DPOCaptionYesOrNoWithGivenCritiqueFormat,
        "dpo_caption_yes_or_no_with_self_critique_last_simple": DPOCaptionYesOrNoWithSelfCritiqueLastSimpleFormat,
        "dpo_caption_yes_or_no_with_self_critique_first_simple": DPOCaptionYesOrNoWithSelfCritiqueFirstSimpleFormat,
        "dpo_critique_yes_or_no": DPOCritiqueYesOrNoFormat,
        "dpo_caption_correction": DPOCaptionCorrectionFormat,
        "dpo_caption_correction_with_given_critique": DPOCaptionCorrectionWithGivenCritiqueFormat,
        "dpo_caption_correction_with_self_critique": DPOCaptionCorrectionWithSelfCritiqueFormat,
    }
}

# Our method: everything minus pairwise
# Baseline: everything minuse pairwise minus everything with critiques


# Format name to number mapping for conciseness of the output file name - i.e. training_data_generator_v2.py
FORMAT_NAME_TO_NUM = {
    "caption_generation": "f01",
    "critique_generation": "f02",
    "caption_yes_or_no": "f03",
    "caption_yes_or_no_with_given_critique": "f04",
    "caption_yes_or_no_with_self_critique": "f05",
    "critique_yes_or_no": "f06",
    "critique_yes_or_no_v2": "f07",
    "caption_correction": "f08",
    "caption_correction_with_given_critique": "f09",
    "caption_correction_with_self_critique": "f10",
    "caption_scoring": "f11",
    "caption_scoring_with_given_critique": "f12",
    "caption_scoring_with_self_critique": "f13",
    "pairwise_caption_selection": "f14",
    "pairwise_critique_selection": "f15",
    "pairwise_caption_selection_with_given_critique": "f16",
    "pairwise_caption_selection_with_self_critique": "f17",
    "dpo_preference": "f18",
    "rlhf_v": "f19"
}

# Number to format name mapping (reverse lookup)
NUM_TO_FORMAT_NAME = {v: k for k, v in FORMAT_NAME_TO_NUM.items()}

def get_format_number(format_name: str) -> str:
    """Get numerical code for a format name."""
    return FORMAT_NAME_TO_NUM.get(format_name, format_name)

def get_format_name_from_number(format_num: str) -> str:
    """Get format name from numerical code."""
    return NUM_TO_FORMAT_NAME.get(format_num, format_num)

def convert_format_names_to_numbers(format_names: list) -> list:
    """Convert a list of format names to their numerical codes."""
    return [get_format_number(name) for name in format_names]

def convert_format_numbers_to_names(format_numbers: list) -> list:
    """Convert a list of format numbers back to their names."""
    return [get_format_name_from_number(num) for num in format_numbers]

# Necessary format functions:
def get_available_formats() -> Dict[str, Dict[str, Any]]:
    """Return the hierarchical dictionary of all available formats."""
    return AVAILABLE_FORMATS

def get_format_by_name(format_name: str) -> Optional[BaseFormat]:
    """Get a format instance by its format name."""
    for category in AVAILABLE_FORMATS.values():
        if format_name in category:
            format_class = category[format_name]
            # return format_class()
            return format_class
    return None

def list_all_format_names() -> List[str]:
    """Get a flat list of all format names."""
    format_names = []
    for category in AVAILABLE_FORMATS.values():
        format_names.extend(category.keys())
    return format_names
