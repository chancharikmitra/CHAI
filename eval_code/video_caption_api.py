#!/usr/bin/env python3
"""
Single-video API for caption generation and reward scoring.
Leverages HLC codebase formats, constants, and task instructions directly.
"""
from dotenv import load_dotenv
load_dotenv()

from typing import Dict, Any, Optional, List, Tuple
import json
import inspect
from pathlib import Path

# Import from HLC codebase
from eval_code.formats import get_format_by_name, list_all_format_names, get_available_formats, BaseFormat
from eval_code.constants import get_constants_from_config, CONSTANTS


# Mapping of format names to their scoring configuration
# This defines how to extract the Yes/No token probability for each format
SCORING_CONFIG = {
    # Standard Yes/No formats - answer is first generated token
    "caption_yes_or_no": {
        "score_position": "start",
        "add_space_before_answer": False,
    },
    "caption_yes_or_no_with_given_critique": {
        "score_position": "start",
        "add_space_before_answer": False,
    },
    "critique_yes_or_no": {
        "score_position": "start",
        "add_space_before_answer": False,
    },
    
    # Self-critique last: "Yes. {critique}" - answer is first generated token
    "caption_yes_or_no_with_self_critique_last_simple": {
        "score_position": "start",
        "add_space_before_answer": False,
    },
    
    # Self-critique first: "{critique} The answer is: Yes" - answer comes after generated text
    "caption_yes_or_no_with_self_critique_first_simple": {
        "score_position": "end",
        "add_space_before_answer": True,
    },
}


REWARD_MODES = {
    "direct": {
        "method": "caption_reward",
        "kwargs": {},
        "passes_per_rollout": 1,
    },
    "critique_first": {
        "method": "caption_reward_with_self_critique",
        "kwargs": {"critique_first": True, "mode": "score"},
        "passes_per_rollout": 1,
    },
    "critique_last": {
        "method": "caption_reward_with_self_critique",
        "kwargs": {"critique_first": False, "mode": "score"},
        "passes_per_rollout": 1,
    },
    "self_critique_conditioned": {
        "method": "self_critique_conditioned_reward",
        "kwargs": {},
        "passes_per_rollout": 2,
    },
    "human_critique_conditioned": {
        "method": "caption_reward_with_given_critique",
        "kwargs": {},  # caller must supply critique=
        "passes_per_rollout": 1,
    },
}

# Default max_new_tokens by model type
_API_MODEL_MAX_NEW_TOKENS = 65536
_LOCAL_MODEL_MAX_NEW_TOKENS = 2048


class VideoCaptionAPI:
    """
    API for running inference on single videos using HLC formats.
    
    Uses the same format classes as training data generation,
    ensuring prompt consistency between training and inference.
    """
    
    def __init__(self,
                 constants_config: Dict[str, List[str]],
                 task_instructions: Dict[str, Dict[str, str]],
                 model: Any = None,
                 model_name: Optional[str] = None,
                 checkpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 project_id: Optional[str] = None,):
        """
        Initialize the API.
        
        Args:
            constants_config: Dict mapping constant names to variant lists
                Example: {"VIDEO_PREFIX": ["angle_bracket"], "YES_TOKEN": ["capital"]}
            task_instructions: Dict with structure {style: {caption_type: instruction}}
            model: Pre-initialized model (if provided, model_name/checkpoint ignored)
            model_name: Name of model to initialize (e.g., 'qwen2.5-vl-7b', 'qwen3-vl-8b')
            checkpoint: Optional checkpoint path for local models
            api_key: Optional API key for API-based models
            project_id: Optional project_id for Gemini Vertex AI Models
        """
        # Resolve constants to actual string values
        self.constants = get_constants_from_config(constants_config)
        self.task_instructions = task_instructions
        
        # Store raw config for overrides
        self._constants_config = constants_config
        
        # Store model name for default parameter selection
        self._model_name = model_name
        
        # Cache for format instances
        self._format_cache: Dict[str, BaseFormat] = {}
        
        # Initialize model
        if model is not None:
            self.model = model
        elif model_name is not None:
            self._init_model(model_name, checkpoint, api_key, project_id)
        else:
            self.model = None
    
    def _is_api_model(self) -> bool:
        """Check if the current model is an API-based model (Gemini, GPT)."""
        if self._model_name is None:
            return False
        name = self._model_name.lower()
        return name.startswith("gemini") or name.startswith("gpt")
    
    def _default_max_new_tokens(self, format_name: str, mode: str) -> int:
        """
        Get the default max_new_tokens based on model type, format, and mode.
        
        For generate mode:
            - API models (Gemini/GPT): 65536
            - Local models: 2048
        For score mode:
            - self-critique formats: 512 (need tokens for critique + answer)
            - simple Yes/No formats: 1
        """
        if mode == "score":
            # Score mode defaults are functional, not arbitrary caps
            if "self_critique" in format_name:
                return 512
            else:
                return 1
        
        # Generate mode: use model-appropriate ceiling
        if self._is_api_model():
            return _API_MODEL_MAX_NEW_TOKENS
        return _LOCAL_MODEL_MAX_NEW_TOKENS
    
    def _init_model(self, model_name: str, checkpoint: Optional[str], api_key: Optional[str], project_id: Optional[str] = None):
        try:
            import t2v_metrics
        except ImportError:
            raise ImportError("t2v_metrics required for model inference.")

        print(f"Initializing model: {model_name}")
        if checkpoint:
            print(f"Checkpoint: {checkpoint}")

        checkpoint = self._resolve_checkpoint(checkpoint)

        kwargs = {}
        if api_key:
            kwargs['api_key'] = api_key
        if checkpoint:
            kwargs['checkpoint'] = checkpoint
        if project_id:
            kwargs['project_id'] = project_id

        self.model = t2v_metrics.VQAScore(model=model_name, **kwargs)
        print(f"Model initialized successfully")

    @staticmethod
    def _resolve_checkpoint(checkpoint: Optional[str]) -> Optional[str]:
        """
        Resolve a checkpoint path for HuggingFace from_pretrained compatibility.
        
        Handles three cases:
            - None → returned as-is
            - Local path (e.g., /data/checkpoints/global_step250) → returned as-is
            - Standard HF repo (e.g., namespace/repo) → returned as-is
            - HF repo with subfolder → raises error with guidance
        
        # TODO: Support remote intermediate checkpoints. This requires either:
        # 1. Training with save_safetensors=true so intermediate checkpoints are
        #    saved in HF format (not raw DeepSpeed ZeRO shards), or
        # 2. Passing subfolder directly to from_pretrained (requires changes in
        #    t2v_metrics / qwen3vl_model.py to forward subfolder kwarg).
        """
        if checkpoint is None:
            return None
        
        # Local absolute or relative path — return as-is
        if checkpoint.startswith("/") or checkpoint.startswith("./") or Path(checkpoint).exists():
            return checkpoint
        
        # HF-style path: split on "/"
        parts = checkpoint.split("/")
        if len(parts) <= 2:
            # Standard "namespace/repo" or just "repo" — valid HF repo ID
            return checkpoint
        
        # More than 2 segments — not a valid HF repo ID
        raise ValueError(
            f"Checkpoint '{checkpoint}' looks like a HF repo with a subfolder path. "
            f"This is not directly supported by from_pretrained.\n"
            f"Options:\n"
            f"  1. Use the final consolidated checkpoint: '{'/'.join(parts[:2])}'\n"
            f"  2. Use a local path to converted weights (e.g., /path/to/global_step250/)\n"
            f"  3. Convert on the training machine: "
            f"python zero_to_fp32.py <checkpoint_dir> <output_dir> --tag <step> --safe_serialization"
        )
            
    def get_format(self, 
                   format_name: str, 
                   overrides: Optional[Dict[str, List[str]]] = None) -> BaseFormat:
        """
        Get a format instance with resolved constants.
        
        Args:
            format_name: Name of the format (e.g., "caption_yes_or_no")
            overrides: Optional constant overrides for this format
                Example: {"CAPTION_EVALUATION_CONTEXT": ["given_critique"]}
        
        Returns:
            Initialized format instance
        """
        # Create cache key
        override_key = tuple(sorted((k, tuple(v)) for k, v in overrides.items())) if overrides else None
        cache_key = (format_name, override_key)
        
        if cache_key not in self._format_cache:
            # Start with base constants
            constants = dict(self.constants)
            
            # Apply overrides if provided
            if overrides:
                override_resolved = get_constants_from_config(overrides)
                constants.update(override_resolved)
            
            # Get format class and instantiate
            format_class = get_format_by_name(format_name)
            if format_class is None:
                available = list_all_format_names()
                raise ValueError(f"Unknown format: '{format_name}'. Available: {available}")
            
            self._format_cache[cache_key] = format_class(constants_config=constants)
        
        return self._format_cache[cache_key]
    
    def get_task_instruction(self, 
                             instruction_style: str, 
                             caption_type: str) -> str:
        """
        Get task instruction text.
        
        Args:
            instruction_style: Style key (e.g., 'human', 'human_short')
            caption_type: Caption type (e.g., 'subject', 'scene', 'motion')
        
        Returns:
            Task instruction string
        """
        if instruction_style not in self.task_instructions:
            available = list(self.task_instructions.keys())
            raise ValueError(f"Unknown instruction style: '{instruction_style}'. Available: {available}")
        
        style_instructions = self.task_instructions[instruction_style]
        
        if caption_type not in style_instructions:
            available = list(style_instructions.keys())
            raise ValueError(f"Unknown caption type: '{caption_type}'. Available: {available}")
        
        return style_instructions[caption_type]
    
    def format_prompt(self,
                      format_name: str,
                      instruction_style: str,
                      caption_type: str,
                      overrides: Optional[Dict[str, List[str]]] = None,
                      **kwargs) -> str:
        """
        Format a prompt for inference.
        
        Args:
            format_name: Name of the format (e.g., "caption_generation", "caption_yes_or_no")
            instruction_style: Task instruction style (e.g., "human", "human_short")
            caption_type: Caption type (e.g., "subject", "scene", "motion", "camera", "spatial")
            overrides: Optional constant overrides for this format
            **kwargs: Format-specific inputs:
                - caption: For evaluation/correction/scoring formats
                - critique: For formats that use critiques
        
        Returns:
            Formatted prompt string ready for model input
        """
        format_instance = self.get_format(format_name, overrides)
        task_instruction = self.get_task_instruction(instruction_style, caption_type)
        
        return format_instance.format_prompt(task_instruction=task_instruction, **kwargs)
    
    def generate(self,
                 video_path: str,
                 prompt: str,
                 fps: float = 8.0,
                 max_new_tokens: int = 2048,
                 **generate_kwargs) -> str:
        """
        Generate a response from the model.
        
        Args:
            video_path: Path to video file
            prompt: Formatted prompt string
            fps: Frames per second for video processing
            max_new_tokens: Maximum tokens to generate
            **generate_kwargs: Additional arguments for model.generate()
        
        Returns:
            Model response string
        """
        if self.model is None:
            raise RuntimeError("No model initialized. Pass model to __init__ or set model_name.")
        
        response = self.model.model.generate(
            images=[video_path],
            texts=[prompt],
            fps=fps,
            max_new_tokens=max_new_tokens,
            **generate_kwargs
        )
        
        return response[0] if isinstance(response, list) else response
        
    def score(self,
            video_path: str,
            prompt: str,
            answer: str,
            fps: float = 8.0,
            max_new_tokens: int = 1,
            temperature: float = 1.0,
            score_position: str = "start",
            add_space_before_answer: bool = False,
            do_sample: bool = False,
            sample_temperature: float = 0.7) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Compute VQA score (probability of answer token) with trace.
        
        Args:
            video_path: Path to video file
            prompt: Formatted prompt string
            answer: Expected answer token (e.g., "Yes", "No")
            fps: Frames per second for video processing
            max_new_tokens: Maximum tokens to generate for scoring
            temperature: Post-hoc temperature for probability calibration
            score_position: Where to extract the score from:
                - "start": Score the first n tokens of generation
                - "end": Score the last n tokens
            add_space_before_answer: If True, prepend space to answer template
            do_sample: If True, use stochastic sampling during generation.
                Required for self-consistency (different outputs per rollout).
            sample_temperature: Sampling temperature when do_sample=True.
                Controls diversity of generated text. Ignored when do_sample=False.
        
        Returns:
            Tuple of (score, trace_dict)
                - score: Probability of answer token (float between 0 and 1)
                - trace: Dictionary with trace information from forward_with_trace
        """
        if self.model is None:
            raise RuntimeError("No model initialized. Pass model to __init__ or set model_name.")
        
        # Build answer template
        answer_template = f" {answer}" if add_space_before_answer else answer
        
        score_tensor, traces = self.model.model.forward_with_trace(
            images=[video_path],
            texts=[prompt],
            answer_template=answer_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            score_position=score_position,
            do_sample=do_sample,
            sample_temperature=sample_temperature,
        )
        
        # Extract trace (first and only trace)
        trace = traces[0] if traces else None
        
        return float(score_tensor.item()), trace
        
    def _get_scoring_config(self, format_name: str) -> Dict[str, Any]:
        """
        Get the scoring configuration for a format.
        """
        if format_name in SCORING_CONFIG:
            return SCORING_CONFIG[format_name]
        
        # Default configuration for unknown formats
        return {
            "score_position": "start",
            "add_space_before_answer": False,
        }
    
    def _is_scorable_format(self, format_name: str) -> bool:
        """Check if a format supports Yes/No token probability scoring."""
        scorable_formats = [
            "caption_yes_or_no",
            "caption_yes_or_no_with_given_critique",
            "caption_yes_or_no_with_self_critique",
            "caption_yes_or_no_with_self_critique_last_simple",
            "caption_yes_or_no_with_self_critique_first_simple",
            "critique_yes_or_no",
        ]
        return format_name in scorable_formats
    
    def run(self,
            video_path: str,
            format_name: str,
            instruction_style: str,
            caption_type: str,
            overrides: Optional[Dict[str, List[str]]] = None,
            mode: str = "generate",
            fps: float = 8.0,
            max_new_tokens: Optional[int] = None,
            temperature: float = None,
            **kwargs) -> Dict[str, Any]:
        """
        Run inference with a specific format.
        
        This is a convenience method that combines format_prompt + generate/score.
        
        Args:
            video_path: Path to video file
            format_name: Name of the format
            instruction_style: Task instruction style
            caption_type: Caption type
            overrides: Optional constant overrides
            mode: "generate" for text generation, "score" for VQA scoring
            fps: Frames per second for video processing
            max_new_tokens: Maximum tokens to generate. If None, uses model-aware
                defaults: 2048 for local models, 65536 for API models (generate mode);
                1 or 512 for score mode depending on format.
            temperature: Temperature during generation
            **kwargs: Format-specific inputs (caption, critique, etc.)
                For mode="score", also accepts:
                - answer: The answer token to score (default: uses YES_TOKEN from constants)
        
        Returns:
            Dict containing:
                - prompt: The formatted prompt
                - response: Generated text (if mode="generate")
                - score: VQA score (if mode="score")
                - trace: Trace dict from forward_with_trace (if mode="score")
                - format_name, instruction_style, caption_type: Echo of inputs
        
        Raises:
            RuntimeError: If no model is initialized
            ValueError: If format_name or mode is invalid
            Exception: Any model/inference errors propagate with full traceback
        """
        result = {
            "format_name": format_name,
            "instruction_style": instruction_style,
            "caption_type": caption_type,
            "video_path": video_path,
            "prompt": None,
        }
        # IMPORTANT: Handle Default Temperature In Case it Isn't Passed:
        if temperature is None:
            temperature = 1.0 if mode == "score" else 0.0
            
        # Get format instance
        format_instance = self.get_format(format_name, overrides)
        
        # Extract answer before passing kwargs to format_prompt
        answer = kwargs.pop("answer", None)
        
        # Extract sampling params before passing kwargs to format_prompt
        # (format classes don't accept these — they'd cause TypeError)
        do_sample = kwargs.pop("do_sample", False)
        sample_temperature = kwargs.pop("sample_temperature", 0.7)
        
        # Build prompt
        prompt = self.format_prompt(format_name, instruction_style, caption_type, overrides, **kwargs)
        result["prompt"] = prompt
        
        # Resolve max_new_tokens: user override > model-aware default
        resolved_max_new_tokens = max_new_tokens if max_new_tokens is not None else self._default_max_new_tokens(format_name, mode)
        
        if mode == "generate":
            generate_extra = {}
            if do_sample:
                generate_extra["do_sample"] = True
                resolved_temperature = sample_temperature if sample_temperature is not None else temperature
            else:
                resolved_temperature = temperature
            
            result["response"] = self.generate(
                video_path, prompt, fps=fps,
                max_new_tokens=resolved_max_new_tokens,
                temperature=resolved_temperature,
                **generate_extra,
            )
        
        elif mode == "score":
            # Get answer token (default to YES_TOKEN)
            if answer is None:
                answer = format_instance.yes_token
            result["answer"] = answer
            
            # Get scoring configuration for this format
            scoring_config = self._get_scoring_config(format_name)
            
            # Get score and trace
            score_value, trace = self.score(
                video_path=video_path,
                prompt=prompt,
                answer=answer,
                fps=fps,
                max_new_tokens=resolved_max_new_tokens,
                temperature=temperature,
                score_position=scoring_config["score_position"],
                add_space_before_answer=scoring_config["add_space_before_answer"],
                do_sample=do_sample,
                sample_temperature=sample_temperature,
            )
            result["score"] = score_value
            result["trace"] = trace
        
        else:
            raise ValueError(f"Unknown mode: '{mode}'. Use 'generate' or 'score'.")
        
        return result
    
    # =========================================================================
    # Convenience methods for common operations
    # =========================================================================
    
    def caption(self,
                video_path: str,
                instruction_style: str,
                caption_type: str,
                fps: float = 8.0,
                max_new_tokens: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate a caption for a video.
        
        Convenience wrapper around run() with format_name="caption_generation".
        
        Returns:
            Dict containing:
                - response: Generated caption text
                - prompt: The formatted prompt
        """
        return self.run(
            video_path=video_path,
            format_name="caption_generation",
            instruction_style=instruction_style,
            caption_type=caption_type,
            mode="generate",
            fps=fps,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    def critique(self,
                 video_path: str,
                 instruction_style: str,
                 caption_type: str,
                 caption: str,
                 fps: float = 8.0,
                 max_new_tokens: Optional[int] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Generate a critique for a caption.
        
        Convenience wrapper around run() with format_name="critique_generation".
        
        Returns:
            Dict containing:
                - response: Generated critique text
                - prompt: The formatted prompt
        """
        return self.run(
            video_path=video_path,
            format_name="critique_generation",
            instruction_style=instruction_style,
            caption_type=caption_type,
            mode="generate",
            fps=fps,
            max_new_tokens=max_new_tokens,
            caption=caption,
            **kwargs
        )

    def caption_reward(self,
                    video_path: str,
                    instruction_style: str,
                    caption_type: str,
                    caption: str,
                    fps: float = 8.0,
                    **kwargs) -> Dict[str, Any]:
        """
        Compute reward score for a caption (Yes/No alignment).
        
        Uses token probability scoring with format_name="caption_yes_or_no".
        Scores the probability of the "Yes" token at the start of generation.
        
        Returns:
            Dict containing:
                - score: Float between 0 and 1 (probability of "Yes" token)
                - trace: Trace dict with generated_text, scored_tokens, probability, etc.
                - prompt: The formatted prompt
        """
        return self.run(
            video_path=video_path,
            format_name="caption_yes_or_no",
            instruction_style=instruction_style,
            caption_type=caption_type,
            mode="score",
            fps=fps,
            caption=caption,
            **kwargs
        )
    def self_critique_conditioned_reward(self,
                                        video_path: str,
                                        instruction_style: str,
                                        caption_type: str,
                                        caption: str,
                                        fps: float = 8.0,
                                        critique_temperature: float = 0.7,
                                        critique_max_new_tokens: Optional[int] = None,
                                        score_temperature: float = 1.0,
                                        **kwargs) -> Dict[str, Any]:
        """
        Two-step reward: generate a critique, then score P(Yes | critique).
        
        Step 1: Generate a critique via critique_generation format.
        Step 2: Score the caption conditioned on the generated critique
                via caption_yes_or_no_with_given_critique format.
        
        This corresponds to "Self-Critique-Conditioned" in Table 9
        (2 forward passes per rollout).
        
        Args:
            video_path: Path to video file.
            instruction_style: Task instruction style (e.g., "human").
            caption_type: Caption dimension (e.g., "subject").
            caption: The caption to evaluate.
            fps: Frames per second for video processing.
            critique_temperature: Temperature for critique generation (step 1).
                Higher values produce more diverse critiques across rollouts.
            critique_max_new_tokens: Max tokens for critique generation.
                If None, uses model-aware default.
            score_temperature: Temperature for scoring (step 2).
            **kwargs: Additional kwargs forwarded to both steps.
        
        Returns:
            Dict containing:
                - score: Float between 0 and 1 (P(Yes | generated critique))
                - trace: Trace dict from the scoring step
                - critique: The generated critique text
                - critique_prompt: Prompt used for critique generation
                - score_prompt: Prompt used for scoring
                - video_path, instruction_style, caption_type: Echo of inputs
        """
        # Step 1: Generate critique
        critique_result = self.critique(
            video_path=video_path,
            instruction_style=instruction_style,
            caption_type=caption_type,
            caption=caption,
            fps=fps,
            temperature=critique_temperature,
            max_new_tokens=critique_max_new_tokens,
            **kwargs,
        )
        generated_critique = critique_result["response"]
        
        # Step 2: Score P(Yes | generated critique)
        score_result = self.caption_reward_with_given_critique(
            video_path=video_path,
            instruction_style=instruction_style,
            caption_type=caption_type,
            caption=caption,
            critique=generated_critique,
            fps=fps,
            temperature=score_temperature,
            **kwargs,
        )
        
        return {
            "score": score_result["score"],
            "trace": score_result["trace"],
            "critique": generated_critique,
            "critique_prompt": critique_result["prompt"],
            "score_prompt": score_result["prompt"],
            "video_path": video_path,
            "instruction_style": instruction_style,
            "caption_type": caption_type,
        }

    def scaled_caption_reward(self,
                              video_path: str,
                              instruction_style: str,
                              caption_type: str,
                              caption: str,
                              mode: str = "direct",
                              n_rollouts: int = 1,
                              temperature: float = 0.7,
                              fps: float = 8.0,
                              threshold: float = 0.5,
                              critique: Optional[str] = None,
                              return_all: bool = False,
                              **kwargs) -> Dict[str, Any]:
        """
        Unified inference-time scaling for VQAScore reward modeling.
        
        Runs a reward scoring mode N times with sampling temperature and
        aggregates via self-consistency (empirical accept ratio).
        
        When n_rollouts=1, returns P(Yes) from the single rollout directly.
        
        Modes (matching Table 9):
            "direct"                    - caption_reward (1 pass)
            "critique_first"            - self-critique, critique before answer (1 pass)
            "critique_last"             - self-critique, answer before critique (1 pass)
            "self_critique_conditioned" - generate critique then score (2 passes)
            "human_critique_conditioned"- score conditioned on given critique (1 pass)
        
        Self-Consistency aggregation (n_rollouts > 1):
            score = (1/N) * sum_{i=1}^{N} 1[P_i(Yes) > threshold]
        
        Total forward passes = n_rollouts * passes_per_rollout.
        
        Args:
            video_path: Path to video file.
            instruction_style: Task instruction style.
            caption_type: Caption dimension.
            caption: The caption to evaluate.
            mode: One of the REWARD_MODES keys.
            n_rollouts: Number of independent rollouts. 1 = no self-consistency.
            temperature: Sampling temperature for generation steps.
            fps: Frames per second for video processing.
            threshold: Accept threshold for self-consistency (default 0.5).
            critique: Required for mode="human_critique_conditioned".
            return_all: If True, include per-rollout details in result.
            **kwargs: Additional kwargs forwarded to the reward method.
        
        Returns:
            Dict containing:
                - score: Float. P(Yes) if n_rollouts=1, else empirical accept ratio.
                - n_rollouts: Number of rollouts performed.
                - total_passes: Total forward passes used.
                - mode: The reward mode used.
                - rollouts: (only if return_all) List of per-rollout result dicts.
                - per_rollout_scores: (only if return_all) List of per-rollout P(Yes).
                - threshold: (only if return_all and n_rollouts>1) Accept threshold.
                - accept_count: (only if return_all and n_rollouts>1) Number accepted.
        
        Raises:
            ValueError: If mode is unknown or critique missing when required.
        """
        if mode not in REWARD_MODES:
            raise ValueError(
                f"Unknown reward mode: '{mode}'. "
                f"Choose from: {list(REWARD_MODES.keys())}"
            )
        
        mode_config = REWARD_MODES[mode]
        method_name = mode_config["method"]
        mode_kwargs = dict(mode_config["kwargs"])  # copy to avoid mutation
        passes_per_rollout = mode_config["passes_per_rollout"]
        
        # Validate critique requirement
        if mode == "human_critique_conditioned":
            if critique is None:
                raise ValueError(
                    "mode='human_critique_conditioned' requires critique= argument."
                )
            mode_kwargs["critique"] = critique
        
        # Build common kwargs for the underlying method
        call_kwargs = {
            "video_path": video_path,
            "instruction_style": instruction_style,
            "caption_type": caption_type,
            "caption": caption,
            "fps": fps,
            **mode_kwargs,
            **kwargs,
        }
        
        # Set temperature and sampling based on mode type
        if mode == "self_critique_conditioned":
            # Two-step: critique generation uses sampling, scoring does not
            call_kwargs["critique_temperature"] = temperature
            call_kwargs["score_temperature"] = 1.0
            if n_rollouts > 1:
                # critique() forwards do_sample via **kwargs to run(mode="generate")
                call_kwargs["do_sample"] = True
                call_kwargs["sample_temperature"] = temperature
        elif mode in ("critique_first", "critique_last"):
            # Self-critique modes: generate critique+answer in one pass via score path
            # Need sampling for diverse critiques across rollouts
            call_kwargs["temperature"] = temperature
            if n_rollouts > 1:
                call_kwargs["do_sample"] = True
                call_kwargs["sample_temperature"] = temperature
        else:
            # direct, human_critique_conditioned: single-token scoring
            # Sampling 1 token doesn't help — P(Yes) is the same each time
            call_kwargs["temperature"] = temperature
        
        # Get the bound method
        method = getattr(self, method_name)
        
        # Run rollouts
        rollout_results = []
        for i in range(n_rollouts):
            result = method(**call_kwargs)
            rollout_results.append(result)
        
        # Aggregate
        scores = [r["score"] for r in rollout_results]
        
        if n_rollouts == 1:
            final_score = scores[0]
        else:
            # Self-consistency: empirical accept ratio
            accepts = [1.0 if s > threshold else 0.0 for s in scores]
            final_score = sum(accepts) / len(accepts)
        
        output = {
            "score": final_score,
            "n_rollouts": n_rollouts,
            "total_passes": n_rollouts * passes_per_rollout,
            "mode": mode,
            "video_path": video_path,
            "instruction_style": instruction_style,
            "caption_type": caption_type,
        }
        
        if return_all:
            output["rollouts"] = rollout_results
            output["per_rollout_scores"] = scores
            if n_rollouts > 1:
                output["threshold"] = threshold
                output["accept_count"] = sum(1 for s in scores if s > threshold)
        
        return output
    def caption_reward_with_given_critique(self,
                                           video_path: str,
                                           instruction_style: str,
                                           caption_type: str,
                                           caption: str,
                                           critique: str,
                                           fps: float = 8.0,
                                           **kwargs) -> Dict[str, Any]:
        """
        Compute reward score for a caption given a critique.
        
        Uses token probability scoring with format_name="caption_yes_or_no_with_given_critique".
        Scores the probability of the "Yes" token at the start of generation.
        
        Returns:
            Dict containing:
                - score: Float between 0 and 1 (probability of "Yes" token)
                - trace: Trace dict with generated_text, scored_tokens, probability, etc.
                - prompt: The formatted prompt
        """
        return self.run(
            video_path=video_path,
            format_name="caption_yes_or_no_with_given_critique",
            instruction_style=instruction_style,
            caption_type=caption_type,
            overrides={"CAPTION_EVALUATION_CONTEXT": ["given_critique"]},
            mode="score",
            fps=fps,
            caption=caption,
            critique=critique,
            **kwargs
        )
    
    def caption_reward_with_self_critique(self,
                                          video_path: str,
                                          instruction_style: str,
                                          caption_type: str,
                                          caption: str,
                                          critique_first: bool = True,
                                          fps: float = 8.0,
                                          mode: str = "score",
                                          **kwargs) -> Dict[str, Any]:
        """
        Compute reward score or generate response for caption with self-critique.
        
        Args:
            video_path: Path to video file
            instruction_style: Task instruction style
            caption_type: Caption type
            caption: The caption to evaluate
            critique_first: If True, uses critique-first format (answer after "The answer is:");
                           if False, uses answer-first format (answer at start)
            fps: Frames per second for video processing
            mode: "score" for token probability scoring, "generate" for full text generation
        
        For mode="score":
            - critique_first=True: Scores "Yes" token after "The answer is:" marker
            - critique_first=False: Scores "Yes" token at start of generation
        
        Returns:
            Dict containing:
                - score: Float between 0 and 1 (if mode="score")
                - response: Generated text with critique and answer (if mode="generate")
                - trace: Trace dict (if mode="score")
                - prompt: The formatted prompt
        """
        if critique_first:
            format_name = "caption_yes_or_no_with_self_critique_first_simple"
            overrides = {
                "CAPTION_EVALUATION_CONTEXT": ["self_critique_first_simple"],
                "ANSWER_LABEL": ["standard"],
            }
        else:
            format_name = "caption_yes_or_no_with_self_critique_last_simple"
            overrides = {
                "CAPTION_EVALUATION_CONTEXT": ["self_critique_last_simple"],
                "YES_TOKEN": ["capital_period"],
                "NO_TOKEN": ["capital_period"],
            }
        
        return self.run(
            video_path=video_path,
            format_name=format_name,
            instruction_style=instruction_style,
            caption_type=caption_type,
            overrides=overrides,
            mode=mode,
            fps=fps,
            caption=caption,
            **kwargs
        )
    
    def critique_reward(self,
                        video_path: str,
                        instruction_style: str,
                        caption_type: str,
                        caption: str,
                        critique: str,
                        fps: float = 8.0,
                        **kwargs) -> Dict[str, Any]:
        """
        Compute reward score for a critique (Yes/No accuracy/constructiveness).
        
        Uses token probability scoring with format_name="critique_yes_or_no".
        Scores the probability of the "Yes" token at the start of generation.
        
        Returns:
            Dict containing:
                - score: Float between 0 and 1 (probability of "Yes" token)
                - trace: Trace dict with generated_text, scored_tokens, probability, etc.
                - prompt: The formatted prompt
        """
        return self.run(
            video_path=video_path,
            format_name="critique_yes_or_no",
            instruction_style=instruction_style,
            caption_type=caption_type,
            mode="score",
            fps=fps,
            caption=caption,
            critique=critique,
            **kwargs
        )
    
    def correct_caption(self,
                        video_path: str,
                        instruction_style: str,
                        caption_type: str,
                        caption: str,
                        fps: float = 8.0,
                        max_new_tokens: Optional[int] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Generate an improved caption (direct correction without critique).
        
        Convenience wrapper around run() with format_name="caption_correction".
        
        Returns:
            Dict containing:
                - response: Corrected caption text
                - prompt: The formatted prompt
        """
        return self.run(
            video_path=video_path,
            format_name="caption_correction",
            instruction_style=instruction_style,
            caption_type=caption_type,
            mode="generate",
            fps=fps,
            max_new_tokens=max_new_tokens,
            caption=caption,
            **kwargs
        )
    
    def correct_caption_with_self_critique(self,
                                           video_path: str,
                                           instruction_style: str,
                                           caption_type: str,
                                           caption: str,
                                           fps: float = 8.0,
                                           max_new_tokens: Optional[int] = None,
                                           **kwargs) -> Dict[str, Any]:
        """
        Generate an improved caption with self-generated critique.
        
        The model first generates a critique, then produces an improved caption.
        Response format: "Critique: ...\nImproved Caption: ..."
        
        Convenience wrapper around run() with format_name="caption_correction_with_self_critique".
        
        Returns:
            Dict containing:
                - response: Text containing critique and improved caption
                - prompt: The formatted prompt
        """
        return self.run(
            video_path=video_path,
            format_name="caption_correction_with_self_critique",
            instruction_style=instruction_style,
            caption_type=caption_type,
            overrides={
                "CAPTION_CORRECTION_CONTEXT": ["self_critique"],
                "PERFECT_CAPTION_STATEMENT": ["self_critique"]
            },
            mode="generate",
            fps=fps,
            max_new_tokens=max_new_tokens,
            caption=caption,
            **kwargs
        )
    
    def score_caption(self,
                      video_path: str,
                      instruction_style: str,
                      caption_type: str,
                      caption: str,
                      fps: float = 8.0,
                      max_new_tokens: Optional[int] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Generate a 1-5 score for a caption.
        
        Convenience wrapper around run() with format_name="caption_scoring".
        
        Returns:
            Dict containing:
                - response: Score from 1-5 (as string)
                - prompt: The formatted prompt
        """
        return self.run(
            video_path=video_path,
            format_name="caption_scoring",
            instruction_style=instruction_style,
            caption_type=caption_type,
            mode="generate",
            fps=fps,
            max_new_tokens=max_new_tokens,
            caption=caption,
            **kwargs
        )
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    @staticmethod
    def list_compatible_formats() -> List[str]:
        """
        List all formats that have implemented format_prompt() and can be used with the API.
        
        Returns:
            List of format names that support sample-level API usage
        """
        compatible_formats = []
        available_formats = get_available_formats()
        
        for category_name, formats in available_formats.items():
            for format_name, format_class in formats.items():
                # Check if the class has its own format_prompt method (not just inherited)
                if 'format_prompt' in format_class.__dict__:
                    compatible_formats.append(format_name)
        
        return compatible_formats
    
    @staticmethod
    def list_scorable_formats() -> List[str]:
        """
        List all formats that support Yes/No token probability scoring.
        
        Returns:
            List of format names that can be used with mode="score"
        """
        return list(SCORING_CONFIG.keys())
    
    @staticmethod
    def get_format_info(format_name: str) -> Dict[str, Any]:
        """
        Get information about a format's API compatibility and required inputs.
        
        Args:
            format_name: Name of the format
            
        Returns:
            Dict containing:
                - name: Format name
                - description: Format description
                - output_type: What the format outputs
                - api_compatible: Whether format_prompt() is implemented
                - scorable: Whether token probability scoring is supported
                - scoring_config: Scoring configuration (if scorable)
                - required_inputs: List of parameter names for format_prompt() (if compatible)
        """
        format_class = get_format_by_name(format_name)
        if format_class is None:
            raise ValueError(f"Unknown format: {format_name}")
        
        info = {
            "name": format_name,
            "description": getattr(format_class, 'description', ''),
            "output_type": getattr(format_class, 'output_type', ''),
            "api_compatible": 'format_prompt' in format_class.__dict__,
            "scorable": format_name in SCORING_CONFIG,
            "scoring_config": SCORING_CONFIG.get(format_name),
            "required_inputs": None
        }
        
        # If api_compatible, extract the parameters from format_prompt
        if info["api_compatible"]:
            sig = inspect.signature(format_class.format_prompt)
            # Skip 'self' and 'task_instruction' (always required)
            params = [
                p.name for p in sig.parameters.values() 
                if p.name not in ('self', 'task_instruction')
            ]
            info["required_inputs"] = params
        
        return info
    
    @staticmethod
    def print_compatible_formats():
        """Print a formatted table of all API-compatible formats and their required inputs."""
        print("=" * 100)
        print("API-Compatible Formats")
        print("=" * 100)
        print(f"{'Format Name':<50} {'Scorable':<10} {'Required Inputs':<35}")
        print("-" * 100)
        
        available_formats = get_available_formats()
        
        for category_name, formats in available_formats.items():
            compatible_in_category = []
            
            for format_name, format_class in formats.items():
                if 'format_prompt' in format_class.__dict__:
                    sig = inspect.signature(format_class.format_prompt)
                    params = [
                        p.name for p in sig.parameters.values() 
                        if p.name not in ('self', 'task_instruction')
                    ]
                    scorable = "Yes" if format_name in SCORING_CONFIG else "No"
                    compatible_in_category.append((format_name, scorable, params))
            
            if compatible_in_category:
                print(f"\n[{category_name}]")
                for format_name, scorable, params in compatible_in_category:
                    params_str = ", ".join(params) if params else "(none)"
                    print(f"  {format_name:<48} {scorable:<10} {params_str:<35}")
        
        print("\n" + "=" * 100)
        print("Notes:")
        print("  - All formats require 'task_instruction' (provided via instruction_style + caption_type)")
        print("  - 'Scorable' = supports mode='score' for Yes/No token probability extraction")
        print("=" * 100)


# =============================================================================
# Factory functions
# =============================================================================

def load_api_from_experiment(experiment_config: Dict[str, Any],
                             task_instructions_file: str = "eval_code/task_instructions.json",
                             **model_kwargs) -> VideoCaptionAPI:
    """
    Create API instance from an experiment configuration.
    
    Args:
        experiment_config: Experiment config dict (like those in data/experiments/)
        task_instructions_file: Path to task instructions JSON
        **model_kwargs: Passed to VideoCaptionAPI (model_name, checkpoint, api_key)
    
    Returns:
        Configured VideoCaptionAPI instance
    
    Example:
        from data.experiments.my_experiment import EXPERIMENTS, STANDARD_CONSTANTS
        
        api = load_api_from_experiment(
            experiment_config={"constants": STANDARD_CONSTANTS},
            model_name="qwen2.5-vl-7b"
        )
    """
    # Load task instructions
    with open(task_instructions_file, 'r') as f:
        task_instructions = json.load(f)
    
    # Get constants config
    constants_config = experiment_config.get("constants", {})
    
    return VideoCaptionAPI(
        constants_config=constants_config,
        task_instructions=task_instructions,
        **model_kwargs
    )


def load_api_with_standard_constants(task_instructions_file: str = "eval_code/task_instructions.json",
                                     **model_kwargs) -> VideoCaptionAPI:
    """
    Create API instance with standard constants (most common use case).
    
    Args:
        task_instructions_file: Path to task instructions JSON
        **model_kwargs: Passed to VideoCaptionAPI (model_name, checkpoint, api_key)
    
    Returns:
        Configured VideoCaptionAPI instance
    """
    # Standard constants (same as in experiments)
    STANDARD_CONSTANTS = {
        "VIDEO_PREFIX": ["angle_bracket"],
        "TASK_INSTRUCTION_HEADER": ["standard"],
        "CAPTION_HEADER": ["standard"],
        "CRITIQUE_HEADER": ["standard"],
        "YESNO_SUFFIX": ["polite_full"],
        "CHOICE_SUFFIX": ["explicit"],
        "PERFECT_CAPTION_CRITIQUE": ["accurate_no_edits"],
        "PERFECT_CAPTION_STATEMENT": ["standard"],
        "YES_TOKEN": ["capital"],
        "NO_TOKEN": ["capital"],
        "CHOICE_1_TOKEN": ["capital"],
        "CHOICE_2_TOKEN": ["capital"],
        "SCORE_SCALE_HINT": ["scale_1_5_compact"],
        "ANSWER_LABEL": ["standard"],
        "CAPTION_EVALUATION_CONTEXT": ["standard"],
        "CRITIQUE_GENERATION_CONTEXT": ["simple"],
        "CRITIQUE_EVALUATION_CONTEXT": ["standard"],
        "CAPTION_CORRECTION_CONTEXT": ["standard"],
        "CAPTION_SCORING_CONTEXT": ["standard"],
    }
    
    # Load task instructions
    with open(task_instructions_file, 'r') as f:
        task_instructions = json.load(f)
    
    return VideoCaptionAPI(
        constants_config=STANDARD_CONSTANTS,
        task_instructions=task_instructions,
        **model_kwargs
    )


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Example: Initialize API with standard constants (no model for prompt-only mode)
    api = load_api_with_standard_constants(
        task_instructions_file="eval_code/task_instructions.json"
    )
    
    # Print compatible formats with scoring info
    print("\n=== Compatible Formats ===\n")
    VideoCaptionAPI.print_compatible_formats()
    
    # Print scorable formats
    print("\n=== Scorable Formats (Yes/No Token Probability) ===\n")
    for fmt in VideoCaptionAPI.list_scorable_formats():
        config = SCORING_CONFIG[fmt]
        print(f"  {fmt}")
        print(f"    score_position: {config['score_position']}")
        # if config['score_after_text']:
        #     print(f"    score_after_text: '{config['score_after_text']}'")
        # if config['score_after_text']:
        #     print(f"    score_after_text: '{config['score_after_text']}'")
        print(f"    add_space_before_answer: {config['add_space_before_answer']}")
        print()
    
    # Example prompts
    print("\n=== Example Prompts ===\n")
    
    # Caption generation prompt
    prompt = api.format_prompt(
        format_name="caption_generation",
        instruction_style="human",
        caption_type="subject"
    )
    print("Caption Generation Prompt:")
    print(f"  {prompt[:100]}...\n")
    
    # Caption Yes/No prompt (scorable)
    prompt = api.format_prompt(
        format_name="caption_yes_or_no",
        instruction_style="human",
        caption_type="subject",
        caption="A dog runs across a grassy field."
    )
    print("Caption Yes/No Prompt (scorable at start):")
    print(f"  {prompt[:150]}...\n")
    
    # Self-critique first prompt (scorable after marker)
    api_with_override = load_api_with_standard_constants(
        task_instructions_file="eval_code/task_instructions.json"
    )
    prompt = api_with_override.format_prompt(
        format_name="caption_yes_or_no_with_self_critique_first_simple",
        instruction_style="human",
        caption_type="subject",
        overrides={
            "CAPTION_EVALUATION_CONTEXT": ["self_critique_first_simple"],
            "ANSWER_LABEL": ["standard"],
        },
        caption="A dog runs across a grassy field."
    )
    print("Self-Critique First Prompt (scorable after 'The answer is:'):")
    print(f"  {prompt[:200]}...\n")
