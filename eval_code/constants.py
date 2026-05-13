# constants.py
from typing import Dict, List

# Type alias for better readability
ConstantVariants = Dict[str, str]
Constants = Dict[str, ConstantVariants]

CONSTANTS: Constants = {

    # ============================================================================
    # VIDEO PREFIX
    # ============================================================================
    "VIDEO_PREFIX": {
        "angle_bracket": "<video>",
        "empty": "",
    },
    
    
    # ============================================================================
    # PROMPT HEADERS
    # ============================================================================
    "TASK_INSTRUCTION_HEADER": {
        "standard": "Task Instruction:",
        "prompt": "Prompt:",
        "task": "Task:",
        "question": "Question:",
        "empty": "",
    },
    
    "CAPTION_HEADER": {
        "standard": "Caption:",
        "given": "Given Caption:",
        "empty": "",
    },
    
    "CRITIQUE_HEADER": {
        "standard": "Critique:",
        "feedback": "Feedback:",
        "empty": "",
    },

    # ============================================================================
    # INSTRUCTION SUFFIXES
    # ============================================================================
    "YESNO_SUFFIX": {
        "polite_full": "Please answer Yes or No only.",
        "tight": "Answer Yes or No only.",
        "minimal": "Yes or No?",
        "empty": "",
    },
    
    "CHOICE_SUFFIX": {
        "explicit": "Choose Caption 1 or Caption 2 by outputting either 1 or 2 respectively to indicate your choice.",
        "simple": "Choose 1 or 2.",
        "empty": "",
    },
    
    # ============================================================================
    # PERFECT CAPTION CRITIQUE
    # ============================================================================
    # "PERFECT_CAPTION_CRITIQUE_INSTRUCTION": {
    #     "if_perfect_standard": "If the caption is already accurate, output:",
    #     "if_perfect_brief": "If accurate, output:",
    #     "empty": "",
    # },

    "PERFECT_CAPTION_STATEMENT": {
        "standard": "If the original caption is already accurate, keep it exactly the same with no edits.",
        "no_changes": "If the caption already follows the task instruction well, make no changes.",
        "keep_unchanged": "If the caption is already correct, leave it unchanged.",
        "self_critique": "And keep the original caption exactly the same with no edits."
    },

    "PERFECT_CAPTION_CRITIQUE": {
        "standard": "The caption is perfect and requires no changes.",
        "no_edits": "The caption requires no edits, so leave it unchanged.",
        "accurate_same": "The caption is accurate and should stay exactly the same.",
        "accurate_no_edits": "The caption is accurate and requires no edits, so it should remain exactly the same."
    },
    
    # ============================================================================
    # YES/NO TOKENS
    # ============================================================================
    "YES_TOKEN": {
        "capital": "Yes",
        "lower": "yes",
        "capital_period": "Yes.",
    },
    
    "NO_TOKEN": {
        "capital": "No",
        "lower": "no",
        "capital_period": "No.",
    },
    
    # ============================================================================
    # CHOICE TOKENS (for pairwise selection)
    # ============================================================================
    "CHOICE_1_TOKEN": {
        "numeric": "1",
        "capital": "Caption 1",
        "ordinal": "first",
    },
    
    "CHOICE_2_TOKEN": {
        "numeric": "2",
        "capital": "Caption 2",
        "ordinal": "second",
    },

    # ============================================================================
    # SCORING SCALE HINTS
    # ============================================================================
    "SCORE_SCALE_HINT": {
        "scale_1_5_plain": "Use a scale of 1-5 where 1 is poor and 5 is excellent.",
        "scale_1_5_compact": "Rate 1–5 (1=poor, 5=excellent).",
        "scale_1_5_detailed": (
            "5 points: Completely correct; no changes needed. "
            "4 points: Excellent, with only minor adjustments required (e.g., (1) a few inaccurate words need correction, "
            "(2) a few missing words need addition, or (3) a few hallucinated words need deletion). Overall, about one "
            "sentence's worth of content may need to be added, changed, or deleted. "
            "3 points: Mostly correct, but with some important omissions, hallucinations, or errors (e.g., around two "
            "sentences need rewriting, deleting, or adding). "
            "2 points: Mostly incorrect, requiring significant revisions (e.g., about half of the caption is incorrect "
            "and needs to be redone). "
            "1 point: Almost entirely wrong, requiring a complete rewrite (e.g., the entire caption is irrelevant)."
        ),
    },

    # ============================================================================
    # TEMPLATE CONTEXTS (for assembling task instructions)
    # ============================================================================
    "CAPTION_EVALUATION_CONTEXT": { 
        "standard": "Does the video's caption accurately follow the task instruction?",
        "given_critique": "Does the video's caption accurately follow the task instruction? You are provided a critique of the caption with respect to the task instruction as context.",
        "self_critique": "Does the video's caption accurately follow the task instruction? First, provide a critique after 'Critique:', then on a new line answer only Yes or No after 'Answer:'.",
        "self_critique_last_simple": "Does the video's caption accurately follow the task instruction? Answer Yes or No followed by a critique.",
        "self_critique_first_simple": "Does the video's caption accurately follow the task instruction? Provide a critique and then answer Yes or No.",
    },
    "CRITIQUE_GENERATION_CONTEXT": {
        "simple": "Provide a critique of the video's caption based on how accurately it follows the task instruction. Point out what is wrong or missing and how to fix it. If the caption is already accurate, output:",
        "detailed": "Provide a critique of the video's caption based on how accurately it follows the task instruction. Point out what is wrong or missing in the caption and explain how to correct it by: (1) fixing inaccurate words, (2) adding necessary missing information, and (3) removing hallucinated content. Expand brief captions with descriptive details so someone who hasn't seen the video can picture it. If the caption is already accurate, output:",
        "strict": "Provide a critique of the video's caption based on how accurately it follows the task instruction. Point out precisely what is wrong or missing in the caption and explain exactly how to correct it. If the caption is already accurate, output:",
    },
    "CRITIQUE_EVALUATION_CONTEXT": {
        "standard": "Does this critique of the video’s caption provide accurate and constructive feedback to help the caption better follow the task instruction?",
        "detailed": "Does this critique of the video’s caption provide accurate and constructive (i.e. suggest corrective measures for video-caption misalignment) feedback to help the caption better follow the task instruction?",
        # "brief": "Is this critique of the video's caption accurate and constructive?",
    },

    "CAPTION_CORRECTION_CONTEXT": {
        "standard": f"Provide an improved caption that better follows the task instruction. {{perfect_caption_statement}}",
        "given_critique": f"Provide an improved caption that better follows the task instruction. You are provided a critique of the caption with respect to the task instruction as context. {{perfect_caption_statement}}",
        "self_critique": f"Provide an improved caption that better follows the task instruction, after first writing a critique. If the caption is already accurate enough, the critique should be: '{{perfect_caption_critique}}' {{perfect_caption_statement}} Write your critique after 'Critique:', then on a new line write the improved caption after 'Improved Caption:'",
    },
    
    "CAPTION_SCORING_CONTEXT": {
        "standard": "Score the video's caption based on how well it follows the task instruction.",
        "given_critique": "Score the video's caption based on how well it follows the task instruction. You are provided a critique of the caption with respect to the task instruction as context.",
        "self_critique": f"Score the video's caption based on how well it follows the task instruction. First, write a critique of how well the video's caption follows the task instruction. Then, for the score: {{score_scale_hint}} Please provide your critique after 'Critique:', and your score on a new line after 'Score:'",
    },
    
    # No more pairwise format.
    # "PAIRWISE_CONTEXT": {
    #     "caption": "Which caption better follows the task instruction:",
    #     "critique": "Which critique better evaluates the caption's alignment with the task instruction:",
    #     "self_critique": "Which caption better follows the task instruction? First provide critiques for both provided captions with respect to their alignment with the task instruction, then choose which caption is better. Provide critique for Caption 1 after 'Critique 1:', critique for Caption 2 after 'Critique 2:', and finally your choice (1 or 2) after 'Choice:'.",
    # },

    # ============================================================================
    # TEMPLATE CONTEXTS FOR SELF_CRITIQUE (Deprecated - moved above)
    # ============================================================================

    # "SELF_CRITIQUE_FORMAT_YESNO": {
    #     "json_standard": "Does the caption follow the task instruction? Format your response as a JSON with 'critique' and 'yes_or_no' fields.",
    #     "json_brief": "Does the caption follow the task instruction? Return JSON: {'critique': ..., 'yes_or_no': ...}",
    #     "nl_standard": "Does the caption follow the task instruction? First provide your critique after 'Critique:', then on a new line answer only Yes or No after 'Answer:'.",
    #     "nl_brief": "Does the caption follow the task instruction? Critique first, then answer only Yes or No.",
    # },

    # "SELF_CRITIQUE_FORMAT_SCORE": {
    #     "json_standard": "First provide a critique of the caption, then rate how well it follows the task instruction. Use a scale of 1-5 where 1 is poor and 5 is excellent. Format your response as a JSON with 'critique' and 'score' fields.",
    #     "json_brief": "Critique, then rate 1-5 (1=poor, 5=excellent). Return JSON: {'critique': ..., 'score': ...}",
    #     "nl_standard": "First provide a critique of the caption, then rate how well it follows the task instruction. Use a scale of 1-5 where 1 is poor and 5 is excellent. Provide your critique after 'Critique:', then your score after 'Score:'.",
    #     "nl_brief": "Critique first, then rate 1-5 (1=poor, 5=excellent).",
    # },

    # "SELF_CRITIQUE_FORMAT_IMPROVED": {
    #     "json_standard": f"First provide a critique of the caption, then provide an improved version. Format your response as a JSON with 'critique' and 'improved_caption' fields.",
    #     "json_brief": f"Critique, then improve. Return as a JSON: {{'critique': ..., 'improved_caption': ...}}",
    #     "nl_standard": f"First provide a critique of the caption, then provide an improved version. Provide your critique after 'Critique:', then the improved caption after 'Improved Caption:'.",
    #     "nl_brief": f"Critique first, then improve.",
    # },

    # "SELF_CRITIQUE_FORMAT_CHOICE": {
    #     "json_standard": "First provide critiques for both captions, then choose which caption is better. Choose Caption 1 or Caption 2 by outputting either 1 or 2 respectively to indicate your choice. Format your response as a JSON with 'critique_1', 'critique_2', and 'choice' fields.",
    #     "json_brief": "Critique both, then choose 1 or 2. Return JSON: {'critique_1': ..., 'critique_2': ..., 'choice': ...}",
    #     "nl_standard": "First provide critiques for both captions, then choose which caption is better. Provide critique for Caption 1 after 'Critique 1:', critique for Caption 2 after 'Critique 2:', and finally your choice (1 or 2) after 'Choice:'.",
    #     "nl_brief": "Critique both captions, then state your choice (1 or 2).",
    # },
    
    # "RESPONSE_ORDER_CRITIQUE_FIRST": {
    #     "standard": "First provide a critique, then",
    #     "brief": "Critique first, then",
    # },

    # ============================================================================
    # ANSWER LABELS (for simple self-critique formats)
    # ============================================================================
    "ANSWER_LABEL": {
        "standard": "The answer is:",
        "brief": "Answer:",
        "minimal": "A:",
        "empty": "",
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_constant(constant_name: str, variant: str = None) -> str:
    """
    Get a constant value by name and optional variant.
    
    Args:
        constant_name: The name of the constant group (e.g., "PERFECT_CRITIQUE")
        variant: The specific variant to use. If None, returns the first variant.
        
    Returns:
        The constant string value
        
    Raises:
        KeyError: If constant_name or variant not found
    """
    if constant_name not in CONSTANTS:
        raise KeyError(f"Constant '{constant_name}' not found")
    
    variants = CONSTANTS[constant_name]
    
    if variant is None:
        # Return first variant by default
        return next(iter(variants.values()))
    
    if variant not in variants:
        available = list(variants.keys())
        raise KeyError(f"Variant '{variant}' not found in '{constant_name}'. Available: {available}")
    
    return variants[variant]


def get_constants_from_config(config: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Extract constant values from an experiment configuration.
    
    Args:
        config: Dictionary mapping constant names to lists of variant names
        
    Returns:
        Dictionary mapping constant names to their resolved string values
        
    Example:
        >>> config = {"PERFECT_CRITIQUE": ["standard"], "YES_TOKEN": ["capital"]}
        >>> get_constants_from_config(config)
        {'PERFECT_CRITIQUE': 'The caption is perfect and requires no changes.', 'YES_TOKEN': 'Yes'}
    """
    resolved = {}
    
    for constant_name, variants in config.items():
        if not variants:
            # If empty list, use first variant
            resolved[constant_name] = get_constant(constant_name)
        else:
            # Use first variant specified (could extend to handle multiple variants)
            resolved[constant_name] = get_constant(constant_name, variants[0])
    
    return resolved


def validate_experiment_config(config: Dict[str, List[str]]) -> List[str]:
    """
    Validate that all constants in an experiment config are valid.
    
    Args:
        config: Experiment configuration with constant selections
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    for constant_name, variants in config.items():
        if constant_name not in CONSTANTS:
            errors.append(f"Unknown constant: '{constant_name}'")
            continue
        
        available_variants = list(CONSTANTS[constant_name].keys())
        
        for variant in variants:
            if variant not in CONSTANTS[constant_name]:
                errors.append(
                    f"Unknown variant '{variant}' for constant '{constant_name}'. "
                    f"Available: {available_variants}"
                )
    
    return errors