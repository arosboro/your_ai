"""
Shared utilities for model loading and generation.

This module provides common helpers used by both evaluate_checkpoint.py
and validate_model.py to avoid code duplication.
"""


def generate_with_chat_template(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    generate_fn=None,
) -> str:
    """
    Generate response with proper chat template formatting.

    Applies the tokenizer's chat template if available, otherwise uses
    the raw prompt. Works with both MLX and transformers backends.

    Args:
        model: The loaded model instance
        tokenizer: The tokenizer instance
        prompt: The user prompt to generate from
        max_tokens: Maximum tokens to generate
        generate_fn: Optional custom generate function. If None, assumes MLX
            generate is available. For transformers, pass a custom function.

    Returns:
        str: The generated response text
    """
    # Format prompt with chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    # Use provided generate function or default to MLX
    if generate_fn is not None:
        return generate_fn(model, tokenizer, formatted, max_tokens)

    # Default MLX path
    from mlx_lm import generate

    return generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens)


def create_transformers_generate_fn(model, tokenizer):
    """
    Create a generate function for transformers models.

    Returns a function compatible with generate_with_chat_template's generate_fn parameter.

    Args:
        model: The transformers model instance
        tokenizer: The transformers tokenizer instance

    Returns:
        Callable that generates text given (model, tokenizer, formatted_prompt, max_tokens)
    """

    def generate_fn(model, tokenizer, formatted_prompt, max_tokens):
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generate_fn
