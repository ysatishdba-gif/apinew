"""
Generation config utilities for LLM calls.

Provides:
- build_generation_config: Merges user config with base config
- extract_usage_metadata: Extracts token usage for audit logging
"""

from typing import Any

from google.genai import types

# Allowed keys for generation config
_ALLOWED_GEN_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
    "stop_sequences",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "thinking_config",
}


def _safe_int(value: Any) -> int:
    """Safely convert a value to int, returning 0 on failure."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_thinking_config(value: Any) -> Any:
    """
    Build a ThinkingConfig from a dict with thinking_budget.

    Args:
        value: Dict with thinking_budget key

    Returns:
        types.ThinkingConfig instance

    Raises:
        ValueError: If thinking_level is used or thinking_budget is missing
    """
    if not isinstance(value, dict):
        return value

    # Reject thinking_level explicitly
    if "thinking_level" in value:
        raise ValueError(
            "thinking_level is not allowed. Only 'thinking_budget' is supported."
        )

    # Extract budget
    budget = value.get("thinking_budget")

    if budget is None:
        raise ValueError("thinking_config must include 'thinking_budget'")

    try:
        return types.ThinkingConfig(thinking_budget=int(budget))
    except Exception as e:
        raise ValueError(f"Invalid thinking_config: {e}") from e


def build_generation_config(
    user_cfg: dict[str, Any] | None,
    base_cfg: dict[str, Any],
    *,
    max_output_limit: int | None = None,
) -> dict[str, Any]:
    """
    Merge a user-supplied generation_config into a base config dict.

    Only a restricted set of keys is allowed; any others will raise.

    Args:
        user_cfg: User-provided generation config (optional)
        base_cfg: Base/default generation config
        max_output_limit: Optional cap for max_output_tokens

    Returns:
        Merged config dict ready for GenerateContentConfig

    Raises:
        ValueError: If user_cfg contains unsupported keys
    """
    cfg = dict(base_cfg)

    if not user_cfg:
        return cfg

    # Validate allowed keys
    invalid = set(user_cfg.keys()) - _ALLOWED_GEN_KEYS
    if invalid:
        raise ValueError(
            f"generation_config contains unsupported keys: {sorted(invalid)}. "
            f"Allowed keys are: {sorted(_ALLOWED_GEN_KEYS)}"
        )

    for key, val in user_cfg.items():
        if key == "thinking_config":
            val = _build_thinking_config(val)
        if key == "response_schema":
            continue
        cfg[key] = val

    # Cap max_output_tokens to model limit
    if max_output_limit is not None:
        user_max = cfg.get("max_output_tokens")
        if user_max is not None:
            try:
                cfg["max_output_tokens"] = min(int(user_max), int(max_output_limit))
            except (TypeError, ValueError):
                cfg["max_output_tokens"] = max_output_limit

    return cfg


def extract_usage_metadata(response: Any) -> dict[str, int]:
    """
    Extract token usage metadata from LLM response for audit logging.

    Args:
        response: LLM response object

    Returns:
        Dict with token counts:
        - prompt_token_count: Tokens in the input prompt
        - candidates_token_count: Tokens in the generated output (response text)
        - total_token_count: Total tokens used (prompt + candidates + thinking)
        - thinking_token_count: Tokens used for thinking/reasoning (if enabled)

    Note:
        candidates_token_count represents the tokens in the model's generated response.
        This is the output text that was produced by the model.
    """
    md = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)

    if md is None:
        return {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
            "thinking_token_count": 0,
        }

    # Thinking tokens (different SDK versions use different names)
    thinking = getattr(md, "thinking_tokens", None) or getattr(
        md, "thoughts_token_count", None
    )

    return {
        "prompt_token_count": _safe_int(getattr(md, "prompt_token_count", 0)),
        "candidates_token_count": _safe_int(getattr(md, "candidates_token_count", 0)),
        "total_token_count": _safe_int(getattr(md, "total_token_count", 0)),
        "thinking_token_count": _safe_int(thinking if thinking is not None else 0),
    }
