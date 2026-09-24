"""Action Finder — classifies a raw clinical query into action types.

Used in-process by /v2/retrieval-signals to gate each query BEFORE the
retrieval pipeline runs (see app.utils.action_gate). The classification goes
through the shared ContextualIntentPipeline model client so it inherits the
service's model allow-list, location routing, credentials, tracing and
LLM error classification instead of building a second Vertex client.

Output contract (consumed by action_gate.evaluate_gate_async):

    {"actions": [str, ...], "is_processable": bool, "usage_metadata": {...}}
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app import config
from app.prompts.v2.action_finder import ACTION_FINDER_PROMPT
from app.utils.action_gate import ACTION_TYPES, PROCESSABLE_ACTION, normalize_actions

# Classification output is a handful of tokens; the budget only needs to cover
# any model-side thinking ahead of the JSON body.
ACTION_FINDER_MAX_TOKENS = 8192


class ActionFinderOutput(BaseModel):
    """Structured-output schema for the Action Finder call."""

    actions: list[str] = Field(
        default_factory=list, description="Selected action types, primary first"
    )
    is_processable: bool = Field(
        False, description="true iff information_retrieval is among actions"
    )


class ActionFinderError(RuntimeError):
    """The Action Finder could not produce a usable classification."""


def normalize_output(raw: Any) -> dict[str, Any]:
    """Model output -> {"actions": [...], "is_processable": bool}, invariants enforced.

    Raises ActionFinderError when the response cannot be interpreted.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ActionFinderError(f"response is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ActionFinderError(f"unexpected response type: {type(raw).__name__}")

    actions: list[str] = [
        a for a in normalize_actions(raw.get("actions")) if a in ACTION_TYPES
    ]
    if not actions:
        raise ActionFinderError("model returned no recognizable actions")

    # out_of_scope is exclusive by taxonomy rule; enforce it here too.
    if "out_of_scope" in actions:
        actions = ["out_of_scope"]

    return {
        "actions": actions,
        "is_processable": PROCESSABLE_ACTION in actions,
    }


class ActionFinder:
    """Runs the Action Finder prompt through the pipeline's model client."""

    def __init__(self, pipeline, logger=None):
        self._pipeline = pipeline
        self.lg = logger

    def resolve_model(self, model_name: str | None = None) -> str:
        """ACTION_FINDER_MODEL override > request model > service default."""
        return config.ACTION_FINDER_MODEL or model_name or self._pipeline.model_name

    async def find_actions_async(
        self,
        query_text: str,
        model_name: str | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Classify one raw query.

        Returns {"actions": [...], "is_processable": bool, "usage_metadata": {...}}.
        Raises ActionFinderError on an unusable response and propagates LLM
        errors; the retrieval-signals gate applies its fail-open policy to both.
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ActionFinderError("query_text is empty")

        prompt = ACTION_FINDER_PROMPT.format(query=query_text.strip())
        raw_response, usage_metadata = await self._pipeline._call_model_async(
            prompt,
            model_name=self.resolve_model(model_name),
            # Deterministic classification: the request's generation_config
            # applies to the retrieval pipeline, not to the gate.
            generation_config=None,
            location=location,
            temperature=0.0,
            max_tokens=ACTION_FINDER_MAX_TOKENS,
            step_name="action_finder",
            response_schema=ActionFinderOutput,
        )
        result = normalize_output(raw_response)
        result["usage_metadata"] = usage_metadata or {}
        return result
