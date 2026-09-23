"""Location resolution for routing LLM calls to a GCP region.

Precedence (highest first):
  1. Explicit per-request ``location``
  2. Model-family map (longest matching prefix) - MODEL_LOCATION_DEFAULTS
  3. ``config_location`` (e.g. GCP_LOCATION)
  4. ``fallback`` (DEFAULT_LOCATION)
"""

VALID_LOCATIONS = {"us", "us-central1"}

DEFAULT_LOCATION = "us-central1"

# Model-name-prefix -> location overrides for model families that need a
# different region than DEFAULT_LOCATION.
# gemini-2.5+ is available in the global "us" endpoint (resolved to us-central1).
MODEL_LOCATION_DEFAULTS: dict[str, str] = {
    "gemini-3": "us",
}


def resolve_model_location(
    model,
    location,
    config_map,
    config_location,
    fallback=DEFAULT_LOCATION,
    logger=None,
):
    """Return (resolved_model, resolved_location).

    Location precedence:
      1. Explicit per-request ``location``
      2. Model-family map (longest matching prefix)
      3. ``config_location`` (e.g. GCP_LOCATION)
      4. ``fallback`` (default ``us-central1``)
    """
    resolved_model = model

    if location:
        return resolved_model, location

    mapped = None
    if model and config_map:
        matches = [k for k in config_map if isinstance(k, str) and model.startswith(k)]
        if matches:
            best = max(matches, key=len)
            mapped = config_map[best]

    if mapped:
        if config_location and config_location != mapped and logger:
            from aie_logging import Severity

            logger.log_text(
                f"model_locations map selected location {mapped!r} for model {model!r}, "
                f"overriding configured GCP_LOCATION {config_location!r}; pass location= "
                "explicitly to override",
                severity=Severity.WARNING,
            )
        return resolved_model, mapped

    if config_location:
        return resolved_model, config_location

    return resolved_model, fallback
