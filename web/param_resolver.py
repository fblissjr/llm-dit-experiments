"""Unified parameter resolution: client value > config.toml > schema default.

Uses Pydantic v2's model_fields_set to distinguish explicit client values
from schema defaults. This lets config.toml act as the server-wide default
layer, overridable per-request.

The precedence rule:
    1. Client sends value explicitly  ->  use it (even if falsy: 0, 0.0, False, "")
    2. Client omits field             ->  use config.toml value
    3. Config.toml omits field        ->  schema default (already on the request object)
"""

from typing import Any

from pydantic import BaseModel


def resolve_param(
    request: BaseModel,
    field: str,
    config_value: Any,
    skip_none: bool = False,
) -> Any:
    """Resolve a generation parameter with proper precedence.

    Args:
        request: Pydantic request model instance.
        field: Field name on the request model.
        config_value: Value from the pipeline's config dataclass.
        skip_none: If True and client sends None, treat as "use config default"
                   (for Optional fields where None means "no override").
    """
    if field in request.model_fields_set:
        val = getattr(request, field)
        if skip_none and val is None:
            return config_value
        return val
    return config_value


def csv_to_int_list(csv_string: str) -> list[int]:
    """Convert a comma-separated string to a list of ints.

    Used for config fields stored as CSV strings (e.g., stg_blocks = "29,30").

    Returns empty list for empty/whitespace-only strings.
    """
    stripped = csv_string.strip()
    if not stripped:
        return []
    return [int(b.strip()) for b in stripped.split(",")]
