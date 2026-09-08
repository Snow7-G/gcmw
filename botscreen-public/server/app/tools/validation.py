"""JSON-Schema subset validator for ToolGateway (issue #57).

The gateway validates tool parameters and results against declarative JSON
Schemas before any executor runs. Only a documented, dependency-free subset is
supported — schemas in the read-only whitelist are authored against it:

- ``type``: single value (string/integer/number/boolean/object/array/null)
- ``enum``: closed value list
- object: ``required`` / ``properties`` / ``additionalProperties: false``
- array: ``items`` (single schema) / ``minItems`` / ``maxItems``
- string: ``minLength`` / ``maxLength``
- number/integer: ``minimum`` / ``maximum``

Errors are reported as ``path: keyword`` pairs. Values are never echoed into
error text so tool parameters/results cannot leak into messages or logs.
"""

from __future__ import annotations

from typing import Any

_JSON_TYPES = ("string", "integer", "number", "boolean", "object", "array", "null")


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return False


def _check(value: Any, schema: Any, path: str, errors: list[str]) -> None:
    """Validate ``value`` against ``schema``, appending ``path: keyword``
    errors. ``schema`` must be a plain dict or ``True``/``False``."""
    if schema is True or schema == {}:
        return
    if schema is False:
        errors.append(f"{path}: schema-false")
        return
    if not isinstance(schema, dict):
        errors.append(f"{path}: invalid-schema")
        return

    type_spec = schema.get("type")
    if type_spec is not None:
        expected_types = [type_spec] if isinstance(type_spec, str) else list(type_spec)
        if not any(_type_matches(value, t) for t in expected_types):
            errors.append(f"{path}: type")
            return  # remaining keywords only apply once the type matches

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        errors.append(f"{path}: enum")
        return

    if isinstance(value, str):
        min_len = schema.get("minLength")
        max_len = schema.get("maxLength")
        if min_len is not None and len(value) < min_len:
            errors.append(f"{path}: minLength")
        if max_len is not None and len(value) > max_len:
            errors.append(f"{path}: maxLength")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            errors.append(f"{path}: minimum")
        if maximum is not None and value > maximum:
            errors.append(f"{path}: maximum")

    if isinstance(value, dict):
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}.{required}: required")
        properties = schema.get("properties", {})
        for key, item in value.items():
            prop_schema = properties.get(key)
            if prop_schema is None:
                if schema.get("additionalProperties") is False:
                    errors.append(f"{path}.{key}: additionalProperties")
                continue
            _check(item, prop_schema, f"{path}.{key}", errors)

    if isinstance(value, list):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"{path}: minItems")
        if max_items is not None and len(value) > max_items:
            errors.append(f"{path}: maxItems")
        items_schema = schema.get("items")
        if items_schema is not None:
            for index, item in enumerate(value):
                _check(item, items_schema, f"{path}[{index}]", errors)


def validate(schema: dict[str, Any] | None, value: Any) -> list[str]:
    """Return the list of violations (empty when the value is valid).

    ``None`` schema means "anything is accepted".
    """
    if not schema:
        return []
    errors: list[str] = []
    _check(value, schema, "$", errors)
    return errors
