"""Consistent API error sanitization at the final HTTP boundary."""

from __future__ import annotations


def sanitize_error_response(app, response, request_id: str):
    """Strip internal diagnostics from JSON 5xx responses by default."""
    if (
        response.status_code < 500
        or not response.is_json
        or app.config.get("EXPOSE_INTERNAL_ERRORS", False)
    ):
        return response

    payload = response.get_json(silent=True)
    if not isinstance(payload, dict):
        return response

    payload.pop("traceback", None)
    payload.setdefault("success", False)
    payload.setdefault("error", "Internal server error")
    payload.setdefault("error_code", "internal_error")
    payload.setdefault("request_id", request_id)
    response.set_data(app.json.dumps(payload))
    response.content_type = "application/json"
    return response
