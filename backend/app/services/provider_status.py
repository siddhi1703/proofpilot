"""Translate known provider failures without exposing provider response bodies."""


def public_provider_problem(error: BaseException) -> str | None:
    current = error
    for _ in range(6):
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            details = body.get("error", body)
            code = str(details.get("code", "")) if isinstance(details, dict) else ""
            if code in {"AllocationQuota.FreeTierOnly", "Arrearage", "InsufficientBalance", "insufficient_quota"}:
                return "The AI service quota is exhausted. The service owner must restore provider access before audits can run."
        status = getattr(current, "status_code", None)
        if status == 429:
            return "The AI service is busy or rate limited. Please try again later."
        if status in {401, 403}:
            return "The AI service credentials or permissions need attention from the service owner."
        if status is not None and status >= 500:
            return "The AI provider is temporarily unavailable. Please try again later."
        if current.__cause__ is None:
            return None
        current = current.__cause__
    return None
