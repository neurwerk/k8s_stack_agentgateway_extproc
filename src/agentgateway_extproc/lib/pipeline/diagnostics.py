"""Bounded, credential-redacted values for the existing DEBUG failure log."""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qsl, urlsplit

from neurwerk_request_segments import UnsupportedFeatureError
from neurwerk_request_segments.reasoning import CHAT_REASONING_FIELDS
from pydantic import ValidationError

from agentgateway_extproc.lib.json_limits import MAX_JSON_DEPTH, strict_json_loads

_REDACTED = "[REDACTED]"
_MAX_BYTES = 65_536
_MAX_ERRORS = 100
_MAX_SECRETS = 256
_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>\"']+")
_AUTH = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9_~+./=-]+")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_PEM = re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----.*?(?:-----END [^-]*PRIVATE KEY-----|$)", re.S)
_SECRET_LINE = re.compile(
    r"(?im)\b(?:[\w-]*authorization|[\w-]*api[_ -]?key|[\w-]*token|password|passwd|"
    r"client[_ -]?secret|cookie|set-cookie|recovery(?:[_ -]?(?:key|phrase|codes?))?|"
    r"unseal[_ -]?keys?|seed[_ -]?phrase|mnemonic)\s*[:=]\s*[^\r\n]+"
)


def _secret_key(key: str) -> bool:
    name = re.sub(r"[^a-z0-9]", "", key.lower())
    return (
        name.startswith(("recovery", "unseal"))
        or name
        in {
            "auth",
            "authentication",
            "recovery",
            "mnemonic",
            "seedphrase",
        }
        or name.endswith(
            (
                "authorization",
                "apikey",
                "token",
                "secret",
                "secretkey",
                "secretaccesskey",
                "password",
                "passwd",
                "cookie",
                "cookies",
                "credentials",
                "privatekey",
                "recoverykey",
                "recoverykeys",
                "recoverycode",
                "recoverycodes",
                "recoveryphrase",
                "unsealkey",
                "unsealkeys",
            )
        )
    )


def _credential_url(match: re.Match[str]) -> str:
    value = match.group()
    try:
        url = urlsplit(value)
        if (
            url.username is not None
            or url.password is not None
            or any(
                _secret_key(key)
                or key.lower() in {"key", "sig", "code"}
                or "signature" in key.lower()
                or "credential" in key.lower()
                for key, _ in parse_qsl(url.query) + parse_qsl(url.fragment)
            )
        ):
            return _REDACTED
    except ValueError:
        return _REDACTED
    return value


def _text(value: str) -> str:
    value = _PEM.sub(_REDACTED, value)
    value = _URL.sub(_credential_url, value)
    value = _AUTH.sub(_REDACTED, value)
    value = _JWT.sub(_REDACTED, value)
    return _SECRET_LINE.sub(_REDACTED, value)


def _remember(value: object, secrets: set[str]) -> None:
    if len(secrets) > _MAX_SECRETS:
        return
    if isinstance(value, str) and value:
        secrets.add(value)
        if re.fullmatch(r"(?i)(?:bearer|basic)\s+\S+", value):
            secrets.add(value.split(None, 1)[1])
    elif isinstance(value, dict):
        for key, child in value.items():
            if key not in {"type", "role", "status", "format"}:
                _remember(child, secrets)
    elif isinstance(value, list):
        for child in value:
            _remember(child, secrets)


def _opaque(value: dict[str, object]) -> bool:
    return value.get("type") in ("reasoning.encrypted", "redacted_thinking") or any(
        re.sub(r"[^a-z]", "", key.lower()) in {"signature", "thoughtsignature", "encryptedcontent"}
        and item is not None
        for key, item in value.items()
    )


def _redact(value: object, secrets: set[str], depth: int = 0) -> object:
    if depth > MAX_JSON_DEPTH:
        return {"truncated": True, "reason": "depth"}
    if isinstance(value, dict):
        if _opaque(value):
            _remember(value, secrets)
            return _REDACTED
        result = {}
        for key, item in value.items():
            if _secret_key(key) or (
                value.get("reasoning_signature") is not None and key in CHAT_REASONING_FIELDS
            ):
                _remember(item, secrets)
                result[_text(key)] = _REDACTED
            else:
                result[_text(key)] = _redact(item, secrets, depth + 1)
        return result
    if isinstance(value, list | tuple):
        return [_redact(item, secrets, depth + 1) for item in value]
    if isinstance(value, str):
        return _encoded_text(value, secrets, depth)
    return value


def _encoded_text(value: str, secrets: set[str], depth: int) -> str:
    # Function arguments and other JSON-encoded content need the same redaction.
    if value.lstrip().startswith(("{", "[", '"')):
        try:
            nested = strict_json_loads(value)
        except ValueError:
            # A partial encoded credential/carrier must not leak its unparsed tail.
            if re.search(
                r'(?i)"[^"\n]*(?:token|secret|cookie|signature|encrypted|recovery|key|'
                r"password|passwd|auth|credential|mnemonic|seed)|redacted_thinking|reasoning\.encrypted",
                value,
            ):
                return _REDACTED
        else:
            return json.dumps(_redact(nested, secrets, depth + 1), ensure_ascii=True)
    return _text(value)


def _scrub(value: object, secrets: list[str]) -> object:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, _REDACTED)
            value = value.replace(json.dumps(secret, ensure_ascii=True)[1:-1], _REDACTED)
        return value
    if isinstance(value, list):
        return [_scrub(item, secrets) for item in value]
    if isinstance(value, dict):
        return {str(_scrub(key, secrets)): _scrub(item, secrets) for key, item in value.items()}
    return value


def diagnostic_json(value: object, *, maximum: int) -> str:
    """Redact before truncation, including echoes of credential or opaque carrier values."""
    secrets: set[str] = set()
    redacted = _redact(value, secrets)
    if len(secrets) > _MAX_SECRETS:
        return '{"truncated":true,"reason":"redaction_limit"}'
    encoded = json.dumps(
        _scrub(redacted, sorted(secrets, key=len, reverse=True)),
        ensure_ascii=True,
        default=lambda _: _REDACTED,
    )
    limit = min(maximum, _MAX_BYTES)
    if len(encoded) <= limit:
        return encoded
    return json.dumps(
        {"truncated": True, "bytes": len(encoded), "prefix": encoded[: max(0, (limit - 128) // 6)]}
    )


def validation_details(exc: Exception) -> dict[str, object]:
    """Keep exact paths and indices, standard codes and messages; never Pydantic inputs."""
    if not isinstance(exc, ValidationError):
        return {"message": str(exc)}
    if exc.error_count() > _MAX_ERRORS:
        return {"count": exc.error_count(), "truncated": True}
    details = []
    for error in exc.errors(include_input=False, include_url=False)[:_MAX_ERRORS]:
        path = list(error["loc"])
        cause = error.get("ctx", {}).get("error")
        if isinstance(cause, UnsupportedFeatureError):
            path.append(cause.field)
        details.append({"path": path, "code": error["type"], "message": error["msg"]})
    return {
        "errors": details,
        "count": exc.error_count(),
        "truncated": exc.error_count() > len(details),
    }
