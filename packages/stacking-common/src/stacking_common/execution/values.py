"""Small validation helpers for durable execution values."""

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


def required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def sha256_text(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def plain_json(value):
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("record keys must be strings")
        return {key: plain_json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return plain_json(value.tolist())
    if isinstance(value, np.generic):
        return plain_json(value.item())
    if isinstance(value, (tuple, list)):
        return [plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported record value: {type(value).__name__}")


def canonical_json(value):
    return json.dumps(plain_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def freeze_json(value):
    if not isinstance(value, Mapping):
        raise TypeError("diagnostics/profiles must be a mapping")
    values = json.loads(canonical_json(value))

    def freeze(item):
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(values)
