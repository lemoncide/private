from __future__ import annotations


def normalize_base_url(base_url: str) -> str:
    value = (base_url or "").strip()
    for wrapper in ("`", '"', "'"):
        if len(value) >= 2 and value[0] == wrapper and value[-1] == wrapper:
            value = value[1:-1].strip()
    value = value.strip("`").strip()
    return value
