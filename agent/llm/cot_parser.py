import json
import re
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class CoTParseError(ValueError):
    def __init__(self, message: str, *, preview: str = "", length: int = 0):
        super().__init__(message)
        self.preview = preview
        self.length = length


def extract_thinking(text: str) -> Optional[str]:
    matches = re.findall(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    joined = "\n\n".join(m.strip() for m in matches if m and m.strip())
    return joined or None


def _balanced_json_substring(text: str) -> Optional[str]:
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False

    for j in range(start, len(text)):
        c = text[j]
        if in_string:
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = False
            continue

        if c == '"':
            in_string = True
            continue

        if c == opening:
            depth += 1
            continue
        if c == closing:
            depth -= 1
            if depth == 0:
                return text[start : j + 1]

    return None


def extract_json_candidate(text: str) -> str:
    fenced_json = re.search(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced_json:
        return fenced_json.group(1).strip()

    fenced_any = re.search(r"```\s*([\s\S]*?)```", text)
    if fenced_any:
        candidate = fenced_any.group(1).strip()
        balanced = _balanced_json_substring(candidate)
        return (balanced or candidate).strip()

    balanced = _balanced_json_substring(text)
    return (balanced or text).strip()


def parse_structured(text: str, model_class: Type[T]) -> T:
    json_str = extract_json_candidate(text)
    try:
        return model_class.model_validate_json(json_str)
    except Exception:
        try:
            obj = json.loads(json_str)
            return model_class.model_validate(obj)
        except Exception as e:
            preview = (text or "")[:800]
            raise CoTParseError(f"Invalid structured output: {e}", preview=preview, length=len(text or "")) from e
