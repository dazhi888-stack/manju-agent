"""JSON 文件读写 + Markdown 代码块清洗。

LLM 有时会返回 ```json ... ```；所有落盘/读取都先走这里，
保证本地 JSON 不混入 Markdown 围栏。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def strip_code_fence(text: str) -> str:
    """去掉 ```json / ``` 围栏，并取第一段 JSON。"""
    value = str(text or "").strip()
    fence = re.search(
        r"```(?:json)?\s*(.*?)```",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence:
        value = fence.group(1).strip()
    if value.startswith("{") and value.endswith("}"):
        return value
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        return value[start : end + 1]
    return value


def read_json(path: str | Path, default: Any = None) -> Any:
    file = Path(path)
    if not file.exists():
        return default
    try:
        return json.loads(strip_code_fence(file.read_text(encoding="utf-8")))
    except Exception:
        return default


def write_json(path: str | Path, data: Any) -> Path:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return file
