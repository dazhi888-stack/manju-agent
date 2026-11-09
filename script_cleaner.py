"""剧本输出的正则清洗：只处理台词行里的重复角色名，不改剧情内容。"""

from __future__ import annotations

import re

DIALOGUE_PREFIX = "【台词】"
# 允许出现在重复姓名之间的分隔符，例如“林昭：林昭”“林昭 林昭”“林昭（林昭）”。
_NAME_SEPARATORS = r"\s*[：:、,，/|｜（）()·\-\s]+\s*"


def clean_speaker_name(speaker: str) -> str:
    """把“林昭：林昭”“林昭 林昭”“林昭（林昭）”收敛成一个姓名。"""
    value = str(speaker or "").strip()
    if not value:
        return value

    # 先去掉“林昭（林昭）”这种括号重复。
    wrapped = re.fullmatch(r"(.{1,8}?)\s*[（(]\s*\1\s*[）)]", value)
    if wrapped:
        value = wrapped.group(1).strip()

    # 再处理“林昭：林昭”“林昭、林昭、林昭”等重复串。
    repeated = re.compile(
        rf"^(.{{1,8}}?)(?:{_NAME_SEPARATORS}\1)+$"
    )
    for _ in range(3):
        match = repeated.fullmatch(value)
        if not match:
            break
        collapsed = match.group(1).strip()
        if collapsed == value:
            break
        value = collapsed

    return value.strip(" ：:、,，/|｜")


def _strip_duplicate_name_from_content(content: str, speaker: str) -> str:
    """清掉台词正文开头重复的“角色名：”。"""
    name = re.escape(str(speaker or "").strip())
    if not name:
        return content
    value = content.lstrip()
    # “林昭：林昭：正文” / “林昭 林昭：正文” 这类重复前缀。
    repeated_prefix = re.compile(
        rf"^(?:{name}\s*[：:、,，/|｜\s]+){{2,}}"
    )
    value = repeated_prefix.sub("", value, count=1)
    # 单次“林昭：正文”通常也是渲染时重复加入的说话人，剥掉。
    single_prefix = re.compile(rf"^{name}\s*[：:]\s*")
    value = single_prefix.sub("", value, count=1)
    return value


def clean_script_text(text: str) -> tuple[str, int]:
    """
    清洗剧本 Markdown 中的重复角色名。

    返回 (清洗后文本, 被修改的台词行数)。只动 `【台词】...` 行，
    旁白、动作描述、场景标题和正文剧情都不改。
    """
    changed = 0
    output: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.lstrip()
        if not stripped.startswith(DIALOGUE_PREFIX):
            output.append(line)
            continue

        indent = line[: len(line) - len(stripped)]
        payload = stripped[len(DIALOGUE_PREFIX):]
        separator = re.search(r"[：:]", payload)
        if not separator:
            output.append(line)
            continue

        speaker_raw = payload[: separator.start()]
        content_raw = payload[separator.end():]
        speaker = clean_speaker_name(speaker_raw)
        content = _strip_duplicate_name_from_content(content_raw, speaker)
        cleaned = f"{indent}{DIALOGUE_PREFIX}{speaker}：{content}"
        if cleaned != line:
            changed += 1
        output.append(cleaned)

    return "\n".join(output), changed
