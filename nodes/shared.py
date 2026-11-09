"""节点共享工具：日志、结构化 LLM 调用、NodeError。"""
from __future__ import annotations

import json
import time
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

import config
from services.script_cleaner import clean_script_text

T = TypeVar("T", bound=BaseModel)
SEPARATOR = "=" * 72

class NodeError(RuntimeError):
    """节点执行失败。带上节点名，方便 CLI 直接提示“哪个环节出了问题”。"""

    def __init__(self, node_name: str, message: str) -> None:
        self.node_name = node_name
        super().__init__(f"{node_name} 失败：{message}")

def _print_header(node_index: int, node_name: str, input_summary: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"【节点{node_index}/4】{node_name}")
    print("-" * 72)
    print(f"输入：{input_summary}", flush=True)

def _print_stage(stage_name: str, input_summary: str) -> None:
    """新增分支节点的通用标题；不占用原有“节点1/4~4/4”的编号。"""
    print(f"\n{SEPARATOR}")
    print(f"【{stage_name}】")
    print("-" * 72)
    print(f"输入：{input_summary}", flush=True)

def _print_output(text: str) -> None:
    print("输出：")
    print(text, flush=True)

def _clean_rendered_script(script: str) -> str:
    """剧本生成后、进入分镜/校验前，清理重复角色名。"""
    cleaned, changed = clean_script_text(script)
    if changed:
        print(
            f"[剧本清洗] 修正 {changed} 处重复角色名/重复台词前缀。",
            flush=True,
        )
    return cleaned

def _print_usage(raw_message: Any) -> None:
    """打印 token 用量和缓存命中，方便估算成本。"""
    usage = getattr(raw_message, "usage_metadata", None) if raw_message else None
    if usage:
        print(
            f"[用量] 输入 {usage.get('input_tokens', 0)} tokens"
            f" / 输出 {usage.get('output_tokens', 0)} tokens",
            flush=True,
        )
        details = usage.get("input_token_details") or {}
        cached = (
            details.get("cache_read")
            or details.get("cached_tokens")
            or usage.get("cache_read_input_tokens")
        )
        if cached:
            print(f"[缓存] 命中 {cached} tokens", flush=True)
    if raw_message is not None:
        metadata = getattr(raw_message, "response_metadata", {}) or {}
        cache_hit = metadata.get("prompt_cache_hit_tokens")
        if cache_hit:
            print(f"[缓存] 命中 {cache_hit} tokens", flush=True)

def _print_debug_raw(raw_message: Any) -> None:
    """--debug 时打印原始模型输出；默认不打印，避免刷屏。"""
    if not config.DEBUG or raw_message is None:
        return

    content = getattr(raw_message, "content", "")
    tool_calls = getattr(raw_message, "tool_calls", None)

    print("[调试] 原始 content：", flush=True)
    print((content or "（空；结构化输出可能在 tool_calls 里）")[:2_000], flush=True)
    if tool_calls:
        print("[调试] 原始 tool_calls：", flush=True)
        try:
            print(json.dumps(tool_calls, ensure_ascii=False, indent=2)[:6_000], flush=True)
        except (TypeError, ValueError):
            print(str(tool_calls)[:6_000], flush=True)

def _is_rate_limit_error(exc: Exception) -> bool:
    """识别 429/Too Many Requests，用于免费 Qwen 自动切 Pro。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return " 429" in text or "429 " in text or "too many requests" in text or "rate limit" in text

def _is_script_balance_error(exc: Exception) -> bool:
    """识别脚本模型余额不足/欠费，直接切 DeepSeek，不再重试 Qwen。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    text = str(exc).lower()
    if status in {401, 402, 403}:
        return True
    return any(
        marker in text
        for marker in (
            "insufficient",
            "balance",
            "quota",
            "欠费",
            "余额不足",
            "arrears",
        )
    )

def invoke_structured(
    node_name: str,
    schema: type[T],
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    use_script_llm: bool = False,
) -> T:
    """
    调用 LLM，并强制返回指定 Pydantic 结构。

    - system_prompt 必须是固定前缀，动态摘要/用户输入全部在 user_prompt；
    - 剧本初稿 use_script_llm=True：硅基流动 Qwen，429 自动切兜底；
    - 校验失败后的修复轮 use_script_llm=False：切回 DeepSeek 做实质性修改；
    - 结构化解析失败最多重试一次，仍失败抛 NodeError。
    """
    if use_script_llm:
        current_model = config.SCRIPT_MODEL_FREE
        llm = config.get_script_llm(temperature, model=current_model)
    else:
        current_model = config.MODEL_NAME
        llm = config.get_llm(temperature)
    structured_llm = llm.with_structured_output(schema, include_raw=True)
    used_script_pro = False

    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    last_error: Any = None
    for attempt in range(1, 3):
        print(
            f"[LLM] {node_name} 第{attempt}次结构化调用...｜模型={current_model}",
            flush=True,
        )
        started = time.perf_counter()
        try:
            result = structured_llm.invoke(messages)
        except Exception as exc:
            if use_script_llm and not used_script_pro and _is_script_balance_error(exc):
                # 脚本模型余额不足/欠费：不阻塞新剧本，直接切 DeepSeek。
                print(
                    f"[LLM] 脚本模型 {current_model} 余额/鉴权异常，"
                    f"自动切换 DeepSeek：{config.MODEL_NAME}",
                    flush=True,
                )
                used_script_pro = True
                current_model = config.MODEL_NAME
                try:
                    llm = config.get_llm(temperature)
                    structured_llm = llm.with_structured_output(
                        schema, include_raw=True
                    )
                    result = structured_llm.invoke(messages)
                except Exception as fallback_exc:
                    raise NodeError(
                        node_name,
                        f"DeepSeek 兜底调用失败：{type(fallback_exc).__name__}: {fallback_exc}",
                    ) from fallback_exc
            elif use_script_llm and not used_script_pro and _is_rate_limit_error(exc):
                # 免费 Qwen 429：同一请求立即切 Pro 版，不消耗结构化重试次数。
                print(
                    f"[LLM] 免费模型 {current_model} 触发 429，"
                    f"自动切换 Pro 模型 {config.SCRIPT_MODEL_PRO}",
                    flush=True,
                )
                used_script_pro = True
                current_model = config.SCRIPT_MODEL_PRO
                try:
                    llm = config.get_script_llm(
                        temperature,
                        model=current_model,
                        max_retries=config.LLM_MAX_RETRIES,
                    )
                    structured_llm = llm.with_structured_output(
                        schema, include_raw=True
                    )
                    result = structured_llm.invoke(messages)
                except Exception as pro_exc:
                    raise NodeError(
                        node_name,
                        f"Qwen Pro 兜底调用失败：{type(pro_exc).__name__}: {pro_exc}",
                    ) from pro_exc
            else:
                # 网络/鉴权/参数错误通常重试也没用，直接包装成节点错误。
                raise NodeError(
                    node_name,
                    f"LLM 调用失败：{type(exc).__name__}: {exc}",
                ) from exc

        elapsed = time.perf_counter() - started

        # include_raw=True 时返回 dict；做个兼容，防止未来版本行为变化。
        if not isinstance(result, dict) or "parsed" not in result:
            if isinstance(result, schema):
                print(f"[LLM] 完成，耗时 {elapsed:.1f}s", flush=True)
                return result
            raise NodeError(node_name, "结构化输出返回了未知格式，请检查 langchain 版本。")

        raw_message = result.get("raw")
        parsed = result.get("parsed")
        parse_error = result.get("parsing_error")

        _print_usage(raw_message)
        _print_debug_raw(raw_message)
        print(f"[LLM] 完成，耗时 {elapsed:.1f}s", flush=True)

        if parsed is not None:
            return parsed

        last_error = parse_error
        finish_reason = ""
        if raw_message is not None:
            metadata = getattr(raw_message, "response_metadata", {}) or {}
            finish_reason = str(metadata.get("finish_reason", ""))

        if finish_reason == "length":
            raise NodeError(
                node_name,
                "模型输出被 max_tokens 截断。请缩短章节输入，"
                "或在 .env 中增大 MAX_TOKENS（例如 8192）后重试。",
            )

        print(
            f"[LLM警告] 第{attempt}次结构化解析失败：{str(parse_error)[:300]}",
            flush=True,
        )

        if attempt == 1:
            # 只追加一条“纠错说明”，不要把带 tool_calls 的原始 AIMessage
            # 直接放回消息列表，否则可能缺少对应的 ToolMessage，导致下一轮请求非法。
            messages.append(
                HumanMessage(
                    content=(
                        "你上一次的输出无法通过结构化解析。错误信息："
                        f"{str(parse_error)[:500]}\n"
                        "请严格按照给定的 JSON Schema 重新输出完整结果，"
                        "不要省略字段，不要输出 Markdown 代码块或解释。"
                    )
                )
            )

    raise NodeError(
        node_name,
        f"连续两次结构化解析失败。最后一次错误：{str(last_error)[:500]}。"
        "可先用 --debug 查看原始输出，再检查提示词与 Schema 是否冲突。",
    )

__all__ = [
    "NodeError",
    "SEPARATOR",
    "T",
    "_print_header",
    "_print_stage",
    "_print_output",
    "_clean_rendered_script",
    "_print_usage",
    "_print_debug_raw",
    "_is_rate_limit_error",
    "invoke_structured",
]
