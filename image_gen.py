"""
文生图节点：分镜 -> 绘图 prompt -> 外部图片 API -> 本地文件。

第三方接口只从环境变量读取，不写死 key：
  IMAGE_API_URL / IMAGE_API_KEY / IMAGE_API_MODEL
没有配置时只生成 prompt，不阻塞主流程。
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import config
from character_reference import image_similarity, reference_block, reference_images
from prompts import IMAGE_PROMPT_SYSTEM, build_image_prompt_prompt
from state import ComicState, ImagePromptDraft
from vector_store import get_story_memory

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - langsmith 是可选依赖
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# 固定林昭锚点：即使角色卡缺字段，也保证画风切换时不变脸。
LIN_ZHAO_FALLBACK = (
    "林昭：青年男性，前临安府衙捕快；左眉有一道旧疤，随身无鞘短剑，"
    "深色旧衣；脸型、发型、眉疤在所有画风中必须保持一致。"
)


class ImageApiError(RuntimeError):
    """外部文生图接口错误。"""


def _safe(text: str, fallback: str = "image") -> str:
    value = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(text or "")).strip(" ._")
    value = re.sub(r"\s+", "_", value)
    return value[:40] or fallback


def _character_anchor(state: ComicState) -> str:
    """从状态/角色库取林昭锚点；没有就使用固定兜底。"""
    roles = state.get("canon_profiles") or []
    if not roles:
        story_id = str(state.get("story_id") or "").strip()
        if story_id:
            try:
                roles = get_story_memory().get_roles(story_id)
            except Exception:
                roles = []
    lin = next((r for r in roles if str(r.get("name")) == "林昭"), None)
    if not lin:
        return LIN_ZHAO_FALLBACK
    appearance = str(lin.get("appearance") or "").strip()
    role = str(lin.get("role") or "主角").strip()
    details = [f"林昭（{role}）"]
    if appearance and appearance != "原文未说明":
        details.append(appearance)
    details.append("左眉旧疤、无鞘短剑")
    details.append("脸型/发型/眉疤固定，不因画风改变")
    return "；".join(details)


def _base_negative() -> str:
    return config.IMAGE_NEGATIVE_BASE


def _merge_negative(extra: list[str]) -> str:
    items = [config.IMAGE_NEGATIVE_BASE]
    for item in extra:
        value = str(item or "").strip()
        if value and value not in items:
            items.append(value)
    return "，".join(items)


def _post_json_sync(url: str, payload: dict[str, Any], headers: dict[str, str]) -> tuple[bytes, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.IMAGE_TIMEOUT_SECONDS) as response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        raw = response.read()
    return raw, content_type


async def _download(url: str) -> bytes:
    def _open() -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "manju-agent/1.0"})
        with urllib.request.urlopen(request, timeout=config.IMAGE_TIMEOUT_SECONDS) as response:
            return response.read()

    return await asyncio.to_thread(_open)


@traceable(run_type="tool", name="external_image_api")
async def external_image_api(
    positive_prompt: str,
    negative_prompt: str,
    style_desc: str,
    image_size: str | None = None,
) -> bytes:
    """
    硅基流动文生图接口（异步）。

    约定：
    - URL/Key/模型从 IMAGE_API_URL / IMAGE_API_KEY / IMAGE_API_MODEL 读取；
    - 默认模型 Tongyi-MAI/Z-Image-Turbo；
    - 提交 JSON：model / prompt / negative_prompt / image_size / batch_size；
    - 兼容 images[0].url、data[0].url / b64_json 或直接二进制图片；
    - 没配置时抛 ImageApiError，由节点降级为 prompt_only。
    """
    if not config.IMAGE_API_URL or not config.IMAGE_API_KEY:
        raise ImageApiError(
            "未配置 IMAGE_API_URL / IMAGE_API_KEY；本次只生成绘图 prompt。"
        )

    payload = {
        "model": config.IMAGE_API_MODEL,
        "prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "image_size": image_size or config.IMAGE_SIZE,
        "batch_size": 1,
    }
    headers = {"Authorization": f"Bearer {config.IMAGE_API_KEY}"}
    raw, content_type = await asyncio.to_thread(
        _post_json_sync,
        config.IMAGE_API_URL,
        payload,
        headers,
    )

    if "image/" in content_type or not content_type:
        return raw

    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ImageApiError(f"文生图接口返回无法解析：{exc}") from exc

    # SiliconFlow 返回 {"images":[{"url":...}]}；兼容 OpenAI 风格 data。
    images = data.get("images") or data.get("data") or []
    if images:
        first = images[0] or {}
        if first.get("b64_json"):
            return base64.b64decode(first["b64_json"])
        if first.get("url"):
            return await _download(str(first["url"]))
    raise ImageApiError(f"文生图接口没有返回图片：{str(data)[:300]}")


def _save_image(story_id: str, episode_no: int, style_desc: str, content: bytes) -> Path:
    directory = config.OUTPUT_DIR / _safe(story_id, "default") / "images"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"第{episode_no}章_{_safe(style_desc, 'style')}.png"
    path.write_bytes(content)
    return path


def _fallback_prompt(storyboard: str, style: str, anchor: str) -> str:
    return (
        f"{style}，漫剧关键帧，角色一致性：{anchor}；"
        f"场景与动作参考：{storyboard[:900]}；"
        "电影感构图，清晰五官，干净背景，高质量"
    )


async def image_gen_node(state: ComicState) -> dict[str, Any]:
    """
    异步独立图片节点：
    1. 读取分镜 + 角色记忆 + style_desc；
    2. DeepSeek 生成 positive/negative prompt；
    3. 可选调用外部文生图 API；最多重试 IMAGE_GEN_MAX_RETRIES 次；
    4. 下载/保存本地，写回 state；
    5. 失败只记录日志与状态，继续往下走 persist。
    """
    storyboard = str(state.get("storyboard") or "").strip()
    if not storyboard:
        raise RuntimeError("文生图节点缺少分镜文本。")

    started = time.perf_counter()
    style = str(state.get("style_desc") or config.DEFAULT_STYLE_DESC).strip()
    anchor = _character_anchor(state)
    episode_no = int(state.get("episode_no") or 0)
    story_id = str(state.get("story_id") or "default").strip()

    print("\n" + "=" * 72)
    print("【文生图】")
    print("-" * 72)
    print(
        f"输入：分镜 {len(storyboard)} 字｜画风={style}｜第{episode_no}集"
        f"｜角色锚点 {len(anchor)} 字",
        flush=True,
    )

    user_negative = str(state.get("image_negative_extra") or "").strip()
    user_extra = (
        [item.strip() for item in re.split(r"[，,、;；\n]+", user_negative) if item.strip()]
        if user_negative
        else []
    )

    if config.IMAGE_PROMPT_USE_LLM:
        # 可选：让 DeepSeek 扩写 prompt；默认关闭，生图本身始终走 Z-Image。
        try:
            draft = await asyncio.to_thread(
                __import__("nodes").invoke_structured,
                node_name="文生图Prompt",
                schema=ImagePromptDraft,
                system_prompt=IMAGE_PROMPT_SYSTEM,
                user_prompt=build_image_prompt_prompt(
                    storyboard=storyboard,
                    style_desc=style,
                    character_anchor=anchor,
                    extra_negative=user_negative,
                ),
                temperature=0.4,
            )
            positive = draft.positive_prompt.strip()
            negative = _merge_negative(list(draft.negative_extra) + user_extra)
        except Exception as exc:
            positive = _fallback_prompt(storyboard, style, anchor)
            negative = _merge_negative(user_extra)
            print(f"[文生图警告] Prompt 生成失败，使用兜底：{exc}", flush=True)
    else:
        # 默认不调用 DeepSeek：本地模板直接生成绘图 prompt，实际生图由 Z-Image 完成。
        positive = _fallback_prompt(storyboard, style, anchor)
        negative = _merge_negative(user_extra)
        print("[文生图] 使用本地 prompt 模板（不调用 DeepSeek）。", flush=True)

    ref_block = reference_block(story_id)
    if ref_block:
        positive = ref_block + "\n" + positive
    refs = reference_images(story_id)
    status = "prompt_only"
    image_path = ""
    image_error = ""
    retry_count = 0

    if not config.IMAGE_API_URL or not config.IMAGE_API_KEY:
        print(
            "[文生图] 未配置 IMAGE_API_URL / IMAGE_API_KEY；本次只生成绘图 prompt。",
            flush=True,
        )
    else:
        max_retries = max(0, int(config.IMAGE_GEN_MAX_RETRIES))
        attempts = max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                content = await external_image_api(positive, negative, style)
                path = _save_image(story_id, episode_no, style, content)
                image_path = str(path)
                status = "done"
                image_error = ""
                print(f"[文生图] 图片已保存：{image_path}", flush=True)

                # 有角色基准图时，做 CLIP 图像相似度校验；不足最多重试 2 次。
                if refs:
                    check = image_similarity(story_id, image_path)
                    print(
                        f"[文生图] 角色相似度={check['similarity']}"
                        f"｜阈值={check['threshold']}｜backend={check['embedding_backend']}",
                        flush=True,
                    )
                    if not check["passed"]:
                        image_error = (
                            f"角色样貌 OOC：相似度 {check['similarity']} "
                            f"低于阈值 {check['threshold']}（{check.get('character') or '-'}）"
                        )
                        status = "character_ooc"
                        if attempt < attempts:
                            retry_count = attempt
                            print(f"[文生图] {image_error}，重试中……", flush=True)
                            await asyncio.sleep(min(2 ** (attempt - 1), 4))
                            continue
                        status = "character_ooc_failed"
                        image_path = ""
                        break
                break
            except ImageApiError as exc:
                image_error = str(exc)
                status = "prompt_only"
                print(f"[文生图] {exc}", flush=True)
                break
            except Exception as exc:
                image_error = f"{type(exc).__name__}: {exc}"
                status = "failed"
                print(
                    f"[文生图警告] 第{attempt}/{attempts}次绘图失败：{image_error}",
                    flush=True,
                )
                if attempt < attempts:
                    retry_count = attempt
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    record = {
        "story_id": story_id,
        "episode_no": episode_no,
        "style_desc": style,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "image_path": image_path,
        "status": status,
        "retry_count": retry_count,
        "error": image_error,
        "elapsed_ms": elapsed_ms,
    }
    print(f"[文生图] 状态={status}｜耗时={elapsed_ms}ms", flush=True)
    print(positive, flush=True)

    return {
        "style_desc": style,
        "image_prompt_positive": positive,
        "image_prompt_negative": negative,
        "image_path": image_path,
        "image_status": status,
        "image_retry_count": retry_count,
        "image_error": image_error,
        "image_gen_ms": elapsed_ms,
        "error_message": (
            image_error if status == "character_ooc_failed" else ""
        ),
        "image_records": [record],
        "trace": [
            f"文生图：剧本 {len(str(state.get('script') or ''))} 字，"
            f"分镜 {len(storyboard)} 字，画风={style}，"
            f"prompt {len(positive)} 字，路径={image_path or '-'}，"
            f"状态={status}，重试 {retry_count} 次，耗时 {elapsed_ms}ms"
        ],
    }


async def generate_image_from_text(
    text: str,
    style_desc: str = "",
    story_id: str = "default",
    negative_extra: str = "",
) -> dict[str, Any]:
    """直接输入一段文本出图（菜单/CLI 用），不跑完整剧本链路。"""
    state: ComicState = {
        "storyboard": text,
        "script": text,
        "style_desc": style_desc or config.DEFAULT_STYLE_DESC,
        "story_id": story_id or "default",
        "episode_no": 1,
        "image_negative_extra": negative_extra,
    }
    return await image_gen_node(state)
