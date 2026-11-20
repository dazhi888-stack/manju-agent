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
from services.character_reference import reference_block, reference_paths_for
from services.image_review import expected_characters, review_shot_image
from services.shot_prompt import parse_storyboard_shots
from prompts import IMAGE_PROMPT_SYSTEM, build_image_prompt_prompt
from state import ComicState, ImagePromptDraft
from stores.vector_store import get_story_memory

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - langsmith 是可选依赖
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# 林昭是演示故事的主角：角色卡缺外貌字段时补齐这些标志性特征。
LIN_ZHAO_FALLBACK = (
    "左眉有一道旧疤，随身无鞘短剑，深色旧衣"
)


class ImageApiError(RuntimeError):
    """外部文生图接口错误。"""


def _safe(text: str, fallback: str = "image") -> str:
    value = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(text or "")).strip(" ._")
    value = re.sub(r"\s+", "_", value)
    return value[:40] or fallback


def _load_roles(state: ComicState) -> list[dict[str, Any]]:
    """优先用状态里的角色卡；没有就从向量库角色记忆读取。"""
    roles = list(state.get("canon_profiles") or [])
    if roles:
        return roles
    story_id = str(state.get("story_id") or "").strip()
    if story_id:
        try:
            return list(get_story_memory().get_roles(story_id))
        except Exception:
            return []
    return []


def _character_anchor(state: ComicState, shot_text: str) -> str:
    """
    只为本镜头实际出场的角色生成一致性锚点。

    之前硬编码注入林昭外貌，导致其他故事（如韩月瑶线）每个镜头
    都被塞进林昭的长相——审核器报"该角色不在本镜头出镜名单中"。
    现在：出场名单来自 expected_characters（全名/去姓氏称呼匹配），
    没有角色出场的空镜头返回空串，prompt 不含任何角色。
    """
    roles = _load_roles(state)
    cast_names = expected_characters(shot_text, roles)
    if not cast_names:
        return ""
    by_name = {str(r.get("name") or "").strip(): r for r in roles}
    lines: list[str] = []
    for name in cast_names:
        card = by_name.get(name) or {}
        details = [name]
        appearance = str(card.get("appearance") or "").strip()
        if appearance and appearance != "原文未说明":
            details.append(appearance)
        if name == "林昭" and "疤" not in appearance:
            details.append(LIN_ZHAO_FALLBACK)
        line = "，".join(part for part in details if part)
        lines.append(f"{line}（脸型/发型/服饰固定，不因画风改变）")
    return "；".join(lines)


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


def _encode_reference(path: str) -> str:
    """角色基准图 -> base64 data URL（硅基流动参考图格式）。"""
    content = Path(path).read_bytes()
    suffix = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(suffix, suffix or "png")
    return f"data:image/{mime};base64,{base64.b64encode(content).decode('ascii')}"


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
    reference_images: list[str] | None = None,
    model_override: str = "",
) -> bytes:
    """
    硅基流动生图接口（异步），支持三种用法：

    - 文生图（默认）：IMAGE_API_MODEL（Z-Image-Turbo），prompt 生成整张画面；
    - 图生图（传 reference_images 时）：IMAGE_EDIT_MODEL（Qwen-Image-Edit-2509），
      把角色基准图作为参考图真正喂给模型，锁定长相/发型/服装，实现角色一致性。
      最多 3 张（image / image2 / image3，base64 data URL）；
    - model_override：指定其他模型（如基准图用 Qwen/Qwen-Image），
      此时用最小 payload，只发 model/prompt/negative_prompt/image_size。

    约定：
    - URL/Key/模型从 IMAGE_API_URL / IMAGE_API_KEY / IMAGE_API_MODEL 读取；
    - 兼容 images[0].url、data[0].url / b64_json 或直接二进制图片；
    - 没配置时抛 ImageApiError，由节点降级为 prompt_only；
    - 图生图请求失败时自动退回纯文生图重试一次，不让一致性功能拖垮出图链路。
    """
    if not config.IMAGE_API_URL or not config.IMAGE_API_KEY:
        raise ImageApiError(
            "未配置 IMAGE_API_URL / IMAGE_API_KEY；本次只生成绘图 prompt。"
        )

    use_edit = bool(reference_images) and config.IMAGE_EDIT_ENABLED

    def build_payload(with_refs: bool) -> dict[str, Any]:
        if with_refs:
            payload: dict[str, Any] = {
                "model": config.IMAGE_EDIT_MODEL,
                "prompt": positive_prompt,
            }
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt
            for index, path in enumerate(
                list(reference_images or [])[: config.IMAGE_EDIT_MAX_REFS]
            ):
                payload["image" if index == 0 else f"image{index + 1}"] = (
                    _encode_reference(path)
                )
            return payload
        model_name = str(model_override or "").strip() or config.IMAGE_API_MODEL
        payload = {
            "model": model_name,
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "image_size": image_size or config.IMAGE_SIZE,
            # 硅基流动平台级参数：False = 关闭右下角"AI 生成"水印。
            # 基准图带水印会干扰后续图生图参考，必须关；接口不认识会自动忽略。
            "watermark": False,
        }
        if not model_override:
            # 默认 Turbo 路径保留 batch_size；覆盖模型时不发，避免字段不兼容被拒。
            payload["batch_size"] = 1
        return payload

    headers = {"Authorization": f"Bearer {config.IMAGE_API_KEY}"}
    payload = build_payload(use_edit)
    if use_edit:
        print(
            f"[文生图] 图生图模式：附带 {min(len(reference_images or []), config.IMAGE_EDIT_MAX_REFS)}"
            f" 张角色参考图（{config.IMAGE_EDIT_MODEL}）",
            flush=True,
        )
    try:
        raw, content_type = await asyncio.to_thread(
            _post_json_sync,
            config.IMAGE_API_URL,
            payload,
            headers,
        )
    except ImageApiError:
        raise
    except Exception as exc:
        if not use_edit:
            raise
        # 参考图模式失败（模型未开通/限流/网络）：退回纯文生图再试一次。
        print(
            f"[文生图警告] 图生图（参考图）请求失败：{exc}；退回纯文生图重试。",
            flush=True,
        )
        payload = build_payload(False)
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


def _save_image(
    story_id: str,
    episode_no: int,
    style_desc: str,
    content: bytes,
    shot_no: int = 0,
    pending: bool = False,
) -> Path:
    """
    保存生成的图片。

    pending=True 时存入 images/_pending 待定区（不会被分镜卡片收录），
    用户在界面点"采用"后才移动到正式目录；False = 流水线正式产物直接入库。
    """
    sub_dir = "images/_pending" if pending else "images"
    directory = config.OUTPUT_DIR / _safe(story_id, "default") / sub_dir
    directory.mkdir(parents=True, exist_ok=True)
    shot_tag = f"_镜头{shot_no}" if shot_no else ""
    # 待定区文件名带时间戳：同一镜头多次生成互不覆盖，浏览器也不会拿缓存旧图
    stamp = time.strftime("%H%M%S") if pending else ""
    suffix = f"_{stamp}" if stamp else ""
    path = (
        directory
        / f"第{episode_no}章{shot_tag}_{_safe(style_desc, 'style')}{suffix}.png"
    )
    path.write_bytes(content)
    return path


def _fallback_prompt(storyboard: str, style: str, anchor: str) -> str:
    anchor_part = f"角色一致性：{anchor}；" if anchor else ""
    return (
        f"{style}，漫剧关键帧，{anchor_part}"
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
    episode_no = int(state.get("episode_no") or 0)
    story_id = str(state.get("story_id") or "default").strip()

    # 指定镜头号时做"切片隔离"：只保留当前镜头的结构化描述，
    # 其他镜头的文本一律不进入 prompt（一镜一图）。
    # 未指定时保持旧行为：整集文本自动选关键画面。
    shot_no = int(state.get("shot_no") or 0)
    pending = bool(state.get("image_pending"))
    prompt_override = str(state.get("prompt_override") or "").strip()
    if shot_no:
        shots = parse_storyboard_shots(storyboard)
        matched = next((s for s in shots if s["shot_no"] == shot_no), None)
        if not matched:
            raise RuntimeError(
                f"分镜里找不到镜头 #{shot_no}（本集共 {len(shots)} 个镜头）。"
            )
        storyboard = (
            f"镜头#{matched['shot_no']}｜时长：{matched['duration']}｜"
            f"画面：{matched['visual']}｜台词/旁白：{matched['dialogue']}｜"
            f"氛围：{matched['mood']}"
        )

    # 锚点必须用"切片后的镜头文本"算出场角色：先切片、再锚点，顺序不能反。
    anchor = _character_anchor(state, storyboard)

    print("\n" + "=" * 72)
    print("【文生图】")
    print("-" * 72)
    print(
        f"输入：分镜 {len(storyboard)} 字｜镜头="
        f"{f'#{shot_no}' if shot_no else '整集'}｜画风={style}｜第{episode_no}集"
        f"｜角色锚点 {len(anchor)} 字",
        flush=True,
    )

    user_negative = str(state.get("image_negative_extra") or "").strip()
    user_extra = (
        [item.strip() for item in re.split(r"[，,、;；\n]+", user_negative) if item.strip()]
        if user_negative
        else []
    )

    if prompt_override:
        # 审核两次未通过后，人工修改 prompt 直接覆盖，不再走 LLM/模板
        positive = prompt_override
        negative = _merge_negative(user_extra)
        print("[文生图] 使用人工修改后的提示词重绘。", flush=True)
    elif config.IMAGE_PROMPT_USE_LLM:
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

    # 基准图声明同样只注入本镜头出场角色的（不传 names 会注入全部角色，
    # 把没出场的角色长相塞进 prompt 造成"串角色"）。
    cast_names = expected_characters(storyboard, _load_roles(state))
    ref_block = reference_block(story_id, names=cast_names) if cast_names else ""
    if ref_block:
        positive = ref_block + "\n" + positive
    # 真正的角色一致性：把本镜头出场角色的已确认基准图作为参考图传给生图模型
    # （图生图模式）；没有基准图时 ref_paths 为空，走纯文生图。
    ref_paths = reference_paths_for(story_id, cast_names) if cast_names else []
    status = "prompt_only"
    image_path = ""
    image_error = ""
    retry_count = 0
    review: dict[str, Any] = {}

    if not config.IMAGE_API_URL or not config.IMAGE_API_KEY:
        print(
            "[文生图] 未配置 IMAGE_API_URL / IMAGE_API_KEY；本次只生成绘图 prompt。",
            flush=True,
        )
    else:
        # 首次生成 + 最多自动重绘 IMAGE_GEN_MAX_RETRIES 次（默认 2 次）。
        max_retries = max(0, int(config.IMAGE_GEN_MAX_RETRIES))
        attempts = max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                if ref_paths and config.IMAGE_EDIT_ENABLED:
                    # 图生图：基准图作为参考图传入，锁定角色长相。
                    content = await external_image_api(
                        positive, negative, style, reference_images=ref_paths
                    )
                else:
                    content = await external_image_api(positive, negative, style)
                path = _save_image(story_id, episode_no, style, content, shot_no, pending)
                print(f"[文生图] 图片已保存：{path}", flush=True)

                # ---- 审核节点（branch_validate_node 图片层）----
                # 人物数规则 + CLIP 角色一致性 +（可选）LLM OOC，全部同步执行。
                review = await asyncio.to_thread(
                    review_shot_image,
                    state,
                    storyboard,
                    positive,
                    negative,
                    str(path),
                )
                # 采纳审核强化后的 prompt（如单人约束），下一次重绘生效
                positive = str(review.get("prompt") or positive)
                negative = str(review.get("negative") or negative)

                if review.get("passed"):
                    image_path = str(path)
                    status = "done"
                    image_error = ""
                    print(
                        "[文生图] ✅ 审核通过（人物/角色/一致性）", flush=True
                    )
                    break

                image_error = "；".join(review.get("issues") or []) or "审核未通过"
                retry_count = attempt
                if attempt < attempts:
                    status = "review_failed"
                    print(
                        f"[文生图] ❌ 审核未通过（第{attempt}/{attempts}次）："
                        f"{image_error}，自动重绘中……",
                        flush=True,
                    )
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
                    continue

                # 达到自动重绘上限：标记待人工修改 prompt，不再自动生成。
                # 图片保留在待定区供人工查看（坏图不入分镜卡片）。
                status = "needs_manual_prompt"
                image_path = str(path)
                print(
                    f"[文生图] ⚠️ 自动重绘 {max_retries} 次仍未通过审核，"
                    f"镜头已标记为「待人工修改 prompt」。原因：{image_error}",
                    flush=True,
                )
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
    used_edit = bool(ref_paths) and config.IMAGE_EDIT_ENABLED
    record = {
        "story_id": story_id,
        "episode_no": episode_no,
        "shot_no": shot_no,
        "style_desc": style,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "image_path": image_path,
        "status": status,
        "retry_count": retry_count,
        "review_issues": list(review.get("issues") or []),
        "error": image_error,
        "elapsed_ms": elapsed_ms,
        "image_model": config.IMAGE_EDIT_MODEL if used_edit else config.IMAGE_API_MODEL,
        "reference_images": [Path(p).name for p in ref_paths] if used_edit else [],
    }
    print(f"[文生图] 状态={status}｜耗时={elapsed_ms}ms", flush=True)
    print(positive, flush=True)

    return {
        "style_desc": style,
        "shot_no": shot_no,
        "image_pending": pending,
        "needs_manual_prompt": status == "needs_manual_prompt",
        "image_review": review,
        "image_prompt_positive": positive,
        "image_prompt_negative": negative,
        "image_path": image_path,
        "image_status": status,
        "image_retry_count": retry_count,
        "image_error": image_error,
        "image_gen_ms": elapsed_ms,
        "error_message": (
            image_error if status == "needs_manual_prompt" else ""
        ),
        "image_records": [record],
        "trace": [
            f"文生图：剧本 {len(str(state.get('script') or ''))} 字，"
            f"分镜 {len(storyboard)} 字，镜头={f'#{shot_no}' if shot_no else '整集'}，"
            f"画风={style}，prompt {len(positive)} 字，"
            f"参考图 {len(ref_paths) if used_edit else 0} 张"
            f"（{'图生图 ' + config.IMAGE_EDIT_MODEL if used_edit else config.IMAGE_API_MODEL}），"
            f"路径={image_path or '-'}，"
            f"状态={status}，重绘 {retry_count} 次，耗时 {elapsed_ms}ms"
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
