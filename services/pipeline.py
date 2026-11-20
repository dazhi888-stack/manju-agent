"""整集视频流水线：分镜脚本 → 缺图自动补 → 逐镜头图生视频 → ffmpeg 拼接成片。

定位：
- 上一版「成片合成」（services/compose.py）走静态图 + TTS 配音，适合零额度出片；
- 本流水线走真正的图生视频（MiniMax-H3，分镜图作首帧），每个镜头是一段
  会动的视频，最后拼接成完整成片。烧视频额度（¥0.5/秒），烧图额度（缺图时）。

省钱机制：
- 断点续跑：outputs/<story>/videos/ep{NN}_shot{KK}.mp4 已存在时默认直接复用，
  重跑不会重复扣钱；只有 force_video=True 才强制重新生成；
- dry_run：build_pipeline_plan 只读文件算计划，不调用任何收费接口；
- 单镜头失败（视频 API 报错/熔断打开/缺图）只跳过该镜头，不拖垮整批。

无背景音乐：按用户要求当前版本不混 BGM；成片音轨来自 H3 片段自带的音轨。
字幕：不烧录进画面（H3 画面内容已完整），输出外挂 srt，需要时可后期加。
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import config
from services.compose import (
    _probe_duration_seconds,
    _run_ffmpeg,
    _safe_dir,
)
from services.image_gen import image_gen_node
from services.image_manifest import bind_shot, entries_for_episode
from services.shot_prompt import load_episode_context, parse_storyboard_shots
from services.video_gen import call_seedance_async, video_circuit_breaker
from state import ComicState

# 进度回调：pipeline_progress(stage, detail)
ProgressCallback = Callable[[str, str], None]

# MiniMax-H3 报价（¥/秒），只用于预算估算展示，不参与计费
VIDEO_PRICE_PER_SECOND = 0.5
# Z-Image-Turbo 报价（¥/张）下限，Qwen-Image-Edit 约 ¥0.30/张
IMAGE_PRICE_LOW = 0.10
IMAGE_PRICE_HIGH = 0.30


def shot_video_path(story_id: str, episode_no: int, shot_no: int) -> Path:
    """单镜头视频的固定保存路径（断点续跑的依据）。"""
    directory = config.OUTPUT_DIR / _safe_dir(story_id) / "videos"
    return directory / f"ep{int(episode_no):02d}_shot{int(shot_no):02d}.mp4"


def build_shot_video_prompt(shot: dict[str, Any], style_desc: str = "") -> str:
    """
    从分镜行构造视频提示词。

    与绘图 prompt 相反：视频 prompt 保留运镜词（推/拉/摇/跟拍），
    视频模型需要知道镜头怎么动；末尾追加稳定性约束，减少角色变形。
    """
    parts: list[str] = []
    visual = str(shot.get("visual") or "").strip()
    if visual:
        parts.append(visual)
    mood = str(shot.get("mood") or "").strip()
    if mood:
        parts.append(f"氛围：{mood}")
    camera = str(shot.get("camera") or "").strip()
    if camera:
        parts.append(f"运镜：{camera}")
    if style_desc:
        parts.append(f"画风：{style_desc}")
    parts.append("以首帧画面为起点自然运动，人物五官与服装保持稳定，动作连贯流畅，画面不要突变")
    prompt = "。".join(p.rstrip("。") for p in parts if p)
    return prompt[:1000]


def _target_shot_nos(
    shots: list[dict[str, Any]], shot_nos: list[int] | None
) -> list[dict[str, Any]]:
    if not shot_nos:
        return shots
    wanted = {int(n) for n in shot_nos if int(n) > 0}
    return [s for s in shots if int(s.get("shot_no") or 0) in wanted]


def build_pipeline_plan(
    story_id: str,
    episode_no: int,
    shot_nos: list[int] | None = None,
    style_desc: str = "",
) -> dict[str, Any]:
    """
    只读不写的跑批计划（dry_run 用）。

    返回：每个镜头是否有绑定图、是否会生成图/视频、是否复用已有视频、预算。
    """
    context = load_episode_context(story_id, episode_no)
    all_shots = parse_storyboard_shots(context["storyboard"])
    if not all_shots:
        raise ValueError(f"{story_id} 第{episode_no}集没有可用的分镜镜头。")
    targets = _target_shot_nos(all_shots, shot_nos)
    if not targets:
        raise ValueError(f"镜头号不在本集分镜里：{shot_nos}（共 {len(all_shots)} 镜头）")

    bindings = entries_for_episode(story_id, episode_no)
    images_dir = config.OUTPUT_DIR / _safe_dir(story_id) / "images"

    items: list[dict[str, Any]] = []
    new_image_count = 0
    new_video_count = 0
    for shot in targets:
        shot_no = int(shot.get("shot_no") or 0)
        entry = bindings.get(shot_no)
        image_name = str((entry or {}).get("file_name") or "")
        image_exists = bool(image_name) and (images_dir / image_name).is_file()
        video_file = shot_video_path(story_id, episode_no, shot_no)
        reuse_video = video_file.is_file()
        if not image_exists:
            new_image_count += 1
        if not reuse_video:
            new_video_count += 1
        items.append(
            {
                "shot_no": shot_no,
                "visual": str(shot.get("visual") or "")[:60],
                "image_name": image_name if image_exists else "",
                "need_image_gen": not image_exists,
                "reuse_video": reuse_video,
                "video_path": str(video_file) if reuse_video else "",
                "video_prompt": build_shot_video_prompt(shot, style_desc),
            }
        )

    video_cost = round(new_video_count * config.VIDEO_DURATION * VIDEO_PRICE_PER_SECOND, 1)
    image_cost_low = round(new_image_count * IMAGE_PRICE_LOW, 1)
    image_cost_high = round(new_image_count * IMAGE_PRICE_HIGH, 1)
    return {
        "story_id": story_id,
        "episode_no": int(context["episode_no"]),
        "title": str(context.get("title") or ""),
        "model": config.VIDEO_API_MODEL,
        "resolution": config.VIDEO_RESOLUTION,
        "duration_per_shot": config.VIDEO_DURATION,
        "items": items,
        "summary": {
            "total_shots": len(items),
            "need_image_gen": new_image_count,
            "need_video_gen": new_video_count,
            "reuse_video": len(items) - new_video_count,
            "estimated_cost_yuan": f"视频 ≈¥{video_cost}（{new_video_count} 段 × {config.VIDEO_DURATION}s × ¥{VIDEO_PRICE_PER_SECOND}/s）"
            + (
                f"｜图片 ≈¥{image_cost_low}~{image_cost_high}（{new_image_count} 张）"
                if new_image_count
                else ""
            ),
            "total_video_seconds": len(items) * config.VIDEO_DURATION,
        },
    }


async def _ensure_image(
    story_id: str,
    episode_no: int,
    shot_no: int,
    context: dict[str, Any],
    style_desc: str,
    tick: ProgressCallback,
) -> tuple[str, str]:
    """
    确保镜头有已入库的分镜图；缺图时自动生成并绑定。

    返回 (图片绝对路径, 错误信息)；成功时错误信息为空串。
    """
    bindings = entries_for_episode(story_id, episode_no)
    entry = bindings.get(shot_no)
    images_dir = config.OUTPUT_DIR / _safe_dir(story_id) / "images"
    if entry:
        image = images_dir / str(entry.get("file_name") or "")
        if image.is_file():
            return str(image), ""

    tick(f"镜头#{shot_no}", "缺分镜图，自动生成中（含角色参考图锁脸）……")
    state: ComicState = {
        "story_id": story_id,
        "episode_no": episode_no,
        "storyboard": context["storyboard"],
        "script": context["script"],
        "style_desc": style_desc or config.DEFAULT_STYLE_DESC,
        "shot_no": shot_no,
        "canon_profiles": context["characters"],
        "image_pending": True,
    }
    result = await image_gen_node(state)
    status = str(result.get("image_status") or "")
    image_path = str(result.get("image_path") or "")
    if not image_path or not Path(image_path).is_file():
        return "", f"自动出图失败（status={status}）：{result.get('image_error') or '未知原因'}"

    # 待定区 → 正式目录并绑定（与网页「确认入库」同一逻辑）
    source = Path(image_path).resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    target = images_dir / source.name
    if source != target:
        shutil.move(str(source), str(target))
    bind_shot(story_id, episode_no, shot_no, target.name, style_desc)
    tick(f"镜头#{shot_no}", f"分镜图已生成并入库：{target.name}")
    return str(target), ""


async def _ensure_video(
    story_id: str,
    episode_no: int,
    shot_no: int,
    image_path: str,
    prompt: str,
    force: bool,
    tick: ProgressCallback,
) -> tuple[Path, str, bool]:
    """
    确保镜头有视频片段；已有片段默认复用（断点续跑）。

    返回 (视频文件路径, 错误信息, 是否新生成)；错误时首项为空 Path。
    """
    video_file = shot_video_path(story_id, episode_no, shot_no)
    if video_file.is_file() and not force:
        tick(f"镜头#{shot_no}", "已有视频片段，直接复用（不重复扣费）")
        return video_file, "", False

    allowed, circuit_state = video_circuit_breaker.allow_request()
    if not allowed:
        return Path(), f"熔断器状态={circuit_state}，冷却中，跳过本镜头", False

    tick(f"镜头#{shot_no}", f"图生视频生成中（{config.VIDEO_DURATION}s / {config.VIDEO_RESOLUTION}）……")
    started = time.monotonic()
    try:
        content = await call_seedance_async(prompt, "", image_path=image_path)
    except Exception as exc:
        video_circuit_breaker.record_failure()
        return Path(), f"视频生成失败：{type(exc).__name__}: {exc}", False
    video_circuit_breaker.record_success()

    video_file.parent.mkdir(parents=True, exist_ok=True)
    video_file.write_bytes(content)
    elapsed = round(time.monotonic() - started, 1)
    tick(f"镜头#{shot_no}", f"视频生成完成：{video_file.name}｜耗时 {elapsed}s")
    return video_file, "", True


def _concat_clips(clips: list[Path], work_dir: Path, final_name: str) -> Path:
    """ffmpeg concat demuxer + 重编码拼接（兼容编码参数有细微差异的片段）。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        local = work_dir / clip.name
        if clip.resolve() != local.resolve():
            shutil.copy(str(clip), str(local))
    list_file = work_dir / "concat_list.txt"
    list_file.write_text(
        "\n".join(f"file '{clip.name}'" for clip in clips), encoding="utf-8"
    )
    _run_ffmpeg(
        [
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", "concat_list.txt",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            final_name,
        ],
        timeout=1200,
        cwd=work_dir,
    )
    return work_dir / final_name


def _build_srt(entries: list[dict[str, Any]]) -> str:
    """按片段实际时长累加生成外挂字幕（与 compose.py 同格式）。"""
    lines: list[str] = []
    cursor = 0.0
    index = 1

    def _ts(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, rem = divmod(ms, 3600000)
        m, rem = divmod(rem, 60000)
        s, msec = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"

    for item in entries:
        duration = float(item.get("duration") or 0)
        text = str(item.get("subtitle") or "")
        if text and duration > 0:
            lines.append(str(index))
            lines.append(f"{_ts(cursor)} --> {_ts(cursor + min(duration, 6.0))}")
            lines.append(text)
            lines.append("")
            index += 1
        cursor += duration
    return "\n".join(lines)


def _subtitle_of(shot: dict[str, Any]) -> str:
    dialogue = re.sub(r"^[「『\"“”]+|[」』\"“”]+$", "", str(shot.get("dialogue") or "").strip()).strip()
    return dialogue[:42] + ("…" if len(dialogue) > 42 else "")


async def run_video_pipeline(
    story_id: str,
    episode_no: int,
    shot_nos: list[int] | None = None,
    auto_gen_images: bool = True,
    force_video: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    跑整集视频流水线（异步入口；ffmpeg 等同步重活在 to_thread 里）。

    返回报告：{status, output_path, shots, skipped, summary, srt_path}
    """
    started = time.monotonic()

    def tick(stage: str, detail: str = "") -> None:
        if progress:
            progress(stage, detail)
        print(f"[视频流水线] {stage}{('：' + detail) if detail else ''}", flush=True)

    plan = build_pipeline_plan(story_id, episode_no, shot_nos)
    episode_no = int(plan["episode_no"])
    context = load_episode_context(story_id, episode_no)
    all_shots = parse_storyboard_shots(context["storyboard"])
    shot_map = {int(s["shot_no"]): s for s in all_shots}

    shots_report: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    clip_files: list[Path] = []
    srt_entries: list[dict[str, Any]] = []
    new_video_count = 0

    tick("开始", f"{len(plan['items'])} 镜头｜模型 {plan['model']}")

    for item in plan["items"]:
        shot_no = int(item["shot_no"])
        shot = shot_map.get(shot_no) or {}

        # ① 分镜图：已有 → 直接用；缺图且允许自动补 → 生成并绑定
        image_path = ""
        bindings = entries_for_episode(story_id, episode_no)
        entry = bindings.get(shot_no)
        images_dir = config.OUTPUT_DIR / _safe_dir(story_id) / "images"
        if entry:
            candidate = images_dir / str(entry.get("file_name") or "")
            if candidate.is_file():
                image_path = str(candidate)
        error = ""
        if not image_path:
            if not auto_gen_images:
                skipped.append({"shot_no": shot_no, "reason": "没有已入库的分镜图（auto_gen_images=False）"})
                tick(f"镜头#{shot_no}", "跳过：缺分镜图")
                continue
            image_path, error = await _ensure_image(
                story_id, episode_no, shot_no, context, "", tick,
            )
            if error:
                skipped.append({"shot_no": shot_no, "reason": error})
                tick(f"镜头#{shot_no}", f"跳过：{error[:120]}")
                continue

        # ② 图生视频（复用已有片段省钱）
        prompt = item.get("video_prompt") or build_shot_video_prompt(shot)
        video_file, error, generated = await _ensure_video(
            story_id, episode_no, shot_no, image_path, prompt, force_video, tick
        )
        if error:
            skipped.append({"shot_no": shot_no, "reason": error})
            tick(f"镜头#{shot_no}", f"跳过：{error[:120]}")
            continue
        if generated:
            new_video_count += 1

        # ③ 探测时长，登记片段
        duration = await _probe(video_file)
        clip_files.append(video_file)
        shots_report.append(
            {
                "shot_no": shot_no,
                "video": video_file.name,
                "duration": round(duration, 2),
                "subtitle": _subtitle_of(shot),
                "image": Path(image_path).name,
            }
        )
        srt_entries.append({"subtitle": _subtitle_of(shot), "duration": duration})

    if not clip_files:
        raise RuntimeError(
            "没有任何镜头产出视频片段。" + (json_dumps(skipped))
        )

    # ④ 拼接成片 + 外挂字幕
    tick("拼接", f"{len(clip_files)} 个片段")
    work_dir = (
        config.OUTPUT_DIR / _safe_dir(story_id) / "compositions" / f"video_ep_{episode_no:02d}"
    )
    final_name = f"ep_{episode_no:02d}_video.mp4"
    final_path = await _concat_async(clip_files, work_dir, final_name)
    srt_path = work_dir / f"ep_{episode_no:02d}_video.srt"
    srt_path.write_text(_build_srt(srt_entries), encoding="utf-8")

    elapsed = round(time.monotonic() - started, 1)
    total_seconds = round(sum(r["duration"] for r in shots_report), 1)
    tick("完成", f"{final_path.name}｜成片 {total_seconds}s｜耗时 {elapsed}s")
    return {
        "status": "done",
        "story_id": story_id,
        "episode_no": episode_no,
        "output_path": str(final_path),
        "srt_path": str(srt_path),
        "model": plan["model"],
        "video_seconds": total_seconds,
        "elapsed_seconds": elapsed,
        "new_video_segments": new_video_count,
        "estimated_cost_yuan": round(new_video_count * config.VIDEO_DURATION * VIDEO_PRICE_PER_SECOND, 1),
        "shots": shots_report,
        "skipped": skipped,
        "circuit": video_circuit_breaker.snapshot(),
    }


def json_dumps(data: Any) -> str:
    import json

    return json.dumps(data, ensure_ascii=False)


async def _probe(path: Path) -> float:
    """探测视频时长（ffmpeg -i 解析），在线程里跑避免阻塞事件循环。"""
    import asyncio

    return await asyncio.to_thread(_probe_duration_seconds, path)


async def _concat_async(clips: list[Path], work_dir: Path, final_name: str) -> Path:
    import asyncio

    return await asyncio.to_thread(_concat_clips, clips, work_dir, final_name)
