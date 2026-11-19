"""
成片合成服务：分镜 JSON + 已绑定分镜图 → 带配音字幕的完整成片。

流程（每个镜头）：
1. 从 manifest 取该镜头已确认入库的分镜图（无图的镜头跳过并记录）；
2. 台词非空时用 edge-tts（微软免费语音）生成配音 mp3，失败则该镜头静音；
3. 时长 = max(分镜时长, 配音时长 + 0.8s 余量)；
4. ffmpeg 把静态图转成片段并烧录字幕（drawtext，微软雅黑）；
5. concat 拼接全部片段 → outputs/<story_id>/compositions/ep_XX.mp4，
   同时输出 ep_XX.srt 外挂字幕。

降级策略：
- edge-tts 不可用/失败 → 静音 + 字幕照常，不抛异常；
- 镜头无绑定图 → 跳过该镜头，写入报告 skipped；
- 中文字体缺失 → 尝试黑体/宋体，全部缺失则不烧字幕只出画面。

说明：合成是纯本地操作（TTS 走微软免费接口），不消耗图片/视频额度。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import config
from services.image_manifest import entries_for_episode
from stores.shot_store import load_episode_shots

# 进度回调：compose_progress(stage, detail)；供 API 层转发/日志
ProgressCallback = Callable[[str, str], None]

SUBTITLE_MAX_CHARS = 42  # 每条字幕的最大长度（约一行），超出截断
TTS_MAX_CHARS = 200  # 单条 TTS 文本上限，避免长句超时
DEFAULT_VOICE = "zh-CN-YunjianNeural"  # 云健：男声，适合武侠/男频


# ===========================================================================
# 一、基础工具
# ===========================================================================
def _ffmpeg_exe() -> str:
    """imageio-ffmpeg 自带的静态 ffmpeg，无需系统安装。"""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg(args: list[str], timeout: int = 300, cwd: Path | None = None) -> subprocess.CompletedProcess:
    exe = _ffmpeg_exe()
    proc = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败：{proc.stderr[-600:] or proc.returncode}")
    return proc


def _probe_duration_seconds(path: Path) -> float:
    """用 ffmpeg -i 的 stderr 解析媒体时长（imageio-ffmpeg 不带 ffprobe）。"""
    exe = _ffmpeg_exe()
    proc = subprocess.run(
        [exe, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not match:
        return 0.0
    hours, minutes, seconds = (float(g) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _find_chinese_font() -> str | None:
    """按优先级找可用的中文字体（返回绝对路径）。"""
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),  # 微软雅黑
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),  # 黑体
        Path("C:/Windows/Fonts/simsun.ttc"),  # 宋体
        Path("C:/Windows/Fonts/Deng.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _subtitle_text(shot: dict[str, Any]) -> str:
    """字幕文本：优先台词；无台词不配字幕（画面即内容）。"""
    dialogue = str(shot.get("dialogue") or "").strip()
    dialogue = re.sub(r"^[「『\"“”]+|[」』\"“”]+$", "", dialogue).strip()
    if len(dialogue) > SUBTITLE_MAX_CHARS:
        dialogue = dialogue[: SUBTITLE_MAX_CHARS - 1] + "…"
    return dialogue


def _shot_duration_seconds(shot: dict[str, Any]) -> float:
    try:
        value = float(shot.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        value = 0.0
    return value if value > 0 else 4.0  # 分镜没写时长时默认 4 秒


def _safe_dir(story_id: str) -> str:
    value = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(story_id or "default")).strip(" ._")
    return value[:40] or "default"


@contextlib.contextmanager
def _no_proxy():
    """临时清掉代理环境变量：edge-tts 直连微软语音接口，绕开失效的本地代理。"""
    saved: dict[str, str] = {}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    try:
        yield
    finally:
        os.environ.update(saved)


# ===========================================================================
# 二、TTS 配音
# ===========================================================================
def _tts_one(text: str, out_path: Path, voice: str) -> float:
    """生成一条配音，返回音频时长秒数；失败/超时抛异常，由上层重试降级。"""
    import edge_tts

    async def _synth() -> None:
        communicate = edge_tts.Communicate(text, voice, rate="-5%")
        await communicate.save(str(out_path))

    with _no_proxy():
        # 微软语音接口偶发挂起（websocket 不返回），60 秒强制超时走重试
        asyncio.run(asyncio.wait_for(_synth(), timeout=60))
    return _probe_duration_seconds(out_path)


def _tts_clip(
    text: str, out_path: Path, voice: str, attempts: int = 2
) -> tuple[bool, float]:
    """带重试的 TTS；失败降级为静音（返回 False, 0.0）。"""
    for attempt in range(1, attempts + 1):
        try:
            duration = _tts_one(text, out_path, voice)
            if duration > 0:
                return True, duration
        except Exception as exc:
            print(
                f"[成片TTS] 第{attempt}/{attempts}次失败：{type(exc).__name__}: {exc}",
                flush=True,
            )
    return False, 0.0


# ===========================================================================
# 三、片段生成与拼接
# ===========================================================================
def _ff_path_escape(path: Path | str) -> str:
    """ffmpeg filter 参数内的路径转义：正斜杠 + 盘符冒号转义（C:\\x → C\\:/x）。"""
    return str(path).replace("\\", "/").replace(":", "\\:")


def _clip_args(
    image_name: str,
    duration: float,
    subtitle: str,
    clip_name: str,
    font_name: str | None,
    audio_name: str | None,
    shot_no: int,
    work_dir: Path,
) -> list[str]:
    """
    单个片段的完整 ffmpeg 参数（含输出文件）。所有文件都在工作目录内、
    用相对路径传入并配合 cwd=work_dir 运行——彻底规避 Windows 盘符冒号
    在 filter 里的转义问题。
    """
    vf = (
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
    )
    if subtitle and font_name:
        sub_file = work_dir / f"sub_{shot_no:02d}.txt"
        sub_file.write_text(subtitle, encoding="utf-8")
        vf += (
            f",drawtext=fontfile={font_name}"
            f":textfile={sub_file.name}"
            ":fontsize=44:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=18"
            ":x=(w-text_w)/2:y=h-th-72"
        )
    args = [
        "-y",
        "-loop", "1",
        "-t", f"{duration:.2f}",
        "-i", image_name,
    ]
    if audio_name:
        args += ["-i", audio_name]
    else:
        # 无配音镜头补静音音轨：保证所有片段流结构一致，concat 才不会丢音轨
        args += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    args += [
        "-vf", vf,
        "-r", "25",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
    ]
    args += [
        "-map", "0:v", "-map", "1:a",
        "-c:a", "aac", "-b:a", "128k",
        # 输出级时长上限：截断无限静音源（anullsrc），配音结束后的留白靠它兜底
        "-t", f"{duration:.2f}",
    ]
    args.append(clip_name)
    return args


def _build_srt(shots_report: list[dict[str, Any]]) -> str:
    """根据片段实际时长生成外挂字幕 srt（时间轴按拼接顺序累加）。"""
    lines: list[str] = []
    cursor = 0.0
    index = 1

    def _ts(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, rem = divmod(ms, 3600000)
        m, rem = divmod(rem, 60000)
        s, msec = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"

    for item in shots_report:
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


def compose_episode(
    story_id: str,
    episode_no: int,
    voice: str = DEFAULT_VOICE,
    progress: ProgressCallback | None = None,
    burn_subtitles: bool = True,
) -> dict[str, Any]:
    """
    合成第 episode_no 集成片。同步函数（ffmpeg/TTS 在工作线程里跑）。

    burn_subtitles=False 时不在画面上烧字幕（外挂 srt 仍生成）：
    适用于分镜图本身已带台词的旧素材，避免画面出现两层文字。

    返回报告：{status, output_path, shots: [...], skipped: [...], total_seconds}
    """
    started = time.monotonic()

    def _tick(stage: str, detail: str = "") -> None:
        if progress:
            progress(stage, detail)
        print(f"[成片合成] {stage}{('：' + detail) if detail else ''}", flush=True)

    data = load_episode_shots(story_id, episode_no) or {}
    shots = [s for s in (data.get("shots") or []) if s.get("shot_no")]
    if not shots:
        raise RuntimeError(f"{story_id} 第{episode_no}集没有分镜数据，请先生成分镜。")

    images_dir = config.OUTPUT_DIR / _safe_dir(story_id) / "images"
    bindings = entries_for_episode(story_id, episode_no)

    work_dir = (
        config.OUTPUT_DIR / _safe_dir(story_id) / "compositions" / f"ep_{episode_no:02d}"
    )
    # 不清空旧目录：所有产物都是确定性文件名直接覆盖，历史文件留给用户自行清理
    work_dir.mkdir(parents=True, exist_ok=True)

    font_file = _find_chinese_font()
    if font_file:
        # 复制字体进工作目录，用相对路径传给 drawtext，规避盘符冒号转义问题
        shutil.copy(font_file, work_dir / "font.ttf")
        font_arg = "font.ttf"
    else:
        font_arg = None
        _tick("字体", "未找到中文字体，本集不烧录字幕")

    shots_report: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    srt_entries: list[dict[str, Any]] = []
    clip_files: list[Path] = []

    _tick("开始", f"{len(shots)} 镜头｜配音音色 {voice}")

    for shot in shots:
        shot_no = int(shot.get("shot_no") or 0)
        entry = bindings.get(shot_no)
        if not entry:
            reason = "该镜头没有已入库的分镜图"
            skipped.append({"shot_no": shot_no, "reason": reason})
            _tick(f"镜头#{shot_no}", f"跳过：{reason}")
            continue
        image = images_dir / str(entry.get("file_name") or "")
        if not image.exists():
            reason = f"绑定图缺失：{entry.get('file_name')}"
            skipped.append({"shot_no": shot_no, "reason": reason})
            _tick(f"镜头#{shot_no}", f"跳过：{reason}")
            continue
        # 复制进工作目录统一用相对路径，规避 Windows 盘符在 ffmpeg filter 里的转义坑
        local_image = f"img_{shot_no:02d}{image.suffix.lower()}"
        shutil.copy(image, work_dir / local_image)

        subtitle = _subtitle_text(shot)
        base_duration = _shot_duration_seconds(shot)
        tts_ok = False
        tts_duration = 0.0

        if subtitle:
            text = subtitle[:TTS_MAX_CHARS]
            audio_path = work_dir / f"tts_{shot_no:02d}.mp3"
            _tick(f"镜头#{shot_no}", "生成配音中……")
            tts_ok, tts_duration = _tts_clip(text, audio_path, voice)

        duration = max(base_duration, tts_duration + 0.8) if tts_ok else base_duration
        clip_name = f"clip_{shot_no:02d}.mp4"
        audio_name = f"tts_{shot_no:02d}.mp3" if tts_ok else None
        clip_subtitle = subtitle if burn_subtitles else ""
        args = _clip_args(
            local_image,
            duration,
            clip_subtitle,
            clip_name,
            font_arg,
            audio_name,
            shot_no,
            work_dir,
        )
        try:
            _run_ffmpeg(args, cwd=work_dir)
        except Exception as exc:
            reason = f"片段生成失败：{exc}"
            skipped.append({"shot_no": shot_no, "reason": reason})
            _tick(f"镜头#{shot_no}", f"失败：{reason[:120]}")
            continue

        clip_files.append(work_dir / clip_name)
        shots_report.append(
            {
                "shot_no": shot_no,
                "image": image.name,
                "duration": round(duration, 2),
                "subtitle": subtitle,
                "tts_ok": tts_ok,
            }
        )
        srt_entries.append({"subtitle": subtitle, "duration": duration})
        _tick(f"镜头#{shot_no}", f"完成｜{duration:.1f}s｜配音{'✅' if tts_ok else '—'}")

    if not clip_files:
        raise RuntimeError(
            "没有任何镜头可以合成（需要先在分镜卡片确认入库分镜图）。"
            + json.dumps(skipped, ensure_ascii=False)
        )

    # 拼接：concat demuxer + 流复制（所有片段编码参数一致）
    _tick("拼接", f"{len(clip_files)} 个片段")
    list_file = work_dir / "concat_list.txt"
    list_file.write_text(
        "\n".join(f"file '{clip.name}'" for clip in clip_files), encoding="utf-8"
    )
    final_name = f"ep_{episode_no:02d}.mp4"
    _run_ffmpeg(
        [
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", "concat_list.txt",
            "-c", "copy",
            final_name,
        ],
        timeout=600,
        cwd=work_dir,
    )
    final_path = work_dir / final_name

    # 外挂字幕（时间轴按实际片段时长累加）
    srt_path = work_dir / f"ep_{episode_no:02d}.srt"
    srt_path.write_text(_build_srt(srt_entries), encoding="utf-8")

    total_seconds = round(time.monotonic() - started, 1)
    _tick("完成", f"{final_path.name}｜总时长 {total_seconds}s")
    return {
        "status": "done",
        "output_path": str(final_path),
        "srt_path": str(srt_path),
        "video_seconds": round(sum(r["duration"] for r in shots_report), 1),
        "compose_seconds": total_seconds,
        "tts_voice": voice,
        "burn_subtitles": burn_subtitles,
        "shots": shots_report,
        "skipped": skipped,
    }
