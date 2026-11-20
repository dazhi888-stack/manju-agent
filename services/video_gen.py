"""
文生视频节点：Seedance + 熔断 + 降级。

熔断器是进程级单例，整个项目共用：
- CLOSED：正常放行；
- OPEN：连续 3 次失败后打开，冷却 40 秒；
- HALF_OPEN：冷却后只放行 1 次探测，成功恢复，失败重新打开。

降级：
- enable_video=False：不调用 Seedance，只保存视频提示词；
- 熔断打开 / API 报错 / 额度不足：保存提示词，返回降级状态，不抛出异常。
"""

from __future__ import annotations

import re
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

import config
from state import ComicState

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ===========================================================================
# 一、轻量熔断器：3 状态 + 全局唯一实例
# ===========================================================================
class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """线程安全的轻量熔断器；不需要引入第三方库。"""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 40.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at = 0.0
        self.half_open_probe = False
        self._lock = threading.Lock()

    def allow_request(self) -> tuple[bool, str]:
        """返回 (是否放行, 当前状态)。HALF_OPEN 只允许一个探测请求。"""
        with self._lock:
            now = time.monotonic()
            if self.state == CircuitState.OPEN:
                if now - self.opened_at >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_probe = False
                else:
                    return False, self.state.value
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_probe:
                    return False, self.state.value
                self.half_open_probe = True
            return True, self.state.value

    def record_success(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.opened_at = 0.0
            self.half_open_probe = False

    def record_failure(self) -> None:
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                # 半开探测失败：立即重新熔断。
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()
                self.half_open_probe = False
                return
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.opened_at = 0.0
            self.half_open_probe = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "recovery_timeout": self.recovery_timeout,
                "opened_at": self.opened_at,
            }


# 整个项目共用这一个实例。
video_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=40.0,
)


# ===========================================================================
# 二、视频异步 API（硅基流动 MiniMax-H3 图生视频）
# ===========================================================================
class VideoApiError(RuntimeError):
    """视频接口错误异常（由节点捕获并走 fail_degrade）。"""


def _post_json_sync(url: str, payload: dict, key: str) -> tuple[int, bytes]:
    """同步 POST JSON；返回 (http状态码, 响应体)。放在线程里跑，不阻塞事件循环。"""
    import json
    import urllib.error
    import urllib.request

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # 0 = 关闭显式「AI 生成」水印（下游还会做合成/剪辑）。
            "X-Enable-Watermark": "0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _encode_first_frame(image_path: str) -> str:
    """把本地分镜图缩放/转 JPEG 后编码成 base64 data URL，作首帧参考图。"""
    import base64
    import io

    from PIL import Image

    file = Path(image_path)
    image = Image.open(file).convert("RGB")
    width, height = image.size
    max_edge = config.VIDEO_FIRST_FRAME_MAX_EDGE
    scale = min(1.0, max_edge / max(width, height))
    if scale < 1.0:
        image = image.resize((int(width * scale), int(height * scale)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@traceable(run_type="tool", name="video_async_api")
async def call_seedance_async(
    positive_prompt: str,
    style_desc: str,
    image_path: str = "",
) -> bytes:
    """
    硅基流动 MiniMax-H3 图生视频（异步）。

    流程：POST /video/submit 拿 requestId → 轮询 POST /video/status，
    status=Succeed 时从 results.videos[0].url 下载 mp4 二进制返回。

    - 传入 image_path 时作为首帧参考图（first_frame），锁定角色长相/服装；
      不传则退化为纯文生视频。
    - 视频自带立体声音轨，下载后即是可直接拼接的 mp4。
    - 任一环节失败抛 VideoApiError，由 video_gen_node 捕获走 fail_degrade
      （保存提示词、记录熔断失败），不拖垮整条流水线。

    函数名保留 call_seedance_async 是为了不改动节点调用方；
    底层实际走硅基流动，Seedance（火山方舟）当前未启用。
    """
    import asyncio
    import json
    import time
    import urllib.request

    if not config.VIDEO_API_KEY:
        raise VideoApiError("未配置视频 API Key（VIDEO_API_KEY / IMAGE_API_KEY）")

    payload: dict[str, Any] = {
        "model": config.VIDEO_API_MODEL,
        "prompt": (positive_prompt or style_desc or "画面轻微运动，镜头稳定")[:1000],
        "duration": config.VIDEO_DURATION,
        "resolution": config.VIDEO_RESOLUTION,
    }
    if image_path and Path(image_path).exists():
        payload["images"] = [
            {"url": _encode_first_frame(image_path), "role": "first_frame"}
        ]

    # ① 提交任务
    status_code, raw = await asyncio.to_thread(
        _post_json_sync, config.VIDEO_SUBMIT_URL, payload, config.VIDEO_API_KEY
    )
    if status_code != 200:
        raise VideoApiError(
            f"submit {status_code}: {raw.decode('utf-8', 'ignore')[:300]}"
        )
    try:
        request_id = (json.loads(raw.decode("utf-8", "ignore")) or {}).get("requestId")
    except Exception:
        request_id = None
    if not request_id:
        raise VideoApiError("submit 未返回 requestId")

    # ② 轮询状态
    deadline = time.monotonic() + config.VIDEO_TIMEOUT_SECONDS
    video_url = ""
    while time.monotonic() < deadline:
        await asyncio.sleep(config.VIDEO_POLL_INTERVAL)
        code, body = await asyncio.to_thread(
            _post_json_sync,
            config.VIDEO_STATUS_URL,
            {"requestId": request_id},
            config.VIDEO_API_KEY,
        )
        if code != 200:
            continue  # 轮询偶发错误，下一轮再试
        data = json.loads(body.decode("utf-8", "ignore")) or {}
        state = data.get("status")
        if state == "Succeed":
            videos = (data.get("results") or {}).get("videos") or []
            video_url = (videos[0] or {}).get("url", "") if videos else ""
            break
        if state == "Failed":
            raise VideoApiError(f"生成失败：{data.get('reason') or '未知原因'}")
    if not video_url:
        raise VideoApiError("轮询超时或未返回视频 URL")

    # ③ 下载视频（URL 有效期 1 小时，立即取回 bytes）
    def _download() -> bytes:
        with urllib.request.urlopen(video_url, timeout=180) as response:
            return response.read()

    content = await asyncio.to_thread(_download)
    if not content:
        raise VideoApiError("下载到的视频为空")
    return content


# ===========================================================================
# 三、降级保存
# ===========================================================================
def _safe(text: str, fallback: str = "default") -> str:
    value = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(text or "")).strip(" ._")
    return value[:50] or fallback


def _save_prompt_file(state: ComicState, prompt: str, reason: str) -> Path:
    """降级时把视频提示词保存到 outputs/{story_id}/。"""
    story_id = _safe(str(state.get("story_id") or "default"))
    episode_no = int(state.get("episode_no") or 0)
    directory = config.OUTPUT_DIR / story_id
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f"第{episode_no}集" if episode_no > 0 else "video"
    path = directory / f"{suffix}_video_prompt.txt"
    path.write_text(
        f"{prompt}\n\n# 降级原因\n{reason}\n",
        encoding="utf-8",
    )
    return path


def _save_video(state: ComicState, content: bytes) -> Path:
    story_id = _safe(str(state.get("story_id") or "default"))
    episode_no = int(state.get("episode_no") or 0)
    directory = config.OUTPUT_DIR / story_id / "videos"
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f"第{episode_no}集" if episode_no > 0 else "video"
    path = directory / f"{suffix}_seedance.mp4"
    path.write_bytes(content)
    return path


def _degrade(
    state: ComicState,
    prompt: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    prompt_path = _save_prompt_file(state, prompt, reason)
    msg = f"{reason}；提示词已保存：{prompt_path}"
    print(f"[文生视频降级] {status}｜{msg}", flush=True)
    return {
        "video_path": "",
        "video_prompt": prompt,
        "status": status,
        "msg": msg,
    }


# ===========================================================================
# 四、对外节点
# ===========================================================================
async def video_gen_node(
    state: ComicState,
    enable_video: bool = True,
) -> dict[str, Any]:
    """
    文生视频节点（熔断 + 降级）。

    入参 state：video_prompt / image_prompt_positive / style_desc / story_id
    额外参数 enable_video：默认 True；False 时直接跳过，不消耗额度。

    返回固定字段：video_path / video_prompt / status / msg
    status：success / skipped_degrade / fail_degrade / circuit_open
    """
    prompt = str(
        state.get("video_prompt")
        or state.get("image_prompt_positive")
        or ""
    ).strip()
    style = str(state.get("style_desc") or config.DEFAULT_STYLE_DESC).strip()

    if not prompt:
        return _degrade(state, prompt, "fail_degrade", "没有视频提示词")

    # ① 手动关闭：不调用模型，只保存提示词。
    if not enable_video:
        return _degrade(
            state,
            prompt,
            "skipped_degrade",
            "enable_video=False，已跳过视频模型调用",
        )

    # ② 熔断器拦截。
    allowed, circuit_state = video_circuit_breaker.allow_request()
    if not allowed:
        return _degrade(
            state,
            prompt,
            "circuit_open",
            f"熔断器状态={circuit_state}，冷却中，拦截本次请求",
        )

    # ③ 调用视频模型；把已生成的分镜图作为首帧（图生视频锁角色）。
    #    没有图时 API 内部自动退化为纯文生视频。
    image_path = str(state.get("image_path") or "").strip()
    try:
        content = await call_seedance_async(prompt, style, image_path=image_path)
        path = _save_video(state, content)
        video_circuit_breaker.record_success()
        msg = f"视频生成成功：{path}"
        print(f"[文生视频] {msg}", flush=True)
        return {
            "video_path": str(path),
            "video_prompt": prompt,
            "status": "success",
            "msg": msg,
        }
    except Exception as exc:
        video_circuit_breaker.record_failure()
        return _degrade(
            state,
            prompt,
            "fail_degrade",
            f"视频模型调用失败：{type(exc).__name__}: {exc}",
        )


async def video_graph_node(state: ComicState) -> dict[str, Any]:
    """LangGraph 包装：固定返回字段映射到现有 state 字段。"""
    result = await video_gen_node(
        state,
        enable_video=bool(state.get("enable_video", config.VIDEO_GEN_ENABLED)),
    )
    return {
        "video_path": result["video_path"],
        "video_prompt": result["video_prompt"],
        "video_status": result["status"],
        "status": result["status"],
        "msg": result["msg"],
        "trace": [
            f"文生视频：status={result['status']}，{result['msg']}"
        ],
    }
