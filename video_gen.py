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
# 二、Seedance 异步 API 接入位置（故意留空）
# ===========================================================================
class VideoApiError(RuntimeError):
    """视频接口错误/未接入异常。"""


@traceable(run_type="tool", name="seedance_async_api")
async def call_seedance_async(
    positive_prompt: str,
    style_desc: str,
    image_path: str = "",
) -> bytes:
    """
    ===== 在这里填写 Seedance 异步 API 调用代码 =====

    当前故意不接入模型，直接抛 VideoApiError：
    - 视频节点会捕获异常并走 fail_degrade；
    - 提示词保存到 outputs/{story_id}/；
    - Web 页面必须兼容“视频还没接入、只能降级输出提示词”。

    接入时只需要让本函数返回视频二进制 bytes，熔断/降级/文件保存都不需要改。
    """
    raise VideoApiError(
        "Seedance 尚未接入：请在 video_gen.call_seedance_async 中填写真实异步 API"
    )


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

    # ③ 调用 Seedance；失败进入降级并记录熔断失败。
    try:
        content = await call_seedance_async(prompt, style)
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
            f"Seedance 调用失败：{type(exc).__name__}: {exc}",
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
