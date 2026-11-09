"""
全局配置：路径、模型参数、上下文限制、LLM 创建。

设计原则：
1. 所有“可调参数”集中在这里，新手不用到处翻代码。
2. 不在 import 时检查 API Key，这样 --seed-only 等纯本地功能也能运行。
3. ChatDeepSeek 延迟导入，缺少依赖时 main.py 可以先给出友好提示。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 一、路径
# ---------------------------------------------------------------------------
# __file__ 是 manju_agent/config.py，parent 就是 manju_agent 目录。
PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent

CHROMA_DIR = PROJECT_DIR / "chroma_db"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_DIR = PROJECT_DIR / "data"
SAMPLE_CHAPTER_PATH = DATA_DIR / "sample_chapter.txt"
DEMO_CHARACTERS_PATH = DATA_DIR / "demo_characters.json"

# ---------------------------------------------------------------------------
# 二、加载 .env
# ---------------------------------------------------------------------------
# 优先加载项目自己的 .env；如果不存在，再加载工作区根目录的 .env。
# override=False 表示：系统环境变量优先，不会被 .env 意外覆盖。
# 当前工作区根目录已经有 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL，可直接复用。
for _env_path in (PROJECT_DIR / ".env", WORKSPACE_DIR / ".env"):
    if _env_path.exists():
        load_dotenv(_env_path, override=False)


# ---------------------------------------------------------------------------
# 三、小工具：安全读取环境变量
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    """读取整数环境变量；写错时回退默认值并提示，不让程序直接崩溃。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[配置警告] {name}={raw!r} 不是整数，已使用默认值 {default}")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[配置警告] {name}={raw!r} 不是数字，已使用默认值 {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# 四、可调参数
# ---------------------------------------------------------------------------
# 模型名。当前账号已实测 deepseek-v4-flash 可用；
# 如果你的账号不支持，改成 deepseek-chat（在 .env 里设置 DEEPSEEK_MODEL）。
MODEL_NAME = (
    (os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash").strip()
    or "deepseek-v4-flash"
)
DEEPSEEK_BASE_URL = (os.getenv("DEEPSEEK_BASE_URL") or "").strip() or None

# 剧本节点专用：硅基流动 Qwen 免费版，429 时切更强的付费兜底。
# 注意：硅基流动实际注册 ID 是 Qwen/Qwen3-14B；
# 写成 Qwen/Qwen3-14B-Instruct 会返回 400（模型不存在）。
# 兼容你已有的 GJLD_API_KEY / GJLD_BASE_URL 命名。
SCRIPT_PROVIDER = (os.getenv("SCRIPT_PROVIDER") or "siliconflow").strip().lower()
SILICONFLOW_API_KEY = (
    os.getenv("SILICONFLOW_API_KEY")
    or os.getenv("GJLD_API_KEY")
    or ""
).strip()
SILICONFLOW_BASE_URL = (
    os.getenv("SILICONFLOW_BASE_URL")
    or os.getenv("GJLD_BASE_URL")
    or "https://api.siliconflow.cn/v1"
).strip()
SCRIPT_MODEL_FREE = (
    os.getenv("SCRIPT_MODEL_FREE") or "Qwen/Qwen3-14B"
).strip()
# Qwen3-14B 在硅基流动没有 Pro/ 别名；429 时用同系列更大的 32B 付费模型兜底。
SCRIPT_MODEL_PRO = (
    os.getenv("SCRIPT_MODEL_PRO") or "Qwen/Qwen3-32B"
).strip()
SCRIPT_LLM_MAX_RETRIES = _env_int("SCRIPT_LLM_MAX_RETRIES", 0)

# 文生图 / 文生视频（第三方接口只从环境变量读取，不写死 key）
DEFAULT_STYLE_DESC = (os.getenv("STYLE_DESC") or "写实").strip()
# 菜单 11 自动读取的当前故事；不设置时自动选集数最多的故事。
ACTIVE_STORY_ID = (os.getenv("ACTIVE_STORY_ID") or "").strip()
# 全局角色档案优先；为 True 时 Chroma 只保存剧情摘要，不再写完整角色卡。
GLOBAL_PROFILE_PRIMARY = True
IMAGE_NEGATIVE_BASE = (
    os.getenv("IMAGE_NEGATIVE_BASE")
    or "畸形五官，扭曲肢体，多手指，水印，文字，多余物体"
).strip()
IMAGE_API_URL = (
    os.getenv("IMAGE_API_URL")
    or "https://api.siliconflow.cn/v1/images/generations"
).strip()
IMAGE_API_KEY = (
    os.getenv("IMAGE_API_KEY")
    or os.getenv("SILICONFLOW_API_KEY")
    or os.getenv("GJLD_API_KEY")
    or ""
).strip()
IMAGE_API_MODEL = (
    os.getenv("IMAGE_API_MODEL") or "Tongyi-MAI/Z-Image-Turbo"
).strip()
IMAGE_SIZE = (os.getenv("IMAGE_SIZE") or "1920x1080").strip()
# 默认不调用 DeepSeek 生成绘图 prompt；需要 LLM 扩写时设 1。
IMAGE_PROMPT_USE_LLM = _env_bool("IMAGE_PROMPT_USE_LLM", False)
IMAGE_TIMEOUT_SECONDS = _env_float("IMAGE_TIMEOUT_SECONDS", 180.0)
IMAGE_GEN_MAX_RETRIES = _env_int("IMAGE_GEN_MAX_RETRIES", 2)

# 视频：Seedance（火山方舟）
SEEDANCE_API_KEY = (
    os.getenv("SEEDANCE_KEY") or os.getenv("SEEDANCE_API_KEY") or ""
).strip()
SEEDANCE_URL = (
    os.getenv("SEEDANCE_URL")
    or "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
).strip()
SEEDANCE_MODEL = (
    os.getenv("SEEDANCE_MODEL") or "doubao-seedance-2-0-fast-260128"
).strip()
VIDEO_GEN_ENABLED = _env_bool("VIDEO_GEN_ENABLED", False)

# 单次输入上限。中文小说一章通常 2k~8k 字，20k 已经比较宽松。
# 超过后会截断，并在控制台和最终报告里明确提示，避免悄悄丢剧情。
MAX_INPUT_CHARS = _env_int("MAX_INPUT_CHARS", 20_000)

# 校验不通过时最多重写几次剧本。0 表示只跑一轮，适合快速调提示词。
MAX_REVISIONS = _env_int("MAX_REVISIONS", 2)

# 单次 LLM 最大输出 token。剧本/分镜是结构化 JSON，给足空间防止被截断。
MAX_TOKENS = _env_int("MAX_TOKENS", 8_192)
LLM_TIMEOUT_SECONDS = _env_float("LLM_TIMEOUT_SECONDS", 120.0)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)

# deepseek-v4-flash 支持关闭 thinking，能减少无用的推理输出。
# 如果换成不支持该参数的模型并报 400，把 .env 里的 DISABLE_THINKING 设为 0。
DISABLE_THINKING = _env_bool("DISABLE_THINKING", True)

# 四个节点的温度：解析/校验要稳定，剧本要有一点创作感，分镜适中。
PARSE_TEMPERATURE = _env_float("PARSE_TEMPERATURE", 0.1)
SCRIPT_TEMPERATURE = _env_float("SCRIPT_TEMPERATURE", 0.7)
STORYBOARD_TEMPERATURE = _env_float("STORYBOARD_TEMPERATURE", 0.4)
VALIDATE_TEMPERATURE = _env_float("VALIDATE_TEMPERATURE", 0.1)

# 人物设定库
COLLECTION_NAME = "character_settings"
# 故事级记忆库：角色、摘要、剧本、分镜
STORY_COLLECTION_NAME = "story_memory"
EMBEDDING_DIM = 384
# 语义匹配的保守阈值。哈希 embedding 的相似度偏低，默认主要靠姓名/别名精确匹配；
# 换成真实 embedding 后可以把这个值调到 0.7~0.85 左右再试。
SEMANTIC_MATCH_THRESHOLD = _env_float("SEMANTIC_MATCH_THRESHOLD", 0.55)

# 上下文控制：人物卡、校验反馈、历史剧情注入提示词时的最大长度。
MAX_CANON_CHARS_PER_CHARACTER = 900
MAX_FEEDBACK_CHARS = 2_000
MAX_ISSUES_IN_FEEDBACK = 10
MAX_ACCUMULATED_ISSUES = _env_int("MAX_ACCUMULATED_ISSUES", 20)
MAX_BRANCH_FEEDBACK_CHARS = _env_int("MAX_BRANCH_FEEDBACK_CHARS", 6_000)
MAX_HISTORY_CHARS = _env_int("MAX_HISTORY_CHARS", 12_000)
MAX_RECALL_EPISODES = _env_int("MAX_RECALL_EPISODES", 3)

# 新分支温度：意图识别要稳定，局部修改不宜太发散，续写可以更有创作感。
INTENT_TEMPERATURE = _env_float("INTENT_TEMPERATURE", 0.0)
CONTINUE_SCRIPT_TEMPERATURE = _env_float("CONTINUE_SCRIPT_TEMPERATURE", 0.7)
MODIFY_SCRIPT_TEMPERATURE = _env_float("MODIFY_SCRIPT_TEMPERATURE", 0.3)

# 意图置信度硬阈值（按需求固定）：
# >= 0.85 放行；0.60 < c < 0.85 二次确认；<= 0.60 直接 unknown。
INTENT_HIGH_CONFIDENCE = 0.85
INTENT_LOW_CONFIDENCE = 0.60

# 续写/修改分支“剧情忠实度校验”失败后最多重写几次。
MAX_BRANCH_REVISIONS = _env_int("MAX_BRANCH_REVISIONS", 2)
BRANCH_VALIDATE_TEMPERATURE = _env_float("BRANCH_VALIDATE_TEMPERATURE", 0.1)

# 运行期开关，由 main.py 的命令行参数设置。
DEBUG = False


def set_debug(enabled: bool) -> None:
    """由 --debug 调用，打印更多原始输出。"""
    global DEBUG
    DEBUG = enabled


def set_model_override(model_name: str | None) -> None:
    """由 --model 调用，临时覆盖 .env 中的模型名。"""
    global MODEL_NAME
    if model_name:
        MODEL_NAME = model_name.strip()


# ---------------------------------------------------------------------------
# 五、控制台与目录
# ---------------------------------------------------------------------------
def configure_console() -> None:
    """
    尽量让 Windows 控制台按 UTF-8 输出中文。

    如果终端仍然乱码，README 里提供了 chcp 65001 / PyCharm 编码设置的办法。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and getattr(stream, "encoding", "").lower() != "utf-8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # 某些被重定向的流不支持 reconfigure，忽略即可，不影响主流程。
            pass


def ensure_output_dir() -> Path:
    """确保 outputs/ 存在，并返回路径。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# 六、LLM 工厂
# ---------------------------------------------------------------------------
class ConfigError(RuntimeError):
    """配置问题（缺 API Key 等），由 main.py 捕获后给出友好提示。"""


def get_llm(temperature: float):
    """
    创建一个 ChatDeepSeek 实例。

    每次调用都新建实例，避免不同节点共享温度等参数时互相影响。
    真正的网络重试由 SDK 的 max_retries 处理；结构化解析失败的重试在 nodes.py 里做。
    """
    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError(
            "没有找到 DEEPSEEK_API_KEY。\n"
            f"请把它写入以下任一文件：\n"
            f"  1) {PROJECT_DIR / '.env'}\n"
            f"  2) {WORKSPACE_DIR / '.env'}\n"
            "内容示例：DEEPSEEK_API_KEY=sk-xxxx\n"
            "注意不要把 Key 提交到公开仓库。"
        )

    # 延迟导入：--seed-only 等本地功能不需要 langchain-deepseek。
    from langchain_deepseek import ChatDeepSeek

    kwargs = {
        "model": MODEL_NAME,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "timeout": LLM_TIMEOUT_SECONDS,
        "max_retries": LLM_MAX_RETRIES,
    }
    if DEEPSEEK_BASE_URL:
        kwargs["api_base"] = DEEPSEEK_BASE_URL
    if DISABLE_THINKING:
        # 当前 deepseek-v4-flash 实测支持；换模型报 400 时把 DISABLE_THINKING 设为 0。
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    return ChatDeepSeek(**kwargs)


def get_script_llm(
    temperature: float,
    model: str | None = None,
    max_retries: int | None = None,
):
    """
    剧本节点专用 LLM：硅基流动 OpenAI 兼容接口。

    - 默认免费模型 Qwen/Qwen3-14B；
    - 429 时由 nodes.invoke_structured 切换到更大的 Qwen/Qwen3-32B 付费兜底；
    - 免费模型 max_retries 默认 0，避免 SDK 自己重试 429 导致无法及时切兜底。
    """
    if SCRIPT_PROVIDER != "siliconflow":
        raise ConfigError(
            f"暂不支持的 SCRIPT_PROVIDER={SCRIPT_PROVIDER!r}，"
            "当前只实现了 siliconflow。"
        )
    if not SILICONFLOW_API_KEY:
        raise ConfigError(
            "没有找到 SILICONFLOW_API_KEY（或兼容的 GJLD_API_KEY）。\n"
            f"请写入 {PROJECT_DIR / '.env'} 或 {WORKSPACE_DIR / '.env'}，例如：\n"
            "SILICONFLOW_API_KEY=sk-xxxx\n"
            "GJLD_BASE_URL=https://api.siliconflow.cn/v1"
        )

    from langchain_openai import ChatOpenAI

    model_name = (model or SCRIPT_MODEL_FREE).strip()
    if max_retries is None:
        max_retries = 0 if model_name == SCRIPT_MODEL_FREE else LLM_MAX_RETRIES

    return ChatOpenAI(
        model=model_name,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=max_retries,
    )
