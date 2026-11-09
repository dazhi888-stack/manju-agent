"""
LangSmith 配置与检查。

LangGraph / LangChain 会自动读取 LANGSMITH_* / LANGCHAIN_* 环境变量并上报 trace。
这个文件做两件事：
1. 把 .env 里的配置同步成 LangChain 也识别的别名；
2. 提供一个可直接使用的 LangSmith Client 实例和连接检查，方便确认 trace 是否可上报。
"""

from __future__ import annotations

import os
from typing import Any


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip().strip('"').strip("'")


def _enabled() -> bool:
    return _env("LANGSMITH_TRACING").lower() in {"1", "true", "yes", "y", "on"}


def _api_key() -> str:
    return _env("LANGSMITH_API_KEY") or _env("LANGCHAIN_API_KEY")


def _project() -> str:
    return _env("LANGSMITH_PROJECT") or _env("LANGCHAIN_PROJECT") or "AImanju"


def _endpoint() -> str:
    return (
        _env("LANGSMITH_ENDPOINT")
        or _env("LANGCHAIN_ENDPOINT")
        or "https://api.smith.langchain.com"
    )


def configure_langsmith() -> bool:
    """
    把环境变量补齐为 LangChain/LangGraph 能识别的形式。

    返回 True 表示 tracing 已开启且存在 API Key。
    注意：这个函数要在导入 graph 之前调用。
    """
    if not _enabled():
        return False
    key = _api_key()
    if not key:
        return False

    project = _project()
    endpoint = _endpoint()

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", key)
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)

    # LangChain 旧变量名兼容
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", key)
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
    return True


_CLIENT: Any | None = None


def get_langsmith_client() -> Any:
    """返回可复用的 LangSmith Client；没有 Key 时抛 RuntimeError。"""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    key = _api_key()
    if not key:
        raise RuntimeError(
            "没有 LANGSMITH_API_KEY。请在 .env 中设置 "
            "LANGSMITH_TRACING=true 和 LANGSMITH_API_KEY=lsv2_..."
        )

    from langsmith import Client

    _CLIENT = Client(api_url=_endpoint(), api_key=key)
    return _CLIENT


def langsmith_status() -> dict[str, Any]:
    """返回当前配置状态，不打印密钥。"""
    return {
        "tracing": _enabled(),
        "key_present": bool(_api_key()),
        "project": _project(),
        "endpoint": _endpoint(),
    }


def check_langsmith() -> bool:
    """检查 LangSmith 依赖、配置和网络连接；返回是否可用。"""
    status = langsmith_status()
    print("=" * 72)
    print("LangSmith 检查")
    print("-" * 72)
    print(f"tracing：{status['tracing']}")
    print(f"API Key：{'已配置' if status['key_present'] else '未配置'}")
    print(f"项目：{status['project']}")
    print(f"Endpoint：{status['endpoint']}")

    try:
        import importlib.metadata as metadata

        print(f"langsmith 版本：{metadata.version('langsmith')}")
    except Exception:
        print("langsmith 版本：未知")

    if not status["tracing"]:
        print("结果：未开启。请在 .env 设置 LANGSMITH_TRACING=true。")
        print("=" * 72)
        return False
    if not status["key_present"]:
        print("结果：缺少 LANGSMITH_API_KEY。")
        print("=" * 72)
        return False

    configure_langsmith()
    try:
        client = get_langsmith_client()
        projects = list(client.list_projects(limit=10))
        print(f"连接：成功｜可见项目 {len(projects)} 个")
        for project in projects[:10]:
            print(f"  - {project.name}")
        if not any(str(project.name) == status["project"] for project in projects):
            print(
                f"提示：当前项目 {status['project']} 还没有出现在列表里；"
                "第一次上报 trace 后会自动出现。"
            )
        print("结果：可用。运行一次 Agent 后即可在 LangSmith 查看 trace。")
        print("=" * 72)
        return True
    except Exception as exc:
        print(f"连接：失败｜{type(exc).__name__}: {exc}")
        print("=" * 72)
        return False
