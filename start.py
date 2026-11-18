"""一键启动脚本：同时拉起 FastAPI 后端 + Streamlit 前端。

用法（在项目根目录）：
    python start.py

脚本会依次检查：
1. Python 版本 >= 3.10
2. 依赖是否已安装（缺什么列什么）
3. .env 是否存在（不存在给出配置指引，不打印任何密钥）

通过后自动启动两个服务并打开浏览器，Ctrl+C 一键全部退出。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

BACKEND_CMD = [sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"]
FRONTEND_CMD = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "web_app.py",
    "--server.port",
    "8501",
    "--server.headless",
    "true",
]

REQUIRED_PACKAGES = [
    "langgraph",
    "langchain",
    "langchain_core",
    "langchain_deepseek",
    "langchain_openai",
    "openai",
    "fastapi",
    "uvicorn",
    "streamlit",
    "requests",
    "chromadb",
    "dotenv",
    "pydantic",
]


def check_python() -> bool:
    if sys.version_info < (3, 10):
        print("[启动失败] 需要 Python >= 3.10，当前版本：", sys.version)
        return False
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_packages() -> bool:
    missing = [
        name
        for name in REQUIRED_PACKAGES
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        print("[启动失败] 缺少依赖：", ", ".join(missing))
        print("请先执行：pip install -r requirements.txt")
        return False
    print("[OK] 依赖完整")
    return True


def check_env() -> bool:
    # 与 config.py 的加载逻辑保持一致：项目目录或其上级目录任一存在即可
    if (PROJECT_DIR / ".env").exists() or (PROJECT_DIR.parent / ".env").exists():
        print("[OK] 已找到 .env 配置")
        return True
    print("[启动失败] 项目根目录（或其上级目录）没有 .env 文件")
    print("请复制 .env.example 为 .env，并填入你自己的 API Key：")
    print("    copy .env.example .env")
    print("（脚本不会读取/打印密钥内容）")
    return False


def main() -> None:
    print("=" * 60)
    print("AI 漫剧 Agent 一键启动")
    print("=" * 60)

    if not (check_python() and check_packages() and check_env()):
        sys.exit(1)

    print("[启动] 后端 FastAPI -> http://127.0.0.1:8000")
    backend = subprocess.Popen(BACKEND_CMD, cwd=PROJECT_DIR)

    print("[启动] 前端 Streamlit -> http://127.0.0.1:8501")
    frontend = subprocess.Popen(FRONTEND_CMD, cwd=PROJECT_DIR)

    print("[等待] 服务就绪中（约 8 秒）...")
    time.sleep(8)

    if backend.poll() is not None:
        print("[启动失败] 后端已退出，请检查上方后端报错信息")
        frontend.terminate()
        sys.exit(1)
    if frontend.poll() is not None:
        print("[启动失败] 前端已退出，请检查上方前端报错信息")
        backend.terminate()
        sys.exit(1)

    print("=" * 60)
    print("启动完成！")
    print("  前端界面：http://127.0.0.1:8501")
    print("  接口文档：http://127.0.0.1:8000/docs")
    print("按 Ctrl+C 可同时退出前后端")
    print("=" * 60)

    webbrowser.open("http://127.0.0.1:8501")

    try:
        # 任一子进程退出就整体退出，避免出现"半死"状态
        while True:
            if backend.poll() is not None or frontend.poll() is not None:
                print("[警告] 有服务异常退出，正在关闭全部进程...")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n[退出] 收到 Ctrl+C，正在关闭...")
    finally:
        for proc in (frontend, backend):
            if proc.poll() is None:
                proc.terminate()
        print("[完成] 前后端已关闭")


if __name__ == "__main__":
    main()
