@echo off
rem AI 漫剧 Agent 一键启动（Windows）
rem 双击运行：自动创建虚拟环境、安装依赖、启动前后端
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [启动失败] 未检测到 Python，请先安装 Python 3.10+ 并勾选 "Add to PATH"
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [初始化] 首次运行，创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo [启动失败] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo [初始化] 安装依赖（约 2-5 分钟，仅首次）...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [启动失败] 依赖安装失败，请检查网络后重试
        pause
        exit /b 1
    )
)

if not exist ".env" (
    if exist ".env.example" (
        echo [提示] 未找到 .env，已从 .env.example 复制一份，请填入你自己的 API Key 后重新运行
        copy .env.example .env >nul
        notepad .env
        pause
        exit /b 0
    )
)

".venv\Scripts\python.exe" start.py
pause
