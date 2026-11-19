@echo off
rem AI 漫剧 Agent 一键启动（Windows）
rem 双击运行：自动创建虚拟环境、安装依赖、启动前后端
chcp 65001 >nul
cd /d "%~dp0"

set "PYEXE=python"
python --version >nul 2>nul
if errorlevel 1 (
    set "PYEXE=py -3"
    py -3 --version >nul 2>nul
)
%PYEXE% --version >nul 2>nul
if errorlevel 1 (
    echo [启动失败] 未检测到可用的 Python，请先安装 Python 3.10+ 并勾选 "Add to PATH"
    echo [提示] 安装时可勾选 py launcher；安装完成后重新双击本文件
    pause
    exit /b 1
)
echo [环境] 使用 Python: %PYEXE%

if not exist ".venv\Scripts\python.exe" (
    echo [初始化] 首次运行，创建虚拟环境...
    %PYEXE% -m venv .venv
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
        echo [提示] 未找到 .env，已从 .env.example 复制一份
        echo [提示] 请在弹出的记事本里填入你的 API Key，保存并关闭记事本后按任意键继续启动
        copy .env.example .env >nul
        notepad .env
        echo [检查] 正在确认 .env 是否已填写...
        findstr /r "^DEEPSEEK_API_KEY=sk-." .env >nul
        if errorlevel 1 (
            echo [启动失败] .env 中还没有有效的 DEEPSEEK_API_KEY（应以 DEEPSEEK_API_KEY=sk- 开头），请重新运行并填写
            notepad .env
            pause
            exit /b 1
        )
        echo [检查] .env 已填写，继续启动...
    )
)

".venv\Scripts\python.exe" start.py
pause
