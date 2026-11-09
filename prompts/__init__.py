"""提示词集中管理：核心节点（core）、全局角色抽取（profile）、分镜（shots）。

`from prompts import X` 兼容旧写法：核心提示词由 __init__ 统一 re-export。
"""

from prompts.core import *  # noqa: F401,F403
