#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文档一致性校验脚本（CI 用，也可本地直接运行）。

校验三件事，任何一项失败都以非零码退出：
1. README / 技术文档中的相对链接都指向仓库内真实存在的文件或目录；
2. docs/训练日志/README.md 里所有 `文件名:行号` 引用都能定位到该日志的真实行；
3. README 里公开的目差区间与实际日志统计出的区间一致（防止文档与日志漂移）。

设计原则：只读仓库文件，不访问网络，因此可在任意机器/CI 上离线运行。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "docs" / "训练日志"

# 需要检查相对链接的文档；缺失的文件直接跳过，避免耦合仓库结构变动
LINK_DOCS = [
    ROOT / "README.md",
    ROOT / "README.en.md",
    ROOT / "docs" / "训练日志" / "README.md",
    ROOT / "docs" / "技术文档" / "成果说明.md",
    ROOT / "docs" / "技术文档" / "训练日志总结.md",
]

# Markdown 行内链接：[文本](目标)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# 日志引用：形如 `train_out_scale.txt:50`
CITATION_RE = re.compile(r"`([A-Za-z0-9_.]+\.txt):(\d+)`")
# 日志中的评估结果行
EVAL_RE = re.compile(r"评估1局 vs KataGo.*?目差=(-?[0-9.]+)目")

errors: list[str] = []


def check_links(path: Path) -> None:
    """检查单个 Markdown 文件中所有相对链接的目标是否存在。"""
    text = path.read_text(encoding="utf-8")
    for target in LINK_RE.findall(text):
        # 跳过外链、页内锚点和邮件链接
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        # 去掉链接里的锚点/查询串后解析目标路径
        pure = target.split("#", 1)[0].split("?", 1)[0]
        if not pure:
            continue
        # README 中使用了少数站内绝对路径（/SpaceGo/...），跳过
        if pure.startswith("/"):
            continue
        resolved = (path.parent / pure).resolve()
        if not resolved.exists():
            errors.append(f"[链接] {path.relative_to(ROOT)} -> {target} 不存在")


def check_citations() -> None:
    """检查训练日志索引中的 文件:行号 引用是否落在文件的真实行号范围内。"""
    index = LOG_DIR / "README.md"
    if not index.exists():
        errors.append("[日志] docs/训练日志/README.md 缺失，无法校验引用")
        return
    text = index.read_text(encoding="utf-8")
    for name, line_no in CITATION_RE.findall(text):
        log_path = LOG_DIR / name
        if not log_path.exists():
            errors.append(f"[日志] 引用 {name}:{line_no}，但 {name} 不在 docs/训练日志/ 中")
            continue
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        number = int(line_no)
        if number < 1 or number > len(lines):
            errors.append(f"[日志] {name}:{line_no} 越界（该文件共 {len(lines)} 行）")
            continue
        if not lines[number - 1].strip():
            errors.append(f"[日志] {name}:{line_no} 是空行，引用无效")


def check_benchmark_numbers() -> None:
    """把日志里实测出的目差区间与 README 公开的数字对齐。"""
    gaps: list[float] = []
    for log in sorted(LOG_DIR.glob("*.txt")):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = EVAL_RE.search(line)
            if match:
                gaps.append(float(match.group(1)))
    if not gaps:
        errors.append("[数值] 训练日志中没有找到任何 KataGo 评估结果行")
        return

    best = max(gaps)   # 最接近 0（数值最大）的一次
    worst = min(gaps)  # 差距最大的一次
    # README 中应当出现这两个数字（保留一位小数）
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for label, value in (("最优", best), ("下界", worst)):
        token = f"{value:.1f}"
        if token not in readme:
            errors.append(f"[数值] 日志统计的{label}目差 {token} 未出现在 README.md 中")
    print(f"评测样本 {len(gaps)} 次，目差区间 {worst:.1f} ~ {best:.1f}")


def main() -> int:
    for doc in LINK_DOCS:
        if doc.exists():
            check_links(doc)
    check_citations()
    check_benchmark_numbers()

    if errors:
        print("发现以下不一致：")
        for item in errors:
            print("  - " + item)
        return 1
    print("文档一致性校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
