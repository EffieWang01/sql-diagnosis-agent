"""SQL 文本处理工具：去噪、归一化、只读判定。

被沙箱（安全校验）和奖励（重复查询检测）共用，必须保证两边口径一致。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- 去噪
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"--[^\n]*")
# 单引号字符串（含 '' 转义）与双引号标识符
_SINGLE_QUOTED = re.compile(r"'(?:[^']|'')*'")
_DOUBLE_QUOTED = re.compile(r'"(?:[^"]|"")*"')
_DOLLAR_QUOTED = re.compile(r"\$\$.*?\$\$", re.DOTALL)


def strip_sql_noise(sql: str) -> str:
    """去掉注释与字符串字面量，只保留「结构性」文本。

    这样后续关键字检查不会被 `WHERE note = 'DROP'` 这类内容误伤。
    """
    s = _DOLLAR_QUOTED.sub("''", sql)
    s = _BLOCK_COMMENT.sub(" ", s)
    s = _LINE_COMMENT.sub(" ", s)
    s = _SINGLE_QUOTED.sub("''", s)
    s = _DOUBLE_QUOTED.sub('""', s)
    return s


def split_statements(sql: str) -> list[str]:
    """按分号切分多语句（同样先剥离字符串与注释，避免误切）。"""
    s = strip_sql_noise(sql)
    return [p.strip() for p in s.split(";") if p.strip()]


def first_keyword(sql: str) -> str:
    """取第一条语句的首个关键字（大写）。"""
    s = strip_sql_noise(sql).strip()
    # 去掉开头的括号，兼容 (SELECT ...) UNION ...
    s = s.lstrip("( \t\r\n")
    m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s)
    return m.group(0).upper() if m else ""


def find_forbidden(sql: str, keywords: tuple[str, ...]) -> str | None:
    """在去噪后的 SQL 中查找被禁关键字（整词匹配），返回首个命中的词。"""
    body = strip_sql_noise(sql)
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", body, re.IGNORECASE):
            return kw
    return None


def find_forbidden_function(sql: str, funcs: tuple[str, ...]) -> str | None:
    """查找被禁函数调用，要求后接左括号，避免命中同名列。"""
    body = strip_sql_noise(sql)
    for fn in funcs:
        if re.search(rf"\b{re.escape(fn)}\s*\(", body, re.IGNORECASE):
            return fn
    return None


# ---------------------------------------------------------------- 归一化
_WS = re.compile(r"\s+")
_NUM_LITERAL = re.compile(r"\b\d+(?:\.\d+)?\b")


def normalize_sql(sql: str, mask_numbers: bool = False) -> str:
    """把 SQL 归一化成可比较的形式，用于「重复查询」检测。

    做四件事：去注释/字符串噪声 → 统一空白 → 转小写 → 去尾部/内部分号。
    mask_numbers=True 时把数字字面量替换成占位符（用于识别「只改了阈值」的近似重复）。
    """
    s = strip_sql_noise(sql).lower()
    s = _WS.sub(" ", s).strip().strip(";").strip()
    if mask_numbers:
        s = _NUM_LITERAL.sub("?", s)
    # 统一空格：逗号、括号前后
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\(\s*", "(", s)
    s = re.sub(r"\s*\)", ")", s)
    return s


def sql_similarity(a: str, b: str, mask_numbers: bool = False) -> float:
    """归一化后的序列相似度（0~1），用 difflib 即可，无需额外依赖。"""
    from difflib import SequenceMatcher

    na, nb = normalize_sql(a, mask_numbers), normalize_sql(b, mask_numbers)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()
