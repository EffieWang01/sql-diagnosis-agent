"""语义上下文层：业务指标文档 + 已验证 SQL 的检索。

设计约束（来自技术方案，别越界）：
  - 上下文里**只有语义**：指标口径、字段含义、表间关系、已验证的查询范式。
  - 上下文里**绝不出现业务数值**。否则模型可以直接从上下文抄答案，
    可验证奖励就失效了。
  - 检索是纯本地的词法检索，不调外部模型——保证轨迹可复现、零成本。

为什么自己写检索而不用向量库：本项目上下文规模只有几十到几百条，
词法匹配足够且零依赖；引入 embedding 只会增加复现成本。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..utils.io import ensure_dir, read_json, write_json

# ---------------------------------------------------------------------- #
_CJK = re.compile(r"[\u4e00-\u9fff]")
_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+")


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文/标识符按词切，中文按字符二元组切。

    不引入 jieba，是为了让检索结果在任何环境下完全一致。
    """
    text = text.lower()
    tokens: list[str] = [t for t in _TOKEN.findall(text)]
    cjk_chars = _CJK.findall(text)
    tokens.extend(
        cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)
    )
    tokens.extend(cjk_chars)
    return tokens


@dataclass
class ContextEntry:
    """一条语义上下文条目。"""

    id: str
    kind: str  # "metric" | "verified_sql" | "table_note" | "relation"
    title: str
    text: str
    tags: list[str] = field(default_factory=list)
    # 已验证 SQL 可携带其所属表，便于按表过滤
    tables: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        head = f"[{self.kind}] {self.title}"
        return f"{head}\n{self.text.strip()}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContextEntry":
        return cls(
            id=d["id"], kind=d.get("kind", "metric"), title=d.get("title", ""),
            text=d.get("text", ""), tags=list(d.get("tags", [])),
            tables=list(d.get("tables", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "title": self.title,
            "text": self.text, "tags": self.tags, "tables": self.tables,
        }


class ContextStore:
    """语义上下文库 + 词法检索（TF-IDF + 标签加权）。"""

    def __init__(self, entries: Iterable[ContextEntry] | None = None) -> None:
        self.entries: list[ContextEntry] = list(entries or [])
        self._df: Counter[str] = Counter()
        self._tokens: list[list[str]] = []
        self._reindex()

    # ------------------------------------------------------------------ #
    def _reindex(self) -> None:
        self._df = Counter()
        self._tokens = []
        for e in self.entries:
            toks = tokenize(f"{e.title} {e.text} {' '.join(e.tags)}")
            self._tokens.append(toks)
            self._df.update(set(toks))

    def add(self, entry: ContextEntry) -> None:
        self.entries.append(entry)
        self._reindex()

    def extend(self, entries: Iterable[ContextEntry]) -> None:
        self.entries.extend(entries)
        self._reindex()

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------ #
    def _idf(self, tok: str) -> float:
        n = max(len(self.entries), 1)
        return math.log((n + 1) / (self._df.get(tok, 0) + 1)) + 1.0

    def search(
        self,
        query: str,
        top_k: int = 3,
        kind: str | None = None,
        tables: list[str] | None = None,
    ) -> list[tuple[ContextEntry, float]]:
        """返回 (条目, 得分) 列表，按得分降序。

        打分 = TF-IDF 余弦 + 标签精确命中加权 + 表名命中加权。
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_tf = Counter(q_tokens)
        q_vec = {t: (1 + math.log(c)) * self._idf(t) for t, c in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_lower = query.lower()

        scored: list[tuple[ContextEntry, float]] = []
        for e, toks in zip(self.entries, self._tokens):
            if kind and e.kind != kind:
                continue
            if tables and e.tables and not (set(tables) & set(e.tables)):
                continue

            tf = Counter(toks)
            dot = sum(
                (1 + math.log(c)) * self._idf(t) * q_vec.get(t, 0.0)
                for t, c in tf.items()
            )
            d_norm = math.sqrt(
                sum(((1 + math.log(c)) * self._idf(t)) ** 2 for t, c in tf.items())
            ) or 1.0
            score = dot / (q_norm * d_norm)

            # 标签/标题的精确命中额外加权——业务术语常常是专有名词
            for tag in e.tags:
                if tag.lower() in q_lower:
                    score += 0.25
            if e.title and e.title.lower() in q_lower:
                score += 0.35

            if score > 0:
                scored.append((e, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def render(self, hits: list[tuple[ContextEntry, float]]) -> str:
        """把检索结果渲染成回灌给模型的文本。"""
        if not hits:
            return "未检索到相关的指标定义或已验证查询。可先用 DESCRIBE 查看表结构。"
        parts = []
        for e, _ in hits:
            parts.append(e.to_text())
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_json(cls, path: str | Path) -> "ContextStore":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = read_json(p)
        items = raw.get("entries", raw) if isinstance(raw, dict) else raw
        return cls(ContextEntry.from_dict(d) for d in items)

    def to_json(self, path: str | Path) -> Path:
        return write_json(
            {"entries": [e.to_dict() for e in self.entries]}, ensure_dir(Path(path).parent) / Path(path).name
        )
