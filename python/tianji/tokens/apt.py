"""Agent-pretraining tokenizer (APT): vocab build, special-token encode,
AST extraction, and deterministic embedding.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

SPECIAL_TAGS = [
    "<tool_call>", "</tool_call>",
    "<bash_output>", "</bash_output>",
    "<system>", "</system>",
    "<cot>", "</cot>",
    "<diff>", "</diff>",
    # Decision Transformer: return-conditioned generation tokens.
    # High RTG biases model toward good agent behavior during inference.
    "<rtg_high>", "<rtg_low>",
]

SPECIAL_IDS = {tag: i for i, tag in enumerate(SPECIAL_TAGS)}


@dataclass
class EncodeOut:
    ids: List[int]
    ast_nodes: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class Vocab:
    tokens: List[str]
    size: int
    dim: int
    ast_dim: int
    char_ids: dict = field(default_factory=dict)
    embeddings: np.ndarray = None

    @classmethod
    def build(cls, corpus, target_size: int = 128, dim: int = 16, ast_dim: int = 8) -> "Vocab":
        rng = np.random.default_rng(0)
        size = max(target_size, len(SPECIAL_TAGS) + 1)
        n_special = len(SPECIAL_TAGS)
        tokens: List[str] = list(SPECIAL_TAGS)
        chars: set = set()
        for line in corpus:
            for ch in line:
                chars.add(ch)
        char_list = sorted(chars)
        for ch in char_list:
            if len(tokens) >= size:
                break
            tokens.append(ch)
        while len(tokens) < size:
            tokens.append(f"\u0001{len(tokens)}")
        tokens = tokens[:size]
        char_ids = {}
        for i, ch in enumerate(char_list):
            if n_special + i >= size:
                break
            char_ids[ch] = n_special + i
        embeddings = rng.standard_normal((size, dim + ast_dim)).astype(np.float32)
        embeddings[:n_special] = 0.0
        return cls(tokens=tokens, size=size, dim=dim, ast_dim=ast_dim, char_ids=char_ids, embeddings=embeddings)


def _encode_text_pieces(text: str, vocab: Vocab) -> List[int]:
    ids: List[int] = []
    i = 0
    n = len(text)
    while i < n:
        matched = None
        for tag in SPECIAL_TAGS:
            if text.startswith(tag, i):
                matched = tag
                break
        if matched is not None:
            ids.append(SPECIAL_IDS[matched])
            i += len(matched)
            continue
        ch = text[i]
        ids.append(vocab.char_ids.get(ch, SPECIAL_IDS["<system>"]))
        i += 1
    return ids


def encode(text: str, vocab: Vocab, parse_ast: bool = False, lang: Optional[str] = None) -> EncodeOut:
    ids = _encode_text_pieces(text, vocab)
    ast_nodes: List[Tuple[str, str]] = []
    if parse_ast and text:
        ast_nodes = _extract_ast(text, lang)
    return EncodeOut(ids=ids, ast_nodes=ast_nodes)


def _extract_ast(text: str, lang: Optional[str]) -> List[Tuple[str, str]]:
    nodes: List[Tuple[str, str]] = []
    try:
        lang = (lang or "python").lower()
        if lang in ("python", "javascript", "typescript", "js", "ts"):
            nodes = _ts_ast(text, lang)
            if nodes:
                return nodes
    except Exception:
        pass
    return _regex_ast(text, lang)


def _ts_ast(text: str, lang: str) -> List[Tuple[str, str]]:
    try:
        import tree_sitter
        import tree_sitter_python
        import tree_sitter_javascript
    except Exception:
        return []
    if lang in ("python",):
        parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_python.language()))
    else:
        parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_javascript.language()))
    tree = parser.parse(text.encode("utf-8"))
    out = []

    def walk(node):
        name = node.type
        snippet = text[node.start_byte:node.end_byte]
        out.append((name, snippet))
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return out


def _regex_ast(text: str, lang: Optional[str]) -> List[Tuple[str, str]]:
    nodes: List[Tuple[str, str]] = []
    lang = (lang or "python").lower()
    if lang == "python":
        for m in re.finditer(r"def\s+([A-Za-z_]\w*)\s*\(", text):
            nodes.append(("function_definition", m.group(0)))
        for m in re.finditer(r"class\s+([A-Za-z_]\w*)\s*", text):
            nodes.append(("class_definition", m.group(0)))
    else:
        for m in re.finditer(r"function\s+\w*\s*\(", text):
            nodes.append(("function_declaration", m.group(0)))
    return nodes


def embed(out: EncodeOut, vocab: Vocab) -> np.ndarray:
    dim = vocab.dim + vocab.ast_dim
    if not out.ids:
        return np.zeros((0, dim), dtype=np.float32)
    mat = vocab.embeddings[np.asarray(out.ids, dtype=np.int64)]
    return mat.astype(np.float32)


def decode(ids, vocab: Vocab) -> str:
    """Decode token IDs back to text, filtering padding placeholder tokens."""
    out = []
    for i in ids:
        if not (0 <= i < len(vocab.tokens)):
            continue
        tok = vocab.tokens[i]
        # Skip padding placeholders (\\x01NNN) that aren't real tokens.
        if tok.startswith("\x01"):
            continue
        out.append(tok)
    return "".join(out)
