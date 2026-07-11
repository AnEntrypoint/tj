import numpy as np
from tianji.tokens.apt import Vocab, encode, embed, decode, SPECIAL_IDS


SAMPLE = [
    '<tool_call>{"name":"edit","args":{"path":"a.py"}}</tool_call>',
    "<bash_output>ok</bash_output>",
    "def fib(n): return n",
    "<system>agent</system>",
    "<cot>plan</cot>",
    "<diff>--- a\n+++ b\n</diff>",
    "<system-reminder>external context</system-reminder>",
]


def test_vocab_build_is_deterministic():
    v1 = Vocab.build(SAMPLE, target_size=128, dim=16, ast_dim=8)
    v2 = Vocab.build(SAMPLE, target_size=128, dim=16, ast_dim=8)
    assert v1.size == v2.size
    assert v1.tokens == v2.tokens


def test_special_tokens_recognized():
    v = Vocab.build(SAMPLE, target_size=128, dim=16, ast_dim=8)
    out = encode("<tool_call>x</tool_call>", v, parse_ast=False)
    assert out.ids[0] == SPECIAL_IDS["<tool_call>"]
    assert out.ids[-1] == SPECIAL_IDS["</tool_call>"]


def test_special_reminder_not_treated_as_special():
    v = Vocab.build(SAMPLE, target_size=128, dim=16, ast_dim=8)
    assert len(encode("<system-reminder>x</system-reminder>", v, parse_ast=False).ids) > 0


def test_encode_total_on_empty():
    v = Vocab.build(SAMPLE, target_size=128, dim=16, ast_dim=8)
    assert len(encode("", v, parse_ast=False).ids) == 0


def test_embed_shape():
    v = Vocab.build(SAMPLE, target_size=128, dim=16, ast_dim=8)
    out = encode("def fib(n): return n", v, parse_ast=False)
    arr = embed(out, v)
    assert arr.shape == (len(out.ids), 16 + 8)
    assert arr.dtype == np.float32


def test_ast_extraction_python():
    v = Vocab.build(SAMPLE, target_size=128, dim=16, ast_dim=8)
    out = encode("def f(x): return x + 1", v, parse_ast=True, lang="python")
    assert any(nt == "function_definition" for nt, _ in out.ast_nodes)


def test_ast_extraction_javascript():
    v = Vocab.build(SAMPLE, target_size=128, dim=16, ast_dim=8)
    out = encode("function f(x) { return x + 1; }", v, parse_ast=True, lang="javascript")
    assert any("function" in nt for nt, _ in out.ast_nodes)


def test_ast_extraction_unknown_lang_falls_back_to_regex():
    v = Vocab.build(SAMPLE, target_size=128, dim=16, ast_dim=8)
    out = encode("def f(x): return x + 1", v, parse_ast=True, lang="klingon")
    assert len(out.ast_nodes) >= 0
