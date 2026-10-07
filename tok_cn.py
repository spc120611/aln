# -*- coding: utf-8 -*-
"""中文 byte-level BPE tokenizer（零依赖，与训练完全一致）"""
import json, os, re
try:
    import regex as _re_mod
    # 与 llama.cpp LLAMA_VOCAB_PRE_TYPE_GPT2 完全一致的分词正则
    PAT = _re_mod.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
    EXACT = True
except ImportError:
    # 降级：Python 内置 re 不支持 \p{L}，用中日韩+拉丁范围近似。
    # 分词结果会与 llama.cpp 不一致，请执行 pip install regex
    PAT = re.compile(
        "[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7afA-Za-z]+|[0-9]+"
        "|[^\s\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7afA-Za-z0-9]+|\s+")
    EXACT = False
    import warnings
    warnings.warn("未安装 regex，分词已退化为近似模式，结果与 llama.cpp 不一致；"
                  "请执行 pip install regex", RuntimeWarning)
_d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_cn.json")))
VOCAB_STR = {int(k): v for k, v in _d["vocab"].items()}
VOCAB = len(VOCAB_STR)
TOK2ID = {v: k for k, v in VOCAB_STR.items()}
rank, mid = {}, {}
for r, (a, b) in enumerate(_d["merges"]):
    if a in TOK2ID and b in TOK2ID:
        rank[(TOK2ID[a], TOK2ID[b])] = r
        mid[(TOK2ID[a], TOK2ID[b])] = TOK2ID[a + b]


def _bpe(ids):
    ids = list(ids)
    while len(ids) > 1:
        br, bi = 1 << 30, -1
        for i in range(len(ids) - 1):
            r = rank.get((ids[i], ids[i + 1]))
            if r is not None and r < br:
                br, bi = r, i
        if bi < 0:
            break
        ids[bi:bi + 2] = [mid[(ids[bi], ids[bi + 1])]]
    return ids


def encode(text):
    out = []
    for c in PAT.findall(text):
        out.extend(_bpe(tuple(c.encode("utf-8"))))
    return out


def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


U2B = {ord(v): k for k, v in _bytes_to_unicode().items()}


def decode(ids):
    out = bytearray()
    for i in ids:
        for ch in VOCAB_STR.get(int(i), ""):
            out.append(U2B.get(ord(ch), 63))
    return out.decode("utf-8", errors="replace")
