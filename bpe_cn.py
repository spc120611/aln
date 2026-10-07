# -*- coding: utf-8 -*-
"""中文 byte-level BPE 词表训练（GPT-2 / Llama 同款的 byte-to-unicode 映射）
输出: tokenizer_cn.json (vocab + merges)、tokens_cn.npy (语料编码)
"""
import json, re, os, time
import regex
from collections import Counter
import numpy as np

VOCAB_SIZE = 3072
CORPUS = "corpus_arona.txt"
SUB = 1_500_000          # BPE 训练用子样本字符数

# ---- 预切分：与 llama.cpp LLAMA_VOCAB_PRE_TYPE_GPT2 完全一致 ----
# src/llama-vocab.cpp: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
GPT2_PAT = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
try:
    PAT = regex.compile(GPT2_PAT)
    _IS_REGEX = True
except Exception:
    PAT = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+|[0-9]{1,3}|[^\s\u4e00-\u9fffA-Za-z0-9]+|\s+")
    _IS_REGEX = False
print("预切分正则: %s" % ("llama.cpp GPT2 完全一致 (regex \\p{L})" if _IS_REGEX else "退化 re"))


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


B2U = bytes_to_unicode()


def bpe_str(b: bytes) -> str:
    return "".join(B2U[x] for x in b)


t0 = time.time()
text = open(CORPUS, encoding="utf-8").read()
print("corpus chars:", len(text))
sub = text[:SUB]
chunks = PAT.findall(sub)
freq = Counter(chunks)
print("train chunks:", len(chunks), "| unique:", len(freq))

words = {tuple(c.encode("utf-8")): f for c, f in freq.items()}   # 以字节 tuple 为初始符号
pair_words = {}
pairs = Counter()
for w, f in words.items():
    for i in range(len(w) - 1):
        p = (w[i], w[i + 1])
        pairs[p] += f
        pair_words.setdefault(p, set()).add(w)

vocab = {i: (i,) for i in range(256)}      # id -> byte tuple
merges = []                                # (a_id, b_id, new_id)
nxt = 256
while len(vocab) < VOCAB_SIZE and pairs:
    best = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
    new_id = nxt
    nxt += 1
    vocab[new_id] = vocab[best[0]] + vocab[best[1]]
    merges.append((best[0], best[1], new_id))
    affected = pair_words.pop(best, set())
    for w in affected:
        f = words.pop(w)
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and w[i] == best[0] and w[i + 1] == best[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(w[i])
                i += 1
        nw = tuple(out)
        words[nw] = words.get(nw, 0) + f
        for j in range(len(nw) - 1):
            p = (nw[j], nw[j + 1])
            pairs[p] = pairs.get(p, 0) + f
            pair_words.setdefault(p, set()).add(nw)
        for j in range(len(w) - 1):
            p = (w[j], w[j + 1])
            if p in pairs:
                pairs[p] -= f
                if pairs[p] <= 0:
                    del pairs[p]
                if p in pair_words:
                    pair_words[p].discard(w)
print("vocab:", len(vocab), "| merges:", len(merges), "| %.0fs" % (time.time() - t0))

VOCAB_STR = {i: bpe_str(bytes(vocab[i])) for i in range(len(vocab))}
json.dump({"vocab": {str(k): v for k, v in VOCAB_STR.items()},
           "merges": [[VOCAB_STR[a], VOCAB_STR[b]] for a, b, _ in merges]},
          open("tokenizer_cn.json", "w"), ensure_ascii=False)

# ---- 编码全语料 ----
merge_id = {}
rank = {}
for r, (a, b, nid) in enumerate(merges):
    merge_id[(a, b)] = nid
    rank[(a, b)] = r


def bpe_chunk(ids):
    ids = list(ids)
    while len(ids) > 1:
        br, bi = 1 << 30, -1
        for i in range(len(ids) - 1):
            r = rank.get((ids[i], ids[i + 1]))
            if r is not None and r < br:
                br, bi = r, i
        if bi < 0:
            break
        ids[bi:bi + 2] = [merge_id[(ids[bi], ids[bi + 1])]]
    return ids


cache = {}
ids = []
t1 = time.time()
for c in PAT.findall(text):
    r = cache.get(c)
    if r is None:
        r = cache[c] = bpe_chunk(tuple(c.encode("utf-8")))
    ids.extend(r)
    if len(ids) % 500000 < 1:
        print("  encoded", len(ids), "%.0fs" % (time.time() - t1), flush=True)
ids = np.array(ids, dtype=np.int32)
np.save("tokens_cn.npy", ids)
print("tokens:", len(ids), "| chars/token: %.2f" % (len(text.encode()) / len(ids)),
      "| total %.0fs" % (time.time() - t0))

# ---- 抽样查看分词效果 ----
sample = "今天天气很好，我和小明一起去公园玩。"
toks = []
for c in PAT.findall(sample):
    toks.extend(bpe_chunk(tuple(c.encode("utf-8"))))
print("sample:", sample)
print("pieces:", [VOCAB_STR[i] for i in toks])
