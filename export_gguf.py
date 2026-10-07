# -*- coding: utf-8 -*-
"""导出 GGUF（llama.cpp 格式，v3）+ llama2.c v1 bin
用法: python export_gguf.py
"""
import json, os, struct, math
import numpy as np

OUT = os.environ.get("OUT_DIR", ".")
os.makedirs(OUT, exist_ok=True)
z = np.load("ckpt_cn.npz")
STEP = int(z["__step"])

# ---- 架构常量（需与 llama_cn.CFG 一致）----
DIM, NL, NH, NKV, HID, VOCAB, SEQ = 192, 8, 8, 4, 224, 3072, 512
HD = DIM // NH
KVD = NKV * HD
EPS, THETA = 1e-5, 10000.0

P = {"tok_emb": z["tok_emb"], "rms_final": z["rms_final"]}
for l in range(NL):
    P.setdefault("wq", []).append(z["wqkv"][l][:DIM])
    P.setdefault("wk", []).append(z["wqkv"][l][DIM:DIM + KVD])
    P.setdefault("wv", []).append(z["wqkv"][l][DIM + KVD:])
    P["wo"] = P.get("wo", []) + [z["wo"][l]]
    P.setdefault("w1", []).append(z["w13"][l][:HID])
    P.setdefault("w3", []).append(z["w13"][l][HID:])
    P.setdefault("w2", []).append(z["w2"][l])
    P.setdefault("rms_att", []).append(z["rms_att"][l])
    P.setdefault("rms_ffn", []).append(z["rms_ffn"][l])

def gt(a):   # 数据保持 numpy 行优先原样 (out, in)；GGUF dims 采用列优先 reversed 语义
    return np.ascontiguousarray(a, np.float32)


# tokenizer 实际大小可能与 CFG.vocab 不同（BPE 合并提前耗尽），以 tokenizer 为准并对齐 embedding
_tv = json.load(open("tokenizer_cn.json"))
VOCAB = len(_tv["vocab"])
P["tok_emb"] = P["tok_emb"][:VOCAB]
print("tokenizer vocab = %d, embedding 已对齐" % VOCAB)

tensors = {"token_embd.weight": gt(P["tok_emb"]),
           "output_norm.weight": gt(P["rms_final"])}
for l in range(NL):
    tensors[f"blk.{l}.attn_norm.weight"] = gt(P["rms_att"][l])
    tensors[f"blk.{l}.attn_q.weight"] = gt(P["wq"][l])
    tensors[f"blk.{l}.attn_k.weight"] = gt(P["wk"][l])
    tensors[f"blk.{l}.attn_v.weight"] = gt(P["wv"][l])
    tensors[f"blk.{l}.attn_output.weight"] = gt(P["wo"][l])
    tensors[f"blk.{l}.ffn_norm.weight"] = gt(P["rms_ffn"][l])
    tensors[f"blk.{l}.ffn_gate.weight"] = gt(P["w1"][l])
    tensors[f"blk.{l}.ffn_down.weight"] = gt(P["w2"][l])
    tensors[f"blk.{l}.ffn_up.weight"] = gt(P["w3"][l])
# 不写 output.weight -> llama.cpp 自动绑定 token_embd（tied embedding）

# ---- tokenizer ----
_t = json.load(open("tokenizer_cn.json"))
VOCAB_STR = {int(k): v for k, v in _t["vocab"].items()}
MERGES = _t["merges"]


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


U2B = {ord(v): k for k, v in bytes_to_unicode().items()}


def raw_bytes(s):
    return bytes(U2B.get(ord(ch), 63) for ch in s)


TOKENS = [VOCAB_STR[i] for i in range(VOCAB)]
MERGE_STR = [a + " " + b for a, b in MERGES]
SCORES = [1000.0 - i if i >= 256 else 0.0 for i in range(VOCAB)]
TOK_TYPES = [6 if i < 256 else 1 for i in range(VOCAB)]   # 6 = BYTE


# ================= GGUF 写出 =================
# （内联写出，删除未使用的类）


def write_gguf(path, tensors, tokens, scores, merges, tok_types, dtype_code, quantize=None,
               file_type=0):
    """dtype_code: 0 = F32, 8 = Q8_0；quantize: 数组 -> 字节（当 dtype_code!=0）"""
    # 1) metadata
    import io
    meta = io.BytesIO()

    def s(x, buf=meta):
        b = x.encode("utf-8")
        buf.write(struct.pack("<Q", len(b)) + b)

    kvcount = [0]

    def kv(key, vtype, value):
        kvcount[0] += 1
        s(key)
        meta.write(struct.pack("<I", vtype))
        if vtype == 8:  # STRING
            s(value)
        elif vtype == 4:
            meta.write(struct.pack("<I", value))
        elif vtype == 5:
            meta.write(struct.pack("<i", value))
        elif vtype == 6:
            meta.write(struct.pack("<f", value))
        elif vtype == 7:
            meta.write(struct.pack("<B", 1 if value else 0))
        elif vtype == 9:  # ARRAY
            etype, items = value
            meta.write(struct.pack("<I", etype))
            meta.write(struct.pack("<Q", len(items)))
            for it in items:
                if etype == 8:
                    s(it)
                elif etype == 6:
                    meta.write(struct.pack("<f", it))
                elif etype == 5:
                    meta.write(struct.pack("<i", it))

    kv("general.architecture", 8, "llama")
    kv("general.name", 8, "arona-2m5")
    kv("general.file_type", 4, file_type)
    kv("general.quantization_version", 4, 2)
    kv("llama.block_count", 4, NL)
    kv("llama.context_length", 4, SEQ)
    kv("llama.embedding_length", 4, DIM)
    kv("llama.feed_forward_length", 4, HID)
    kv("llama.attention.head_count", 4, NH)
    kv("llama.attention.head_count_kv", 4, NKV)
    kv("llama.attention.layer_norm_rms_epsilon", 6, EPS)
    kv("llama.rope.dimension_count", 4, HD)
    kv("llama.rope.freq_base", 6, THETA)
    kv("llama.vocab_size", 4, VOCAB)
    kv("tokenizer.ggml.model", 8, "gpt2")
    kv("tokenizer.ggml.tokens", 9, (8, tokens))
    kv("tokenizer.ggml.scores", 9, (6, scores))
    kv("tokenizer.ggml.merges", 9, (8, merges))
    kv("tokenizer.ggml.token_type", 9, (5, tok_types))
    kv("tokenizer.ggml.bos_token_id", 4, 0)
    kv("tokenizer.ggml.eos_token_id", 4, 0)
    kv("tokenizer.ggml.unknown_token_id", 4, 0)
    kv("tokenizer.ggml.padding_token_id", 4, 0)
    kv("tokenizer.ggml.add_bos_token", 7, False)
    kv("tokenizer.ggml.add_eos_token", 7, False)
    meta_bytes = meta.getvalue()

    # 2) tensor infos（需要先知道数据长度以对齐；offset 相对数据区起点）
    infos = io.BytesIO()
    blobs, offset = [], 0
    for name, arr in tensors.items():
        use_q = quantize is not None and (arr.ndim == 1 or arr.shape[-1] % 32 == 0)
        data = arr.tobytes(order="C") if not use_q else quantize(arr)
        pad = (-len(data)) % 32
        infos.write(struct.pack("<Q", len(name.encode())) + name.encode())
        infos.write(struct.pack("<I", len(arr.shape)))
        for d in reversed(arr.shape):      # GGUF dims 为列优先，须反转
            infos.write(struct.pack("<Q", d))
        infos.write(struct.pack("<I", dtype_code if use_q else 0))
        infos.write(struct.pack("<Q", offset))
        blobs.append(data + b"\0" * pad)
        offset += len(data) + pad
    infos_bytes = infos.getvalue()

    # 3) 组装
    head = io.BytesIO()
    head.write(struct.pack("<II", 0x46554747, 3))       # magic "GGUF" + version 3
    head.write(struct.pack("<QQ", len(tensors), kvcount[0]))
    head.write(meta_bytes)
    head.write(infos_bytes)
    pad_head = (-head.tell()) % 32
    head.write(b"\0" * pad_head)

    with open(path, "wb") as f:
        f.write(head.getvalue())
        for b in blobs:
            f.write(b)
    return os.path.getsize(path)


def quantize_q8_0(a):
    x = np.ascontiguousarray(a, np.float32).reshape(-1)
    assert x.size % 32 == 0, x.size
    nb = x.size // 32
    blocks = x.reshape(nb, 32)
    amax = np.abs(blocks).max(axis=1)
    d = (amax / 127.0).astype(np.float32)
    d[amax == 0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d[:, None] > 0, np.round(blocks / np.where(d[:, None] > 0, d[:, None], 1.0)), 0)
    q = np.clip(q, -127, 127).astype(np.int8)
    d16 = d.astype(np.float16)
    out = bytearray()
    for i in range(nb):                                  # 每块: f16 缩放因子 + 32 个 int8
        out += struct.pack("<e", d[i])
        out += q[i].tobytes()
    return bytes(out)


# ================= 主流程 =================
n_params = sum(int(np.prod(v.shape)) for v in tensors.values())
f32 = write_gguf(f"{OUT}/arona-2m5-f32.gguf", tensors, TOKENS, SCORES, MERGE_STR,
                 TOK_TYPES, 0, None, 0)
q8 = write_gguf(f"{OUT}/arona-2m5-q8_0.gguf", tensors, TOKENS, SCORES, MERGE_STR,
                TOK_TYPES, 8, quantize_q8_0, 7)

# ---- llama2.c v1 bin + tokenizer.bin（供 C 推理器）----
order = [P["tok_emb"], *(P["rms_att"][l] for l in range(NL)),
         *(P["wq"][l] for l in range(NL)), *(P["wk"][l] for l in range(NL)),
         *(P["wv"][l] for l in range(NL)), *(P["wo"][l] for l in range(NL)),
         *(P["rms_ffn"][l] for l in range(NL)), *(P["w1"][l] for l in range(NL)),
         *(P["w2"][l] for l in range(NL)), *(P["w3"][l] for l in range(NL)),
         P["rms_final"]]
with open(f"{OUT}/model_cn_llama2c.bin", "wb") as f:
    f.write(struct.pack("<ii", 0x616B3432, 1))
    f.write(struct.pack("<iiiiiii", DIM, HID, NL, NH, NKV, VOCAB, SEQ))
    f.write(struct.pack("<B", 1))
    for a in order:
        f.write(np.ascontiguousarray(a, np.float32).tobytes())

with open(f"{OUT}/tokenizer_cn_llama2c.bin", "wb") as f:
    f.write(struct.pack("<i", max(len(raw_bytes(t)) for t in TOKENS)))
    for i, t in enumerate(TOKENS):
        b = raw_bytes(t)
        f.write(struct.pack("<fi", SCORES[i], len(b)))
        f.write(b)

stats = dict(steps=STEP, params=n_params, dim=DIM, layers=NL, heads=NH, kv_heads=NKV,
             head_dim=HD, ffn=HID, vocab=VOCAB, seq=SEQ,
             gguf_f32_bytes=f32, gguf_f32_MB=round(f32 / 1e6, 3),
             gguf_q8_0_bytes=q8, gguf_q8_0_MB=round(q8 / 1e6, 3))
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
