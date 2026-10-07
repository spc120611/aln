# -*- coding: utf-8 -*-
"""独立解析并校验导出的 GGUF 文件（不依赖第三方库）"""
import struct, sys, os
import numpy as np

TYPES = {0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32", 5: "INT32",
         6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY", 10: "UINT64",
         11: "INT64", 12: "FLOAT64"}
DTYPES = {0: ("F32", 4, np.float32), 1: ("F16", 2, np.float16), 8: ("Q8_0", 34, None),
          2: ("Q4_0", 18, None), 3: ("Q4_1", 20, None)}
RS = {0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32",
      6: "float32", 7: "bool", 10: "uint64", 11: "int64", 12: "float64"}


class R:
    def __init__(self, b): self.b, self.i = b, 0

    def u8(self): v = self.b[self.i]; self.i += 1; return v

    def u32(self): v = struct.unpack_from("<I", self.b, self.i)[0]; self.i += 4; return v

    def u64(self): v = struct.unpack_from("<Q", self.b, self.i)[0]; self.i += 8; return v

    def raw(self, n): v = self.b[self.i:self.i + n]; self.i += n; return v

    def s(self):
        n = self.u64(); v = self.raw(n).decode("utf-8"); return v

    def val(self, t):
        if t == 8: return self.s()
        if t == 7: return bool(self.u8())
        if t in (0, 1, 2, 3, 4, 5, 6, 10, 11, 12):
            cf = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
                  5: ("i", 4), 6: ("f", 4), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}[t]
            v = struct.unpack_from("<" + cf[0], self.b, self.i)[0]
            self.i += cf[1]
            return v
        if t == 9:
            et = self.u32(); n = self.u64()
            return (TYPES.get(et), [self.val(et) for _ in range(n)])
        raise ValueError("type %d" % t)


def parse(path):
    b = open(path, "rb").read()
    r = R(b)
    magic, ver = r.u32(), r.u32()
    assert magic == 0x46554747, hex(magic)
    ntensor, nkv = r.u64(), r.u64()
    kv = {}
    for _ in range(nkv):
        k = r.s()
        t = r.u32()
        kv[k] = (TYPES.get(t), r.val(t))
    infos = []
    for _ in range(ntensor):
        name = r.s()
        nd = r.u32()
        shape = [r.u64() for _ in range(nd)]
        dt = r.u32()
        off = r.u64()
        infos.append((name, tuple(shape), dt, off))
    pad = (-r.i) % 32
    r.i += pad
    start = r.i
    return dict(magic=hex(magic), version=ver, kv_count=nkv, kv=kv,
                tensors=infos, data_start=start, blob=b, size=len(b))


def read_tensor(info, g):
    name, shape, dt, off = info
    b = g["blob"]
    n = int(np.prod(shape))
    if dt == 0:
        np_shape = tuple(reversed(shape))       # GGUF dims 列优先 -> numpy 行优先
        return np.frombuffer(b, np.float32, n, g["data_start"] + off).reshape(np_shape)
    if dt == 8:
        assert n % 32 == 0
        raw = b[g["data_start"] + off: g["data_start"] + off + 34 * (n // 32)]
        out = np.empty(n, np.float32)
        for i in range(n // 32):
            d = struct.unpack_from("<e", raw, i * 34)[0]
            q = np.frombuffer(raw, np.int8, 32, i * 34 + 2)
            out[i * 32:(i + 1) * 32] = q * d
        return out.reshape(tuple(reversed(shape)))
    raise ValueError(dt)


if __name__ == "__main__":
    path = sys.argv[1]
    g = parse(path)
    print("=== %s ===" % path)
    print("magic %s version %d  kv=%d  tensors=%d  size %.3f MB" %
          (g["magic"], g["version"], g["kv_count"], len(g["tensors"]), g["size"] / 1e6))
    for k in ["general.architecture", "llama.block_count", "llama.context_length",
              "llama.embedding_length", "llama.feed_forward_length",
              "llama.attention.head_count", "llama.attention.head_count_kv",
              "llama.rope.dimension_count", "llama.vocab_size", "general.file_type",
              "llama.attention.layer_norm_rms_epsilon", "llama.rope.freq_base"]:
        v = g["kv"].get(k)
        print("  %-42s %s" % (k, (v[1] if v else "MISSING")))
    def arr(k):
        v = g["kv"].get(k)
        return v[1][1] if v else None          # (ARRAY_TYPE, (ELEM_TYPE, [items]))
    tk = arr("tokenizer.ggml.tokens")
    mg = arr("tokenizer.ggml.merges")
    sc = arr("tokenizer.ggml.scores")
    tt = arr("tokenizer.ggml.token_type")
    print("  tokens n=%d | merges n=%d | scores n=%d | token_type n=%d" %
          (len(tk or []), len(mg or []), len(sc or []), len(tt or [])))
    print("  tokens[0:3]=%r  tokens[-2:]=%r" % ((tk or [])[:3], (tk or [])[-2:]))
    print("  merges[0]=%r merges[-1]=%r | 合并对格式校验: %s" %
          ((mg or [""])[0], (mg or [""])[-1],
           all(len(m.split(" ")) == 2 for m in (mg or ["a b"]))))
    # llama.cpp 必需字段核对
    need = ["general.architecture", "llama.block_count", "llama.context_length",
            "llama.embedding_length", "llama.feed_forward_length",
            "llama.attention.head_count", "llama.attention.head_count_kv",
            "llama.rope.dimension_count", "llama.vocab_size",
            "llama.attention.layer_norm_rms_epsilon", "llama.rope.freq_base",
            "tokenizer.ggml.model", "tokenizer.ggml.tokens", "tokenizer.ggml.merges"]
    miss = [k for k in need if k not in g["kv"]]
    print("  llama.cpp 必需字段缺失:", miss or "无")
    print("  general.quantization_version:", g["kv"].get("general.quantization_version", "缺失(可选,默认2)"))
    print("  output.weight 存在?", any(t[0] == "output.weight" for t in g["tensors"]),
          "(tied embedding: llama.cpp 会自动回退到 token_embd)")

    # 与训练权重比对
    z = np.load("ckpt_cn.npz")
    NL, DIM, HID, NKV, HD = 8, 192, 224, 4, 24
    import json as _j
    _nv = len(_j.load(open("tokenizer_cn.json"))["vocab"])
    ref = {"token_embd.weight": z["tok_emb"][:_nv], "output_norm.weight": z["rms_final"]}
    for l in range(NL):
        ref[f"blk.{l}.attn_norm.weight"] = z["rms_att"][l]
        ref[f"blk.{l}.attn_q.weight"] = z["wqkv"][l][:DIM]
        ref[f"blk.{l}.attn_k.weight"] = z["wqkv"][l][DIM:DIM + NKV * HD]
        ref[f"blk.{l}.attn_v.weight"] = z["wqkv"][l][DIM + NKV * HD:]
        ref[f"blk.{l}.attn_output.weight"] = z["wo"][l]
        ref[f"blk.{l}.ffn_norm.weight"] = z["rms_ffn"][l]
        ref[f"blk.{l}.ffn_gate.weight"] = z["w13"][l][:HID]
        ref[f"blk.{l}.ffn_up.weight"] = z["w13"][l][HID:]
        ref[f"blk.{l}.ffn_down.weight"] = z["w2"][l]
    print("  ref tensors: %d" % len(ref))
    missing = set(ref) - {t[0] for t in g["tensors"]}
    extra = {t[0] for t in g["tensors"]} - set(ref)
    print("  missing:", missing or "none", "| extra:", extra or "none")
    maxerr, worst = 0.0, None
    nd = {}
    for name, shape, dt, off in g["tensors"]:
        nd.setdefault(DTYPES[dt][0], 0)
        nd[DTYPES[dt][0]] += 1
    print("  dtype 统计:", nd)
    for name, shape, dt, off in g["tensors"]:
        if name not in ref:
            continue
        a = read_tensor((name, shape, dt, off), g)
        if a.shape != ref[name].shape:
            print("  SHAPE MISMATCH", name, a.shape, ref[name].shape)
            continue
        e = float(np.abs(a - ref[name]).max())
        rel = float((np.abs(a - ref[name]) / (np.abs(ref[name]) + 1e-6)).mean())
        if rel > maxerr:
            maxerr, worst = rel, (name, e, rel)
    print("  max mean-rel-error vs fp32: %.5f  worst: %s" % (maxerr, worst))
