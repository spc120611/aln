# -*- coding: utf-8 -*-
"""从 GGUF 直接加载并推理（纯 numpy，不需要 llama.cpp / torch）
用法:
    python generate_cn.py --gguf llama-cn-2m5-f32.gguf --prompt "有一天" --steps 120
    python generate_cn.py --gguf llama-cn-2m5-q8_0.gguf --prompt "你好，" --temp 0.7
"""
import argparse, os, sys
import numpy as np
from verify_gguf import parse, read_tensor
import tok_cn

DIR = os.path.dirname(os.path.abspath(__file__))


def load_gguf(path):
    g = parse(path)
    kv = g["kv"]
    C = dict(
        dim=int(kv["llama.embedding_length"][1]),
        nl=int(kv["llama.block_count"][1]),
        nh=int(kv["llama.attention.head_count"][1]),
        nkv=int(kv["llama.attention.head_count_kv"][1]),
        hid=int(kv["llama.feed_forward_length"][1]),
        seq=int(kv["llama.context_length"][1]),
        eps=float(kv["llama.attention.layer_norm_rms_epsilon"][1]),
        theta=float(kv["llama.rope.freq_base"][1]),
    )
    C["hd"] = C["dim"] // C["nh"]
    C["group"] = C["nh"] // C["nkv"]
    T = {}
    for name, shape, dt, off in g["tensors"]:
        a = read_tensor((name, shape, dt, off), g)   # read_tensor 已按 reversed(dims) 还原
        T[name] = np.ascontiguousarray(a).astype(np.float32)

    p = {"tok_emb": T["token_embd.weight"], "rms_final": T["output_norm.weight"]}
    for l in range(C["nl"]):
        p.setdefault("rms_att", []).append(T[f"blk.{l}.attn_norm.weight"])
        p.setdefault("wq", []).append(T[f"blk.{l}.attn_q.weight"])
        p.setdefault("wk", []).append(T[f"blk.{l}.attn_k.weight"])
        p.setdefault("wv", []).append(T[f"blk.{l}.attn_v.weight"])
        p.setdefault("wo", []).append(T[f"blk.{l}.attn_output.weight"])
        p.setdefault("rms_ffn", []).append(T[f"blk.{l}.ffn_norm.weight"])
        p.setdefault("w1", []).append(T[f"blk.{l}.ffn_gate.weight"])
        p.setdefault("w2", []).append(T[f"blk.{l}.ffn_down.weight"])
        p.setdefault("w3", []).append(T[f"blk.{l}.ffn_up.weight"])
    for k in ["rms_att", "wq", "wk", "wv", "wo", "rms_ffn", "w1", "w2", "w3"]:
        p[k] = np.stack(p[k])
    return C, p


def rms(x, w, eps):
    return (x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps)) * w


def forward(C, p, ids):
    T = len(ids)
    D, NH, NKV, HD, G = C["dim"], C["nh"], C["nkv"], C["hd"], C["group"]
    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
    x = p["tok_emb"][ids].astype(np.float32)
    cos, sin = rope_tables(T, HD, C["theta"])
    for l in range(C["nl"]):
        h = rms(x, p["rms_att"][l], C["eps"])
        q = (h @ p["wq"][l].T).reshape(T, NH, HD).transpose(1, 0, 2)
        k = (h @ p["wk"][l].T).reshape(T, NKV, HD).transpose(1, 0, 2)
        v = (h @ p["wv"][l].T).reshape(T, NKV, HD).transpose(1, 0, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        kk, vv = np.repeat(k, G, 0), np.repeat(v, G, 0)
        att = (q @ kk.transpose(0, 2, 1)) / np.sqrt(HD) + mask
        e = np.exp(att - att.max(-1, keepdims=True))
        att = e / e.sum(-1, keepdims=True)
        x = x + (att @ vv).transpose(1, 0, 2).reshape(T, D) @ p["wo"][l].T
        h2 = rms(x, p["rms_ffn"][l], C["eps"])
        u, v3 = h2 @ p["w1"][l].T, h2 @ p["w3"][l].T
        x = x + ((u / (1 + np.exp(-u))) * v3) @ p["w2"][l].T
    return rms(x, p["rms_final"], C["eps"]) @ p["tok_emb"].T


def rope_tables(L, hd, theta):
    pos = np.arange(L, dtype=np.float32)
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, np.float32) / hd))
    ang = np.outer(pos, inv)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope(x, cos, sin):
    x2 = x.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x2[..., 0], x2[..., 1]
    c, s = cos[None, None, :, :], sin[None, None, :, :]
    return np.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).reshape(x.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.path.join(DIR, "llama-cn-2m5-f32.gguf"))
    ap.add_argument("--prompt", default="有一天")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    C, p = load_gguf(a.gguf)
    rng = np.random.default_rng(a.seed)
    ids = tok_cn.encode(a.prompt) or [0]
    for _ in range(a.steps):
        lg = forward(C, p, np.array(ids[-C["seq"]:], np.int64))[-1].astype(np.float64)
        pr = np.exp(lg - lg.max())
        pr = pr ** (1 / a.temp)
        pr /= pr.sum()
        if a.top_p < 1:
            o = np.argsort(-pr)
            keep = int(np.searchsorted(np.cumsum(pr[o]), a.top_p)) + 1
            m = np.zeros_like(pr)
            m[o[:keep]] = 1
            pr = pr * m
            pr /= pr.sum()
        ids.append(int(rng.choice(len(pr), p=pr)))
    print(tok_cn.decode(ids))
