# -*- coding: utf-8 -*-
"""标准 Llama 架构的可配置纯 numpy 实现（前向 + 手写反向 + Adam）
组件: RMSNorm / RoPE / GQA / SwiGLU / tied embedding，与 Llama 2/3 一致
为速度考虑，qkv 与 gate/up 投影做了融合（导出时再拆开）
"""
import numpy as np

CFG = dict(dim=192, n_layers=8, n_heads=8, n_kv_heads=4, ffn=224,
           vocab=3072, seq=512, eps=1e-5, rope_theta=10000.0)


def setup(cfg=None):
    global DIM, NL, NH, NKV, HD, HID, VOCAB, SEQ, EPS, THETA, KVD, GROUP, QKV_DIM
    if cfg:
        CFG.update(cfg)
    DIM = CFG["dim"]; NL = CFG["n_layers"]; NH = CFG["n_heads"]; NKV = CFG["n_kv_heads"]
    HID = CFG["ffn"]; VOCAB = CFG["vocab"]; SEQ = CFG["seq"]; EPS = CFG["eps"]
    THETA = CFG["rope_theta"]
    HD = DIM // NH
    KVD = NKV * HD
    GROUP = NH // NKV
    QKV_DIM = DIM + 2 * KVD


setup()


def init_params(seed=42, scale=0.02):
    rng = np.random.default_rng(seed)
    s = scale
    rs = scale / np.sqrt(2 * NL)
    return {
        "tok_emb": (rng.standard_normal((VOCAB, DIM)) * s).astype(np.float32),
        "rms_att": np.ones((NL, DIM), dtype=np.float32),
        "rms_ffn": np.ones((NL, DIM), dtype=np.float32),
        "rms_final": np.ones(DIM, dtype=np.float32),
        "wqkv": (rng.standard_normal((NL, QKV_DIM, DIM)) * s).astype(np.float32),
        "wo": (rng.standard_normal((NL, DIM, DIM)) * rs).astype(np.float32),
        "w13": (rng.standard_normal((NL, 2 * HID, DIM)) * s).astype(np.float32),
        "w2": (rng.standard_normal((NL, DIM, HID)) * rs).astype(np.float32),
    }


def rope_tables(T):
    pos = np.arange(T, dtype=np.float32)
    inv = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
    ang = np.outer(pos, inv)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope_fwd(x, cos, sin):
    x2 = x.reshape(*x.shape[:-1], HD // 2, 2)
    x0, x1 = x2[..., 0], x2[..., 1]
    c, s = cos[None, None, :, :], sin[None, None, :, :]
    return np.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).reshape(x.shape).astype(np.float32)


def rope_bwd(dy, cos, sin):
    d2 = dy.reshape(*dy.shape[:-1], HD // 2, 2)
    d0, d1 = d2[..., 0], d2[..., 1]
    c, s = cos[None, None, :, :], sin[None, None, :, :]
    return np.stack([d0 * c + d1 * s, -d0 * s + d1 * c], -1).reshape(dy.shape).astype(np.float32)


def rms_fwd(x, w):
    m = np.mean(x * x, axis=-1, keepdims=True)
    sc = (1.0 / np.sqrt(m + EPS)).astype(np.float32)
    xn = (x * sc).astype(np.float32)
    return (xn * w).astype(np.float32), (x, sc, w, xn, m)


def rms_bwd(dy, cache):
    x, sc, w, xn, m = cache
    dxn = dy * w
    dw = np.sum(dy * xn, axis=tuple(range(dy.ndim - 1)))
    dm = np.sum(dxn * x, axis=-1, keepdims=True) * (-0.5) * (m + EPS) ** (-1.5)
    dx = dxn * sc + dm * (2.0 * x / x.shape[-1])
    return dx.astype(np.float32), dw.astype(np.float32)


def forward(p, idx, targets=None):
    B, T = idx.shape
    cos, sin = rope_tables(T)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    x = p["tok_emb"][idx].astype(np.float32)
    caches = []
    for l in range(NL):
        h, c1 = rms_fwd(x, p["rms_att"][l])
        qkv = h @ p["wqkv"][l].T                       # (B,T,QKV_DIM)
        q = qkv[..., :DIM].reshape(B, T, NH, HD).transpose(0, 2, 1, 3)
        k = qkv[..., DIM:DIM + KVD].reshape(B, T, NKV, HD).transpose(0, 2, 1, 3)
        v = qkv[..., DIM + KVD:].reshape(B, T, NKV, HD).transpose(0, 2, 1, 3)
        q = rope_fwd(q, cos, sin)
        k = rope_fwd(k, cos, sin)
        kk = np.repeat(k, GROUP, axis=1)
        vv = np.repeat(v, GROUP, axis=1)
        att = (q @ kk.transpose(0, 1, 3, 2)) * np.float32(1.0 / np.sqrt(HD)) + mask
        att = att - att.max(axis=-1, keepdims=True)
        e = np.exp(att)
        att = (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)
        out = (att @ vv).transpose(0, 2, 1, 3).reshape(B, T, DIM)
        x = x + out @ p["wo"][l].T
        h2, c2 = rms_fwd(x, p["rms_ffn"][l])
        g = h2 @ p["w13"][l].T                         # (B,T,2H)
        u = g[..., :HID]
        v3 = g[..., HID:]
        sig = (1.0 / (1.0 + np.exp(-u))).astype(np.float32)
        sil = (u * sig).astype(np.float32)
        hh = (sil * v3).astype(np.float32)
        x = x + hh @ p["w2"][l].T
        caches.append(dict(c1=c1, h=h, qkv=qkv, q=q, k=k, kk=kk, vv=vv, att=att,
                           out=out, c2=c2, h2=h2, u=u, v3=v3, sig=sig, sil=sil, hh=hh))
    xf, cfin = rms_fwd(x, p["rms_final"])
    logits = (xf @ p["tok_emb"].T).astype(np.float32)
    bundle = dict(caches=caches, cfin=cfin, xf=xf, cos=cos, sin=sin, B=B, T=T, idx=idx)
    loss = None
    if targets is not None:
        lg = logits - logits.max(axis=-1, keepdims=True)
        ex = np.exp(lg)
        probs = (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float32)
        ab, at = np.arange(B)[:, None], np.arange(T)
        loss = float(-np.mean(np.log(probs[ab, at, targets] + 1e-9)))
        d = probs.copy()
        d[ab, at, targets] -= 1.0
        bundle["dlogits"] = (d / (B * T)).astype(np.float32)
    return logits, loss, bundle


def backward(p, bundle):
    cs = bundle["caches"]
    B, T = bundle["B"], bundle["T"]
    cos, sin = bundle["cos"], bundle["sin"]
    g = {k: np.zeros_like(v) for k, v in p.items()}
    d = bundle["dlogits"]
    xf = bundle["xf"]
    g["tok_emb"] += d.reshape(-1, VOCAB).T @ xf.reshape(-1, DIM)
    dx = d @ p["tok_emb"]
    dx, g["rms_final"] = rms_bwd(dx.astype(np.float32), bundle["cfin"])
    for l in range(NL - 1, -1, -1):
        c = cs[l]
        # FFN
        g["w2"][l] += dx.reshape(-1, DIM).T @ c["hh"].reshape(-1, HID)
        dh = dx @ p["w2"][l]
        dv3 = dh * c["sil"]
        dsil = dh * c["v3"]
        du = dsil * (c["sig"] + c["u"] * c["sig"] * (1 - c["sig"]))
        dg = np.concatenate([du, dv3], axis=-1)                # (B,T,2H)
        g["w13"][l] += dg.reshape(-1, 2 * HID).T @ c["h2"].reshape(-1, DIM)
        dh2 = dg @ p["w13"][l]
        dx2, g["rms_ffn"][l] = rms_bwd(dh2.astype(np.float32), c["c2"])
        dx = dx + dx2
        # Attention
        g["wo"][l] += dx.reshape(-1, DIM).T @ c["out"].reshape(-1, DIM)
        dout = (dx @ p["wo"][l]).reshape(B, T, NH, HD).transpose(0, 2, 1, 3)
        dvv = c["att"].transpose(0, 1, 3, 2) @ dout
        datt = dout @ c["vv"].transpose(0, 1, 3, 2)
        ds = c["att"] * (datt - np.sum(datt * c["att"], axis=-1, keepdims=True))
        dq = (ds @ c["kk"]) * np.float32(1.0 / np.sqrt(HD))
        dkk = (ds.transpose(0, 1, 3, 2) @ c["q"]) * np.float32(1.0 / np.sqrt(HD))
        dq = rope_bwd(dq.astype(np.float32), cos, sin)
        dkk = rope_bwd(dkk.astype(np.float32), cos, sin)
        dk = dkk.reshape(B, NKV, GROUP, T, HD).sum(axis=2)
        dv = dvv.reshape(B, NKV, GROUP, T, HD).sum(axis=2)
        dqkv = np.concatenate([dq.transpose(0, 2, 1, 3).reshape(B, T, DIM),
                               dk.transpose(0, 2, 1, 3).reshape(B, T, KVD),
                               dv.transpose(0, 2, 1, 3).reshape(B, T, KVD)], axis=-1)
        g["wqkv"][l] += dqkv.reshape(-1, QKV_DIM).T @ c["h"].reshape(-1, DIM)
        dh_in = dqkv @ p["wqkv"][l]
        dx1, g["rms_att"][l] = rms_bwd(dh_in.astype(np.float32), c["c1"])
        dx = dx + dx1
    return g


class Adam:
    def __init__(self, p, lr=2e-3, b1=0.9, b2=0.95, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0

    def step(self, p, g, clip=1.0):
        self.t += 1
        tot = float(np.sqrt(sum(float(np.sum(x * x)) for x in g.values())))
        sc = clip / (tot + 1e-8) if tot > clip else 1.0
        for k in p:
            gk = g[k] * sc
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p[k] = (p[k] - self.lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)
        return tot
