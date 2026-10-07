# -*- coding: utf-8 -*-
"""中文 Llama 训练（分段可续跑）
用法: STEPS=300 TOTAL=2400 timeout 290 python train_cn.py
"""
import os, time, math, json
import numpy as np
import llama_cn as M
import tok_cn

STEPS = int(os.environ.get("STEPS", 300))
TOTAL = int(os.environ.get("TOTAL", 2400))
B = int(os.environ.get("B", 24))
T = int(os.environ.get("T", 96))
LR_MAX = float(os.environ.get("LR", 3e-3))
LR_MIN, WARMUP, CLIP = 3e-4, 150, 1.0

data = np.load("tokens_cn.npy")
split = int(len(data) * 0.97)
train, val = data[:split], data[split:]
print(f"params {sum(a.size for a in M.init_params(0).values()):,} | "
      f"tokens {len(data):,} (train {len(train):,} val {len(val):,})", flush=True)

p = M.init_params(42)
opt = M.Adam(p, lr=LR_MAX)
step0 = 0
if os.path.exists("ckpt_cn.npz"):
    z = np.load("ckpt_cn.npz")
    for k in p:
        p[k] = z[k]
    step0 = int(z["__step"])
if os.path.exists("opt_cn.npz"):
    z = np.load("opt_cn.npz")
    for k in opt.m:
        opt.m[k] = z["m_" + k]
        opt.v[k] = z["v_" + k]
    opt.t = int(z["__t"])
    step0 = int(z["__step"])
print(f"resume {step0} + {STEPS} -> {step0 + STEPS}/{TOTAL}", flush=True)
rng = np.random.default_rng(1234 + step0)


def batch(split_data):
    ix = rng.integers(0, len(split_data) - T - 1, B)
    return (np.stack([split_data[i:i + T] for i in ix]).astype(np.int64),
            np.stack([split_data[i + 1:i + T + 1] for i in ix]).astype(np.int64))


def evaluate(n=8):
    ls = []
    for _ in range(n):
        x, y = batch(val)
        _, l, _ = M.forward(p, x, y)
        ls.append(l)
    return float(np.mean(ls))


def sample(prompt="老师：早上好\n阿罗娜：", n=120, temp=0.85, seed=0):
    r = np.random.default_rng(seed)
    ids = tok_cn.encode(prompt)
    for _ in range(n):
        lg, _, _ = M.forward(p, np.array([ids[-M.SEQ:]], dtype=np.int64))
        pr = lg[0, -1].astype(np.float64)
        pr = np.exp(pr - pr.max())
        pr = pr ** (1.0 / temp)
        pr /= pr.sum()
        ids.append(int(r.choice(len(pr), p=pr)))
    return tok_cn.decode(ids)


t0 = time.time()
for i in range(STEPS):
    step = step0 + i
    lr = LR_MAX if step < WARMUP else LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, (step - WARMUP) / max(1, TOTAL - WARMUP))))
    opt.lr = lr
    x, y = batch(train)
    _, loss, bundle = M.forward(p, x, y)
    g = M.backward(p, bundle)
    gn = opt.step(p, g)
    if i % 25 == 0 or i == STEPS - 1:
        el = time.time() - t0
        print(f"step {step:5d}/{TOTAL} train {loss:.4f} lr {lr:.2e} gn {gn:.2f} "
              f"{el:.0f}s eta {el / (i + 1) * (STEPS - i - 1):.0f}s", flush=True)
    if i % 150 == 0 or i == STEPS - 1:
        v = evaluate()
        print(f"     >> val loss {v:.4f} ppl {math.exp(min(20, v)):.2f}", flush=True)

d = {k: v for k, v in p.items()}
d["__step"] = np.array(step0 + STEPS)
np.savez("ckpt_cn.npz", **d)
od = {"__t": np.array(opt.t), "__step": np.array(step0 + STEPS)}
for k in opt.m:
    od["m_" + k] = opt.m[k]
    od["v_" + k] = opt.v[k]
np.savez("opt_cn.npz", **od)
for pr_, sd in [("老师：早上好\n阿罗娜：", 0), ("老师：你是谁\n阿罗娜：", 1), ("老师：我有点累了\n阿罗娜：", 2)]:
    print("---\n" + sample(pr_, 90, 0.8, sd)[:300], flush=True)
print("saved", step0 + STEPS, "%.0fs" % (time.time() - t0), flush=True)
