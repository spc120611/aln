# -*- coding: utf-8 -*-
"""阿罗娜对话脚本 —— 直接从 GGUF 加载，纯 numpy，不需要 llama.cpp / torch

用法:
    python chat_arona.py                        # 进入交互对话
    python chat_arona.py --gguf arona-2m5-q8_0.gguf --temp 0.7
    python chat_arona.py --once "你是谁"         # 单轮

对话格式：输入即「老师：…」，模型续写「阿罗娜：…」；
遇到模型自己又生成「老师：」时会截断，避免自问自答。
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_cn import load_gguf, forward   # 复用 GGUF 加载与前向
import tok_cn

DIR = os.path.dirname(os.path.abspath(__file__))
STOP = ["老师：", "【系统】", "\n\n", "\n阿罗娜：", "阿罗娜：", "老师，"]


def _sample_from_logits(lg, temp, top_p, rng, top_k, min_p):
    """复刻 llama.cpp 采样链顺序: top-k -> top-p -> min-p -> temp -> dist"""
    lg = lg.astype(np.float64)
    pr = np.exp(lg - lg.max())
    pr /= pr.sum()
    # top-k
    if top_k > 0 and top_k < len(pr):
        cut = np.sort(pr)[-top_k]
        pr = np.where(pr >= cut, pr, 0.0)
    # top-p
    if top_p < 1.0:
        o = np.argsort(-pr)
        keep = int(np.searchsorted(np.cumsum(pr[o]), top_p)) + 1
        m = np.zeros_like(pr)
        m[o[:keep]] = 1
        pr = pr * m
    # min-p
    if min_p > 0:
        pmax = pr.max()
        pr = np.where(pr >= min_p * pmax, pr, 0.0)
    if pr.sum() <= 0:
        pr = np.where(pr > 0, pr, np.exp(lg - lg.max()))
    pr = pr / pr.sum()
    # temp（最后作用）
    if temp and temp > 0:
        pr = pr ** (1.0 / temp)
        pr /= pr.sum()
    return int(rng.choice(len(pr), p=pr))


def sample_once(C, p, ids, temp, top_p, rng, top_k=40, min_p=0.05):
    lg = forward(C, p, np.array(ids[-C["seq"]:], np.int64))[-1]
    return _sample_from_logits(lg, temp, top_p, rng, top_k, min_p)


def sample_once_banned(C, p, ids, temp, top_p, rng, top_k, min_p, banned):
    lg = forward(C, p, np.array(ids[-C["seq"]:], np.int64))[-1]
    lg = lg.astype(np.float64).copy()
    for b in banned:
        lg[b] = -1e9
    return _sample_from_logits(lg, temp, top_p, rng, top_k, min_p)


def reply(C, p, user_text, max_tokens=90, temp=0.8, top_p=0.95, seed=None, history=None, top_k=40, min_p=0.05):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    ctx = ""
    if history:
        ctx = "".join("老师：%s\n阿罗娜：%s\n" % (u, a) for u, a in history[-3:])
    prompt = ctx + "老师：%s\n阿罗娜：" % user_text
    ids = tok_cn.encode(prompt)
    if not ids:
        ids = [0]
    out = []
    n_teacher = tok_cn.encode("老师")[0]
    n_colon = tok_cn.encode("：")[0]
    n_nl = tok_cn.encode("\n")[0]
    for step in range(max_tokens):
        # 首 token 禁止直接续写「老师」/换行，避免模型自问自答
        if step == 0:
            nid = sample_once_banned(C, p, ids, temp, top_p, rng, top_k, min_p,
                                     [n_teacher, n_nl])
        else:
            nid = sample_once(C, p, ids, temp, top_p, rng, top_k, min_p)
        out.append(nid)
        ids.append(nid)
        text = tok_cn.decode(out)
        hit = None
        for s in STOP:
            i = text.find(s)
            if s in ("阿罗娜：", "老师，") and i <= 0:
                continue             # 出现在开头属正常开场白，不截断
            if i >= 0:
                hit = i if hit is None else min(hit, i)
        if hit is not None:
            text = text[:hit]
            break
    text = text.strip()
    # 去除 2.5M 小模型常见的"先复述提问再回答"回声
    for echo in (user_text, user_text + "？", user_text + "?"):
        if echo and text.startswith(echo):
            text = text[len(echo):].lstrip("，,。.、:：\n")
            break
    if not text:
        # 兜底：模型一上来就想扮演老师，重新采样一次
        rng2 = np.random.default_rng()
        ids2 = list(ids[:len(tok_cn.encode(prompt))])
        out2 = []
        for step in range(max_tokens):
            if step == 0:
                nid = sample_once_banned(C, p, ids2, temp, top_p, rng2, top_k, min_p,
                                         [n_teacher, n_nl])
            else:
                nid = sample_once(C, p, ids2, temp, top_p, rng2, top_k, min_p)
            out2.append(nid)
            ids2.append(nid)
            t2 = tok_cn.decode(out2)
            h2 = None
            for s in STOP:
                i = t2.find(s)
                if i >= 0:
                    h2 = i if h2 is None else min(h2, i)
            if h2 is not None and h2 > 0:
                text = t2[:h2].strip()
                break
        else:
            text = tok_cn.decode(out2).strip()
    for echo in (user_text, user_text + "？", user_text + "?"):
        if echo and text.startswith(echo):
            text = text[len(echo):].lstrip("，,。.、:：\n")
            break
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.path.join(DIR, "arona-2m5-f32.gguf"))
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=90)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min_p", type=float, default=0.05)
    ap.add_argument("--once", default=None)
    a = ap.parse_args()

    print("[加载] %s" % os.path.basename(a.gguf), flush=True)
    C, p = load_gguf(a.gguf)
    print("[就绪] 阿罗娜在线。输入 quit 退出\n", flush=True)

    if a.once is not None:
        print("老师：%s" % a.once)
        print("阿罗娜：%s" % reply(C, p, a.once, a.max_tokens, a.temp, a.top_p, a.seed, None, a.top_k, a.min_p))
        sys.exit(0)

    hist = []
    while True:
        try:
            u = input("老师：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n阿罗娜：老师，下次再见 (｡･ω･｡)")
            break
        if u.lower() in ("quit", "exit", "q", "退出"):
            print("阿罗娜：老师，下次再见 (｡･ω･｡)")
            break
        if not u:
            continue
        ans = reply(C, p, u, a.max_tokens, a.temp, a.top_p, a.seed, hist, a.top_k, a.min_p)
        print("阿罗娜：%s\n" % ans, flush=True)
        hist.append((u, ans))
