# Arona-2.5M

一个**扮演阿罗娜**的微型角色语言模型：标准 Llama 架构，纯 numpy 在 CPU 上手写前向/反向训练完成，导出为 **GGUF**，可被 llama.cpp / LM Studio / Ollama 直接加载。

> ⚠️ 语料为**程序化模板合成的仿写文本**，不是游戏原文，也不代表官方设定与台词。详见 [NOTICE.md](NOTICE.md)。

## 亮点

- **真训练过**：800 万字符合成语料 → 193 万 token，Adam + 余弦调度，1350+ 步，验证困惑度约 5.2
- **GGUF v3**：fp32 与 Q8_0 两个版本，llama.cpp 官方二进制实测可加载并生成
- **零依赖推理**：不需要 PyTorch / transformers，只要 numpy 就能跑
- **颜文字 72 种，全部是单个 token**：不会被 BPE 拆成乱码碎片
- **极小**：fp32 权重 10.26 MB，Q8_0 仅 2.89 MB

## 快速开始

### 1. 下载模型

从 Releases 页面下载 `arona-2m5-f32.gguf`（或更小的 `arona-2m5-q8_0.gguf`）。

### 2. 用 llama.cpp 跑（推荐）

```bash
./llama-cli -m arona-2m5-f32.gguf \
  -p "老师：早上好
阿罗娜：" -n 80 -t 0.9 -c 512 \
  --reverse-prompt "老师："
```

`--reverse-prompt "老师："` 是关键，能让模型在开始自己扮演"老师"时停下。
也可用 `llama-server` 起 OpenAI 兼容接口。

### 3. 用 Python 跑（纯 numpy）

```bash
pip install -r requirements.txt
python chat_arona.py --gguf arona-2m5-f32.gguf
```

对话格式固定为 `老师：…` → `阿罗娜：`。

实测输出（llama-cli，temp 0.9）：

```
老师：早上好       → 晚上好，老师 (・∀・；)
老师：你是谁       → 我是阿罗娜，什亭之箱的系统管理人员 (＞▽＜)
老师：我有点累了   → 阿罗娜帮老师准备了甜面包，请慢用 (・∀・｀)
老师：今天有什么安排 → 老师，今天的日程是这样的：文件整理…还有学生的联络事项 (∀｀)
```

## 模型规格

| 项目 | 数值 |
|---|---|
| 参数量 | 2,506,560 |
| dim / layers | 192 / 8 |
| 注意力头 | 8（head_dim 24），KV 头 4（GQA） |
| FFN | 224（SwiGLU） |
| 词表 | 3072（byte-level BPE） |
| 上下文 | 512 |
| 组件 | RMSNorm pre-norm、RoPE(θ=10000)、GQA、SwiGLU、tied embedding、无 bias |

## 仓库内容

| 文件 | 作用 |
|---|---|
| `make_corpus_arona.py` | 合成阿罗娜风格语料（约 800 万字符） |
| `bpe_cn.py` | 训练 byte-level BPE 词表 |
| `llama_cn.py` | 模型定义 + 手写前向/反向传播 |
| `train_cn.py` | 训练主循环（可续训） |
| `export_gguf.py` | 导出 GGUF（含 Q8_0 量化）与 llama2.c 格式 |
| `verify_gguf.py` | GGUF 结构解析与权重校验 |
| `generate_cn.py` | 从 GGUF 加载并推理（纯 numpy） |
| `chat_arona.py` | 交互对话脚本 |
| `run_v1.c` | 极简 C 推理器（llama2.c v1 格式） |
| `tok_cn.py` | 零依赖中文 BPE 分词器 |
| `tokenizer_cn.json` | 词表 + 合并规则 |
| `val_arona.txt` | 验证集文本 |
| `kaomoji_sel.json` | 颜文字池（72 种） |

## 从零复现

全部在 CPU 完成，无需 GPU：

```bash
pip install -r requirements.txt

python make_corpus_arona.py   # 1. 生成语料 corpus_arona.txt
python bpe_cn.py              # 2. 训练 BPE，产出 tokenizer_cn.json + tokens_cn.npy
python train_cn.py            # 3. 训练（反复执行可续训，靠 ckpt_cn.npz / opt_cn.npz）
python export_gguf.py         # 4. 导出 GGUF 与 llama2.c 二进制
python verify_gguf.py arona-2m5-f32.gguf   # 5. 校验
```

训练步数由环境变量控制：`STEPS=150 TOTAL=1500 python train_cn.py`。
想换成自己的语料，替换 `make_corpus_arona.py` 生成的 `corpus_arona.txt` 即可，架构与导出流程无需改动。

## Releases 资产

| 文件 | 说明 |
|---|---|
| `arona-2m5-f32.gguf` | fp32 权重，10.26 MB |
| `arona-2m5-q8_0.gguf` | Q8_0 量化，2.89 MB（74/74 tensor 全量化） |
| `arona-2m5-llama2c.bin` | llama2.c v1 格式，配 `run_v1.c` 使用 |
| `arona-2m5-tokenizer-llama2c.bin` | llama2.c 分词器二进制 |
| `arona-2m5-ckpt.npz` | 训练权重（numpy 格式，用于续训或自行导出） |

## 已知局限

- 2.5M 参数 + 合成语料：只在语料覆盖的话题上表现合理，**问语料外的知识会乱答**
- 偶有答非所问（问题与答案的匹配是概率性的）
- 不支持多语言，以中文为主
- `chat_arona.py` 的纯 numpy 采样质量**不如 llama.cpp**，请以 llama-cli 为准

## 验证情况

- llama.cpp 官方源码编译后实跑 `llama-cli`，生成正确的中文与完整颜文字
- GGUF 结构：magic `GGUF`、version 3、25 项 metadata、74 个 tensor，必需字段零缺失
- fp32 版所有 tensor 与训练权重逐元素误差 0；Q8_0 平均相对误差约 3%
- 分词正则与 llama.cpp 的 GPT-2 预切分规则对齐
- 72 种颜文字编码后均为单 token

## 许可

代码与模型权重采用 **MIT License**，详见 [LICENSE](LICENSE)。

角色名与世界观相关权利归原权利人所有，本项目非官方、未授权，
详见 [NOTICE.md](NOTICE.md)——**二次分发时请一并保留该声明**。
## 联系方式

作者QQ：3784781354
