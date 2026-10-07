/* run_v1.c — llama2.c version-1 fp32 模型的极简 C 推理器（仅依赖 libc / libm）
 * 编译: cc -O2 -fopenmp -o run_v1 run_v1.c -lm
 * 运行: ./run_v1 model.bin -z tokenizer.bin -t 0.85 -n 120 -s 3 -p "有一天"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, shared; } Config;
typedef struct {
    float *tok_emb, *rms_att, *rms_ffn, *rms_final;
    float *wq, *wk, *wv, *wo, *w1, *w2, *w3;
} Weights;
typedef struct { char **vocab; float *scores; int n; } Tokenizer;

static void die(const char *m) { fprintf(stderr, "error: %s\n", m); exit(1); }

static float *read_floats(FILE *f, size_t n) {
    float *p = malloc(n * sizeof(float) + 16);
    if (!p || fread(p, sizeof(float), n, f) != n) die("read weights");
    return p;
}

static void load_model(const char *path, Config *c, Weights *w) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open model");
    int magic, ver, h[7];
    unsigned char sh;
    if (fread(&magic, 4, 1, f) != 1) die("hdr");
    if (magic != 0x616b3432) { fprintf(stderr, "bad magic 0x%x (expect 0x616b3432)\n", magic); exit(1); }
    if (fread(&ver, 4, 1, f) != 1) die("ver");
    if (ver != 1) { fprintf(stderr, "only version 1 supported (got %d)\n", ver); exit(1); }
    if (fread(h, sizeof(int), 7, f) != 7) die("cfg");
    c->dim = h[0]; c->hidden_dim = h[1]; c->n_layers = h[2]; c->n_heads = h[3];
    c->n_kv_heads = h[4]; c->vocab_size = h[5]; c->seq_len = h[6];
    if (fread(&sh, 1, 1, f) != 1) die("shared");
    c->shared = sh;
    int L = c->n_layers, D = c->dim, HD = D / c->n_heads, KVD = c->n_kv_heads * HD;
    w->tok_emb   = read_floats(f, (size_t)c->vocab_size * D);
    w->rms_att   = read_floats(f, (size_t)L * D);
    w->wq        = read_floats(f, (size_t)L * D * D);
    w->wk        = read_floats(f, (size_t)L * KVD * D);
    w->wv        = read_floats(f, (size_t)L * KVD * D);
    w->wo        = read_floats(f, (size_t)L * D * D);
    w->rms_ffn   = read_floats(f, (size_t)L * D);
    w->w1        = read_floats(f, (size_t)L * c->hidden_dim * D);
    w->w2        = read_floats(f, (size_t)L * D * c->hidden_dim);
    w->w3        = read_floats(f, (size_t)L * c->hidden_dim * D);
    w->rms_final = read_floats(f, (size_t)D);
    fclose(f);
}

static void load_tokenizer(const char *path, Tokenizer *t, int vocab_size) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open tokenizer");
    int maxlen;
    if (fread(&maxlen, sizeof(int), 1, f) != 1) die("tok hdr");
    t->n = vocab_size;
    t->vocab = malloc(sizeof(char *) * t->n);
    t->scores = malloc(sizeof(float) * t->n);
    for (int i = 0; i < t->n; i++) {
        int len;
        if (fread(t->scores + i, sizeof(float), 1, f) != 1) die("tok score");
        if (fread(&len, sizeof(int), 1, f) != 1) die("tok len");
        t->vocab[i] = malloc(len + 1);
        if (len && fread(t->vocab[i], 1, len, f) != (size_t)len) die("tok bytes");
        t->vocab[i][len] = 0;
    }
    fclose(f);
}

static int str_lookup(const char *s, Tokenizer *t) {
    for (int i = 0; i < t->n; i++) if (strcmp(s, t->vocab[i]) == 0) return i;
    return -1;
}

/* UTF-8 解码 */
static int utf8_next(const char *s, int i, int len, int *cp) {
    unsigned char c = (unsigned char)s[i];
    int n, v;
    if (c < 0x80) { n = 1; v = c; }
    else if (c >= 0xF0) { n = 4; v = c & 0x07; }
    else if (c >= 0xE0) { n = 3; v = c & 0x0F; }
    else if (c >= 0xC0) { n = 2; v = c & 0x1F; }
    else { n = 1; v = c; }
    if (i + n > len) { n = 1; v = c; }
    for (int k = 1; k < n; k++) v = (v << 6) | ((unsigned char)s[i + k] & 0x3F);
    *cp = v;
    return n;
}

static int is_cjk(int cp) { return cp >= 0x4E00 && cp <= 0x9FFF; }

/* 单个预切分块内做 byte-level BPE（贪心按 score 合并，与 Python 端 rank 一致）*/
static int encode_chunk(Tokenizer *t, const char *s, int slen, int *out, int max) {
    char *pieces[4096];
    int np = 0;
    for (int i = 0; i < slen && np < 4096;) {
        int cp, clen = utf8_next(s, i, slen, &cp);
        char *p = malloc(clen + 1);
        memcpy(p, s + i, clen); p[clen] = 0;
        pieces[np++] = p;
        i += clen;
    }
    while (1) {
        float best = -1e30f; int bi = -1;
        for (int i = 0; i < np - 1; i++) {
            char merged[512];
            snprintf(merged, sizeof(merged), "%s%s", pieces[i], pieces[i + 1]);
            int id = str_lookup(merged, t);
            if (id >= 0 && t->scores[id] > best) { best = t->scores[id]; bi = i; }
        }
        if (bi < 0) break;
        size_t l1 = strlen(pieces[bi]), l2 = strlen(pieces[bi + 1]);
        char *nm = malloc(l1 + l2 + 1);
        memcpy(nm, pieces[bi], l1); memcpy(nm + l1, pieces[bi + 1], l2); nm[l1 + l2] = 0;
        free(pieces[bi]); free(pieces[bi + 1]);
        pieces[bi] = nm;
        for (int i = bi + 1; i < np - 1; i++) pieces[i] = pieces[i + 1];
        np--;
    }
    int n = 0;
    for (int i = 0; i < np && n < max; i++) {
        int id = str_lookup(pieces[i], t);
        if (id >= 0) out[n++] = id;
    }
    for (int i = 0; i < np; i++) free(pieces[i]);
    return n;
}

/* 预切分正则（等价 Python 端 PAT）：CJK+ | [A-Za-z]+ | [0-9]{1,3} | 其他 | 空白 */
static int encode(Tokenizer *t, const char *text, int *tokens, int max_tokens) {
    int len = (int)strlen(text), n = 0, i = 0;
    while (i < len && n < max_tokens) {
        int cp, clen = utf8_next(text, i, len, &cp);
        int j = i + clen, kind;
        if (is_cjk(cp)) kind = 1;
        else if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) kind = 2;
        else if (cp >= '0' && cp <= '9') kind = 3;
        else if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') kind = 4;
        else kind = 5;
        if (kind == 1 || kind == 2 || kind == 4) {
            while (j < len) {
                int cp2, cl2 = utf8_next(text, j, len, &cp2);
                if (kind == 1 && !is_cjk(cp2)) break;
                if (kind == 2 && !((cp2 >= 'A' && cp2 <= 'Z') || (cp2 >= 'a' && cp2 <= 'z'))) break;
                if (kind == 4 && !(cp2 == ' ' || cp2 == '\t' || cp2 == '\n' || cp2 == '\r')) break;
                j += cl2;
            }
        } else if (kind == 3) {
            int cnt = 1;
            while (j < len && cnt < 3) {
                int cp2, cl2 = utf8_next(text, j, len, &cp2);
                if (cp2 < '0' || cp2 > '9') break;
                j += cl2; cnt++;
            }
        }
        n += encode_chunk(t, text + i, j - i, tokens + n, max_tokens - n);
        i = j;
    }
    return n;
}

static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = w[i] * (ss * x[i]);
}

static void softmax(float *x, int n) {
    float m = x[0]; for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void matmul(float *o, const float *x, const float *w, int n, int d) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < d; i++) {
        const float *wi = w + (size_t)i * n;
        float val = 0;
        for (int j = 0; j < n; j++) val += wi[j] * x[j];
        o[i] = val;
    }
}

/* kv cache 布局: [layer][k|v][kv_head][seq][hd] */
static float *forward(Config *c, Weights *w, int token, int pos, float *kcache, float *vcache, float *buf) {
    int D = c->dim, HD = D / c->n_heads, KVD = c->n_kv_heads * HD;
    int NH = c->n_heads, NKV = c->n_kv_heads, G = NH / NKV, H = c->hidden_dim, S = c->seq_len;
    float *x = buf, *hb = buf + D, *hbf = buf + 2 * D;
    float *q = buf + 3 * D, *xb = buf + 4 * D, *xb2 = buf + 5 * D;
    float *k = buf + 6 * D, *v = k + KVD, *u = v + KVD, *u3 = u + H;
    memcpy(x, w->tok_emb + (size_t)token * D, D * sizeof(float));
    for (int l = 0; l < c->n_layers; l++) {
        rmsnorm(hb, x, w->rms_att + (size_t)l * D, D);
        matmul(q, hb, w->wq + (size_t)l * D * D, D, D);
        matmul(k, hb, w->wk + (size_t)l * KVD * D, D, KVD);
        matmul(v, hb, w->wv + (size_t)l * KVD * D, D, KVD);
        for (int i = 0; i < D; i += 2) {                       /* RoPE on q */
            int j = (i % HD) / 2;
            float freq = 1.0f / powf(10000.0f, (float)j / (float)HD);
            float ang = pos * freq, fcr = cosf(ang), fsr = sinf(ang);
            float a = q[i], b = q[i + 1];
            q[i] = a * fcr - b * fsr; q[i + 1] = a * fsr + b * fcr;
        }
        for (int h = 0; h < NKV; h++) for (int i = 0; i < HD; i += 2) {   /* RoPE on k + 写缓存 */
            float freq = 1.0f / powf(10000.0f, (float)(i / 2) / (float)HD);
            float ang = pos * freq, fcr = cosf(ang), fsr = sinf(ang);
            float a = k[h * HD + i], b = k[h * HD + i + 1];
            k[h * HD + i] = a * fcr - b * fsr; k[h * HD + i + 1] = a * fsr + b * fcr;
            size_t off = ((size_t)l * NKV + h) * S * HD + (size_t)pos * HD;
            kcache[off + i] = k[h * HD + i];
            kcache[off + i + 1] = k[h * HD + i + 1];
            vcache[off + i] = v[h * HD + i];
            vcache[off + i + 1] = v[h * HD + i + 1];
        }
        for (int h = 0; h < NH; h++) {
            int kvh = h / G;
            const float *qh = q + h * HD;
            size_t base = ((size_t)l * NKV + kvh) * S * HD;
            float *att = u3 + H;                                /* 复用缓冲尾部 */
            for (int t = 0; t <= pos; t++) {
                const float *kt = kcache + base + (size_t)t * HD;
                float s = 0; for (int i = 0; i < HD; i++) s += qh[i] * kt[i];
                att[t] = s / sqrtf((float)HD);
            }
            softmax(att, pos + 1);
            float *xh = xb + h * HD;
            for (int i = 0; i < HD; i++) xh[i] = 0;
            for (int t = 0; t <= pos; t++) {
                const float *vt = vcache + base + (size_t)t * HD;
                for (int i = 0; i < HD; i++) xh[i] += att[t] * vt[i];
            }
        }
        matmul(xb2, xb, w->wo + (size_t)l * D * D, D, D);
        for (int i = 0; i < D; i++) x[i] += xb2[i];
        rmsnorm(hbf, x, w->rms_ffn + (size_t)l * D, D);
        matmul(u, hbf, w->w1 + (size_t)l * H * D, D, H);
        matmul(u3, hbf, w->w3 + (size_t)l * H * D, D, H);
        for (int i = 0; i < H; i++) u[i] = u[i] / (1.0f + expf(-u[i])) * u3[i];
        matmul(xb2, u, w->w2 + (size_t)l * D * H, H, D);
        for (int i = 0; i < D; i++) x[i] += xb2[i];
    }
    rmsnorm(x, x, w->rms_final, D);
    float *logits = malloc((size_t)c->vocab_size * sizeof(float));
    matmul(logits, x, w->tok_emb, D, c->vocab_size);   /* tied embedding */
    return logits;
}

int main(int argc, char **argv) {
    const char *mpath = "model.bin", *tpath = "tokenizer.bin", *prompt = "";
    float temp = 0.85f; int steps = 120, seed = 1;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') { mpath = argv[i]; continue; }
        if (i + 1 >= argc) break;
        if (argv[i][1] == 'z') tpath = argv[++i];
        else if (argv[i][1] == 't') temp = (float)atof(argv[++i]);
        else if (argv[i][1] == 'n') steps = atoi(argv[++i]);
        else if (argv[i][1] == 's') seed = atoi(argv[++i]);
        else if (argv[i][1] == 'p') prompt = argv[++i];
    }
    Config c; Weights w; Tokenizer tk;
    load_model(mpath, &c, &w);
    load_tokenizer(tpath, &tk, c.vocab_size);
    int D = c.dim, HD = D / c.n_heads, KVD = c.n_kv_heads * HD, S = c.seq_len, H = c.hidden_dim;
    printf("[model] dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq=%d shared=%d\n",
           c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len, c.shared);
    size_t kv_sz = (size_t)c.n_layers * c.n_kv_heads * S * HD;
    float *kcache = calloc(kv_sz, sizeof(float));
    float *vcache = calloc(kv_sz, sizeof(float));
    float *buf = malloc(sizeof(float) * (8 * D + 2 * KVD + 2 * H + S + 8));
    int *tokens = malloc(sizeof(int) * (S + 8));
    int ntok = encode(&tk, prompt, tokens, S);
    if (ntok == 0) { tokens[0] = 0; ntok = 1; }
    printf("[prompt %d tokens] ", ntok);
    for (int i = 0; i < ntok; i++) printf("%s", tk.vocab[tokens[i]]);
    printf("\n[out] ");
    fflush(stdout);
    srand((unsigned)seed);
    int pos = 0;
    for (int i = 0; i < steps; i++) {
        float *logits = forward(&c, &w, tokens[pos], pos, kcache, vcache, buf);
        int next;
        if (temp <= 0) { next = 0; for (int j = 1; j < c.vocab_size; j++) if (logits[j] > logits[next]) next = j; }
        else {
            for (int j = 0; j < c.vocab_size; j++) logits[j] /= temp;
            softmax(logits, c.vocab_size);
            float r = (float)rand() / (float)RAND_MAX, cum = 0; next = c.vocab_size - 1;
            for (int j = 0; j < c.vocab_size; j++) { cum += logits[j]; if (r < cum) { next = j; break; } }
        }
        printf("%s", tk.vocab[next]);
        fflush(stdout);
        pos++;
        tokens[pos] = next;
        free(logits);
        if (pos >= S - 1) break;
    }
    printf("\n");
    return 0;
}
