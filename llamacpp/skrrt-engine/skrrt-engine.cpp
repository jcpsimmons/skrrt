// skrrt-engine: load a GGUF once, then serve steering / capture / scoring commands on stdin.
// Built inside a llama.cpp tree by llamacpp/build.sh; driven by skrrt/backends/llamacpp.py.
//
// usage: skrrt-engine -m model.gguf [-ngl N] [-s n_seq] [-c n_ctx_per_seq] [-ub n_ubatch] [-t threads]
//                     [-stop TEXT]...
//   -ngl   layers to offload to the GPU (0 = CPU only, default 999 = all)
//   -s     sequences decoded in parallel (default 12)
//   -c     context per sequence (default 768); the context is n_seq * n_ctx_per_seq tokens, shared
//   -ub    physical batch size (default 512)
//   -stop  extra stop token, given as text that tokenizes to exactly one token. Generation always
//          stops on the vocabulary's end-of-generation tokens; this adds to them.
//
// On startup it prints one line:   READY <n_layer> <n_embd>
// <n_layer> is the number of graph layers ("slots"): the il of the "l_out-<il>" tensors and of the
// control-vector layers. For looped architectures it is larger than the number of physical layers.
//
// Commands: one per line, fields separated by tabs. Bulk data goes through files. A "records" file
// is a list of UTF-8 strings, EACH terminated by the byte 0x1e. Prompts are tokenized as given, with
// special tokens parsed and no BOS added: the caller sends complete chat-formatted prompts.
//
//   GEN   <vec|-> <il_start> <il_end> <max_tokens> <prompts> <out>
//         Greedy-decode a reply to every prompt record with the steer <vec> active; write the replies
//         to <out> as records. Prompts run in parallel, up to n_seq at a time.
//   CAP   <vec|-> <il_start> <il_end> <entries> <out>
//         Each entry record is "prefix 0x1f continuation". Teacher-force prefix+continuation with the
//         steer <vec> active ("-" = none) and write
//           <out>.mean  float32 [n_entries][n_layer][n_embd]  mean l_out over the continuation tokens
//           <out>.norm  float32 [n_tokens-1 per entry][n_layer]  ||l_out|| of every token but the first
//           <out>.ntok  int32   [n_entries][2]  tokens in the entry, tokens in the prefix
//   SCORE <records> <yes> <no> <out>
//         For each record, logit(first token of <yes>) - logit(first token of <no>) at the next-token
//         position, unsteered. One number per line in <out>.
//   QUIT
//
// Steer <vec>: a raw float32 file, [n_layer-1][n_embd], row k for graph layer k+1 (the layout of
// llama_set_adapter_cvec; llama.cpp has no control-vector slot for layer 0). Only layers
// il_start..il_end (inclusive) are applied. The vector for layer il is added at every token where
// the architecture's graph calls build_cvec(cur, il): the residual leaving layer il in ordinary
// pre-norm models, the MoE output in an inkling2 tree built with patches/inkling2-cvec.patch.
//
// Every command answers one line on stdout: "OK" or "ERR <message>". Logs go to stderr.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static const char REC = '\x1e', UNIT = '\x1f';

// ---- residual capture through the eval callback
static bool g_capture = false;
static int g_n_layer = 0, g_n_embd = 0;
static int g_cap_tokens = 0;
static std::vector<float> g_lout;  // [n_layer][n_tokens][n_embd] for the entry being captured
static std::vector<int> g_fill;    // tokens received so far per layer (an entry may span several ubatches)

static bool cb_eval(struct ggml_tensor * t, bool ask, void *) {
    if (!g_capture) return !ask;
    if (strncmp(t->name, "l_out-", 6) != 0) return !ask;
    char * end = nullptr;
    const int il = (int) strtol(t->name + 6, &end, 10);
    if (end == t->name + 6 || *end != 0) return !ask;  // derived tensors such as "l_out-5 (reshaped)" are not layer outputs
    if (ask) return true;
    if (il < 0 || il >= g_n_layer) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[0] != g_n_embd) return true;
    const int nt = (int) t->ne[1];
    if (g_fill[il] + nt > g_cap_tokens) { g_fill[il] = g_cap_tokens + 1; return true; }  // flagged as a mismatch below
    ggml_backend_tensor_get(t, g_lout.data() + ((size_t) il * g_cap_tokens + g_fill[il]) * g_n_embd, 0,
                            (size_t) nt * g_n_embd * sizeof(float));
    g_fill[il] += nt;
    return true;
}

static std::string slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

// records are each terminated by REC (a missing final terminator is tolerated)
static std::vector<std::string> records(const std::string & s) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a < s.size()) {
        size_t b = s.find(REC, a);
        if (b == std::string::npos) b = s.size();
        out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return out;
}

static std::vector<std::string> split_tabs(const std::string & s) {
    std::vector<std::string> out(1);
    for (char c : s) { if (c == '\t') out.emplace_back(); else out.back() += c; }
    return out;
}

struct Ctx {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_seq = 12, n_ctx = 0, n_vocab = 0;
    std::vector<llama_token> stops;
};

static std::vector<llama_token> tok(Ctx & C, const std::string & s) {
    if (s.empty()) return {};
    int n = -llama_tokenize(C.vocab, s.c_str(), s.size(), nullptr, 0, false, true);
    std::vector<llama_token> t(std::max(n, 0));
    if (n > 0) llama_tokenize(C.vocab, s.c_str(), s.size(), t.data(), n, false, true);
    return t;
}

static std::string piece(Ctx & C, llama_token t) {
    char buf[512];
    int n = llama_token_to_piece(C.vocab, t, buf, sizeof(buf), 0, false);
    std::string s = n > 0 ? std::string(buf, n) : std::string();
    std::replace(s.begin(), s.end(), REC, ' ');
    return s;
}

static void clear(Ctx & C) { llama_memory_clear(llama_get_memory(C.ctx), true); }

static bool is_stop(Ctx & C, llama_token t) {
    return llama_vocab_is_eog(C.vocab, t) || std::find(C.stops.begin(), C.stops.end(), t) != C.stops.end();
}

static void set_steer(Ctx & C, const std::string & vec, const std::string & il0, const std::string & il1) {
    if (vec == "-") {
        llama_set_adapter_cvec(C.ctx, nullptr, 0, g_n_embd, -1, -1);
        return;
    }
    std::string raw = slurp(vec);
    if (raw.size() != (size_t) (g_n_layer - 1) * g_n_embd * sizeof(float)) throw std::runtime_error("steer file has the wrong size");
    if (llama_set_adapter_cvec(C.ctx, (const float *) raw.data(), raw.size() / sizeof(float), g_n_embd,
                               atoi(il0.c_str()), atoi(il1.c_str()))) throw std::runtime_error("llama_set_adapter_cvec failed");
}

static void clear_steer(Ctx & C) { llama_set_adapter_cvec(C.ctx, nullptr, 0, g_n_embd, -1, -1); }

// Pack items into rounds of at most n_seq sequences whose token need fits the shared context.
static std::vector<std::pair<size_t, size_t>> rounds(Ctx & C, const std::vector<std::vector<llama_token>> & toks, int extra) {
    std::vector<std::pair<size_t, size_t>> out;
    size_t a = 0;
    while (a < toks.size()) {
        size_t b = a;
        long used = 0;
        while (b < toks.size() && (int) (b - a) < C.n_seq) {
            long need = (long) toks[b].size() + extra;
            if (toks[b].empty()) throw std::runtime_error("empty prompt (record " + std::to_string(b) + ")");
            if (need > C.n_ctx) throw std::runtime_error("record " + std::to_string(b) + " needs " + std::to_string(need) +
                                                         " tokens, the context holds " + std::to_string(C.n_ctx) + " (raise n_ctx)");
            if (used + need > C.n_ctx) break;
            used += need; b++;
        }
        out.emplace_back(a, b);
        a = b;
    }
    return out;
}

// decode toks[a..b) as sequences 0.., asking for logits at each last token; returns their batch indices
static std::vector<int> decode_prompts(Ctx & C, const std::vector<std::vector<llama_token>> & toks, size_t a, size_t b, llama_batch & batch) {
    clear(C);
    std::vector<int> last(b - a);
    batch.n_tokens = 0;
    for (size_t i = a; i < b; i++) {
        for (size_t j = 0; j < toks[i].size(); j++) {
            int k = batch.n_tokens++;
            batch.token[k] = toks[i][j]; batch.pos[k] = j; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = i - a;
            batch.logits[k] = j + 1 == toks[i].size();
            if (batch.logits[k]) last[i - a] = k;
        }
    }
    if (llama_decode(C.ctx, batch)) throw std::runtime_error("llama_decode failed");
    return last;
}

static llama_token argmax(Ctx & C, const float * l) {
    int best = 0;
    for (int i = 1; i < C.n_vocab; i++) if (l[i] > l[best]) best = i;
    return best;
}

static void cmd_gen(Ctx & C, const std::vector<std::string> & f) {
    const int max_tokens = atoi(f[4].c_str());
    auto prompts = records(slurp(f[5]));
    std::vector<std::vector<llama_token>> toks;
    for (auto & p : prompts) toks.push_back(tok(C, p));
    auto rs = rounds(C, toks, max_tokens);
    set_steer(C, f[1], f[2], f[3]);
    llama_batch batch = llama_batch_init(C.n_ctx, 0, 1);
    std::string all;
    try {
        for (auto [a, b] : rs) {
            const int n = b - a;
            std::vector<int> last = decode_prompts(C, toks, a, b, batch);
            std::vector<std::string> out(n);
            std::vector<int> pos(n);
            std::vector<bool> done(n, false);
            for (int i = 0; i < n; i++) pos[i] = toks[a + i].size();
            for (int step = 0; step < max_tokens; step++) {
                std::vector<llama_token> next(n, -1);
                for (int i = 0; i < n; i++) {
                    if (done[i]) continue;
                    llama_token t = argmax(C, llama_get_logits_ith(C.ctx, last[i]));
                    if (is_stop(C, t)) { done[i] = true; continue; }
                    out[i] += piece(C, t);
                    next[i] = t;
                }
                batch.n_tokens = 0;
                if (step + 1 == max_tokens) break;
                for (int i = 0; i < n; i++) {
                    if (done[i]) continue;
                    int k = batch.n_tokens++;
                    batch.token[k] = next[i]; batch.pos[k] = pos[i]++; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = i; batch.logits[k] = true;
                    last[i] = k;
                }
                if (batch.n_tokens == 0) break;
                if (llama_decode(C.ctx, batch)) throw std::runtime_error("llama_decode failed");
            }
            for (auto & o : out) { all += o; all += REC; }
        }
    } catch (...) {
        llama_batch_free(batch); clear_steer(C);
        throw;
    }
    llama_batch_free(batch);
    clear_steer(C);
    std::ofstream(f[6], std::ios::binary) << all;
}

static void cmd_cap(Ctx & C, const std::vector<std::string> & f) {
    auto ents = records(slurp(f[4]));
    std::ofstream fm(f[5] + ".mean", std::ios::binary), fn(f[5] + ".norm", std::ios::binary), ft(f[5] + ".ntok", std::ios::binary);
    set_steer(C, f[1], f[2], f[3]);
    llama_batch batch = llama_batch_init(C.n_ctx, 0, 1);
    try {
        for (size_t e = 0; e < ents.size(); e++) {
            const std::string & s = ents[e];
            size_t cut = s.find(UNIT);
            auto all = tok(C, s.substr(0, cut));
            const int s0 = all.size();
            if (cut != std::string::npos) {
                auto c = tok(C, s.substr(cut + 1));
                all.insert(all.end(), c.begin(), c.end());
            }
            const int nt = all.size();
            if (nt == 0) throw std::runtime_error("empty entry " + std::to_string(e));
            if (nt > C.n_ctx) throw std::runtime_error("entry " + std::to_string(e) + " has " + std::to_string(nt) +
                                                       " tokens, the context holds " + std::to_string(C.n_ctx) + " (raise n_ctx)");
            clear(C);
            g_cap_tokens = nt;
            g_lout.assign((size_t) g_n_layer * nt * g_n_embd, 0.f);
            g_fill.assign(g_n_layer, 0);
            // ask for an output at every token: some graphs drop the non-output rows before the last l_out
            batch.n_tokens = nt;
            for (int k = 0; k < nt; k++) {
                batch.token[k] = all[k]; batch.pos[k] = k; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = 0; batch.logits[k] = true;
            }
            g_capture = true;
            int rc = llama_decode(C.ctx, batch);
            llama_synchronize(C.ctx);
            g_capture = false;
            if (rc) throw std::runtime_error("llama_decode failed");
            for (int il = 0; il < g_n_layer; il++)
                if (g_fill[il] != nt) throw std::runtime_error("no usable l_out-" + std::to_string(il) + " tensor (got " + std::to_string(g_fill[il]) +
                                                               " of " + std::to_string(nt) + " tokens): this architecture is not supported for capture");
            std::vector<float> mean((size_t) g_n_layer * g_n_embd, 0.f);
            const int cnt = std::max(1, nt - s0);
            for (int il = 0; il < g_n_layer; il++)
                for (int t = s0; t < nt; t++) {
                    const float * v = g_lout.data() + ((size_t) il * nt + t) * g_n_embd;
                    float * m = mean.data() + (size_t) il * g_n_embd;
                    for (int d = 0; d < g_n_embd; d++) m[d] += v[d] / cnt;
                }
            fm.write((char *) mean.data(), mean.size() * sizeof(float));
            for (int t = 1; t < nt; t++)
                for (int il = 0; il < g_n_layer; il++) {
                    const float * v = g_lout.data() + ((size_t) il * nt + t) * g_n_embd;
                    double s2 = 0;
                    for (int d = 0; d < g_n_embd; d++) s2 += (double) v[d] * v[d];
                    float nv = sqrt(s2);
                    fn.write((char *) &nv, sizeof(float));
                }
            int nn[2] = {nt, s0};
            ft.write((char *) nn, sizeof(nn));
        }
    } catch (...) {
        g_capture = false;
        llama_batch_free(batch); clear_steer(C);
        throw;
    }
    llama_batch_free(batch);
    clear_steer(C);
}

static void cmd_score(Ctx & C, const std::vector<std::string> & f) {
    clear_steer(C);
    auto recs = records(slurp(f[1]));
    auto y = tok(C, f[2]), n = tok(C, f[3]);
    if (y.empty() || n.empty()) throw std::runtime_error("yes/no does not tokenize");
    std::vector<std::vector<llama_token>> toks;
    for (auto & r : recs) toks.push_back(tok(C, r));
    auto rs = rounds(C, toks, 0);
    llama_batch batch = llama_batch_init(C.n_ctx, 0, 1);
    std::ostringstream os;
    os.precision(9);
    try {
        for (auto [a, b] : rs) {
            std::vector<int> last = decode_prompts(C, toks, a, b, batch);
            for (size_t i = a; i < b; i++) {
                const float * l = llama_get_logits_ith(C.ctx, last[i - a]);
                os << (l[y[0]] - l[n[0]]) << "\n";
            }
        }
    } catch (...) {
        llama_batch_free(batch);
        throw;
    }
    llama_batch_free(batch);
    std::ofstream(f[4]) << os.str();
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_seq = 12, n_ctx_seq = 768, ngl = 999, n_ubatch = 512, threads = 0;
    std::vector<std::string> stop_text;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); return 2; }
        if      (a == "-m")    model_path = argv[++i];
        else if (a == "-s")    n_seq = atoi(argv[++i]);
        else if (a == "-c")    n_ctx_seq = atoi(argv[++i]);
        else if (a == "-ngl")  ngl = atoi(argv[++i]);
        else if (a == "-ub")   n_ubatch = atoi(argv[++i]);
        else if (a == "-t")    threads = atoi(argv[++i]);
        else if (a == "-stop") stop_text.push_back(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || n_seq < 1 || n_ctx_seq < 8 || n_ubatch < 1) {
        fprintf(stderr, "usage: skrrt-engine -m model.gguf [-ngl N] [-s n_seq] [-c n_ctx_per_seq] [-ub n_ubatch] [-t threads] [-stop TEXT]...\n");
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    Ctx C;
    C.model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!C.model) { fprintf(stderr, "skrrt-engine: model load failed\n"); return 1; }
    C.n_seq = n_seq;
    C.n_ctx = n_seq * n_ctx_seq;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = C.n_ctx;
    cp.n_batch = C.n_ctx;
    cp.n_ubatch = std::min(n_ubatch, C.n_ctx);
    cp.n_seq_max = n_seq;
    cp.kv_unified = true;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = nullptr;
    if (threads > 0) { cp.n_threads = threads; cp.n_threads_batch = threads; }
    C.ctx = llama_init_from_model(C.model, cp);
    if (!C.ctx) { fprintf(stderr, "skrrt-engine: context creation failed\n"); return 1; }
    C.n_ctx = llama_n_ctx(C.ctx);
    C.vocab = llama_model_get_vocab(C.model);
    C.n_vocab = llama_vocab_n_tokens(C.vocab);
    g_n_layer = llama_model_n_layer(C.model);
    g_n_embd = llama_model_n_embd(C.model);
    for (auto & s : stop_text) {
        auto t = tok(C, s);
        if (t.size() == 1) C.stops.push_back(t[0]);
        else fprintf(stderr, "skrrt-engine: -stop '%s' is %zu tokens, ignored\n", s.c_str(), t.size());
    }
    fprintf(stderr, "skrrt-engine ready: n_layer %d n_embd %d n_vocab %d n_ctx %d n_seq %d ngl %d extra stops %zu\n",
            g_n_layer, g_n_embd, C.n_vocab, C.n_ctx, n_seq, ngl, C.stops.size());
    printf("READY %d %d\n", g_n_layer, g_n_embd);
    fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        auto f = split_tabs(line);
        if (line.empty()) continue;
        if (f[0] == "QUIT") break;
        try {
            if      (f[0] == "GEN"   && f.size() == 7) cmd_gen(C, f);
            else if (f[0] == "CAP"   && f.size() == 6) cmd_cap(C, f);
            else if (f[0] == "SCORE" && f.size() == 5) cmd_score(C, f);
            else throw std::runtime_error("bad command");
            printf("OK\n");
        } catch (std::exception & e) {
            std::string m = e.what();
            std::replace(m.begin(), m.end(), '\n', ' ');
            printf("ERR %s\n", m.c_str());
        }
        fflush(stdout);
    }
    llama_free(C.ctx);
    llama_model_free(C.model);
    llama_backend_free();
    return 0;
}
