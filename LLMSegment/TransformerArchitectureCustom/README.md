# TransformerArchitectureCustom

A decoder-only (GPT-style) transformer built from near-scratch (numpy only, no ML framework) — the first concrete step toward LLM-shaped TCA segments. The goal here is a fully understood, fully modifiable baseline: every operation (projections, per-head attention scores, masking, softmax, residual adds) is a separate, inspectable step rather than one opaque framework call, so "custom manipulation of architecture internals" has something real to diverge from, and tracing hooks have real sub-layer boundaries to attach to.

## Status: working, gradient-verified, confirmed to actually learn

- **`layers.py`** — core primitives (`Linear`, `LayerNorm`, `Embedding`, `GELU`, `softmax`, `cross_entropy_loss`), each with manual forward + backward. Same discipline as TCA's `ProcessingNode`: forward caches what backward needs, backward accumulates parameter gradients.
- **`attention.py`** — `MultiHeadSelfAttention`, causal-masked (decoder-only). Q/K/V projections, per-head scaled dot-product scores, masking, softmax, and the output projection are all separate steps — this is the main surface meant to be swapped/modified later.
- **`transformer.py`** — `FeedForward`, `TransformerBlock` (pre-norm residual: `LN → attn → +x`, `LN → ffn → +x`), and `GPTStyleTransformer` (token + positional embedding → N blocks → final LN → output head → cross-entropy).
- **`trace.py`** — `TraceRecorder`: opt-in per-sublayer recording (attention weights per head, Q/K/V/output norms, FFN activation norms), dumped to JSONL. A *new* schema, not the existing `visualizer/`'s `trace.jsonl` — that one is geometric/segment-routing-specific (node positions, signal paths) and has no notion of layers or attention heads. Includes a quick attention-entropy summary (per-head mean entropy of the attention distribution) as a first cheap interpretability signal.
- **`grad_check.py`** — numerical (finite-difference) gradient checks for every layer in isolation, including `MultiHeadSelfAttention`. All pass to float-precision (~1e-8).
- **`grad_check_full_model.py`** — end-to-end check on the *composed* model (embeddings → 2 blocks → final LN → head → loss), catching wiring mistakes at residual/composition boundaries that per-layer checks can't see (e.g. a dropped or double-counted gradient contribution across a residual add). All pass (~1e-7).
- **`train_toy.py`** — the real test: a synthetic delayed-copy task (`token[i] = token[i-k]`, which requires genuine attention to a fixed offset, not just a per-position bias) trains from random-guess loss (`ln(vocab_size)`) down to a real fit, with **100% held-out accuracy** on determined positions. The attention-entropy trace already shows interpretable structure — layer-0 heads specializing (lower, more varied entropy) vs. layer-1 heads more diffuse.

## Design notes / what's deliberately simple right now

- Decoder-only, causal self-attention only — no encoder side, no cross-attention. Chosen because next-token prediction is the most direct fit for later LLM-segment work, and it's the smallest architecture that's still a real language model.
- SGD, no Adam/momentum/LR schedule yet — the toy task converges fine without it; add this when a real training run needs it, not preemptively.
- No tokenizer — operates on integer token ids directly. A real tokenizer is a separate concern for whenever this touches real text.
- `trace.py`'s records currently capture norms/entropy, not full raw activation dumps by default (attention weights are the one full-tensor exception, since they're the main "what is this head doing" signal) — extend `record()` calls in `attention.py`/`transformer.py` as specific interpretability questions come up, rather than logging everything unconditionally.

## Next steps (not yet done)

- Actual "custom manipulation" of the attention/architecture internals — this scaffold is the baseline to modify, not the end state.
- Deciding how this eventually plugs into a TCA segment type (deferred per project direction — see `LLMSegment/README.md`).
