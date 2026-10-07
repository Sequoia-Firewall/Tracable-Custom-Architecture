"""
Tracing utility -- in the spirit of TCA's own "tracable" design goal, but a
new schema (not the existing visualizer's trace.jsonl, which is geometric/
segment-routing-specific and has no notion of layers, heads, or attention --
see TransformerArchitectureCustom/README.md). Each forward pass that opts
in collects one record per instrumented sub-layer; export to JSONL for
offline inspection.
"""
import json
import numpy as np


def _to_jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


class TraceRecorder:
    def __init__(self):
        self.records = []
        self._layer_counter = {}

    def record(self, kind, data):
        idx = self._layer_counter.get(kind, 0)
        self._layer_counter[kind] = idx + 1
        self.records.append({"kind": kind, "index": idx, **data})

    def reset(self):
        self.records = []
        self._layer_counter = {}

    def to_jsonable(self):
        out = []
        for rec in self.records:
            out.append({k: _to_jsonable(v) for k, v in rec.items()})
        return out

    def dump_jsonl(self, path, extra_meta=None):
        with open(path, "a") as f:
            entry = {"meta": extra_meta or {}, "records": self.to_jsonable()}
            f.write(json.dumps(entry) + "\n")

    def attention_summary(self):
        """Quick human-readable summary: per attention-layer, mean entropy
        of each head's attention distribution (low entropy = peaky/focused,
        high entropy = diffuse/uniform) -- a cheap first signal for 'what is
        this head doing' without staring at raw matrices."""
        summaries = []
        for rec in self.records:
            if rec["kind"] != "attention":
                continue
            attn = np.array(rec["attn_weights"])  # (b, h, s, s)
            eps = 1e-12
            entropy = -(attn * np.log(attn + eps)).sum(axis=-1)  # (b, h, s)
            per_head_mean_entropy = entropy.mean(axis=(0, 2))  # (h,)
            summaries.append({
                "layer_index": rec["index"],
                "per_head_mean_entropy": per_head_mean_entropy.tolist(),
            })
        return summaries

    def weight_update_summary(self):
        """One row per (layer, param) with a 'weight_update' record for this
        step -- grad/update/weight norms and their ratio. Covers every
        parameterized layer (embeddings, every LayerNorm and Linear inside
        every block, final LayerNorm, output head), not just attention/ffn."""
        rows = []
        for rec in self.records:
            if rec["kind"] != "weight_update":
                continue
            rows.append({
                "layer": rec["layer"],
                "param": rec["param"],
                "grad_norm": rec["grad_norm"],
                "update_norm": rec["update_norm"],
                "weight_norm": rec["weight_norm"],
                "ratio": rec["ratio"],
            })
        return rows
