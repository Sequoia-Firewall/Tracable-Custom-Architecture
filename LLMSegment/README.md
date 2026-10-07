# LLMSegment

Scaffold for adapting TCA to LLM-shaped tasks. Work sequence (per project direction, 2026-10-07):

1. **`TransformerArchitectureCustom/`** — custom manipulation of a transformer architecture first, standalone from TCA's segment machinery. Not yet started.
2. A dedicated LLM segment type (a `ProcessingNode`/segment variant that wraps or integrates the above) — deferred until (1) has something working to integrate.

See the top-level `README.md`'s Evolution table and `DistanceValueProcessingNode (experimental): findings` section for why a non-regression-shaped node type (pull-toward-a-value rather than additive correction) might matter here: LLM-shaped outputs (tokens, embeddings) aren't a single scalar regression target, so the node-type assumptions baked into `RegressionProcessingNode`/`BayesianProcessingNode` likely don't transfer as-is.
