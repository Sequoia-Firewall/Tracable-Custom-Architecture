# Data-Clustered Sparse Training Across All Layers

Oct 7, 2026 · @George Maurice

## Abstract

We propose **Data-Clustered Sparse Training (DCST)**, a training framework that tests whether the statistical structure of a dataset can organize the parameter topology of a neural network.

DCST clusters training examples by their representations and maps each data cluster to a parameter group in every eligible layer. Each layer holds a small shared group plus one group per cluster. During training, an example activates the shared group and its cluster's group, with optional overlap for ambiguous examples and dense fallback for outliers.

Unlike conventional Mixture-of-Experts (MoE), routing comes from data clustering rather than a learned gating network. Unlike post-training sparsification, DCST exploits sparsity in both the forward and backward passes. Unlike FFN-only specialization, it applies one data-derived organization across FFN, attention and convolutional layers.

The central question is not whether sparse computation or data-dependent routing is possible (both are established), but whether **a data-derived structure can reliably determine useful parameter specialization throughout a network**. The initial target is 40–60% fewer training FLOPs and at least 35% lower peak activation memory than a dense model of equal total parameters, at matched quality.

## 1. Motivation and problem statement

Dense models perform essentially the same parameter computation for every training example. This is most costly in the backward pass, which needs about twice the forward compute plus storage of intermediate activations.

Sparse MoE shows a model can hold far more total parameters than are active per input, but conventional MoE uses learned routers and concentrates sparsity in FFN layers. Prior work has separately shown data- or domain-dependent expert specialization, functional partitioning of parameters, conditional computation in convolution and attention, sparse forward and backward propagation, shared plus specialized experts, fixed routing, and dynamic expert reorganization during training. Each DCST component is therefore feasible; the open question is whether they can be unified around a **common data-derived parameter topology**.

> **Problem:** Can the structure of a training dataset be used to build a layer-wise parameter organization in which each example trains only the parameters relevant to its data cluster, while preserving model quality?

This treats data clustering not merely as a routing mechanism but as a possible source of parameter specialization.

## 2. Research position and novelty

DCST claims no novelty for any individual mechanism. Its novelty hypothesis is the **data-to-parameter correspondence**: one data-derived clustering builds matching parameter groups across heterogeneous layers and drives sparse forward and backward training.

| Prior work | What it establishes | How DCST differs |
| --- | --- | --- |
| c-BTM (Gururangan et al., 2023) | Clusters data with k-means and trains one separate expert model per cluster | DCST partitions layers inside one model, shares a parameter group across clusters, and routes per layer |
| DEMix Layers | Domain-specific FFN experts | Domains are given, not discovered; FFN only |
| MoEfication, EMoE, D2DMoE | Partition FFN neurons into functional experts by weight clustering | Post-training, FFN only, inference savings |
| SLIDE, SAT | Sparse forward and backward computation | No data clustering; fully connected / FFN layers |
| Routing Transformer | Clustering for sparse attention patterns | Clusters tokens for attention, not data for parameters |
| Channel gating, conditional convolution | Input-dependent conv computation | Learned gates; mainly inference savings |
| DeepSeekMoE, HMoE | Shared experts; heterogeneous expert sizes | Learned routing, FFN only |
| Dynamic Expert Clustering (2025) | Re-clusters experts during training | MoE FFN experts only |
| Routing networks, PathNet | Inputs can activate different parameter paths | Learned or evolved paths, not data clusters |

DCST asks whether the pipeline *data structure → parameter topology → sparse training* is a useful alternative to *fixed topology → learned router → expert specialization*.

The closest prior work is c-BTM. The distinguishing claims to test are (1) a single model with a shared group, (2) partitioning beyond FFNs, and (3) reclustering on the model's own hidden states during training.

## 3. Research questions and hypotheses

Thresholds are initial targets, fixed before the main runs and revised only after the pilot.

| # | Question | Hypothesis (falsifiable form) |
| --- | --- | --- |
| RQ1 | Do data clusters determine useful parameter specialization? | **H1:** At equal active compute, DCST is within 1% validation perplexity/accuracy of top-2 MoE, and at least 1% better than the shuffled-cluster control. Group load stays within ±20% of its allocated share with no balancing loss. |
| RQ2 | Does partitioning beyond FFNs add useful savings? | **H2:** Partitioning attention heads and conv filters adds at least 15% training-FLOP reduction over FFN-only DCST, with under 1% quality loss. |
| RQ3 | Does useful specialization follow learned representations, and is one map enough? | **H3a:** Periodic reclustering on hidden states beats fixed input-embedding clusters by at least 0.5% quality. **H3b:** Per-depth cluster maps beat a single shared map in layers past the network midpoint. |
| RQ4 | Do sharing, overlap and outlier fallback prevent over-specialization? | **H4:** Removing the shared group or dense fallback lowers rare-cluster and OOD accuracy by at least 2%, while in-distribution accuracy changes by under 0.5%. |
| RQ5 | Do FLOP savings become real savings? | **H5:** Wall-clock time to target loss drops by at least 30% and peak activation memory by at least 35% versus the equal-total-parameter dense model. |

## 4. Proposed method

### 4.1 Data clustering

Each example is embedded with a small frozen encoder: sequence embeddings for text, pooled pretrained-CNN features for images. Embeddings are reduced to 64–128 dimensions with PCA, then fit with a Gaussian Mixture Model (GMM), K ∈ {8, 16, 32}. The GMM gives hard assignments, soft membership probabilities, likelihoods (outlier signal), cluster sizes and covariances. Balanced k-means, HDBSCAN and a learned VQ codebook are ablations. Clustering itself is not a claimed contribution.

### 4.2 Layer-wise partitioning

Each eligible layer is split into a **shared group** (always active, 10–20% of units) and K **cluster groups**:

- **FFN:** intermediate neurons, with matching rows of W1 and columns of W2.
- **Attention:** heads, with their Q, K, V and output projections.
- **Convolution:** output filters (channels).

Embeddings, normalization layers and the output head stay dense. Groups are stored as contiguous blocks.

### 4.3 Partition initialization and matching

Weight clustering at random initialization is meaningless, so two schemes are compared:

1. **Arbitrary assignment (default).** Group k is assigned to data cluster k at initialization; specialization must emerge from training. This is the cleanest test of the hypothesis.
2. **Dense warm-up.** Train densely for W steps (W = 1–2% of training). For each unit u and cluster k, record mean activation magnitude a(u, k) on that cluster's examples. Assign units to clusters by balanced assignment (Hungarian / min-cost flow) maximizing the sum of a(u, k), subject to the capacity c\_k from 4.5.

### 4.4 Cluster scope: one map or per-depth maps

- **Shared map (primary):** one data clustering routes every layer. This is the strongest form of the data-to-parameter claim.
- **Per-depth maps (H3b):** layers are divided into 2–4 depth blocks; each block's clusters are refit on hidden states at that block's input. This weakens the claim to a per-depth correspondence and is reported separately.

### 4.5 Capacity allocation

Cluster k receives share c\_k of the non-shared units:

```latex
c_k = \frac{n_k^{\alpha} \, \operatorname{tr}(\Sigma_k)^{\beta}}{\sum_j n_j^{\alpha} \, \operatorname{tr}(\Sigma_j)^{\beta}}, \qquad \alpha, \beta \in [0, 1]
```

n\_k is the cluster's example count and Σ\_k its covariance in PCA space. α = 1, β = 0 allocates by size only. Each group gets at least 2% of units.

### 4.6 Routing (training and inference)

- **Confident example:** shared group + primary cluster group.
- **Ambiguous example:** also the second cluster's group when p(C₂ | x) > τ (τ = 0.3 default).
- **Outlier:** GMM log-likelihood in the bottom p% (p = 2–5) → dense layer. If outliers exceed 10% of a batch window, clusters are refit.
- **Granularity:** per example for vision; per 512-token sequence chunk for language (main setting). Per-token routing is an ablation.
- **Inference:** the same frozen encoder, PCA and GMM run on each new input. Their cost (target under 2% of model FLOPs) and routing accuracy on OOD data are reported. With per-depth maps, a small linear probe per depth block predicts the cluster from hidden states, trained on the GMM labels.

### 4.7 Reclustering during training

Every T steps (T = 1,000–5,000): collect hidden states for 50k held-out examples, refit the GMM, match new clusters to existing groups with the Hungarian algorithm, warm-start centroids from the old ones, and skip the update if under 5% of assignments change. A 200-step learning-rate warm-up follows each applied refit.

### 4.8 Sparse forward and backward

Inactive groups are excluded from the autograd graph for that example, so their activations are neither stored nor recomputed. Examples are sorted by cluster within each batch and run through block-sparse grouped GEMMs (MegaBlocks-style); outliers form a separate dense sub-batch. Both FLOPs and wall-clock time are reported, to separate theoretical sparsity from hardware efficiency.

## 5. Experimental design

Experiments start small and scale only configurations that pass the pilot.

### 5.1 Settings

| Setting | Model | Dataset | Partitioned layers |
| --- | --- | --- | --- |
| Vision pilot | ResNet-50 | ImageNet-1k | Conv filters |
| Vision transformer | ViT-B/16 | ImageNet-1k | FFN + attention |
| Language, small | 125M GPT-style decoder | OpenWebText / FineWeb subset (10B tokens) | FFN + attention |
| Language, medium | 760M GPT-style decoder | FineWeb subset (30B tokens) | FFN + attention |
| Multi-domain | 125M GPT-style decoder | The Pile | FFN + attention |

The Pile has labelled domains, so discovered clusters can be compared to known structure.

### 5.2 Baselines

All sparse methods are compared at approximately equal total parameters, active parameters, data and optimizer budget.

1. Dense, equal total parameters.
2. Dense, equal active parameters.
3. Top-2 learned-router MoE with load-balancing loss.
4. Expert Choice routing.
5. Hash routing (fixed, data-agnostic).
6. c-BTM (separate expert models per data cluster).
7. RigL dynamic sparse training at matched sparsity.
8. Channel gating (conv) and head-routing (attention) conditional-computation baselines.
9. **Shuffled-cluster control:** identical to DCST (same group sizes, overlap, outlier handling, reclustering schedule) but cluster labels are randomly permuted across examples, preserving cluster sizes. This isolates the effect of data structure from generic sparsity.
10. **Cluster-initialized learned router:** a learned router initialized to DCST's cluster assignments and then trained, testing whether learning improves on the data map.

### 5.3 Ablations

| Factor | Values |
| --- | --- |
| Clustering method | GMM, balanced k-means, HDBSCAN, learned VQ |
| Number of clusters K | 8, 16, 32 |
| Layer scope | FFN only, FFN + attention, all eligible layers |
| Shared group size | 0%, 10%, 20%, 30% |
| Overlap τ | top-1 only, 0.3, 0.5 |
| Outlier handling | dense fallback, none, p = 2%, 5% |
| Partition init | arbitrary, dense warm-up |
| Cluster scope | shared map, per-depth maps |
| Reclustering | fixed input, periodic input, periodic hidden-state |
| Capacity allocation | equal, size only, size + spread |
| Routing granularity | per sample, per sequence, per token |

Main results use 3 seeds and report mean ± standard deviation.

### 5.4 Metrics

| Category | Metrics |
| --- | --- |
| Quality | Validation perplexity; top-1 accuracy; zero-shot HellaSwag, ARC, PIQA |
| Efficiency | Training FLOPs per example; wall-clock time to target loss; peak activation memory; routing overhead (encoder + GMM + reclustering) |
| Load | Coefficient of variation of group utilization vs allocated share; fraction of groups under 2% of assignments; routing entropy |
| Robustness | Rare-cluster accuracy; ImageNet-R; held-out Pile domains |
| Specialization | Adjusted mutual information between clusters and Pile domains; per-group ablation impact by cluster (does removing group k hurt cluster k most?) |

## 6. Primary scientific test and success criteria

The central comparison is **data-derived parameter organization vs learned routing vs the shuffled-cluster control**, at approximately equal total parameters, active parameters, training data and optimizer budget.

DCST succeeds only if all three hold:

| Criterion | Requirement |
| --- | --- |
| Computational | Wall-clock time and peak activation memory below the **equal-total-parameter** dense model, by the H5 margins |
| Statistical | Quality within 1% of the **equal-total-parameter** dense model, above the **equal-active-parameter** dense model, and within 1% of top-2 MoE at equal active compute |
| Scientific | DCST beats the shuffled-cluster control (H1), and per-group ablations show group k matters most for cluster k |

Informative partial outcomes:

- **Hidden-state clusters work, input clusters do not:** specialization follows learned representations rather than raw data structure.
- **DCST ≈ shuffled control:** gains come from generic structured sparsity, and the data-to-parameter hypothesis is rejected.
- **Cluster-initialized router beats DCST:** data structure is a useful prior for routing but not sufficient on its own.

## 7. Risks and mitigations

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Data clusters do not map to useful specialization | Central hypothesis fails | Shuffled-cluster control and cluster-initialized router make this a measurable, publishable outcome |
| Arbitrary initial assignment never specializes | Groups behave like random sparsity | Dense warm-up with activation-based matching (4.3) |
| Input clusters unsuitable for deep layers | Layers encode different abstractions | Hidden-state reclustering; per-depth maps (4.4) |
| Reclustering destabilizes training | Groups suddenly receive new data | Hungarian matching, warm starts, 5% change threshold, 200-step LR warm-up |
| Over-specialization | Poor generalization | Shared group, overlap τ, dense outlier fallback |
| Outlier share grows | Dense fallback erodes savings | Monitor; refit GMM above 10% |
| Unequal cluster sizes | Poor hardware utilization | Capacity formula with 2% floor; balanced k-means ablation |
| Routing overhead at inference | Encoder + GMM cost offsets savings | Measure it; replace with per-depth linear probes if above 2% of FLOPs |
| Sparse kernels give no real speedup | FLOP savings stay theoretical | Block-contiguous groups, grouped GEMMs, report wall-clock |
| Clusters capture superficial features | Specialization is meaningless | Compare to Pile domains; inspect clusters; controlled perturbations |
| No gain over simpler MoE | Complexity not justified | Equal-active-compute comparisons and full ablations |

## 8. Contributions

1. **A data-to-parameter training framework** that builds layer-wise specialization from the structure of the training distribution.
2. **One organization across heterogeneous layers:** FFN, attention and convolution partitioned from the same data clusters.
3. **Training-time sparsity:** reduced forward and backward compute and activation storage, not only inference savings.
4. **An alternative to learned routing** that avoids router collapse and auxiliary balancing losses by construction.
5. **A controlled test of the hypothesis** via the shuffled-cluster control, separating data-driven specialization from generic sparsity.
6. **Analysis across depth** of whether useful specialization follows raw data structure or learned representations.

## 9. Conclusion

DCST organizes established mechanisms (data clustering, conditional computation, sparse backpropagation, shared experts, dynamic reorganization) around one hypothesis: **a training dataset's structure can define a parameter topology that stays useful throughout training**.

By mapping data clusters to parameter groups across layer types, routing examples through them, updating the map from learned representations, and training sparsely in both passes, DCST tests that hypothesis directly. The key result is evidence for or against a useful correspondence between data structure and parameter specialization, with lower training cost as the practical payoff. More broadly, it asks whether the structure of a learning problem can determine the structure of the network that solves it.

## References

Linked papers were opened during research; unlinked ones are cited from memory and should be verified before submission.

- Gururangan et al., 2023. Scaling Expert Language Models with Unsupervised Domain Discovery (c-BTM).
- Gururangan et al., 2022. DEMix Layers: Disentangling Domains for Modular Language Modeling. NAACL.
- Zhang et al., 2022. [MoEfication: Transformer Feed-forward Layers are Mixtures of Experts](https://arxiv.org/pdf/2110.01786). Findings of ACL.
- Qiu et al., 2023. [Unlocking Emergent Modularity in Large Language Models (EMoE)](https://arxiv.org/pdf/2310.10908).
- Piórczyński et al., 2023. [Exploiting Activation Sparsity with Dense to Dynamic-k MoE Conversion (D2DMoE)](https://arxiv.org/pdf/2310.04361).
- 2025. [Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](https://arxiv.org/pdf/2510.02345).
- Chen et al., 2020. SLIDE: In Defense of Smart Algorithms over Hardware Acceleration. MLSys. Described in [Distributed SLIDE](https://arxiv.org/pdf/2201.12667).
- Ma et al., 2024. [Sparsity-Accelerated Training for Large Language Models (SAT)](https://finn.lub.lu.se/EdsRecord/edsarx,edsarx.2406.01392). Findings of ACL.
- Roy et al., 2021. Efficient Content-Based Sparse Attention with Routing Transformers. TACL.
- Hua et al., 2019. [Channel Gating Neural Networks](https://neurips.cc/virtual/2019/poster/13394). NeurIPS.
- Bejnordi et al., 2020. [Batch-Shaping for Learning Conditional Channel Gated Networks](https://arxiv.org/pdf/1907.06627). ICLR.
- Rosenbaum et al., 2018. Routing Networks: Adaptive Selection of Non-Linear Functions for Multi-Task Learning. ICLR.
- Fernando et al., 2017. PathNet: Evolution Channels Gradient Descent in Super Neural Networks.
- Dai et al., 2024. DeepSeekMoE.
- Wang et al., 2024. HMoE: Heterogeneous Mixture of Experts.
- Roller et al., 2021. Hash Layers for Large Sparse Models. NeurIPS.
- Zhou et al., 2022. Mixture-of-Experts with Expert Choice Routing. NeurIPS.
- Evci et al., 2020. Rigging the Lottery (RigL). ICML.
- Gale et al., 2023. MegaBlocks: Efficient Sparse Training with Mixture-of-Experts. MLSys.
