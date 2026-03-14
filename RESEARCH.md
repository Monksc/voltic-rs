# Voltic Research: Alternative Architectures

This document explores alternative architectures to standard Transformer/GPT models for potential implementation in Voltic.

## Table of Contents
1. [State Space Models (Mamba)](#state-space-models)
2. [Mixture of Experts (MoE)](#mixture-of-experts)
3. [Your Ideas: Tensor & Multi-Matrix Approaches](#your-ideas)
4. [Retrieval-Augmented Generation (RAG)](#rag)

---

## State Space Models

### Overview
State Space Models (SSMs), particularly Mamba, represent a major alternative to Transformers. They achieve **O(n)** linear time complexity vs Transformer's **O(n²)** quadratic complexity.

### Key Equations
```
h'(t) = A * h(t) + B * u(t)    # State equation
y(t) = C * h(t) + D * u(t)      # Output equation
```

Where:
- `A` = state transition matrix (controls memory)
- `B` = input projection
- `C` = output projection  
- `u(t)` = input at time t
- `h(t)` = hidden state

### Evolution
| Year | Model | Key Innovation |
|------|-------|----------------|
| 2020 | HiPPO | Mathematical framework for long-range dependencies |
| 2021 | S4 | First practical SSM for deep learning |
| 2023 | Mamba | Selective state spaces, data-dependent |
| 2024 | Mamba-2 | State Space Duality (SSM ↔ attention) |
| 2024-25 | Jamba, Bamba | Hybrid attention + SSM |

### Hybrid is the Winner
Pure SSMs struggle with:
- In-context learning
- Precise recall of distant information

**Best approach**: Interleave Transformer attention with SSM layers
- SSM for long-range efficiency
- Attention for precise retrieval

**Industry consensus 2025-26**: Hybrid models offer best of both worlds.

### Implementation in Voltic
Would require:
- New `SSMOp` with selective scan kernel
- Parallel scan algorithm for GPU efficiency
- Hybrid layer pattern

---

## Mixture of Experts (MoE)

### Overview
MoE enables **sparse activation** - only a subset of "experts" process each token, giving massive parameter count with constant compute cost.

### Architecture
```
Input → [Gate Network] → Expert 1
                  → Expert 2
                  → Expert 3
                  → ... → Output
```

### Key Equations
```
MoE(x) = Σ_{i ∈ TopK(p)} p_i · Expert_i(x)

Where:
- TopK = selects k best experts based on gating scores
- p_i = gating probability for expert i
- Expert_i = specialized sub-network
```

### Famous MoE Models (2024-25)
| Model | Active Params | Total Params | Experts |
|-------|--------------|---------------|---------|
| Mixtral 8x7B | 12B | 46B | 8 |
| DeepSeek-R1 | 37B | 671B | 64 |
| LLaMA-4 | 22B | 400B | 16+ |
| GPT-OSS | 20B | 120B | 128 |

### Advantages
- Massive parameter capacity without increased compute
- Specialized experts for different tasks/domains
- Better inference cost scaling

### Implementation in Voltic
Would require:
- `MoELayer` with multiple `Linear` experts
- `GatingNetwork` (simple softmax or top-k)
- Load balancing loss (to prevent expert collapse)

---

## Your Ideas: Tensor & Multi-Matrix Approaches

### Idea 1: Multi-Matrix Transformation
Take the concept of Q @ K^T @ V from attention but extend it:

**Current (Attention):**
```
Q @ K^T → [batch, heads, seq, seq] → Softmax → @ V
```

**Extended (Multi-Matrix):**
```
Instead of 3 matrices (Q,K,V), use N matrices M1, M2, ..., MN

H = M1 @ X
H = M2 * H  (element-wise or matmul)
...
Output = Softmax(H) @ M_final
```

This creates **learnable tensor transformations** where:
- Each matrix can capture different aspects of the data
- Non-linear combinations via element-wise multiplication
- Potentially captures higher-order interactions

### Idea 2: Tensor Embedding with Softmax Decomposition

Instead of softmax over sequence positions, apply softmax across **embedding dimensions** to get "attention-like" tensors:

```
Input: [batch, seq, embed_dim]

# Approach 1: Channel-wise attention
embed = embed.reshape([batch, seq, groups, dim_per_group])
attn = softmax(embed, axis=-1)  # [batch, seq, groups, dim_per_group]
output = embed * attn  # reweight dimensions

# Approach 2: Tensor decomposition
# Factorize attention into multiple smaller tensors
# Similar to tensor train / CP decomposition concepts
```

### Idea 3: Database-Connected Embeddings

Hook embeddings to a "neural database":

```
Query Embedding → Similarity Search (FAISS/vector DB)
                  ↓
         Retrieved Context Vectors
                  ↓
         [Gating Network] → Which context to use
                  ↓
         Combine: query + weighted_context
```

**Integration points for Voltic:**
1. `EmbeddingLayer` + FAISS index
2. Learnable "database" as large constant buffer
3. Differentiable nearest-neighbor lookup (straight-through estimator)

### Idea 4: Grouped Matrix Multiplication

Similar to MoE but with matrix groups:

```
# Group Linear - each group handles different feature subspace
Input: [batch, seq, dim]
Split dim into G groups: [batch, seq, G, dim/G]

For each group g:
    W_g = learnable_weight[dim/G, dim/G]
    Output_g = Input_g @ W_g
    
# Optional: Cross-group communication via attention
# Optional: Softmax to weight groups
```

### Idea 5: Multi-Resolution Matmul

Your idea about "space then word" tokenization + matrix ops:

```
# Hierarchical processing
Text → Word-level embeddings → [batch, num_words, word_dim]
           ↓
    Matrix multiply with word-pattern matrix
           ↓
Phrase-level → [batch, num_phrases, phrase_dim]
           ↓
    Matrix multiply with phrase-pattern matrix
           ↓
Document-level → Final representation
```

This is similar to hierarchical attention but with explicit matrix transformations at each level.

---

## Retrieval-Augmented Generation (RAG)

### Overview
RAG combines a language model with external knowledge retrieval. The model doesn't just rely on training data - it can query databases at inference time.

### Architecture
```
User Query → [Embedding Model] → query_vector
                           ↓
              [Vector Database] ← Documents (pre-embedded)
                           ↓
              Retrieved Context
                           ↓
              [Combine: query + context]
                           ↓
              [LLM] → Generated Response
```

### Voltic Integration

For implementing RAG in Voltic:

1. **Embedding Layer** (already have `Embedding`)
2. **Vector Similarity Search** - use crate like `faiss` or `qdrant` client
3. **Context Combination** - concatenate query + retrieved
4. **Generation** - use GPT/LM forward pass

### Advanced RAG Patterns (2024-25)

| Pattern | Description |
|---------|-------------|
| **GraphRAG** | Build knowledge graph, retrieve via graph traversal |
| **Hybrid Search** | Combine dense (vector) + sparse (keyword) retrieval |
| **Adaptive RAG** | RL to decide when to retrieve |
| **Self-RAG** | Model decides to retrieve or not per token |

---

## Recommendations for Voltic

Based on research, here are prioritized recommendations:

### Phase 1: Quick Wins (Low Effort, High Impact)
1. **MoE Layer** - Replace MLP in GPT with sparse MoE
2. **RAG Helper** - Embed + similarity search wrapper

### Phase 2: Medium Effort
3. **Multi-Matrix Attention** - Extend current attention with multiple Q/K/V
4. **Grouped MatMul** - Channel-wise transformations

### Phase 3: Advanced (High Effort)
5. **Hybrid Transformer-Mamba** - Interleave attention with SSM layers
6. **Neural Database** - Learnable indexed memory with differentiable lookup

---

## References

- Mamba: Linear-time Sequence Modeling with Selective State Spaces (Gu et al., 2023)
- Mamba-2: State Space Duality (2024)
- Jamba: AI21's Hybrid SSM-Attention Model (2024)
- Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al., 2021)
- Mixtral: Mixture of Experts (2024)
- GraphRAG: Microsoft's Graph-based RAG (2024)
- BlendedRAG: IBM's Hybrid Search RAG (2024)
