# Test Suite

This document lists all test cases in the Voltic codebase.

## Running Tests

```bash
cargo test -- --test-threads=1
```

> Tests share a global GPU context and must run serially.

## Test Categories

### Core Operations (src/lib.rs)

| Test Name | Description |
|-----------|-------------|
| `simple_matrix_multiply` | Basic 4x2 × 2x1 matrix multiplication |
| `simple_mse` | MSE loss forward pass |
| `mse_backward` | MSE loss backward pass gradient check |
| `single_layer` | Single linear layer training with MSE |
| `xor_sgd` | XOR problem with SGD optimizer |
| `xor_adam_tanh` | XOR with Adam optimizer and tanh activation |
| `xor_adam_softmax` | XOR with Adam optimizer and softmax |
| `softmax_simple_test` | Softmax along dimension 1 |

### GPT/Transformer (src/lib.rs)

| Test Name | Description |
|-----------|-------------|
| `gpt_forward_16tokens` | GPT forward pass with 16 tokens, single batch |
| `gpt_forward_16tokens_batched` | GPT with 16 batches of 16 tokens |

### Group Operations (src/lib.rs)

| Test Name | Description |
|-----------|-------------|
| `group_mul_test` | Group multiplication: [1,2,3,4,5,6] with group_size=3 → [6, 120] |
| `group_add_test` | Group addition: [1,0,1,1,0,0] with group_size=3 → [2, 1] |
| `group_max_test` | Group max: [1,5,3,4,2,6] with group_size=3 → [5, 6] |

### Convolutional Layers (src/lib.rs)

| Test Name | Description |
|-----------|-------------|
| `conv2d_forward_test` | Conv2d forward pass with 4×4 input |
| `conv2d_training_test` | Conv2d forward + backward with gradients |
| `transposed_conv2d_test` | TransposedConv2d forward pass |

### Image Operations (src/lib.rs)

| Test Name | Description |
|-----------|-------------|
| `upsample_test` | Upsample operation |
| `downsample_test` | Downsample operation |
| `downsample_actual_test` | Actual downsample with reshape/permute |

### Layers (src/lib.rs)

| Test Name | Description |
|-----------|-------------|
| `embedding_forward_test` | Embedding layer forward pass |
| `layer_norm_forward_test` | LayerNorm forward pass |
| `batch_norm_test` | BatchNorm forward pass |

### Model Architectures (src/lib.rs)

| Test Name | Description |
|-----------|-------------|
| `vae_forward_test` | VAE forward pass |
| `moe_forward_test` | Mixture of Experts forward pass |
| `hybrid_mamba_transformer_test` | Hybrid Mamba-Transformer forward pass |

### Utilities (src/lib.rs)

| Test Name | Description |
|-----------|-------------|
| `learnable_memory_test` | LearnableMemory component |
| `neural_database_test` | NeuralDatabase component |
| `rag_helper_test` | RAG (Retrieval-Augmented Generation) helper |
| `tokenizer_bpe_test` | BPE tokenizer training and encoding |
| `tokenizer_wordpiece_test` | WordPiece tokenizer |
| `tokenizer_triettoken_test` | TrieToken tokenizer |

### Checkpointing (src/lib.rs)

| Test Name | Description |
|-----------|-------------|
| `save_restore_test` | Save/restore persistent variables to disk |

### Shape Operations (src/ops/broadcast_shape.rs)

| Test Name | Description |
|-----------|-------------|
| `exact_match` | Broadcast shapes that match exactly |
| `trailing_broadcast` | Broadcasting to trailing dimensions |
| `scalar_broadcast` | Scalar broadcasting |
| `rightmost_ambiguous` | Rightmost dimension matching |
| `explicit_dims` | Explicit broadcast dimensions |
| `incompatible_shapes` | Error on incompatible shapes |

---

## Test Infrastructure

### test_setup()

All GPU tests use the `test_setup()` helper:

```rust
fn test_setup() -> std::sync::MutexGuard<'static, ()> {
    let lock = TEST_LOCK.lock().unwrap();
    Context::clear();
    lock
}
```

- Acquires a global lock to ensure serial execution
- Calls `Context::clear()` to reset GPU state between tests

### Context::clear()

Clears all state in the global Context:
- Clears operations graph
- Clears shapes
- Clears GPU buffers
- Clears persistent variables
- Resets the GpuContext

---

## Test Status

All 38 tests currently pass.

To run a specific test:
```bash
cargo test xor_sgd -- --test-threads=1
```
