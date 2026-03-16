# Changelog

All notable changes to Voltic will be documented in this file.

## Recent Work (2026)

### Test Fix: embedding_forward_test
- Fixed failing test that was missing `test_setup()` 
- Added proper Context initialization order (init after allocate_buffers)
- Tests now pass in both parallel and serial modes

### Checkpoint Save/Restore
- Added `collect_persistent()`, `save()`, `restore()` methods to Context
- Enables saving/loading model weights to disk
- Commits: `6925ad8`

### Conv2d Backward Pass Fix
- Fixed critical bug in col2im shader where incorrect indexing caused zero gradients
- Added backward pass to `src/ops/im2col.rs` with `col2im.wgsl` shader
- Commits: `04e0a94`

### Test Infrastructure
- Added `Context::clear()` for clean GPU state between tests
- Added `test_setup()` helper that acquires lock and clears context
- All tests share global GPU context and must run serially (`--test-threads=1`)
- Commits: `e412790`, `6a9558e`

### Agent Communication
- Notified picaso-obama agent about Conv2d fix via `~/Projects/agentcommunication/messages.md`

### Code Cleanup Exploration (in progress)
- Investigating Layer trait for consistency across layers
- Checking for repetitive patterns across ops files
- Reviewed: `bias.rs`, `activation.rs`, `group_ops.rs`, `linear.rs`

---

## Historical

### v0.1.0
- Initial release
- GPU-accelerated ML library using WGSL compute shaders
- Support for: Linear, Conv2d, LayerNorm, BatchNorm, Embedding
- Optimizers: SGD, Adam
- Activations: Tanh, ReLU, Sigmoid, GELU, Softmax
- Tokenizers: BPE, WordPiece, TrieToken
- Components: VAE, MoE, RAG, NeuralDatabase, LearnableMemory, HybridMambaTransformer
