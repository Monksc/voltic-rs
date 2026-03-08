# Voltic

A GPU-accelerated machine learning library written in Rust, built on [wgpu](https://github.com/gfx-rs/wgpu).

Voltic runs compute entirely on the GPU via WGSL compute shaders — no CUDA, no platform lock-in. It works on Vulkan, Metal, DX12, and WebGPU.

```
epoch     0 — loss: 0.261748
epoch    25 — loss: 0.249146
epoch    50 — loss: 0.235033
...
epoch   250 — loss: 0.000343

XOR predictions:
  0 XOR 0 = 0.0183  (expected 0.0)
  1 XOR 0 = 0.9821  (expected 1.0)
  0 XOR 1 = 0.9819  (expected 1.0)
  1 XOR 1 = 0.0204  (expected 0.0)
```

---

## Features

- **Pure GPU execution** — forward and backward passes run entirely as WGSL compute shaders
- **Automatic differentiation** — reverse-mode autograd through a static operation graph
- **Tiled matrix multiplication** — 16×16 workgroup tiling for forward, weight, and input gradients
- **Activations** — Tanh, ReLU, Sigmoid, GELU (all with analytic backward shaders)
- **Layers** — `Linear` (with optional bias)
- **Losses** — MSE
- **Optimizers** — SGD
- **Cross-platform** — runs on any backend wgpu supports (Vulkan, Metal, DX12, WebGPU)

---

## Quick Start

```toml
[dependencies]
voltic = { path = "." }
```

### XOR in ~0.14s

```rust
use voltic::{Context, Linear, Sgd, Var};

fn main() {
    Context::init_gpu().unwrap();

    let x      = Var::with_shape(vec![4, 2]);
    let y_true = Var::with_shape(vec![4, 1]);

    // Build graph
    let h1     = Linear::new(8).forward(&x).unwrap().tanh().unwrap();
    let y_pred = Linear::new(1).forward(&h1).unwrap();
    let loss   = y_pred.mse(y_true).unwrap();

    Context::allocate_buffers().unwrap();

    x.load(vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ]).unwrap();
    y_true.load(vec![
        vec![0.0], vec![1.0], vec![1.0], vec![0.0],
    ]).unwrap();

    Context::prepare().unwrap();
    let mut sgd = Sgd::new(0.1);

    for _ in 0..250 {
        Context::run().unwrap();
        Context::backward().unwrap();
        sgd.step().unwrap();
    }

    let preds = y_pred.to_cpu().unwrap();
    println!("0 XOR 0 = {:.4}", preds[0]);
    println!("1 XOR 0 = {:.4}", preds[1]);
    println!("0 XOR 1 = {:.4}", preds[2]);
    println!("1 XOR 1 = {:.4}", preds[3]);
}
```

---

## Architecture

Voltic uses a **define-then-run** execution model. You build a static computation graph by calling ops on `Var` handles, then run it repeatedly in a training loop.

```
Var ops  →  Context (graph + shapes)  →  GpuContext (buffers + pipelines)
```

| Concept | Description |
|---|---|
| `Var` | A handle to a tensor. Owns an `ID` and registers ops into the global `Context`. |
| `Context` | Global singleton (`RwLock`). Stores the operation graph, shapes, and the `GpuContext`. |
| `GpuContext` | Owns the wgpu `Device`, `Queue`, `CommandEncoder`, and all GPU buffers and pipelines. |
| `Op` trait | Each operation implements `forward_gpu`, `backward`, `buffers_needed`, and `create_pipelines`. |
| `BufferKey` | `(ID, &'static str)` — indexes into `training_buffers` for grad/momentum/variance slots. |

### Training lifecycle

```
Context::init_gpu()          // acquire device + queue
Context::allocate_buffers()  // create GPU buffers for all vars and grad slots
var.load(data)               // upload initial data
Context::prepare()           // compile and cache all pipelines

loop {
    Context::run()           // forward pass — dispatches all ops
    Context::backward()      // backward pass — dispatches ops in reverse
    optimizer.step()         // update weights in-place on GPU
}
```

---

## Operations

### Tensor ops (via `Var`)

| Method | Description |
|---|---|
| `mat_mul(rhs)` | Matrix multiply — tiled 16×16 WGSL kernel |
| `mse(y_true)` | Mean squared error loss |
| `tanh()` | Element-wise tanh |
| `relu()` | Element-wise ReLU |
| `sigmoid()` | Element-wise sigmoid |
| `gelu()` | Element-wise GELU (tanh approximation) |
| `bias_add(b)` | Row-broadcast add |
| `bias_sub(b)` | Row-broadcast subtract |
| `bias_mul(b)` | Row-broadcast multiply |
| `bias_div(b)` | Row-broadcast divide |
| `scalar_add(s)` | Add scalar |
| `scalar_sub(s)` | Subtract scalar |
| `scalar_mul(s)` | Multiply by scalar |
| `scalar_div(s)` | Divide by scalar |

All ops with trainable inputs implement analytic backward passes.

### Layers

```rust
Linear::new(out_features)          // weights initialised with Xavier uniform
Linear::new(out_features).bias(false)
```

### Optimizers

```rust
Sgd::new(lr)
sgd.update_lr(new_lr);
```

---

## Weight Initialisation

| Function | Use case |
|---|---|
| `init::xavier_flat(n)` | Default — tanh / sigmoid activations |
| `init::he(fan_in)` | ReLU activations |
| `init::zeros()` | Biases |

All initialisers use a fixed seed (`0xD42`) for reproducibility.

---

## Running the Tests

```bash
cargo test -- --test-threads=1
```

> Tests share a global GPU context and must run serially.

Current test suite: `simple_matrix_multiply`, `simple_mse`, `mse_backward`, `single_layer`, `xor`.

---

## Dependencies

| Crate | Purpose |
|---|---|
| [`wgpu`](https://crates.io/crates/wgpu) `27` | GPU abstraction and compute pipeline |
| [`bytemuck`](https://crates.io/crates/bytemuck) `1.25` | Safe cast between Rust types and GPU byte buffers |
| [`pollster`](https://crates.io/crates/pollster) `0.4` | Minimal async executor for wgpu init |
| [`rand`](https://crates.io/crates/rand) `0.10` | Weight initialisation |

---

## Roadmap

- [ ] Adam / AdamW optimizer
- [ ] Gradient zeroing (`zero_grads()`)
- [ ] Cross-entropy loss + softmax
- [ ] Inference mode (`no_grad`)
- [ ] Layer normalisation
- [ ] Minibatch data loader
- [ ] WebAssembly / WebGPU target

---

