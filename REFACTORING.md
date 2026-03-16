# Voltic Code Cleanup Research

Research conducted on repetitive patterns that could benefit from macros or helper functions.

---

## Executive Summary

The voltic codebase has several well-established macro patterns in `src/ops/` but there are still significant opportunities for abstraction in both the ops and layers modules.

---

## 1. Similar Struct Definitions Across Layers

### Pattern Found

All layer structs follow the same pattern: store configuration as primitives, store trainable weights as `Option<Var>`.

| File | Lines | Struct Fields |
|------|-------|---------------|
| `src/layers/linear.rs` | 3-8 | `out_features`, `use_bias`, `weights: Option<Var>`, `bias: Option<Var>` |
| `src/layers/conv2d.rs` | 3-10 | `out_channels`, `kernel_size`, `stride`, `padding`, `weights: Option<Var>`, `bias: Option<Var>` |
| `src/layers/embedding.rs` | 3-7 | `vocab_size`, `d_model`, `weights: Option<Var>` |
| `src/layers/batch_norm.rs` | 3-12 | `num_features`, `_momentum`, `_epsilon`, `gamma`, `beta`, `running_mean`, `running_var`, `training` |
| `src/layers/layer_norm.rs` | 3-8 | `d_model`, `eps`, `gamma: Option<Var>`, `beta: Option<Var>` |

### Suggested Macro: `impl_layer!`

**Location:** New file `src/layers/macros.rs` or extend `src/layers/mod.rs`

**How it would work:**

```rust
macro_rules! impl_layer {
    (
        $( #[$attr:meta] )*
        $name:ident,
        {
            $( $field:ident: $ty:ty, )*
        },
        [$( $param:ident ),*]
    ) => {
        // Generate the struct with config fields and Option<Var> params
        // Generate builder pattern methods
        // Generate parameters() method
    };
}
```

**Example usage:**
```rust
impl_layer!(
    Linear,
    {
        out_features: u32,
        use_bias: bool,
    },
    [weights, bias]
);
```

---

## 2. Similar Method Patterns (new(), init(), forward(), parameters())

### 2a. The `parameters()` Method - Identical Across ALL Layers

**Files and lines:**
- `src/layers/linear.rs:47-56`
- `src/layers/conv2d.rs:127-136`
- `src/layers/embedding.rs:47-52`
- `src/layers/batch_norm.rs:100-115`
- `src/layers/layer_norm.rs:89-98`

**The code (virtually identical):**
```rust
pub fn parameters(&self) -> Vec<&Var> {
    let mut params = vec![];
    if let Some(w) = &self.weights {
        params.push(w);
    }
    if let Some(b) = &self.bias {
        params.push(b);
    }
    // or for LayerNorm:
    if let Some(g) = &self.gamma {
        params.push(g);
    }
    if let Some(b) = &self.beta {
        params.push(b);
    }
    params
}
```

### Suggested Helper: `make_parameters!` Macro

**Location:** `src/layers/macros.rs`

**How it would work:**
```rust
macro_rules! make_parameters {
    ($self:ident, [$( $param:ident ),*]) => {{
        let mut params = vec![];
        $( if let Some(p) = &$self.$param { params.push(p); } )*
        params
    }};
}

// Usage in layers:
pub fn parameters(&self) -> Vec<&Var> {
    make_parameters!(self, [weights, bias])
}
```

---

### 2b. The `init()` Method - Highly Repetitive

**Files and lines:**
- `src/layers/linear.rs:58-71` (xavier for weights, zeros for bias)
- `src/layers/conv2d.rs:112-125` (xavier for weights, zeros for bias)
- `src/layers/embedding.rs:54-60` (xavier for weights)
- `src/layers/batch_norm.rs:80-98` (ones for gamma/running_var, zeros for beta/running_mean)
- `src/layers/layer_norm.rs:77-87` (ones for gamma, zeros for beta)

**The pattern:**
```rust
pub fn init(&self) -> Result<()> {
    if let Some(w) = &self.weights {
        let shape = Context::shape(w.id()).ok_or(VolticError::EmptyShape)?;
        let fan_in = shape[0]; // or shape[1] for conv
        let data = init::xavier_flat(fan_in);
        w.load(vec![data])?;
    }
    if let Some(b) = &self.bias {
        let shape = Context::shape(b.id()).ok_or(VolticError::EmptyShape)?;
        let n: u32 = shape.iter().product();
        b.load(vec![vec![0.0; n as usize]])?;
    }
    Ok(())
}
```

### Suggested Helper: `impl_layer_init!` Macro

**How it would work:**
```rust
macro_rules! impl_layer_init {
    (
        $self:ident,
        weights: xavier_flat($fan_in:expr),
        bias: zeros
    ) => {{
        if let Some(w) = &$self.weights {
            let data = init::xavier_flat($fan_in);
            w.load(vec![data])?;
        }
        if let Some(b) = &$self.bias {
            let shape = Context::shape(b.id()).ok_or(VolticError::EmptyShape)?;
            let n: u32 = shape.iter().product();
            b.load(vec![vec![0.0; n as usize]])?;
        }
        Ok(())
    }};
}
```

---

## 3. Repetitive Shader Creation Code

### Pattern Found

Almost every op file contains identical shader/pipeline creation code:

**Repeated code block (appears in 10+ files):**
```rust
fn create_pipelines(
    &self,
    device: &wgpu::Device,
) -> Vec<(&'static str, wgpu::ComputePipeline)> {
    let make = |label, src: &'static str| {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    // ... use make() for each pipeline
}
```

**Files with this pattern:**
- `src/ops/matmul.rs:105-118` (lines 105-118)
- `src/ops/embedding_op.rs:58-71` (lines 58-71)
- `src/ops/layer_norm_op.rs:85-98` (lines 85-98)
- `src/ops/softmax.rs:59-72` (lines 59-72)
- `src/ops/mse.rs:52-77` (lines 52-77 - slightly different, manually expands both shaders)
- `src/ops/bias.rs:60-73, 263-276` (multiple locations)
- `src/ops/reduce.rs:91-104` (lines 91-104)
- `src/ops/broadcast.rs:123-136` (lines 123-136)

### Suggested Helper: `make_pipeline!` Macro

**Location:** `src/ops/macros.rs` (new file)

**How it would work:**
```rust
macro_rules! make_pipeline {
    ($device:ident, $label:literal, $shader:expr) => {{
        let shader = $device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some($label),
            source: wgpu::ShaderSource::Wgsl($shader.into()),
        });
        $device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some($label),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }};
}

// Usage in create_pipelines:
vec![
    ("my_op", make_pipeline!(device, "my_op", MY_SHADER)),
]
```

---

## 4. Similar Buffer Allocation Patterns

### Pattern Found

The dims buffer creation is repeated in every op's `forward_gpu()` and `backward()`:

**Repeated code (appears in every op):**
```rust
let dims = SomeDims { /* fields */ };
let dims_buf = ctx
    .device
    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(concat!($op_name, "_dims")),
        contents: bytemuck::bytes_of(&dims),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
```

**Examples:**
- `src/ops/matmul.rs:164-170` (MatMulDims)
- `src/ops/embedding_op.rs:110-116` (EmbeddingDims)
- `src/ops/layer_norm_op.rs:134-140` (LayerNormDims)
- `src/ops/softmax.rs:100-106` (SoftmaxDims)
- `src/ops/mse.rs:106-113` (MseDims)
- `src/ops/bias.rs:110-116` (BiasDims)
- `src/ops/reduce.rs:137-143` (ReduceDims)

### Suggested Helper: `create_dims_buffer!` Macro

**How it would work:**
```rust
macro_rules! create_dims_buffer {
    ($ctx:ident, $label:literal, $dims:expr) => {{
        $ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some($label),
            contents: bytemuck::bytes_of(&$dims),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }};
}
```

---

## 5. Bind Group Creation - Extremely Repetitive

### Pattern Found

Every `forward_gpu()` and `backward()` has nearly identical bind group creation:

**Repeated code (in ~15+ locations):**
```rust
let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some(concat!($op_name, "_bind_group")),
    layout: &pipeline.get_bind_group_layout(0),
    entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: buf0.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: buf1.as_entire_binding() },
        // ... more entries
    ],
});
```

### Suggested Helper: `make_bind_group!` Macro

**How it would work:**
```rust
macro_rules! make_bind_group {
    ($ctx:ident, $label:literal, $pipeline:ident, [$( $buf:expr ),*]) => {{
        $ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some($label),
            layout: &$pipeline.get_bind_group_layout(0),
            entries: &[
                $(
                    wgpu::BindGroupEntry {
                        binding: $buf.0,
                        resource: $buf.1.as_entire_binding(),
                    },
                )*
            ],
        })
    }};
}

// Usage:
// make_bind_group!(ctx, "matmul_bg", pipeline, 
//     [(0, lhs_buf), (1, rhs_buf), (2, out_buf), (3, dims_buf)])
```

---

## 6. Existing Macros That Could Be Extended

### 6a. `impl_activation!` (src/ops/activation.rs:4-182)

**What it does:** Creates full Op implementations for activation functions.

**Could extend to:**
- Support for ops that need different numbers of buffers in `buffers_needed`
- Add support for custom dispatch calculations beyond simple `n.div_ceil(256)`

### 6b. `impl_reduce_op!` (src/ops/reduce.rs:14-302)

**Could extend to:**
- LayerNormOp and SoftmaxOp follow similar patterns but aren't using this macro
- Could add a generic "multi-pass reduction" macro that LayerNorm, Softmax, Reduce could all use

### 6c. `impl_binary_op!` (src/ops/broadcast.rs:75-452) and `impl_bias_op!` (src/ops/bias.rs:19-220)

**Observation:** These are excellent examples of the macro pattern. The `impl_binary_op!` in particular handles complex backward scenarios.

**Could extend to:**
- Could be unified - both create 2-input 1-output ops with similar structure
- Could create a higher-level `impl_2input_op!` that handles both broadcast and non-broadcast variants

### 6d. `impl_group_op!` (src/ops/group_ops.rs:6-145)

**Could extend to:**
- Very simple macro, could be combined with similar simple ops like `impl_scalar_op!` in bias.rs

---

## 7. Additional Opportunities

### 7a. Var Trait Methods in var.rs

**Pattern:** Similar method pairs like `argmax` and `sample_with_temperature` (lines 87-181) share significant code for computing outer/reduce/inner from shape.

**Lines:** 96-100 and 136-140 are nearly identical:
```rust
let outer: usize = shape[..axis].iter().map(|&x| x as usize).product();
let reduce: usize = shape[axis] as usize;
let inner: usize = shape[axis + 1..].iter().map(|&x| x as usize).product();
```

**Suggested:** Helper function `compute_reduce_dims(shape: &[u32], axis: usize) -> (usize, usize, usize)`

### 7b. Error Handling Patterns

The pattern `Context::shape(x.id()).ok_or(VolticError::EmptyShape)?` appears frequently (~15+ times).

**Suggested:** Helper method on Context or extension trait:
```rust
trait VarShape {
    fn expect_shape(&self) -> Result<Vec<u32>>;
}
impl VarShape for Var {
    fn expect_shape(&self) -> Result<Vec<u32>> {
        Context::shape(self.id()).ok_or(VolticError::EmptyShape)
    }
}
```

### 7c. Validation Patterns

Dimension validation appears repeatedly:
- `src/layers/conv2d.rs:41-46` - checks shape.len() == 4
- `src/layers/batch_norm.rs:36-41` - checks shape.len() == 4  
- `src/layers/embedding.rs:21-26` - checks shape.len() == 1

**Suggested:** `validate_ndim!` macro

---

## Summary of Recommended Actions

| Priority | Suggestion | Est. Lines Saved |
|----------|------------|------------------|
| High | `make_parameters!` macro for layers | ~40 lines |
| High | `make_pipeline!` helper for ops | ~80 lines |
| High | `create_dims_buffer!` macro | ~50 lines |
| Medium | `impl_layer!` macro for struct + builder + parameters | ~150 lines |
| Medium | Extend `impl_reduce_op!` to cover LayerNorm/Softmax | ~200 lines |
| Low | Var dimension helpers | ~30 lines |

The most impactful would be creating a shared `src/ops/macros.rs` with pipeline and buffer creation helpers, which would reduce approximately 150-200 lines of near-duplicate code across the ops modules.
