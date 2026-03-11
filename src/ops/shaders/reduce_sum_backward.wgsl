// Backward: broadcast grad_out back to input shape
// grad_in[outer, r, inner] = grad_out[outer, inner]

struct Dims {
    outer:  u32,
    reduce: u32,
    inner:  u32,
}

@group(0) @binding(0) var<storage, read>       grad_out: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_in:  array<f32>;
@group(0) @binding(2) var<uniform>             dims:     Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = dims.outer * dims.reduce * dims.inner;
    if i >= total { return; }

    let inner_idx = i % dims.inner;
    let outer_idx = i / (dims.reduce * dims.inner);
    let out_idx   = outer_idx * dims.inner + inner_idx;

    grad_in[i] = grad_out[out_idx];
}
