// Fused softmax backward
// grad_input[i] = softmax_out[i] * (grad_out[i] - dot(grad_out_row, softmax_out_row))
// where dot is computed per (outer, inner) pair over the reduce dimension

struct Dims {
    outer:  u32,
    reduce: u32,
    inner:  u32,
}

@group(0) @binding(0) var<storage, read>       output:     array<f32>; // softmax forward output
@group(0) @binding(1) var<storage, read>       grad_out:   array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform>             dims:       Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = dims.outer * dims.reduce * dims.inner;
    if i >= total { return; }

    let inner_idx = i % dims.inner;
    let r         = (i / dims.inner) % dims.reduce;
    let outer_idx = i / (dims.reduce * dims.inner);

    // Compute dot(grad_out_row, softmax_out_row) for this (outer, inner) pair
    var dot: f32 = 0.0;
    for (var k = 0u; k < dims.reduce; k++) {
        let j = outer_idx * dims.reduce * dims.inner + k * dims.inner + inner_idx;
        dot += grad_out[j] * output[j];
    }

    let s = output[i];
    grad_input[i] = s * (grad_out[i] - dot);
}
