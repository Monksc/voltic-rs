struct Dims { outer: u32, reduce: u32, inner: u32 }

@group(0) @binding(0) var<storage, read>       grad_out:   array<f32>;
@group(0) @binding(1) var<storage, read>       input:      array<f32>;
@group(0) @binding(2) var<storage, read>       rhs:        array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_rhs:   array<f32>;
@group(0) @binding(5) var<uniform>             dims:       Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= dims.outer * dims.inner { return; }

    let inner_idx = col % dims.inner;
    let outer_idx = col / dims.inner;

    var g: f32 = 0.0;
    for (var r = 0u; r < dims.reduce; r++) {
        let i       = outer_idx * dims.reduce * dims.inner + r * dims.inner + inner_idx;
        grad_input[i] = grad_out[i] * rhs[col];
        g            += grad_out[i] * input[i];
    }
    grad_rhs[col] = g;
}
