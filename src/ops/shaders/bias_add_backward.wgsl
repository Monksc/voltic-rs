struct Dims { rows: u32, cols: u32 }

@group(0) @binding(0) var<storage, read>       grad_out:   array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_bias:  array<f32>;
@group(0) @binding(3) var<uniform>             dims:       Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= dims.cols { return; }
    var g: f32 = 0.0;
    for (var row: u32 = 0u; row < dims.rows; row++) {
        let i = row * dims.cols + col;
        grad_input[i] = grad_out[i];
        g += grad_out[i];
    }
    grad_bias[col] = g;
}
