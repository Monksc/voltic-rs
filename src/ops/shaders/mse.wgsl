struct Dims {
    n: u32,
}

@group(0) @binding(0) var<storage, read>       y_pred: array<f32>;
@group(0) @binding(1) var<storage, read>       y_true: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>             dims:   Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < dims.n {
        let diff = y_pred[i] - y_true[i];
        out[i] = diff * diff;
    }
}
