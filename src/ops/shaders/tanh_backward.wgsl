@group(0) @binding(0) var<storage, read>       output:     array<f32>; // tanh output, not input
@group(0) @binding(1) var<storage, read>       grad_out:   array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&grad_input) {
        grad_input[i] = grad_out[i] * (1.0 - output[i] * output[i]);
    }
}
