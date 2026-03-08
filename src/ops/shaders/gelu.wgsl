const SQRT_2_OVER_PI: f32 = 0.7978845608;
const COEFF: f32 = 0.044715;

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&output) {
        let x = input[i];
        let c = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        output[i] = 0.5 * x * (1.0 + tanh(c));
    }
}
