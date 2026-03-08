const SQRT_2_OVER_PI: f32 = 0.7978845608;
const COEFF: f32 = 0.044715;

@group(0) @binding(0) var<storage, read>       input:      array<f32>;
@group(0) @binding(1) var<storage, read>       grad_out:   array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&grad_input) {
        let x    = input[i];
        let c    = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        let tanh_c  = tanh(c);
        let sech2_c = 1.0 - tanh_c * tanh_c;
        let dc_dx   = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x * x);

        let dgelu = 0.5 * (1.0 + tanh_c) + 0.5 * x * sech2_c * dc_dx;
        grad_input[i] = grad_out[i] * dgelu;
    }
}
