struct Dims {
    lr:      f32,
    beta1:   f32,
    beta2:   f32,
    epsilon: f32,
    t:       u32,
    n:       u32,
}

@group(0) @binding(0) var<storage, read_write> weights:  array<f32>;
@group(0) @binding(1) var<storage, read_write> grad:     array<f32>;
@group(0) @binding(2) var<storage, read_write> momentum: array<f32>;
@group(0) @binding(3) var<storage, read_write> variance: array<f32>;
@group(0) @binding(4) var<uniform>             dims:     Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.n { return; }

    let g = grad[i];

    let m = dims.beta1 * momentum[i] + (1.0 - dims.beta1) * g;
    let v = dims.beta2 * variance[i] + (1.0 - dims.beta2) * g * g;

    momentum[i] = m;
    variance[i] = v;

    let t_f = f32(dims.t);
    let m_hat = m / (1.0 - pow(dims.beta1, t_f));
    let v_hat = v / (1.0 - pow(dims.beta2, t_f));

    weights[i] = weights[i] - dims.lr * m_hat / (sqrt(v_hat) + dims.epsilon);

    grad[i] = 0.0;
}
