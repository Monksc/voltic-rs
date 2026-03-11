// Layer norm backward
// grad_gamma[r] = sum over outer of grad_out[outer, r] * x_norm[outer, r]
// grad_beta[r]  = sum over outer of grad_out[outer, r]
// grad_x[i]     = gamma[r] / sqrt(var + eps) * (grad_out[i] 
//                 - mean(grad_out_row) 
//                 - x_norm[i] * mean(grad_out_row * x_norm_row))

struct Dims {
    outer:  u32,
    reduce: u32,
    inner:  u32,
    eps:    f32,
}

@group(0) @binding(0) var<storage, read>       grad_out:    array<f32>;
@group(0) @binding(1) var<storage, read>       x_norm:      array<f32>;
@group(0) @binding(2) var<storage, read>       gamma:       array<f32>;
@group(0) @binding(3) var<storage, read>       partials_var: array<f32>; // reuse from forward
@group(0) @binding(4) var<storage, read>       partials_mean: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_input:  array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_gamma:  array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_beta:   array<f32>;
@group(0) @binding(8) var<uniform>             dims:        Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = dims.outer * dims.reduce * dims.inner;
    if i >= total { return; }

    let inner_idx = i % dims.inner;
    let r         = (i / dims.inner) % dims.reduce;
    let outer_idx = i / (dims.reduce * dims.inner);
    let n_chunks  = (dims.reduce + 255u) / 256u;

    // Recompute mean and var for this row from partials
    var sum:  f32 = 0.0;
    var sum2: f32 = 0.0;
    for (var c = 0u; c < n_chunks; c++) {
        let p = outer_idx * dims.inner * n_chunks + inner_idx * n_chunks + c;
        sum  += partials_mean[p];
        sum2 += partials_var[p];
    }
    let mean = sum / f32(dims.reduce);
    let var_ = sum2 / f32(dims.reduce) - mean * mean;
    let std_ = sqrt(var_ + dims.eps);
    let n    = f32(dims.reduce);

    // Compute row-wise dot products needed for grad_x
    var sum_grad_out:       f32 = 0.0;
    var sum_grad_out_xnorm: f32 = 0.0;
    for (var k = 0u; k < dims.reduce; k++) {
        let j = outer_idx * dims.reduce * dims.inner + k * dims.inner + inner_idx;
        sum_grad_out       += grad_out[j];
        sum_grad_out_xnorm += grad_out[j] * x_norm[j];
    }

    // grad_x
    grad_input[i] = gamma[r] / std_ * (grad_out[i] - sum_grad_out / n - x_norm[i] * sum_grad_out_xnorm / n);

    // grad_gamma and grad_beta — accumulate over outer dimension
    // Only one thread per (r, inner) should write, but since inner=1 for layer norm
    // we accumulate from outer dimension
    if outer_idx == 0u {
        var gg: f32 = 0.0;
        var gb: f32 = 0.0;
        for (var o = 0u; o < dims.outer; o++) {
            let j = o * dims.reduce * dims.inner + r * dims.inner + inner_idx;
            gg += grad_out[j] * x_norm[j];
            gb += grad_out[j];
        }
        grad_gamma[r] = gg;
        grad_beta[r]  = gb;
    }
}
