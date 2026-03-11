// Pass 2: reduce partial sums, compute mean/var, normalize, apply gamma and beta

struct Dims {
    outer:  u32,
    reduce: u32,
    inner:  u32,
    eps:    f32,
}

@group(0) @binding(0) var<storage, read>       input:         array<f32>;
@group(0) @binding(1) var<storage, read>       partials_mean: array<f32>;
@group(0) @binding(2) var<storage, read>       partials_var:  array<f32>;
@group(0) @binding(3) var<storage, read>       gamma:         array<f32>; // [reduce]
@group(0) @binding(4) var<storage, read>       beta:          array<f32>; // [reduce]
@group(0) @binding(5) var<storage, read_write> x_norm:        array<f32>; // stored for backward
@group(0) @binding(6) var<storage, read_write> output:        array<f32>;
@group(0) @binding(7) var<uniform>             dims:          Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = dims.outer * dims.reduce * dims.inner;
    if i >= total { return; }

    let inner_idx = i % dims.inner;
    let r         = (i / dims.inner) % dims.reduce;
    let outer_idx = i / (dims.reduce * dims.inner);
    let n_chunks  = (dims.reduce + 255u) / 256u;

    // Reduce partials to get mean and variance for this row
    var sum:  f32 = 0.0;
    var sum2: f32 = 0.0;
    for (var c = 0u; c < n_chunks; c++) {
        let p = outer_idx * dims.inner * n_chunks + inner_idx * n_chunks + c;
        sum  += partials_mean[p];
        sum2 += partials_var[p];
    }

    let mean = sum / f32(dims.reduce);
    let var_ = sum2 / f32(dims.reduce) - mean * mean;

    // Normalise
    let x_hat = (input[i] - mean) / sqrt(var_ + dims.eps);

    // Store normalised value for backward
    x_norm[i] = x_hat;

    // Scale and shift — gamma and beta are indexed by the reduce dimension
    output[i] = gamma[r] * x_hat + beta[r];
}
