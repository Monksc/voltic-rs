// Pass 2: reduce partial max/sum and write final softmax output

struct Dims {
    outer:  u32,
    reduce: u32,
    inner:  u32,
}

@group(0) @binding(0) var<storage, read>       input:        array<f32>;
@group(0) @binding(1) var<storage, read>       partials_max: array<f32>;
@group(0) @binding(2) var<storage, read>       partials_sum: array<f32>;
@group(0) @binding(3) var<storage, read_write> output:       array<f32>;
@group(0) @binding(4) var<uniform>             dims:         Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = dims.outer * dims.reduce * dims.inner;
    if i >= total { return; }

    let inner_idx = i % dims.inner;
    let outer_idx = i / (dims.reduce * dims.inner);
    let n_chunks  = (dims.reduce + 255u) / 256u;

    // Reduce partials to get final max and sum for this (outer, inner) pair
    var row_max: f32 = -3.402823e+38;
    var row_sum: f32 = 0.0;
    for (var c = 0u; c < n_chunks; c++) {
        let p = outer_idx * dims.inner * n_chunks + inner_idx * n_chunks + c;
        row_max = max(row_max, partials_max[p]);
        row_sum += partials_sum[p];
    }

    // Apply softmax
    output[i] = exp(input[i] - row_max) / row_sum;
}
