// Pass 2: reduce partials to final max per (outer, inner) pair

struct Dims {
    outer:  u32,
    reduce: u32,
    inner:  u32,
}

@group(0) @binding(0) var<storage, read>       partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> output:   array<f32>;
@group(0) @binding(2) var<uniform>             dims:     Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx       = gid.x;
    let n_chunks = (dims.reduce + 255u) / 256u;
    let total_out = dims.outer * dims.inner;
    if idx >= total_out { return; }

    let outer_idx = idx / dims.inner;
    let inner_idx = idx % dims.inner;

    var m: f32 = -3.402823e+38;
    for (var c = 0u; c < n_chunks; c++) {
        let p = outer_idx * dims.inner * n_chunks + inner_idx * n_chunks + c;
        m = max(m, partials[p]);
    }

    output[idx] = m;
}
