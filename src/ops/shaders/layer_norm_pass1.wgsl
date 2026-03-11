// Pass 1: compute mean and variance for each row (outer, inner) pair
// Each workgroup handles one chunk of the reduce dimension

struct Dims {
    outer:  u32,
    reduce: u32,
    inner:  u32,
}

@group(0) @binding(0) var<storage, read>       input:         array<f32>;
@group(0) @binding(1) var<storage, read_write> partials_mean: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials_var:  array<f32>;
@group(0) @binding(3) var<uniform>             dims:          Dims;

var<workgroup> tile_sum:  array<f32, 256>;
var<workgroup> tile_sum2: array<f32, 256>; // sum of squares

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id)  local_id:     vec3<u32>,
    @builtin(workgroup_id)         workgroup_id: vec3<u32>,
) {
    let n_chunks  = (dims.reduce + 255u) / 256u;
    let wg        = workgroup_id.x;
    let inner_idx = wg % dims.inner;
    let outer_idx = (wg / dims.inner) % dims.outer;
    let chunk     = wg / (dims.inner * dims.outer);

    let r = chunk * 256u + local_id.x;

    if r < dims.reduce {
        let i = outer_idx * dims.reduce * dims.inner + r * dims.inner + inner_idx;
        let v = input[i];
        tile_sum[local_id.x]  = v;
        tile_sum2[local_id.x] = v * v;
    } else {
        tile_sum[local_id.x]  = 0.0;
        tile_sum2[local_id.x] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if local_id.x < stride {
            tile_sum[local_id.x]  += tile_sum[local_id.x + stride];
            tile_sum2[local_id.x] += tile_sum2[local_id.x + stride];
        }
        workgroupBarrier();
    }

    if local_id.x == 0u {
        let p = outer_idx * dims.inner * n_chunks + inner_idx * n_chunks + chunk;
        partials_mean[p] = tile_sum[0];
        partials_var[p]  = tile_sum2[0];
    }
}
