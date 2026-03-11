// Pass 1: compute max and sum(exp(x - max)) for each (outer, inner) pair
// Results stored in partials_max and partials_sum scratch buffers

struct Dims {
    outer:  u32,
    reduce: u32,
    inner:  u32,
}

@group(0) @binding(0) var<storage, read>       input:        array<f32>;
@group(0) @binding(1) var<storage, read_write> partials_max: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials_sum: array<f32>;
@group(0) @binding(3) var<uniform>             dims:         Dims;

var<workgroup> tile_max: array<f32, 256>;
var<workgroup> tile_sum: array<f32, 256>;

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
        tile_max[local_id.x] = input[i];
    } else {
        tile_max[local_id.x] = -3.402823e+38;
    }
    workgroupBarrier();

    // Tree reduce for max
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if local_id.x < stride {
            tile_max[local_id.x] = max(tile_max[local_id.x], tile_max[local_id.x + stride]);
        }
        workgroupBarrier();
    }

    let local_max = tile_max[0];

    // Now compute exp(x - max) and sum
    if r < dims.reduce {
        let i = outer_idx * dims.reduce * dims.inner + r * dims.inner + inner_idx;
        tile_sum[local_id.x] = exp(input[i] - local_max);
    } else {
        tile_sum[local_id.x] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if local_id.x < stride {
            tile_sum[local_id.x] += tile_sum[local_id.x + stride];
        }
        workgroupBarrier();
    }

    if local_id.x == 0u {
        let p = outer_idx * dims.inner * n_chunks + inner_idx * n_chunks + chunk;
        partials_max[p] = local_max;
        partials_sum[p] = tile_sum[0];
    }
}
