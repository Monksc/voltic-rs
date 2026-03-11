struct Dims {
    outer:  u32,
    reduce: u32,
    inner:  u32,
}

@group(0) @binding(0) var<storage, read>       input:    array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<uniform>             dims:     Dims;

var<workgroup> tile: array<f32, 256>;

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
        tile[local_id.x] = input[i];
    } else {
        tile[local_id.x] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if local_id.x < stride {
            tile[local_id.x] += tile[local_id.x + stride];
        }
        workgroupBarrier();
    }

    if local_id.x == 0u {
        let p = outer_idx * dims.inner * n_chunks + inner_idx * n_chunks + chunk;
        partials[p] = tile[0];
    }
}
