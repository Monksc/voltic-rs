const TILE: u32 = 16u;

struct Dims {
    batch: u32,
    M:     u32,
    K:     u32,
    N:     u32,
    rhs_batched: u32,
}

@group(0) @binding(0) var<storage, read>       grad_C:  array<f32>; // [batch, M, N]
@group(0) @binding(1) var<storage, read>       B:       array<f32>; // [batch, K, N]
@group(0) @binding(2) var<storage, read_write> grad_A:  array<f32>; // [batch, M, K]
@group(0) @binding(3) var<uniform>             dims:    Dims;

var<workgroup> tileGC: array<array<f32, TILE>, TILE>;
var<workgroup> tileB:  array<array<f32, TILE>, TILE>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    let batch_idx = wg_id.z;
    let row       = global_id.y; // M
    let col       = global_id.x; // K
    let local_row = local_id.y;
    let local_col = local_id.x;

    let gc_offset = batch_idx * dims.M * dims.N;
    let b_offset = batch_idx * dims.K * dims.N * dims.rhs_batched;
    let ga_offset = batch_idx * dims.M * dims.K;

    var acc: f32 = 0.0;
    let num_tiles = (dims.N + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load grad_C tile [M, N] — col dimension is N
        let gc_col = t * TILE + local_col;
        if row < dims.M && gc_col < dims.N {
            tileGC[local_row][local_col] = grad_C[gc_offset + row * dims.N + gc_col];
        } else {
            tileGC[local_row][local_col] = 0.0;
        }

        // Load B^T tile — B is [K, N], B^T is [N, K]
        // we want B[col, t*TILE+local_row] = B[col * N + t*TILE + local_row]
        let b_col = t * TILE + local_row;
        if col < dims.K && b_col < dims.N {
            tileB[local_row][local_col] = B[b_offset + col * dims.N + b_col];
        } else {
            tileB[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE; k++) {
            acc += tileGC[local_row][k] * tileB[k][local_col];
        }

        workgroupBarrier();
    }

    if row < dims.M && col < dims.K {
        grad_A[ga_offset + row * dims.K + col] = acc;
    }
}
