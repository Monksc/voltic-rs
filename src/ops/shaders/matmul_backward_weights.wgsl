const TILE: u32 = 16u;

struct Dims {
    batch: u32,
    M:     u32,
    K:     u32,
    N:     u32,
}

@group(0) @binding(0) var<storage, read>       grad_C:  array<f32>; // [batch, M, N]
@group(0) @binding(1) var<storage, read>       A:       array<f32>; // [batch, M, K]
@group(0) @binding(2) var<storage, read_write> grad_B:  array<f32>; // [batch, K, N]
@group(0) @binding(3) var<uniform>             dims:    Dims;

var<workgroup> tileGC: array<array<f32, TILE>, TILE>;
var<workgroup> tileA:  array<array<f32, TILE>, TILE>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    let batch_idx = wg_id.z;
    let row       = global_id.y; // K
    let col       = global_id.x; // N
    let local_row = local_id.y;
    let local_col = local_id.x;

    let gc_offset = batch_idx * dims.M * dims.N;
    let a_offset  = batch_idx * dims.M * dims.K;
    let gb_offset = batch_idx * dims.K * dims.N;

    var acc: f32 = 0.0;
    let num_tiles = (dims.M + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load grad_C^T tile — grad_C is [M, N], we want row=K so index via col dimension
        // grad_C[t*TILE+local_row, col]
        let gc_row = t * TILE + local_row;
        if gc_row < dims.M && col < dims.N {
            tileGC[local_row][local_col] = grad_C[gc_offset + gc_row * dims.N + col];
        } else {
            tileGC[local_row][local_col] = 0.0;
        }

        // Load A^T tile — A is [M, K], A^T is [K, M]
        // we want A[t*TILE+local_col, row]
        let a_row = t * TILE + local_col;
        if a_row < dims.M && row < dims.K {
            tileA[local_row][local_col] = A[a_offset + a_row * dims.K + row];
        } else {
            tileA[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE; k++) {
            acc += tileA[local_row][k] * tileGC[k][local_col];
        }

        workgroupBarrier();
    }

    if row < dims.K && col < dims.N {
        grad_B[gb_offset + row * dims.N + col] = acc;
    }
}
