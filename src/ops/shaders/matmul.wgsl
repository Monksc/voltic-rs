const TILE: u32 = 16u;

struct Dims {
    batch: u32,
    M:     u32,
    K:     u32,
    N:     u32,
    rhs_batched: u32,
}

@group(0) @binding(0) var<storage, read>       A:    array<f32>; // [batch, M, K]
@group(0) @binding(1) var<storage, read>       B:    array<f32>; // [batch, K, N]
@group(0) @binding(2) var<storage, read_write> C:    array<f32>; // [batch, M, N]
@group(0) @binding(3) var<uniform>             dims: Dims;

var<workgroup> tileA: array<array<f32, TILE>, TILE>;
var<workgroup> tileB: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id:  vec3<u32>,
    @builtin(local_invocation_id)  local_id:   vec3<u32>,
    @builtin(workgroup_id)         wg_id:      vec3<u32>,
) {
    let batch_idx = wg_id.z;
    let row       = global_id.y;
    let col       = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    let a_offset = batch_idx * dims.M * dims.K;
    let b_offset = batch_idx * dims.K * dims.N * dims.rhs_batched;
    let c_offset = batch_idx * dims.M * dims.N;

    var acc: f32 = 0.0;
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        let a_col = t * TILE + local_col;
        if row < dims.M && a_col < dims.K {
            tileA[local_row][local_col] = A[a_offset + row * dims.K + a_col];
        } else {
            tileA[local_row][local_col] = 0.0;
        }

        let b_row = t * TILE + local_row;
        if b_row < dims.K && col < dims.N {
            tileB[local_row][local_col] = B[b_offset + b_row * dims.N + col];
        } else {
            tileB[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE; k++) {
            acc += tileA[local_row][k] * tileB[k][local_col];
        }

        workgroupBarrier();
    }

    if row < dims.M && col < dims.N {
        C[c_offset + row * dims.N + col] = acc;
    }
}
