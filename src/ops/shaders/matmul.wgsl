const TILE: u32 = 16u;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       A:    array<f32>;
@group(0) @binding(1) var<storage, read>       B:    array<f32>;
@group(0) @binding(2) var<storage, read_write> C:    array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

var<workgroup> tileA: array<array<f32, TILE>, TILE>;
var<workgroup> tileB: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id)  global_id:    vec3<u32>,
    @builtin(local_invocation_id)   local_id:     vec3<u32>,
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var acc: f32 = 0.0;
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        // Load tile of A
        let a_col = t * TILE + local_col;
        if row < dims.M && a_col < dims.K {
            tileA[local_row][local_col] = A[row * dims.K + a_col];
        } else {
            tileA[local_row][local_col] = 0.0;
        }

        // Load tile of B
        let b_row = t * TILE + local_row;
        if b_row < dims.K && col < dims.N {
            tileB[local_row][local_col] = B[b_row * dims.N + col];
        } else {
            tileB[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE; k++) {
            acc += tileA[local_row][k] * tileB[k][local_col];
        }

        workgroupBarrier();
    }

    if row < dims.M && col < dims.N {
        C[row * dims.N + col] = acc;
    }
}
