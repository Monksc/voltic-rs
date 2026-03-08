// dL/dX = grad_y @ W^T
// grad_y is [M, N], W is [K, N], dL/dX is [M, K]

const TILE: u32 = 16u;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       grad_y: array<f32>;
@group(0) @binding(1) var<storage, read>       W:      array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_x: array<f32>;
@group(0) @binding(3) var<uniform>             dims:   Dims;

var<workgroup> tileGradY: array<array<f32, TILE>, TILE>;
var<workgroup> tileW:     array<array<f32, TILE>, TILE>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
) {
    let m = global_id.y; // row of dL/dX
    let k = global_id.x; // col of dL/dX

    let local_row = local_id.y;
    let local_col = local_id.x;

    var acc: f32 = 0.0;
    let num_tiles = (dims.N + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        // Load grad_y tile
        let n = t * TILE + local_col;
        if m < dims.M && n < dims.N {
            tileGradY[local_row][local_col] = grad_y[m * dims.N + n];
        } else {
            tileGradY[local_row][local_col] = 0.0;
        }

        // Load W^T tile — W[k, n] accessed as W^T[n, k]
        let n2 = t * TILE + local_row;
        if n2 < dims.N && k < dims.K {
            tileW[local_row][local_col] = W[k * dims.N + n2];
        } else {
            tileW[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE; i++) {
            acc += tileGradY[local_row][i] * tileW[i][local_col];
        }

        workgroupBarrier();
    }

    if m < dims.M && k < dims.K {
        grad_x[m * dims.K + k] = acc;
    }
}
