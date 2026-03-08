// dL/dW = X^T @ grad_y
// X is [M, K], grad_y is [M, N], dL/dW is [K, N]

const TILE: u32 = 16u;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       X:      array<f32>;
@group(0) @binding(1) var<storage, read>       grad_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_w: array<f32>;
@group(0) @binding(3) var<uniform>             dims:   Dims;

var<workgroup> tileX:     array<array<f32, TILE>, TILE>;
var<workgroup> tileGradY: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
) {
    let k = global_id.y; // row of dL/dW
    let n = global_id.x; // col of dL/dW

    let local_row = local_id.y;
    let local_col = local_id.x;

    var acc: f32 = 0.0;
    let num_tiles = (dims.M + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        // Load X^T tile — X[m, k] accessed as X^T[k, m]
        let m = t * TILE + local_col;
        if k < dims.K && m < dims.M {
            tileX[local_row][local_col] = X[m * dims.K + k];
        } else {
            tileX[local_row][local_col] = 0.0;
        }

        // Load grad_y tile
        let m2 = t * TILE + local_row;
        if m2 < dims.M && n < dims.N {
            tileGradY[local_row][local_col] = grad_y[m2 * dims.N + n];
        } else {
            tileGradY[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE; i++) {
            acc += tileX[local_row][i] * tileGradY[i][local_col];
        }

        workgroupBarrier();
    }

    if k < dims.K && n < dims.N {
        grad_w[k * dims.N + n] = acc;
    }
}
