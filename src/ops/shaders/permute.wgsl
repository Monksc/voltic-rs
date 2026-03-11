struct Dims {
    rank:      u32,
    shape:     array<u32, 8>, // input shape
    perm:      array<u32, 8>, // permutation
    out_shape: array<u32, 8>, // output shape
    total:     u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             dims:   Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.total { return; }

    // Convert flat output index -> nd output index
    var nd_out: array<u32, 8>;
    var remaining = i;
    for (var d = dims.rank - 1u; d < dims.rank; d--) {
        nd_out[d] = remaining % dims.out_shape[d];
        remaining = remaining / dims.out_shape[d];
    }

    // Apply inverse permutation to get nd input index
    // perm[d] = which input axis goes to output axis d
    // so input axis perm[d] = nd_out[d]
    var nd_in: array<u32, 8>;
    for (var d = 0u; d < dims.rank; d++) {
        nd_in[dims.perm[d]] = nd_out[d];
    }

    // Convert nd input index -> flat input index
    var flat_in = 0u;
    var stride  = 1u;
    for (var d = dims.rank - 1u; d < dims.rank; d--) {
        flat_in += nd_in[d] * stride;
        stride  *= dims.shape[d];
    }

    output[i] = input[flat_in];
}
