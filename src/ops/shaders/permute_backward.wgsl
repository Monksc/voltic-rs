// Backward: apply inverse permutation to grad_out to get grad_in
// If forward was perm [0,2,1,3], backward is the inverse perm

struct Dims {
    rank:      u32,
    shape:     array<u32, 8>, // original input shape
    perm:      array<u32, 8>, // original forward permutation
    out_shape: array<u32, 8>, // output shape of forward pass
    total:     u32,
}

@group(0) @binding(0) var<storage, read>       grad_out:   array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(2) var<uniform>             dims:       Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.total { return; }

    // Convert flat input index -> nd input index
    var nd_in: array<u32, 8>;
    var remaining = i;
    for (var d = dims.rank - 1u; d < dims.rank; d--) {
        nd_in[d] = remaining % dims.shape[d];
        remaining = remaining / dims.shape[d];
    }

    // Apply forward permutation to get nd output index
    // output axis d = input axis perm[d]
    var nd_out: array<u32, 8>;
    for (var d = 0u; d < dims.rank; d++) {
        nd_out[d] = nd_in[dims.perm[d]];
    }

    // Convert nd output index -> flat output index
    var flat_out = 0u;
    var stride   = 1u;
    for (var d = dims.rank - 1u; d < dims.rank; d--) {
        flat_out += nd_out[d] * stride;
        stride   *= dims.out_shape[d];
    }

    grad_input[i] = grad_out[flat_out];
}
