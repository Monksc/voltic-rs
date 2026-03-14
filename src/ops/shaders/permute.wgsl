struct Dims {
    rank:      u32,
    total:     u32,
    _pad0:     u32,
    _pad1:     u32,
    shape:     array<vec4<u32>, 2>,  // input shape
    perm:      array<vec4<u32>, 2>,  // permutation
    out_shape: array<vec4<u32>, 2>,  // output shape
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             dims:   Dims;

fn get_shape(i: u32) -> u32 {
    return dims.shape[i / 4u][i % 4u];
}

fn get_perm(i: u32) -> u32 {
    return dims.perm[i / 4u][i % 4u];
}

fn get_out_shape(i: u32) -> u32 {
    return dims.out_shape[i / 4u][i % 4u];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.total { return; }

    // flat output index -> nd output index
    var nd_out: array<u32, 8>;
    var remaining = i;
    for (var d = dims.rank - 1u; d < dims.rank; d--) {
        nd_out[d] = remaining % get_out_shape(d);
        remaining = remaining / get_out_shape(d);
    }

    // perm[d] = which input axis goes to output axis d
    // so nd_in[perm[d]] = nd_out[d]
    var nd_in: array<u32, 8>;
    for (var d = 0u; d < dims.rank; d++) {
        nd_in[get_perm(d)] = nd_out[d];
    }

    // nd input index -> flat input index
    var flat_in = 0u;
    var stride  = 1u;
    for (var d = dims.rank - 1u; d < dims.rank; d--) {
        flat_in += nd_in[d] * stride;
        stride  *= get_shape(d);
    }

    output[i] = input[flat_in];
}
