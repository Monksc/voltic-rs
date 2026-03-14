struct Dims {
    rank:      u32,
    total:     u32,
    _pad0:     u32,
    _pad1:     u32,
    shape:     array<vec4<u32>, 2>,  // original input shape
    perm:      array<vec4<u32>, 2>,  // original forward permutation
    out_shape: array<vec4<u32>, 2>,  // output shape of forward pass
}

@group(0) @binding(0) var<storage, read>       grad_out:   array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(2) var<uniform>             dims:       Dims;

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
    // backward iterates over input elements (shape, not out_shape)
    let input_total: u32 = dims.total;
    if i >= input_total { return; }

    // flat input index -> nd input index
    var nd_in: array<u32, 8>;
    var remaining = i;
    for (var d = dims.rank - 1u; d < dims.rank; d--) {
        nd_in[d] = remaining % get_shape(d);
        remaining = remaining / get_shape(d);
    }

    // apply forward permutation to get nd output index
    // output axis d came from input axis perm[d]
    var nd_out: array<u32, 8>;
    for (var d = 0u; d < dims.rank; d++) {
        nd_out[d] = nd_in[get_perm(d)];
    }

    // nd output index -> flat output index
    var flat_out = 0u;
    var stride   = 1u;
    for (var d = dims.rank - 1u; d < dims.rank; d--) {
        flat_out += nd_out[d] * stride;
        stride   *= get_out_shape(d);
    }

    grad_input[i] = grad_out[flat_out];
}
