struct Dims {
    rank:        u32,
    total:       u32,
    _pad0:       u32,
    _pad1:       u32,
    out_shape:   array<vec4<u32>, 2>,
    lhs_strides: array<vec4<u32>, 2>,
    rhs_strides: array<vec4<u32>, 2>,
}

@group(0) @binding(0) var<storage, read>       lhs:  array<f32>;
@group(0) @binding(1) var<storage, read>       rhs:  array<f32>;
@group(0) @binding(2) var<storage, read_write> out:  array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

fn get_shape(i: u32) -> u32 {
    let block = i / 4u;
    let lane  = i % 4u;
    return dims.out_shape[block][lane];
}

fn get_lhs_stride(i: u32) -> u32 {
    let block = i / 4u;
    let lane  = i % 4u;
    return dims.lhs_strides[block][lane];
}

fn get_rhs_stride(i: u32) -> u32 {
    let block = i / 4u;
    let lane  = i % 4u;
    return dims.rhs_strides[block][lane];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.total { return; }

    var remaining = i;
    var lhs_idx: u32 = 0u;
    var rhs_idx: u32 = 0u;

    for (var d = dims.rank - 1u; d < dims.rank; d--) {
        let coord  = remaining % get_shape(d);
        remaining  = remaining / get_shape(d);
        lhs_idx   += coord * get_lhs_stride(d);
        rhs_idx   += coord * get_rhs_stride(d);
    }

    out[i] = lhs[lhs_idx] + rhs[rhs_idx];
}
