struct GroupDims {
    n: u32,
    group_size: u32,
    num_groups: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: GroupDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let group_idx = global_id.x;
    
    if (group_idx >= dims.num_groups) {
        return;
    }
    
    let start = group_idx * dims.group_size;
    var sum = 0.0;
    
    for (var i: u32 = 0u; i < dims.group_size; i = i + 1u) {
        let idx = start + i;
        if (idx < dims.n) {
            sum = sum + input[idx];
        }
    }
    
    output[group_idx] = sum;
}
