struct Im2ColDims {
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    out_height: u32,
    out_width: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: Im2ColDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    let col_w = dims.kernel_size * dims.kernel_size * dims.channels;
    let total_cols = dims.batch * dims.out_height * dims.out_width * col_w;
    
    if (idx >= total_cols) {
        return;
    }
    
    let col_idx = idx / col_w;
    let kernel_idx = idx % col_w;
    
    let batch = col_idx / (dims.out_height * dims.out_width);
    let spatial_idx = col_idx % (dims.out_height * dims.out_width);
    let out_y = spatial_idx / dims.out_width;
    let out_x = spatial_idx % dims.out_width;
    
    let k = dims.kernel_size;
    let c = dims.channels;
    
    let k_y = kernel_idx / (k * c);
    let remainder = kernel_idx % (k * c);
    let k_x = remainder / c;
    let c_in = remainder % c;
    
    let in_y = out_y * dims.stride + k_y - dims.padding;
    let in_x = out_x * dims.stride + k_x - dims.padding;
    
    if (in_y >= dims.height || in_x >= dims.width) {
        output[idx] = 0.0;
        return;
    }
    
    let in_idx = ((batch * c + c_in) * dims.height + in_y) * dims.width + in_x;
    output[idx] = input[in_idx];
}
