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

@group(0) @binding(0) var<storage, read> grad_col: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(2) var<uniform> dims: Im2ColDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    let n_elements = dims.batch * dims.channels * dims.height * dims.width;
    
    if (idx >= n_elements) {
        return;
    }
    
    // Decode input index: [batch, c, in_y, in_x]
    let c_and_spatial = idx / (dims.height * dims.width);
    let spatial = idx % (dims.height * dims.width);
    let c = c_and_spatial % dims.channels;
    let batch = c_and_spatial / dims.channels;
    let in_y = spatial / dims.width;
    let in_x = spatial % dims.width;
    
    var sum = 0.0;
    let col_w = dims.kernel_size * dims.kernel_size * dims.channels;
    
    // For each possible output position, check if this input contributes
    for (var out_y: u32 = 0u; out_y < dims.out_height; out_y = out_y + 1u) {
        for (var out_x: u32 = 0u; out_x < dims.out_width; out_x = out_x + 1u) {
            // Calculate kernel position for this output position
            // Forward: in_y = out_y * stride + k_y - padding
            // Backward: k_y = in_y + padding - out_y * stride
            let k_y = in_y + dims.padding - out_y * dims.stride;
            let k_x = in_x + dims.padding - out_x * dims.stride;
            
            // Check if kernel position is valid (within bounds)
            if (k_y < dims.kernel_size && k_x < dims.kernel_size) {
                // This input position contributes to this output position
                let col_row = batch * dims.out_height * dims.out_width + out_y * dims.out_width + out_x;
                let col_col = k_y * dims.kernel_size * dims.channels + k_x * dims.channels + c;
                let col_idx = col_row * col_w + col_col;
                sum = sum + grad_col[col_idx];
            }
        }
    }
    
    grad_input[idx] = sum;
}
