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
    
    let batch = idx / (dims.channels * dims.height * dims.width);
    let remainder = idx % (dims.channels * dims.height * dims.width);
    let c = remainder / (dims.height * dims.width);
    let spatial = remainder % (dims.height * dims.width);
    let in_y = spatial / dims.width;
    let in_x = spatial % dims.width;
    
    var sum = 0.0;
    
    let col_w = dims.kernel_size * dims.kernel_size * dims.channels;
    
    for (var out_y: u32 = 0u; out_y < dims.out_height; out_y = out_y + 1u) {
        for (var out_x: u32 = 0u; out_x < dims.out_width; out_x = out_x + 1u) {
            let k_y_start = select(0u, dims.padding - out_y * dims.stride, out_y * dims.stride < dims.padding);
            let k_x_start = select(0u, dims.padding - out_x * dims.stride, out_x * dims.stride < dims.padding);
            
            let k_y_end = select(dims.kernel_size, dims.height + dims.padding - out_y * dims.stride, out_y * dims.stride + dims.kernel_size > dims.height + dims.padding);
            let k_x_end = select(dims.kernel_size, dims.width + dims.padding - out_x * dims.stride, out_x * dims.stride + dims.kernel_size > dims.width + dims.padding);
            
            for (var k_y: u32 = k_y_start; k_y < k_y_end; k_y = k_y + 1u) {
                for (var k_x: u32 = k_x_start; k_x < k_x_end; k_x = k_x + 1u) {
                    let in_y_pad = in_y + dims.padding;
                    let in_x_pad = in_x + dims.padding;
                    
                    let ky = in_y_pad - out_y * dims.stride;
                    let kx = in_x_pad - out_x * dims.stride;
                    
                    if (ky == k_y && kx == k_x) {
                        let col_idx = ((batch * dims.out_height * dims.out_width + out_y * dims.out_width + out_x) * col_w + (k_y * dims.kernel_size * dims.channels + k_x * dims.channels + c));
                        sum = sum + grad_col[col_idx];
                    }
                }
            }
        }
    }
    
    grad_input[idx] = sum;
}
