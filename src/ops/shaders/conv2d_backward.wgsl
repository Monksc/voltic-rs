@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> dims: Conv2dDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    let total = dims.batch * dims.channels_in * dims.height * dims.width;
    if (idx >= total) {
        return;
    }
    
    let b = idx / (dims.channels_in * dims.height * dims.width);
    let remainder = idx % (dims.channels_in * dims.height * dims.width);
    let c = remainder / (dims.height * dims.width);
    let remainder2 = remainder % (dims.height * dims.width);
    let h = remainder2 / dims.width;
    let w = remainder2 % dims.width;
    
    var grad_sum = 0.0;
    
    for (var c_out: u32 = 0u; c_out < dims.channels_out; c_out = c_out + 1u) {
        for (var kh: u32 = 0u; kh < dims.kernel_size; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < dims.kernel_size; kw = kw + 1u) {
                let out_h = h + dims.stride - kh - dims.padding;
                let out_w = w + dims.stride - kw - dims.padding;
                
                if (out_h < dims.out_height && out_w < dims.out_width) {
                    let out_idx = ((b * dims.channels_out + c_out) * dims.out_height + out_h) * dims.out_width + out_w;
                    
                    let weight_idx = ((c_out * dims.channels_in + c) * dims.kernel_size + kh) * dims.kernel_size + kw;
                    
                    grad_sum = grad_sum + grad_output[out_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    grad_input[idx] = grad_sum;
}
