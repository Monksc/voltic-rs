struct Dims { n: u32, out: u32 }

@group(0) @binding(0) var<storage, read>       grad_out:  array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_bias: array<f32>;
@group(0) @binding(2) var<uniform>             dims:      Dims;

// One thread per bias element r.
// Sums grad_out[r], grad_out[r + out], grad_out[r + 2*out], ...
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if r >= dims.out { return; }
    var s: f32 = 0.0;
    var i: u32 = r;
    loop {
        if i >= dims.n { break; }
        s += grad_out[i];
        i += dims.out;
    }
    grad_bias[r] = s;
}
