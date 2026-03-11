// Scatter grad_out rows back to the corresponding weight gradient rows
// grad_weights[token_id, :] += grad_out[seq_idx, :]

struct Dims {
    seq_len: u32,
    d_model: u32,
}

@group(0) @binding(0) var<storage, read>       token_ids:    array<f32>;
@group(0) @binding(1) var<storage, read>       grad_out:     array<f32>; // [seq_len, d_model]
@group(0) @binding(2) var<storage, read_write> grad_weights: array<f32>; // [vocab_size, d_model]
@group(0) @binding(3) var<uniform>             dims:         Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; // index into [seq_len * d_model]
    if i >= dims.seq_len * dims.d_model { return; }

    let seq_idx  = i / dims.d_model;
    let dim_idx  = i % dims.d_model;
    let token_id = u32(token_ids[seq_idx]);

    // Dense gradient — accumulate into the weight row for this token
    grad_weights[token_id * dims.d_model + dim_idx] += grad_out[i];
}
