use crate::{Context, LayerNorm, Linear, Result, Var, VolticError};

// ─── Multi-Head Attention ────────────────────────────────────────────────────

pub struct MultiHeadAttention {
    d_model: u32,
    num_heads: u32,
    d_k: u32,

    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
}

impl MultiHeadAttention {
    pub fn new(d_model: u32, num_heads: u32) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(VolticError::Internal(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            )));
        }
        let d_k = d_model / num_heads;
        Ok(Self {
            d_model,
            num_heads,
            d_k,
            linear_q: Linear::new(d_model),
            linear_k: Linear::new(d_model),
            linear_v: Linear::new(d_model),
            linear_out: Linear::new(d_model),
        })
    }

    /// x:    [B, T, d_model]
    /// mask: [T, T] causal mask
    pub fn forward(&mut self, x: &Var, mask: &Var) -> Result<Var> {
        let shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;
        if shape.len() != 3 {
            return Err(VolticError::InvalidDimension {
                dim: shape.len(),
                ndim: 3,
            });
        }

        let b = shape[0];
        let t = shape[1];
        let h = self.num_heads;
        let d_k = self.d_k;

        // Q, K, V projections — [B, T, d_model]
        let q = self.linear_q.forward(x)?;
        let k = self.linear_k.forward(x)?;
        let v = self.linear_v.forward(x)?;

        // Split heads — [B, T, d_model] -> [B, T, H, d_k]
        let q = q.reshape(vec![b, t, h, d_k])?;
        let k = k.reshape(vec![b, t, h, d_k])?;
        let v = v.reshape(vec![b, t, h, d_k])?;

        // [B, T, H, d_k] -> [B, H, T, d_k]
        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        // Attention scores — [B, H, T, T]
        let k_t = k.transpose(2, 3)?; // [B, H, d_k, T]
        let scores = q.mat_mul(k_t)?; // [B, H, T, T]

        // Scale
        let scale = 1.0 / (d_k as f32).sqrt();
        let scores = &scores * scale;

        // Causal mask — mask is [T, T], scores is [B, H, T, T]
        // broadcast add over B and H dims
        let scores = scores.add_bc(mask, &[0, 1]);

        // Softmax over last axis (keys)
        let scores = scores.softmax(3)?; // [B, H, T, T]

        // Weighted sum — [B, H, T, T] @ [B, H, T, d_k] = [B, H, T, d_k]
        let out = scores.mat_mul(v)?;

        // Merge heads — [B, H, T, d_k] -> [B, T, H, d_k] -> [B, T, d_model]
        let out = out.permute(&[0, 2, 1, 3])?;
        let out = out.reshape(vec![b, t, self.d_model])?;

        // Output projection
        self.linear_out.forward(&out)
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut p = vec![];
        p.extend(self.linear_q.parameters());
        p.extend(self.linear_k.parameters());
        p.extend(self.linear_v.parameters());
        p.extend(self.linear_out.parameters());
        p
    }
}

// ─── GPT Block ───────────────────────────────────────────────────────────────

pub struct GptBlock {
    ln1: LayerNorm,
    attn: MultiHeadAttention,
    ln2: LayerNorm,
    ff1: Linear, // d_model -> 4 * d_model
    ff2: Linear, // 4 * d_model -> d_model
}

impl GptBlock {
    pub fn new(d_model: u32, num_heads: u32) -> Result<Self> {
        let ff_dim = d_model * 4;
        Ok(Self {
            ln1: LayerNorm::new(d_model),
            attn: MultiHeadAttention::new(d_model, num_heads)?,
            ln2: LayerNorm::new(d_model),
            ff1: Linear::new(ff_dim),
            ff2: Linear::new(d_model),
        })
    }

    /// x:    [B, T, d_model]
    /// mask: [T, T] causal mask
    pub fn forward(&mut self, x: &Var, mask: &Var) -> Result<Var> {
        // Attention sublayer with residual
        let attn_out = self.attn.forward(&self.ln1.forward(x)?, mask)?;
        let x = x + attn_out;

        // FFN sublayer with residual
        let ffn_in = self.ln2.forward(&x)?;
        let ffn_out = self.ff1.forward(&ffn_in)?;
        let ffn_out = ffn_out.gelu()?;
        let ffn_out = self.ff2.forward(&ffn_out)?;
        let x = x + ffn_out;

        Ok(x)
    }

    /// Must be called after Context::allocate_buffers()
    pub fn init(&self) -> Result<()> {
        self.ln1.init()?;
        self.ln2.init()?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut p = vec![];
        p.extend(self.ln1.parameters());
        p.extend(self.attn.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.ff1.parameters());
        p.extend(self.ff2.parameters());
        p
    }
}
