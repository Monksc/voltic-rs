use crate::{Context, Linear, Result, Var, VolticError};

pub struct MultiMatrixAttention {
    _num_matrices: u32,
    _d_model: u32,
    d_k: u32,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    matrices: Vec<Linear>,
    out_proj: Linear,
}

impl MultiMatrixAttention {
    pub fn new(d_model: u32, num_heads: u32, num_matrices: u32) -> Result<Self> {
        let d_k = d_model / num_heads;

        let mut matrices = Vec::with_capacity(num_matrices as usize);
        for _ in 0..num_matrices {
            matrices.push(Linear::new(d_model));
        }

        Ok(Self {
            _num_matrices: num_matrices,
            _d_model: d_model,
            d_k,
            q_proj: Linear::new(d_model),
            k_proj: Linear::new(d_model),
            v_proj: Linear::new(d_model),
            matrices,
            out_proj: Linear::new(d_model),
        })
    }

    pub fn forward(&mut self, x: &Var, mask: Option<&Var>) -> Result<Var> {
        let shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;
        if shape.len() != 3 {
            return Err(VolticError::InvalidDimension {
                dim: shape.len(),
                ndim: 3,
            });
        }

        let batch = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let num_heads = d_model / self.d_k;

        let q_reshaped = q.reshape(vec![batch, seq_len, num_heads, self.d_k])?;
        let k_reshaped = k.reshape(vec![batch, seq_len, num_heads, self.d_k])?;
        let v_reshaped = v.reshape(vec![batch, seq_len, num_heads, self.d_k])?;

        let q_t = q_reshaped.transpose(1, 2)?;
        let k_t = k_reshaped.transpose(1, 2)?;
        let v_t = v_reshaped.transpose(1, 2)?;

        let scores = q_t.mat_mul(k_t.transpose(2, 3)?)?;

        let scale = 1.0 / (self.d_k as f32).sqrt();
        let scores = scores * scale;

        let scores = if let Some(m) = mask {
            scores + *m
        } else {
            scores
        };

        let attn_weights = scores.softmax(3)?;

        let context = attn_weights.mat_mul(v_t)?;

        let mut combined = context.permute(&[0, 2, 1, 3])?;
        combined = combined.reshape(vec![batch, seq_len, d_model])?;

        for mat in &mut self.matrices {
            let transformed = mat.forward(&combined)?;
            combined = transformed * combined;
        }

        let output = self.out_proj.forward(&combined)?;

        Ok(output)
    }

    pub fn init(&self) -> Result<()> {
        self.q_proj.init()?;
        self.k_proj.init()?;
        self.v_proj.init()?;
        for mat in &self.matrices {
            mat.init()?;
        }
        self.out_proj.init()?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut params = vec![];
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        for mat in &self.matrices {
            params.extend(mat.parameters());
        }
        params.extend(self.out_proj.parameters());
        params
    }
}
