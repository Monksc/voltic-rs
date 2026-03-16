use crate::{Context, LayerNorm, Linear, Result, Var, VolticError};

pub struct SsmLayer {
    _d_model: u32,
    _state_dim: u32,
    x_proj: Linear,
    dt_proj: Linear,
    a_proj: Linear,
    out_proj: Linear,
}

impl SsmLayer {
    pub fn new(d_model: u32, state_dim: u32) -> Self {
        Self {
            _d_model: d_model,
            _state_dim: state_dim,
            x_proj: Linear::new(state_dim),
            dt_proj: Linear::new(state_dim),
            a_proj: Linear::new(state_dim),
            out_proj: Linear::new(d_model),
        }
    }

    pub fn forward_step(&mut self, x: &Var) -> Result<Var> {
        let x_proj_out = self.x_proj.forward(x)?;
        let _dt = self.dt_proj.forward(x)?;
        let a = self.a_proj.forward(x)?;

        let a_sigmoid = a.sigmoid()?;
        let x_sigmoid = x_proj_out.sigmoid()?;

        let combined = x_sigmoid * a_sigmoid;

        let output = self.out_proj.forward(&combined)?;

        Ok(output)
    }

    pub fn init(&self) -> Result<()> {
        self.x_proj.init()?;
        self.dt_proj.init()?;
        self.a_proj.init()?;
        self.out_proj.init()?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut params = vec![];
        params.extend(self.x_proj.parameters());
        params.extend(self.dt_proj.parameters());
        params.extend(self.a_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
}

pub struct HybridMambaTransformer {
    ssm: SsmLayer,
    attn_norm: LayerNorm,
    ssm_norm: LayerNorm,
    ff_norm: LayerNorm,
    ff: Linear,
    _num_heads: u32,
    _d_model: u32,
    causal_mask: Option<Var>,
}

impl HybridMambaTransformer {
    pub fn new(d_model: u32, num_heads: u32, state_dim: u32) -> Result<Self> {
        Ok(Self {
            ssm: SsmLayer::new(d_model, state_dim),
            attn_norm: LayerNorm::new(d_model),
            ssm_norm: LayerNorm::new(d_model),
            ff_norm: LayerNorm::new(d_model),
            ff: Linear::new(d_model * 4),
            _num_heads: num_heads,
            _d_model: d_model,
            causal_mask: None,
        })
    }

    pub fn forward(&mut self, x: &Var, _mask: Option<&Var>) -> Result<Var> {
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

        if self.causal_mask.is_none() {
            let mask_var = Var::with_shape(vec![seq_len, seq_len]);
            self.causal_mask = Some(mask_var);
        }

        let residual = x.clone();
        let x_norm = self.attn_norm.forward(x)?;

        let x_2d = x_norm.reshape(vec![batch * seq_len, d_model])?;
        let ssm_out = self.ssm.forward_step(&x_2d)?;

        let x = residual + ssm_out;

        let residual = x.clone();
        let x = self.ssm_norm.forward(&x)?;
        let x = self.ff.forward(&x)?.gelu()?;
        let x = self.ff_norm.forward(&x)?;
        let x = residual + x;

        Ok(x)
    }

    pub fn init(&self) -> Result<()> {
        self.ssm.init()?;
        self.attn_norm.init()?;
        self.ssm_norm.init()?;
        self.ff_norm.init()?;
        self.ff.init()?;

        if let Some(mask) = &self.causal_mask {
            mask.load_causal_mask()?;
        }

        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut params = vec![];
        params.extend(self.ssm.parameters());
        params.extend(self.attn_norm.parameters());
        params.extend(self.ssm_norm.parameters());
        params.extend(self.ff_norm.parameters());
        params.extend(self.ff.parameters());
        params
    }
}
