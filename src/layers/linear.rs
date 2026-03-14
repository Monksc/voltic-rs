use crate::{init, Context, Result, Var, VolticError};

pub struct Linear {
    out_features: u32,
    use_bias: bool,
    weights: Option<Var>,
    bias: Option<Var>,
}

impl Linear {
    pub fn new(out_features: u32) -> Self {
        Self {
            out_features,
            use_bias: true,
            weights: None,
            bias: None,
        }
    }

    pub fn bias(mut self, enabled: bool) -> Self {
        self.use_bias = enabled;
        self
    }

    pub fn forward(&mut self, x: &Var) -> Result<Var> {
        let x_shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;
        let in_features = x_shape[x_shape.len() - 1];

        if self.weights.is_none() {
            self.weights = Some(Var::with_shape(vec![in_features, self.out_features]));
        }
        let mut out = x.mat_mul(self.weights.as_ref().unwrap().clone())?;

        if self.use_bias {
            if self.bias.is_none() {
                self.bias = Some(Var::with_shape(vec![self.out_features]));
            }
            out = out.add_bc(
                &self.bias.as_ref().unwrap().clone(),
                &(0..out.shape().len() - 1).collect::<Vec<_>>(),
            );
        }

        Ok(out)
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut params = vec![];
        if let Some(w) = &self.weights {
            params.push(w);
        }
        if let Some(b) = &self.bias {
            params.push(b);
        }
        params
    }

    pub fn init(&self) -> Result<()> {
        if let Some(w) = &self.weights {
            let shape = Context::shape(w.id()).ok_or(VolticError::EmptyShape)?;
            let fan_in = shape[0];
            let data = init::xavier_flat(fan_in);
            w.load(vec![data])?;
        }
        if let Some(b) = &self.bias {
            let shape = Context::shape(b.id()).ok_or(VolticError::EmptyShape)?;
            let n: u32 = shape.iter().product();
            b.load(vec![vec![0.0; n as usize]])?;
        }
        Ok(())
    }
}
