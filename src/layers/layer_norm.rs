use crate::{Context, LayerNormOp, Result, Var, VolticError};

pub struct LayerNorm {
    d_model: u32,
    eps: f32,
    gamma: Option<Var>, // scale — init to ones
    beta: Option<Var>,  // shift — init to zeros
}

impl LayerNorm {
    pub fn new(d_model: u32) -> Self {
        Self::with_eps(d_model, 1e-5)
    }

    pub fn with_eps(d_model: u32, eps: f32) -> Self {
        Self {
            d_model,
            eps,
            gamma: None,
            beta: None,
        }
    }

    /// input: Var with shape [..., d_model]
    /// normalises over the last axis
    pub fn forward(&mut self, x: &Var) -> Result<Var> {
        let shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;

        if *shape.last().ok_or(VolticError::EmptyShape)? != self.d_model {
            return Err(VolticError::IncompatibleShapes {
                lhs: shape.clone(),
                rhs: vec![self.d_model],
                op: "layer_norm",
            });
        }

        // gamma init to ones, beta init to zeros
        if self.gamma.is_none() {
            let g = Var::new();
            Context::insert_shape(g.id(), vec![self.d_model]);
            // load ones after allocate_buffers via layer.init()
            self.gamma = Some(g);
        }
        if self.beta.is_none() {
            let b = Var::new();
            Context::insert_shape(b.id(), vec![self.d_model]);
            self.beta = Some(b);
        }

        let gamma = self.gamma.as_ref().unwrap();
        let beta = self.beta.as_ref().unwrap();

        // normalise over last axis
        let axis = shape.len() - 1;
        let outer: u32 = shape[..axis].iter().product();
        let reduce: u32 = shape[axis];
        let inner: u32 = 1;

        let output = Var::new();
        Context::insert_shape(output.id(), shape);
        Context::push_operation(Box::new(LayerNormOp::new(
            x.id(),
            gamma.id(),
            beta.id(),
            output.id(),
            outer,
            reduce,
            inner,
            self.eps,
        )));

        Ok(output)
    }

    /// Must be called after Context::allocate_buffers() to initialise
    /// gamma to ones and beta to zeros
    pub fn init(&self) -> Result<()> {
        if let Some(g) = &self.gamma {
            let ones = vec![vec![1.0f32; self.d_model as usize]];
            g.load(ones)?;
        }
        if let Some(b) = &self.beta {
            let zeros = vec![vec![0.0f32; self.d_model as usize]];
            b.load(zeros)?;
        }
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        parameters!(self, [gamma, beta])
    }
}
