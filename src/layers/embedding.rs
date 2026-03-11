use crate::{Context, EmbeddingOp, Result, Var, VolticError};

pub struct Embedding {
    vocab_size: u32,
    d_model: u32,
    weights: Option<Var>,
}

impl Embedding {
    pub fn new(vocab_size: u32, d_model: u32) -> Self {
        Self {
            vocab_size,
            d_model,
            weights: None,
        }
    }

    /// token_ids: Var with shape [seq_len] containing f32-cast token IDs
    pub fn forward(&mut self, token_ids: &Var) -> Result<Var> {
        let shape = Context::shape(token_ids.id()).ok_or(VolticError::EmptyShape)?;
        if shape.len() != 1 {
            return Err(VolticError::InvalidDimension {
                dim: shape.len(),
                ndim: 1,
            });
        }
        let seq_len = shape[0];

        if self.weights.is_none() {
            self.weights = Some(Var::with_shape(vec![self.vocab_size, self.d_model]));
        }
        let weights = self.weights.as_ref().unwrap();

        let output = Var::new();
        Context::insert_shape(output.id(), vec![seq_len, self.d_model]);
        Context::push_operation(Box::new(EmbeddingOp::new(
            token_ids.id(),
            weights.id(),
            output.id(),
            seq_len,
            self.d_model,
        )));

        Ok(output)
    }

    pub fn parameters(&self) -> Vec<&Var> {
        match &self.weights {
            Some(w) => vec![w],
            None => vec![],
        }
    }
}
