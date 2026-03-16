use crate::{Context, Linear, Result, Var, VolticError};

pub struct MoELayer {
    num_experts: u32,
    _top_k: u32,
    experts: Vec<Linear>,
    gate: Linear,
}

impl MoELayer {
    pub fn new(num_experts: u32, hidden_dim: u32, top_k: u32) -> Self {
        let mut experts = Vec::with_capacity(num_experts as usize);
        for _ in 0..num_experts {
            experts.push(Linear::new(hidden_dim));
        }

        Self {
            num_experts,
            _top_k: top_k,
            experts,
            gate: Linear::new(num_experts),
        }
    }

    pub fn forward(&mut self, x: &Var) -> Result<Var> {
        let x_shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;
        let batch = x_shape[0];
        let seq_len = x_shape.iter().product::<u32>() / batch;

        let x_2d = x.reshape(vec![batch * seq_len, x_shape[x_shape.len() - 1]])?;

        let gating_scores = self.gate.forward(&x_2d)?;
        let _gating_probs = gating_scores.softmax(gating_scores.shape().len() - 1)?;

        let mut outputs = Vec::with_capacity(self.num_experts as usize);
        for expert in &mut self.experts {
            let expert_out = expert.forward(&x_2d)?;
            outputs.push(expert_out);
        }

        let final_output = Var::with_shape(vec![batch * seq_len, x_shape[x_shape.len() - 1]]);

        Ok(final_output)
    }

    pub fn init(&self) -> Result<()> {
        self.gate.init()?;
        for expert in &self.experts {
            expert.init()?;
        }
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut params = vec![];
        params.extend(self.gate.parameters());
        for expert in &self.experts {
            params.extend(expert.parameters());
        }
        params
    }
}
