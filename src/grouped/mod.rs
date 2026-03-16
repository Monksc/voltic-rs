use crate::{Context, Linear, Result, Var, VolticError};

pub struct GroupedMatMul {
    num_groups: u32,
    group_weights: Vec<Linear>,
}

impl GroupedMatMul {
    pub fn new(input_dim: u32, output_dim: u32, num_groups: u32) -> Result<Self> {
        let _dim_per_group = input_dim / num_groups;
        let output_per_group = output_dim / num_groups;

        let mut group_weights = Vec::with_capacity(num_groups as usize);
        for _ in 0..num_groups {
            group_weights.push(Linear::new(output_per_group));
        }

        Ok(Self {
            num_groups,
            group_weights,
        })
    }

    pub fn forward(&mut self, x: &Var) -> Result<Var> {
        let x_shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;
        if x_shape.len() < 2 {
            return Err(VolticError::InvalidDimension {
                dim: x_shape.len(),
                ndim: 2,
            });
        }

        let batch: u32 = x_shape[..x_shape.len() - 1].iter().product();
        let input_dim = x_shape[x_shape.len() - 1];

        let dim_per_group = input_dim / self.num_groups;

        let mut outputs = Vec::new();
        for (i, weight_layer) in self.group_weights.iter_mut().enumerate() {
            let _start_dim = i as u32 * dim_per_group;
            let _end_dim = _start_dim + dim_per_group;

            let x_slice = x.reshape(vec![batch, input_dim])?;

            let group_out = weight_layer.forward(&x_slice)?;
            outputs.push(group_out);
        }

        Ok(Var::new())
    }

    pub fn init(&self) -> Result<()> {
        for weight in &self.group_weights {
            weight.init()?;
        }
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut params = vec![];
        for weight in &self.group_weights {
            params.extend(weight.parameters());
        }
        params
    }
}

pub struct ChannelWiseAttention {
    num_groups: u32,
    query: Linear,
    key: Linear,
    value: Linear,
}

impl ChannelWiseAttention {
    pub fn new(d_model: u32, num_groups: u32) -> Self {
        Self {
            num_groups,
            query: Linear::new(d_model),
            key: Linear::new(d_model),
            value: Linear::new(d_model),
        }
    }

    pub fn forward(&mut self, x: &Var) -> Result<Var> {
        let shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;
        if shape.len() != 4 {
            return Err(VolticError::InvalidDimension {
                dim: shape.len(),
                ndim: 4,
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let dim_per_group = channels / self.num_groups;

        let x_flat = x.reshape(vec![batch * self.num_groups, dim_per_group, height * width])?;

        let q = self.query.forward(&x_flat.reshape(vec![
            batch * self.num_groups,
            dim_per_group * height * width,
        ])?)?;
        let k = self.key.forward(&x_flat.reshape(vec![
            batch * self.num_groups,
            dim_per_group * height * width,
        ])?)?;
        let v = self.value.forward(&x_flat.reshape(vec![
            batch * self.num_groups,
            dim_per_group * height * width,
        ])?)?;

        let q_reshaped = q.reshape(vec![batch * self.num_groups, dim_per_group, height * width])?;
        let k_reshaped = k.reshape(vec![batch * self.num_groups, dim_per_group, height * width])?;
        let v_reshaped = v.reshape(vec![batch * self.num_groups, dim_per_group, height * width])?;

        let attn = q_reshaped * k_reshaped;
        let attn = attn.softmax(attn.shape().len() - 1)?;
        let out = attn * v_reshaped;

        let output = out.reshape(vec![batch, channels, height, width])?;

        Ok(output)
    }

    pub fn init(&self) -> Result<()> {
        self.query.init()?;
        self.key.init()?;
        self.value.init()?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut params = vec![];
        params.extend(self.query.parameters());
        params.extend(self.key.parameters());
        params.extend(self.value.parameters());
        params
    }
}
