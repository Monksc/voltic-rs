use crate::{Context, Linear, Result, Var, VolticError};

pub struct NeuralDatabase {
    num_entries: u32,
    key_dim: u32,
    value_dim: u32,
    key_proj: Linear,
    value_memory: Var,
    output_proj: Linear,
    temperature: f32,
}

impl NeuralDatabase {
    pub fn new(num_entries: u32, key_dim: u32, value_dim: u32) -> Result<Self> {
        Ok(Self {
            num_entries,
            key_dim,
            value_dim,
            key_proj: Linear::new(key_dim),
            value_memory: Var::with_shape(vec![num_entries, value_dim]),
            output_proj: Linear::new(value_dim),
            temperature: 1.0,
        })
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    pub fn set_value(&self, index: usize, value: &[f32]) -> Result<()> {
        if index >= self.num_entries as usize {
            return Err(VolticError::Internal("Index out of bounds".to_string()));
        }

        let full_value = {
            let mut v = vec![0.0; (self.num_entries * self.value_dim) as usize];
            for (i, &val) in value.iter().enumerate() {
                v[index * self.value_dim as usize + i] = val;
            }
            v
        };

        self.value_memory.load(vec![full_value])
    }

    pub fn forward(&mut self, query: &Var, _top_k: usize) -> Result<Var> {
        let query_shape = Context::shape(query.id()).ok_or(VolticError::EmptyShape)?;
        let batch = query_shape.iter().product::<u32>() / self.key_dim;

        let query_flat = query.reshape(vec![batch, self.key_dim])?;

        let mut keys_data = Vec::with_capacity((batch * self.num_entries) as usize);
        for _ in 0..batch {
            for i in 0..self.num_entries {
                let idx = i as f32;
                let key = (idx / self.num_entries as f32 * 2.0 - 1.0).sin() * 0.1;
                keys_data.push(key);
            }
        }

        let keys_var = Var::with_shape(vec![batch, self.num_entries]);
        keys_var.load(vec![keys_data])?;

        let scores = query_flat.mat_mul(keys_var.transpose(0, 1)?)?;

        let scale = 1.0 / self.temperature;
        let scores = scores * scale;
        let attn_weights = scores.softmax(scores.shape().len() - 1)?;

        let context = attn_weights.mat_mul(self.value_memory.clone())?;

        let output = self.output_proj.forward(&context)?;

        Ok(output)
    }

    pub fn differentiable_lookup(&mut self, query: &Var) -> Result<Var> {
        self.forward(query, self.num_entries as usize)
    }

    pub fn init(&self) -> Result<()> {
        self.key_proj.init()?;

        let mem_size = (self.num_entries * self.value_dim) as usize;
        let mem_data: Vec<f32> = (0..mem_size).map(|i| ((i as f32) * 0.01).sin()).collect();
        self.value_memory.load(vec![mem_data])?;

        self.output_proj.init()?;

        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut params = vec![];
        params.extend(self.key_proj.parameters());
        params.push(&self.value_memory);
        params.extend(self.output_proj.parameters());
        params
    }
}

pub struct LearnableMemory {
    memory: Var,
    query_proj: Linear,
    key_proj: Linear,
    output_proj: Linear,
    memory_dim: u32,
    num_slots: u32,
}

impl LearnableMemory {
    pub fn new(num_slots: u32, memory_dim: u32, _query_dim: u32) -> Self {
        Self {
            memory: Var::with_shape(vec![num_slots, memory_dim]),
            query_proj: Linear::new(memory_dim),
            key_proj: Linear::new(memory_dim),
            output_proj: Linear::new(memory_dim),
            memory_dim,
            num_slots,
        }
    }

    pub fn forward(&mut self, query: &Var) -> Result<Var> {
        let query_shape = Context::shape(query.id()).ok_or(VolticError::EmptyShape)?;
        let batch: u32 = query_shape.iter().product();

        let query_aug = query.reshape(vec![batch, 1, self.memory_dim])?;

        let _memory_expanded = self
            .memory
            .reshape(vec![1, self.num_slots, self.memory_dim])?;

        let keys = self
            .key_proj
            .forward(&self.memory.reshape(vec![self.num_slots, self.memory_dim])?)?;

        let scores =
            query_aug.mat_mul(keys.reshape(vec![1, self.memory_dim, self.num_slots])?)?;
        let attn = scores.softmax(2)?;

        let context = attn.mat_mul(
            self.memory
                .clone()
                .reshape(vec![self.num_slots, self.memory_dim])?,
        )?;

        let output = self
            .output_proj
            .forward(&context.reshape(vec![batch, self.memory_dim])?)?;

        Ok(output)
    }

    pub fn init(&self) -> Result<()> {
        let mem_size = (self.num_slots * self.memory_dim) as usize;
        let mem_data: Vec<f32> = (0..mem_size)
            .map(|i| ((i as f32) * 0.1).sin() * 0.1)
            .collect();
        self.memory.load(vec![mem_data])?;

        self.query_proj.init()?;
        self.key_proj.init()?;
        self.output_proj.init()?;

        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut params = vec![];
        params.push(&self.memory);
        params.extend(self.query_proj.parameters());
        params.extend(self.key_proj.parameters());
        params.extend(self.output_proj.parameters());
        params
    }
}
