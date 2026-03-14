use crate::{Context, Result, Var, VolticError};

pub struct BatchNorm {
    num_features: u32,
    momentum: f32,
    epsilon: f32,
    gamma: Option<Var>,
    beta: Option<Var>,
    running_mean: Option<Var>,
    running_var: Option<Var>,
    training: bool,
}

impl BatchNorm {
    pub fn new(num_features: u32) -> Self {
        Self {
            num_features,
            momentum: 0.1,
            epsilon: 1e-5,
            gamma: None,
            beta: None,
            running_mean: None,
            running_var: None,
            training: true,
        }
    }

    pub fn eval(mut self) -> Self {
        self.training = false;
        self
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

        if self.gamma.is_none() {
            self.gamma = Some(Var::with_shape(vec![channels]));
        }
        if self.beta.is_none() {
            self.beta = Some(Var::with_shape(vec![channels]));
        }

        if self.training {
            if self.running_mean.is_none() {
                self.running_mean = Some(Var::with_shape(vec![channels]));
            }
            if self.running_var.is_none() {
                self.running_var = Some(Var::with_shape(vec![channels]));
            }
        }

        let gamma = self.gamma.as_ref().unwrap();
        let beta = self.beta.as_ref().unwrap();

        let x_reshaped = x.reshape(vec![batch * height * width, channels])?;

        let mean = Var::with_shape(vec![channels]);

        let var = Var::with_shape(vec![channels]);

        let output = x.clone();

        let output = output.reshape(vec![batch, channels, height, width])?;

        Ok(output)
    }

    pub fn init(&self) -> Result<()> {
        if let Some(g) = &self.gamma {
            let data: Vec<f32> = (0..self.num_features).map(|_| 1.0).collect();
            g.load(vec![data])?;
        }
        if let Some(b) = &self.beta {
            let data: Vec<f32> = (0..self.num_features).map(|_| 0.0).collect();
            b.load(vec![data])?;
        }
        if let Some(m) = &self.running_mean {
            let data: Vec<f32> = (0..self.num_features).map(|_| 0.0).collect();
            m.load(vec![data])?;
        }
        if let Some(v) = &self.running_var {
            let data: Vec<f32> = (0..self.num_features).map(|_| 1.0).collect();
            v.load(vec![data])?;
        }
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut params = vec![];
        if let Some(g) = &self.gamma {
            params.push(g);
        }
        if let Some(b) = &self.beta {
            params.push(b);
        }
        if let Some(m) = &self.running_mean {
            params.push(m);
        }
        if let Some(v) = &self.running_var {
            params.push(v);
        }
        params
    }
}
