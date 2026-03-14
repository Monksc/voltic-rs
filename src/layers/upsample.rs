use crate::{Context, Result, Var, VolticError};

pub struct Upsample {
    scale_factor: u32,
}

impl Upsample {
    pub fn new(scale_factor: u32) -> Self {
        Self { scale_factor }
    }

    pub fn forward(&self, x: &Var) -> Result<Var> {
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

        let new_height = height * self.scale_factor;
        let new_width = width * self.scale_factor;

        let x_reshaped = x.reshape(vec![batch, channels, height, 1, width, 1])?;
        let expanded = x_reshaped.reshape(vec![batch, channels, new_height, new_width])?;

        Ok(expanded)
    }
}

pub struct Downsample {
    scale_factor: u32,
}

impl Downsample {
    pub fn new(scale_factor: u32) -> Self {
        Self { scale_factor }
    }

    pub fn forward(&self, x: &Var) -> Result<Var> {
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

        let new_height = height / self.scale_factor;
        let new_width = width / self.scale_factor;

        let reshaped = x.reshape(vec![
            batch,
            channels,
            new_height,
            self.scale_factor,
            new_width,
            self.scale_factor,
        ])?;

        let permuted = reshaped.permute(&[0, 1, 2, 4, 3, 5])?;
        let pooled = permuted.reshape(vec![batch, channels, new_height, new_width])?;

        Ok(pooled)
    }
}
