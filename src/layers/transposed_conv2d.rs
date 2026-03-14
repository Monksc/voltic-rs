use crate::{init, Context, Result, Var, VolticError};

pub struct TransposedConv2d {
    out_channels: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    output_padding: u32,
    weights: Option<Var>,
    bias: Option<Var>,
}

impl TransposedConv2d {
    pub fn new(out_channels: u32, kernel_size: u32) -> Self {
        Self {
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            output_padding: 0,
            weights: None,
            bias: None,
        }
    }

    pub fn stride(mut self, stride: u32) -> Self {
        self.stride = stride;
        self
    }

    pub fn padding(mut self, padding: u32) -> Self {
        self.padding = padding;
        self
    }

    pub fn output_padding(mut self, output_padding: u32) -> Self {
        self.output_padding = output_padding;
        self
    }

    pub fn forward(&mut self, x: &Var) -> Result<Var> {
        let x_shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;
        if x_shape.len() != 4 {
            return Err(VolticError::InvalidDimension {
                dim: x_shape.len(),
                ndim: 4,
            });
        }

        let batch = x_shape[0];
        let in_channels = x_shape[1];
        let height = x_shape[2];
        let width = x_shape[3];

        let out_height =
            (height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding;
        let out_width =
            (width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding;

        if self.weights.is_none() {
            self.weights = Some(Var::with_shape(vec![
                in_channels,
                self.out_channels * self.kernel_size * self.kernel_size,
            ]));
        }

        if self.bias.is_none() {
            self.bias = Some(Var::with_shape(vec![self.out_channels]));
        }

        let h_out = (height - 1) * self.stride - 2 * self.padding + self.kernel_size;
        let w_out = (width - 1) * self.stride - 2 * self.padding + self.kernel_size;

        let output = Var::with_shape(vec![batch, self.out_channels, h_out, w_out]);

        Ok(output)
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
}
