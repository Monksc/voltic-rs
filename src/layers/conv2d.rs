use crate::{init, Context, Im2ColOp, Result, Var, VolticError};

pub struct Conv2d {
    out_channels: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    weights: Option<Var>,
    bias: Option<Var>,
}

impl Conv2d {
    pub fn new(out_channels: u32, kernel_size: u32) -> Self {
        Self {
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
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

    pub fn no_bias(mut self) -> Self {
        self.bias = None;
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

        let out_h = (height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (width + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let needs_init = self.weights.is_none();

        if self.weights.is_none() {
            self.weights = Some(Var::with_shape(vec![
                self.out_channels,
                self.kernel_size * self.kernel_size * in_channels,
            ]));
        }

        if self.bias.is_none() {
            self.bias = Some(Var::with_shape(vec![self.out_channels]));
        }

        // Initialize weights if this is the first forward pass
        if needs_init {
            if let Some(w) = &self.weights {
                let shape = Context::shape(w.id()).ok_or(VolticError::EmptyShape)?;
                let fan_in = shape[1];
                let data = init::xavier_flat(fan_in);
                w.load(vec![data])?;
            }
            if let Some(b) = &self.bias {
                let shape = Context::shape(b.id()).ok_or(VolticError::EmptyShape)?;
                let n: u32 = shape.iter().product();
                b.load(vec![vec![0.0; n as usize]])?;
            }
        }

        let col_h = batch * out_h * out_w;
        let col_w = self.kernel_size * self.kernel_size * in_channels;
        let x_col = Var::with_shape(vec![col_h, col_w]);

        Context::push_operation(Box::new(Im2ColOp::new(
            x.id(),
            x_col.id(),
            batch,
            in_channels,
            height,
            width,
            self.kernel_size,
            self.stride,
            self.padding,
        )));

        let w = self.weights.as_ref().unwrap();
        let w_t = w.transpose(0, 1)?;
        let out_flat = x_col.mat_mul(w_t)?;

        let out = out_flat.reshape(vec![batch, self.out_channels, out_h, out_w])?;

        let b = self.bias.as_ref().unwrap();
        let out = out.add_bc(b, &[0, 2, 3]);

        Ok(out)
    }

    pub fn init(&self) -> Result<()> {
        if let Some(w) = &self.weights {
            let shape = Context::shape(w.id()).ok_or(VolticError::EmptyShape)?;
            let fan_in = shape[1];
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
