use crate::{Context, Conv2d, Linear, Result, Upsample, Var, VolticError};

pub struct VaeEncoder {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    fc_mu: Linear,
    fc_log_var: Linear,
    latent_dim: u32,
}

impl VaeEncoder {
    pub fn new(latent_dim: u32, image_channels: u32) -> Self {
        Self {
            conv1: Conv2d::new(32, 3).stride(2).padding(1),
            conv2: Conv2d::new(64, 3).stride(2).padding(1),
            conv3: Conv2d::new(128, 3).stride(2).padding(1),
            conv4: Conv2d::new(256, 3).stride(2).padding(1),
            fc_mu: Linear::new(latent_dim),
            fc_log_var: Linear::new(latent_dim),
            latent_dim,
        }
    }

    pub fn forward(&mut self, x: &Var) -> Result<(Var, Var)> {
        let x = self.conv1.forward(x)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        let x = self.conv3.forward(&x)?.gelu()?;
        let x = self.conv4.forward(&x)?.gelu()?;

        let shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;
        let batch = shape[0];
        let feat_size = shape[1] * shape[2] * shape[3];

        let x_flat = x.reshape(vec![batch, feat_size])?;

        let mu = self.fc_mu.forward(&x_flat)?;
        let log_var = self.fc_log_var.forward(&x_flat)?;

        Ok((mu, log_var))
    }

    pub fn init(&self) -> Result<()> {
        self.conv1.init()?;
        self.conv2.init()?;
        self.conv3.init()?;
        self.conv4.init()?;
        self.fc_mu.init()?;
        self.fc_log_var.init()?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut p = vec![];
        p.extend(self.conv1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.conv3.parameters());
        p.extend(self.conv4.parameters());
        p.extend(self.fc_mu.parameters());
        p.extend(self.fc_log_var.parameters());
        p
    }
}

pub struct VaeDecoder {
    fc: Linear,
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    image_channels: u32,
    output_shape: Option<Vec<u32>>,
}

impl VaeDecoder {
    pub fn new(latent_dim: u32, image_channels: u32, output_shape: Vec<u32>) -> Self {
        Self {
            fc: Linear::new(256 * 4 * 4),
            conv1: Conv2d::new(128, 3).padding(1),
            conv2: Conv2d::new(64, 3).padding(1),
            conv3: Conv2d::new(32, 3).padding(1),
            conv4: Conv2d::new(image_channels, 3).padding(1),
            image_channels,
            output_shape: Some(output_shape),
        }
    }

    pub fn forward(&mut self, z: &Var) -> Result<Var> {
        let z_shape = Context::shape(z.id()).ok_or(VolticError::EmptyShape)?;
        let batch = z_shape[0];

        let x = self.fc.forward(z)?;
        let x = x.reshape(vec![batch, 256, 4, 4])?;

        let x = self.conv1.forward(&x)?.gelu()?;
        let x = Upsample::new(2).forward(&x)?;

        let x = self.conv2.forward(&x)?.gelu()?;
        let x = Upsample::new(2).forward(&x)?;

        let x = self.conv3.forward(&x)?.gelu()?;
        let x = Upsample::new(2).forward(&x)?;

        let x = self.conv4.forward(&x)?.sigmoid()?;

        Ok(x)
    }

    pub fn init(&self) -> Result<()> {
        self.fc.init()?;
        self.conv1.init()?;
        self.conv2.init()?;
        self.conv3.init()?;
        self.conv4.init()?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut p = vec![];
        p.extend(self.fc.parameters());
        p.extend(self.conv1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.conv3.parameters());
        p.extend(self.conv4.parameters());
        p
    }
}

pub struct Vae {
    encoder: VaeEncoder,
    decoder: VaeDecoder,
    latent_dim: u32,
}

impl Vae {
    pub fn new(latent_dim: u32, image_channels: u32, image_height: u32, image_width: u32) -> Self {
        Self {
            encoder: VaeEncoder::new(latent_dim, image_channels),
            decoder: VaeDecoder::new(
                latent_dim,
                image_channels,
                vec![image_channels, image_height, image_width],
            ),
            latent_dim,
        }
    }

    pub fn forward(&mut self, x: &Var) -> Result<(Var, Var, Var)> {
        let x_shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;
        let batch = x_shape[0];

        let (mu, log_var) = self.encoder.forward(x)?;

        let std = log_var * 0.5;
        let std_exp = std.exp()?;

        let epsilon = Var::with_shape(vec![batch, self.latent_dim]);

        let z = mu + std_exp * epsilon;

        let reconstructed = self.decoder.forward(&z)?;

        Ok((reconstructed, mu, log_var))
    }

    pub fn init(&self) -> Result<()> {
        self.encoder.init()?;
        self.decoder.init()?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut p = vec![];
        p.extend(self.encoder.parameters());
        p.extend(self.decoder.parameters());
        p
    }
}
