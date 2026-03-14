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

pub struct FlexibleVaeEncoder {
    channels: Vec<u32>,
    convs: Vec<Conv2d>,
    fc_mu: Linear,
    fc_log_var: Linear,
    latent_dim: u32,
}

impl FlexibleVaeEncoder {
    pub fn new(latent_dim: u32, image_channels: u32, channel_config: &[u32]) -> Self {
        let mut convs = Vec::new();

        let mut in_ch = image_channels;
        for &out_ch in channel_config {
            convs.push(Conv2d::new(out_ch, 3).stride(2).padding(1));
            in_ch = out_ch;
        }

        Self {
            channels: channel_config.to_vec(),
            convs,
            fc_mu: Linear::new(latent_dim),
            fc_log_var: Linear::new(latent_dim),
            latent_dim,
        }
    }

    pub fn forward(&mut self, x: &Var) -> Result<(Var, Var)> {
        let mut x = x.clone();

        for conv in &mut self.convs {
            x = conv.forward(&x)?.gelu()?;
        }

        let shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;
        let batch = shape[0];
        let feat_size = shape[1] * shape[2] * shape[3];

        let x_flat = x.reshape(vec![batch, feat_size])?;

        let mu = self.fc_mu.forward(&x_flat)?;
        let log_var = self.fc_log_var.forward(&x_flat)?;

        Ok((mu, log_var))
    }

    pub fn init(&self) -> Result<()> {
        for conv in &self.convs {
            conv.init()?;
        }
        self.fc_mu.init()?;
        self.fc_log_var.init()?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut p = vec![];
        for conv in &self.convs {
            p.extend(conv.parameters());
        }
        p.extend(self.fc_mu.parameters());
        p.extend(self.fc_log_var.parameters());
        p
    }
}

pub struct FlexibleVaeDecoder {
    fc: Linear,
    convs: Vec<Conv2d>,
    upsample_layers: Vec<Upsample>,
    image_channels: u32,
    init_channels: u32,
    init_size: u32,
}

impl FlexibleVaeDecoder {
    pub fn new(
        latent_dim: u32,
        image_channels: u32,
        channel_config: &[u32],
        init_channels: u32,
        init_size: u32,
    ) -> Self {
        let mut convs = Vec::new();
        let mut upsample_layers = Vec::new();

        let fc_out = init_channels * init_size * init_size;

        for _ in channel_config.windows(2) {
            upsample_layers.push(Upsample::new(2));
        }

        for (i, &out_ch) in channel_config.iter().enumerate() {
            convs.push(Conv2d::new(out_ch, 3).padding(1));
        }
        convs.push(Conv2d::new(image_channels, 3).padding(1));

        Self {
            fc: Linear::new(fc_out),
            convs,
            upsample_layers,
            image_channels,
            init_channels,
            init_size,
        }
    }

    pub fn forward(&mut self, z: &Var) -> Result<Var> {
        let z_shape = Context::shape(z.id()).ok_or(VolticError::EmptyShape)?;
        let batch = z_shape[0];

        let x = self.fc.forward(z)?;
        let x = x.reshape(vec![
            batch,
            self.init_channels,
            self.init_size,
            self.init_size,
        ])?;

        let mut x = x;
        let upsample_count = self.upsample_layers.len();

        for (i, conv) in self.convs.iter_mut().enumerate() {
            if i < upsample_count {
                x = self.upsample_layers[i].forward(&x)?;
            }
            x = conv.forward(&x)?.gelu()?;
        }

        let x = x.sigmoid()?;

        Ok(x)
    }

    pub fn init(&self) -> Result<()> {
        self.fc.init()?;
        for conv in &self.convs {
            conv.init()?;
        }
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut p = vec![];
        p.extend(self.fc.parameters());
        for conv in &self.convs {
            p.extend(conv.parameters());
        }
        p
    }
}

pub struct FlexibleVae {
    encoder: FlexibleVaeEncoder,
    decoder: FlexibleVaeDecoder,
    latent_dim: u32,
    image_channels: u32,
}

impl FlexibleVae {
    pub fn new(
        latent_dim: u32,
        image_channels: u32,
        channel_config: &[u32],
        input_size: u32,
    ) -> Self {
        let num_strides = channel_config.len() as u32;
        let mut size = input_size;
        for _ in 0..num_strides {
            size = (size + 1) / 2;
        }
        let final_channels = *channel_config.last().unwrap_or(&256);

        Self {
            encoder: FlexibleVaeEncoder::new(latent_dim, image_channels, channel_config),
            decoder: FlexibleVaeDecoder::new(
                latent_dim,
                image_channels,
                channel_config,
                final_channels,
                size,
            ),
            latent_dim,
            image_channels,
        }
    }

    pub fn forward(&mut self, x: &Var) -> Result<(Var, Var, Var)> {
        let x_shape = Context::shape(x.id()).ok_or(VolticError::EmptyShape)?;
        let batch = x_shape[0];

        let (mu, log_var) = self.encoder.forward(x)?;

        let std = log_var.clone() * 0.5;
        let std_exp = std.exp()?;

        let epsilon = Var::with_shape(vec![batch, self.latent_dim]);

        let z = mu + std_exp * epsilon;

        let reconstructed = self.decoder.forward(&z)?;

        Ok((reconstructed, mu, log_var))
    }

    pub fn encode(&mut self, x: &Var) -> Result<(Var, Var)> {
        self.encoder.forward(x)
    }

    pub fn decode(&mut self, z: &Var) -> Result<Var> {
        self.decoder.forward(z)
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

    pub fn latent_dim(&self) -> u32 {
        self.latent_dim
    }

    pub fn image_channels(&self) -> u32 {
        self.image_channels
    }
}
