use crate::{Context, Conv2d, Linear, Result, Upsample, Var, VolticError};

pub struct SimpleAutoencoder {
    encoder_conv1: Conv2d,
    encoder_conv2: Conv2d,
    decoder_conv1: Conv2d,
    decoder_conv2: Conv2d,
    latent_channels: u32,
    image_channels: u32,
}

impl SimpleAutoencoder {
    pub fn new(latent_channels: u32, image_channels: u32) -> Self {
        Self {
            encoder_conv1: Conv2d::new(32, 3).stride(1).padding(1),
            encoder_conv2: Conv2d::new(latent_channels, 3).stride(1).padding(1),
            decoder_conv1: Conv2d::new(32, 3).stride(1).padding(1),
            decoder_conv2: Conv2d::new(image_channels, 3).stride(1).padding(1),
            latent_channels,
            image_channels,
        }
    }

    pub fn encode(&mut self, x: &Var) -> Result<Var> {
        let x = self.encoder_conv1.forward(x)?.gelu()?;
        let x = self.encoder_conv2.forward(&x)?.gelu()?;
        Ok(x)
    }

    pub fn decode(&mut self, x: &Var) -> Result<Var> {
        let x = self.decoder_conv1.forward(x)?.gelu()?;
        let x = self.decoder_conv2.forward(&x)?.sigmoid()?;
        Ok(x)
    }

    pub fn forward(&mut self, x: &Var) -> Result<Var> {
        let encoded = self.encode(x)?;
        let decoded = self.decode(&encoded)?;
        Ok(decoded)
    }

    pub fn init(&self) -> Result<()> {
        self.encoder_conv1.init()?;
        self.encoder_conv2.init()?;
        self.decoder_conv1.init()?;
        self.decoder_conv2.init()?;
        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut p = vec![];
        p.extend(self.encoder_conv1.parameters());
        p.extend(self.encoder_conv2.parameters());
        p.extend(self.decoder_conv1.parameters());
        p.extend(self.decoder_conv2.parameters());
        p
    }
}
