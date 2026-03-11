use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

const SEED: u64 = 0xD42; // 0xD42 = 3394

fn rng() -> StdRng {
    StdRng::seed_from_u64(SEED)
}

/// Xavier/Glorot uniform initialisation
/// Recommended for tanh and sigmoid activations
pub fn xavier(fan_in: u32, fan_out: u32) -> impl FnOnce(u32) -> Vec<f32> {
    move |n| {
        let mut rng = rng();
        let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
        (0..n).map(|_| rng.random_range(-limit..=limit)).collect()
    }
}

pub fn xavier_flat(n: u32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let limit = (6.0_f32 / n as f32).sqrt();
    (0..n).map(|_| rng.random_range(-limit..=limit)).collect()
}

/// He/Kaiming initialisation
/// Recommended for ReLU activations  
pub fn he(fan_in: u32) -> impl FnOnce(u32) -> Vec<f32> {
    move |n| {
        let mut rng = rng();
        let std = (2.0_f32 / fan_in as f32).sqrt();
        (0..n).map(|_| rng.random_range(-std..=std)).collect()
    }
}

/// Zeros — for biases
pub fn zeros() -> impl FnOnce(u32) -> Vec<f32> {
    |n| vec![0.0; n as usize]
}

pub fn ones() -> impl FnOnce(u32) -> Vec<f32> {
    |n| vec![1.0; n as usize]
}
