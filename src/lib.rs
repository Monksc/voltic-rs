pub mod var;
pub use var::*;

pub mod ops;
pub use ops::*;

pub mod context;
pub use context::*;

pub mod errors;
pub use errors::*;

pub mod gpu_context;
pub use gpu_context::*;

pub mod optimizers;
pub use optimizers::*;

pub mod layers;
pub use layers::*;

pub mod init;
pub use init::*;

pub type BufferKey = (ID, &'static str);

pub mod buffer_kind {
    pub const GRAD: &str = "grad";
    pub const MOMENTUM: &str = "momentum";
    pub const VARIANCE: &str = "variance";
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use crate::{Context, Linear, Sgd, Var};

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn simple_matrix_multiply() {
        let _lock = TEST_LOCK.lock().unwrap();
        let batch_size = 4;
        let x = Var::with_shape(vec![batch_size, 2]);
        let weights = Var::with_shape(vec![2, 1]);

        let y = x.mat_mul(weights).unwrap();

        Context::init_gpu().unwrap();
        Context::allocate_buffers().unwrap();

        x.load(vec![vec![0., 0.], vec![1., 0.], vec![0., 1.], vec![1., 1.]])
            .unwrap();
        weights.load(vec![vec![1., 1.]]).unwrap();

        println!("Shapes: {:?}", Context::shapes());
        assert_eq!(Context::shape_total(y.id()), Some(4));
        // assert_eq!(Context::shapes_count(), 3);
        // assert_eq!(Context::get().operations().len(), 1);

        Context::prepare().unwrap();
        Context::run().unwrap();

        let x_values = x.to_cpu().unwrap();
        assert_eq!(vec![0., 0., 1., 0., 0., 1., 1., 1.], x_values);

        let weight_values = weights.to_cpu().unwrap();
        assert_eq!(vec![1.0, 1.0], weight_values);

        let y_values = y.to_cpu().unwrap();
        assert_eq!(vec![0.0, 1.0, 1.0, 2.0], y_values);

        y.load(vec![vec![0.0, 0.0, 0.0, 0.0]]).unwrap();
        Context::run().unwrap();

        let y_values = y.to_cpu().unwrap();
        assert_eq!(vec![0.0, 1.0, 1.0, 2.0], y_values);
    }

    #[test]
    fn simple_mse() {
        let _lock = TEST_LOCK.lock().unwrap();
        Context::init_gpu().unwrap();

        let y_pred = Var::with_shape(vec![4, 1]);
        let y_true = Var::with_shape(vec![4, 1]);
        let loss = y_pred.mse(y_true).unwrap();

        Context::allocate_buffers().unwrap();
        y_pred
            .load(vec![vec![1.0], vec![2.0], vec![2.0], vec![3.0]])
            .unwrap();
        y_true
            .load(vec![vec![0.0], vec![1.0], vec![1.0], vec![2.0]])
            .unwrap();

        Context::prepare().unwrap();
        Context::run().unwrap();

        let errors = loss.to_cpu().unwrap();
        let mse = errors.iter().sum::<f32>() / errors.len() as f32;
        assert_eq!(mse, 1.);
    }

    #[test]
    fn mse_backward() {
        let _lock = TEST_LOCK.lock().unwrap();
        Context::init_gpu().unwrap();

        let y_pred = Var::with_shape(vec![4, 1]);
        let y_true = Var::with_shape(vec![4, 1]);
        let loss = y_pred.mse(y_true).unwrap();

        Context::allocate_buffers().unwrap();

        y_pred
            .load(vec![vec![1.0], vec![1.0], vec![-1.0], vec![2.5]])
            .unwrap();
        y_true
            .load(vec![vec![0.0], vec![1.0], vec![1.0], vec![2.0]])
            .unwrap();

        Context::prepare().unwrap();
        Context::run().unwrap();
        Context::backward().unwrap();

        let grad = y_pred.grad().unwrap();
        assert_eq!(grad, vec![0.5, 0., -1., 0.25]);
    }

    #[test]
    fn single_layer() {
        let _lock = TEST_LOCK.lock().unwrap();
        let batch_size = 4;
        let x = Var::with_shape(vec![batch_size, 2]);
        let y_true = Var::with_shape(vec![batch_size, 1]);
        let weights = Var::with_shape(vec![2, 1]);

        let y_pred = x.mat_mul(weights).unwrap();
        let loss = y_pred.mse(y_true).unwrap();

        Context::init_gpu().unwrap();
        Context::allocate_buffers().unwrap();

        let x_data = vec![vec![0., 0.], vec![1., 0.], vec![0., 1.], vec![1., 1.]];
        let y_data = vec![vec![0.], vec![1.], vec![1.], vec![2.]];

        x.load(x_data).unwrap();
        y_true.load(y_data.clone()).unwrap();
        weights.load(vec![vec![0.5, -0.1]]).unwrap();

        Context::prepare().unwrap();

        let mut sgd = Sgd::new(0.01);

        for _ in 0..1_000 {
            Context::run().unwrap();
            Context::backward().unwrap();
            sgd.step().unwrap();
        }

        let y = y_pred.to_cpu().unwrap();
        println!("Y Expected: {:?}", y_data);
        println!("Y Pred    : {:?}", y);
        assert!(
            y[0].powi(2) + (y[1] - 1.).powi(2) + (y[2] - 1.).powi(2) + (y[3] - 2.).powi(2) < 0.1
        );
    }

    #[test]
    fn xor() {
        let _lock = TEST_LOCK.lock().unwrap();

        Context::init_gpu().unwrap();

        let x = Var::with_shape(vec![4, 2]);
        let y_true = Var::with_shape(vec![4, 1]);

        let h1 = Linear::new(8).forward(&x).unwrap().tanh().unwrap();
        let y_pred = Linear::new(1).forward(&h1).unwrap();
        let loss = y_pred.mse(y_true).unwrap();

        Context::allocate_buffers().unwrap();

        x.load(vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ])
        .unwrap();
        y_true
            .load(vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]])
            .unwrap();

        Context::prepare().unwrap();
        let mut sgd = Sgd::new(0.1);

        for epoch in 0..251 {
            Context::run().unwrap();
            Context::backward().unwrap();
            sgd.step().unwrap();

            if epoch % 25 == 0 {
                let errors = loss.to_cpu().unwrap();
                let mse = errors.iter().sum::<f32>() / errors.len() as f32;
                println!("epoch {epoch:5} — loss: {mse:.6}");
            }
        }

        let preds = y_pred.to_cpu().unwrap();
        println!("\nXOR predictions:");
        println!("  0 XOR 0 = {:.4}  (expected 0.0)", preds[0]);
        println!("  1 XOR 0 = {:.4}  (expected 1.0)", preds[1]);
        println!("  0 XOR 1 = {:.4}  (expected 1.0)", preds[2]);
        println!("  1 XOR 1 = {:.4}  (expected 0.0)", preds[3]);

        assert!((preds[0] - 0.0).abs() < 0.2, "0 XOR 0 failed: {}", preds[0]);
        assert!((preds[1] - 1.0).abs() < 0.2, "1 XOR 0 failed: {}", preds[1]);
        assert!((preds[2] - 1.0).abs() < 0.2, "0 XOR 1 failed: {}", preds[2]);
        assert!((preds[3] - 0.0).abs() < 0.2, "1 XOR 1 failed: {}", preds[3]);
    }
}
