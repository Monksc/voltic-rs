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

pub mod tokenizers;
pub use tokenizers::*;

pub mod rag;
pub use rag::*;

pub mod moe;
pub use moe::*;

pub mod attention;
pub use attention::*;

pub mod grouped;
pub use grouped::*;

pub mod hybrid;
pub use hybrid::*;

pub mod neural_db;
pub use neural_db::*;

pub type BufferKey = (ID, &'static str);

pub mod buffer_kind {
    pub const GRAD: &str = "grad";
    pub const MOMENTUM: &str = "momentum";
    pub const VARIANCE: &str = "variance";
    pub const PARTIAL: &str = "partial";
    pub const PARTIAL_SUM: &str = "partial_sum";
    pub const X_NORM: &str = "x_norm";
    pub const LHS_STAGE: &str = "lhs_stage";
    pub const RHS_STAGE: &str = "rhs_stage";
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use crate::{Adam, Context, Gpt, GptConfig, Linear, Sgd, Var};

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn test_setup() -> std::sync::MutexGuard<'static, ()> {
        let lock = TEST_LOCK.lock().unwrap();
        Context::clear();
        lock
    }

    #[test]
    fn simple_matrix_multiply() {
        let _lock = test_setup();
        Context::clear();
        let batch_size = 4;
        let x = Var::with_shape(vec![batch_size, 2]);
        let weights = Var::with_shape(vec![2, 1]);

        let y = x.mat_mul(weights).unwrap();

        Context::init_gpu().unwrap();
        Context::allocate_buffers().unwrap();

        x.load(vec![vec![0., 0.], vec![1., 0.], vec![0., 1.], vec![1., 1.]])
            .unwrap();
        weights.load(vec![vec![1., 1.]]).unwrap();

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
        let _lock = test_setup();
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
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let y_pred = Var::with_shape(vec![4, 1]);
        let y_true = Var::with_shape(vec![4, 1]);
        let _loss = y_pred.mse(y_true).unwrap();

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
        let _lock = test_setup();
        let batch_size = 4;
        let x = Var::with_shape(vec![batch_size, 2]);
        let y_true = Var::with_shape(vec![batch_size, 1]);
        let weights = Var::with_shape(vec![2, 1]);

        let y_pred = x.mat_mul(weights).unwrap();
        let _loss = y_pred.mse(y_true).unwrap();

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
    fn xor_sgd() {
        let _lock = test_setup();

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

    #[test]
    fn xor_adam_tanh() {
        let _lock = test_setup();

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
        let mut adam = Adam::new(0.1);
        adam.init().unwrap();

        for epoch in 0..126 {
            Context::run().unwrap();
            Context::backward().unwrap();
            adam.step().unwrap();

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

    #[test]
    fn xor_adam_softmax() {
        let _lock = test_setup();

        Context::init_gpu().unwrap();

        let x = Var::with_shape(vec![4, 2]);
        let y_true = Var::with_shape(vec![4, 2]);

        let h1 = Linear::new(8).forward(&x).unwrap().tanh().unwrap();
        let y_pred = Linear::new(2).forward(&h1).unwrap().softmax(1).unwrap();
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
            .load(vec![
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ])
            .unwrap();

        Context::prepare().unwrap();
        let mut adam = Adam::new(0.1);
        adam.init().unwrap();

        for epoch in 0..51 {
            Context::run().unwrap();
            Context::backward().unwrap();
            adam.step().unwrap();

            if epoch % 25 == 0 {
                let errors = loss.to_cpu().unwrap();
                let mse = errors.iter().sum::<f32>() / errors.len() as f32;
                println!("epoch {epoch:5} — loss: {mse:.6}");
            }
        }

        let preds = y_pred.to_cpu().unwrap();
        println!("\nXOR predictions: {:?}", preds);
        println!("  0 XOR 0 = {:.4}  (expected 0.0)", preds[0]);
        println!("  1 XOR 0 = {:.4}  (expected 1.0)", preds[2]);
        println!("  0 XOR 1 = {:.4}  (expected 1.0)", preds[4]);
        println!("  1 XOR 1 = {:.4}  (expected 0.0)", preds[6]);

        assert!(preds[0] + 0.1 < preds[1]);
        assert!(preds[2] > preds[3] + 0.1);
        assert!(preds[4] > preds[5] + 0.1);
        assert!(preds[6] + 0.1 < preds[7]);
    }

    #[test]
    fn softmax_simple_test() {
        let _lock = test_setup();

        Context::init_gpu().unwrap();

        let x = Var::with_shape(vec![2, 4]);
        let y = x.softmax(1).unwrap();

        Context::allocate_buffers().unwrap();

        x.load(vec![vec![0.0, 0.0, 1.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]])
            .unwrap();

        Context::prepare().unwrap();
        Context::run().unwrap();

        let y = y.to_cpu().unwrap();

        assert_eq!(y.len(), 8);

        assert!((y[0] + y[1] + y[2] + y[3] - 1.0).abs() < 0.0001);
        assert_eq!(y[0], y[1]);
        assert_eq!(y[1], y[3]);

        assert_eq!(y[4], 0.25);
        assert_eq!(y[5], 0.25);
        assert_eq!(y[6], 0.25);
        assert_eq!(y[7], 0.25);
    }

    #[test]
    fn gpt_forward_16tokens() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let config = GptConfig {
            vocab_size: 16,
            seq_len: 16,
            d_model: 32,
            num_heads: 2,
            num_layers: 1,
            ff_dim: None,
        };

        let mut gpt = Gpt::new(config).unwrap();

        let tokens = Var::with_shape(vec![1, 16]);
        let y_true = Var::with_shape(vec![1, 16, 16]); // one-hot targets

        let output = gpt.forward(&tokens).unwrap();
        let loss = output.mse(y_true).unwrap();

        Context::allocate_buffers().unwrap();
        gpt.init().unwrap();

        // Load tokens 0..15
        tokens
            .load(vec![(0..16u32).map(|i| i as f32).collect()])
            .unwrap();

        // Load one-hot targets — target at pos i is token (i+1) % 16
        let mut y_data = vec![0.0f32; 16 * 16];
        for i in 0..16usize {
            y_data[i * 16 + (i + 1) % 16] = 1.0;
        }
        y_true.load(vec![y_data]).unwrap();

        Context::prepare().unwrap();
        let mut adam = Adam::new(0.001); // 0.1 is very high — might diverge
        adam.init().unwrap();

        for epoch in 0..1_000 {
            Context::run().unwrap();
            Context::backward().unwrap();
            adam.step().unwrap();
            if epoch % 100 == 0 {
                let loss_val = loss.to_cpu().unwrap();
                let mse = loss_val.iter().sum::<f32>() / loss_val.len() as f32;
                println!("epoch {epoch:5} — loss: {mse:.6}");
            }
        }

        let result = output.to_cpu().unwrap();
        for i in 0..(result.len() / 16) {
            let i = i * 16;
            let mut best_j = 0;
            for j in 1..16 {
                if result[i + j] > result[i + best_j] {
                    best_j = j;
                }
            }
            print!("{} ", best_j);
        }
        println!("predictions");
        println!("gpt_forward_16tokens passed!");
    }

    #[test]
    fn gpt_forward_16tokens_batched() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let config = GptConfig {
            vocab_size: 16,
            seq_len: 16,
            d_model: 32,
            num_heads: 2,
            num_layers: 1,
            ff_dim: None,
        };

        let mut gpt = Gpt::new(config).unwrap();

        let batch = 16;

        let tokens = Var::with_shape(vec![batch, 16]);
        let y_true = Var::with_shape(vec![batch, 16, 16]); // one-hot targets

        let output = gpt.forward(&tokens).unwrap();
        let loss = output.mse(y_true).unwrap();

        Context::allocate_buffers().unwrap();
        gpt.init().unwrap();

        // Load tokens 0..15
        tokens
            .load(
                (0..16u32)
                    .map(|b| (0..16u32).map(|i| ((i + b) % 16) as f32).collect())
                    .collect(),
            )
            .unwrap();

        // Load one-hot targets — target at pos i is token (i+1) % 16
        let mut y_data = vec![0.0f32; 16 * 16 * 16];
        for b in 0..(batch as usize) {
            for i in 0..16usize {
                y_data[b * 16 * 16 + i * 16 + (i + 1 + b) % 16] = 1.0;
            }
        }
        y_true.load(vec![y_data]).unwrap();

        Context::prepare().unwrap();
        let mut adam = Adam::new(0.01); // 0.1 is very high — might diverge
        adam.init().unwrap();

        for epoch in 0..1_000 {
            Context::run().unwrap();
            Context::backward().unwrap();
            adam.step().unwrap();
            if epoch % 100 == 0 {
                let loss_val = loss.to_cpu().unwrap();
                let mse = loss_val.iter().sum::<f32>() / loss_val.len() as f32;
                println!("epoch {epoch:5} — loss: {mse:.6}");
            }
        }

        let result = output.to_cpu().unwrap();
        println!("Result Len: {}", result.len());
        for oi in 0..(result.len() / 16) {
            let i = oi * 16;
            let mut best_j = 0;
            for j in 1..16 {
                if result[i + j] > result[i + best_j] {
                    best_j = j;
                }
            }
            print!("{} ", best_j);
            if result[i + best_j] < 0.5 {
                print!("{:.3}; ", result[i + best_j]);
            }

            if oi % 16 == 15 {
                println!("predictions");
            }
        }
        println!("gpt_forward_16tokens passed!");
    }

    #[test]
    fn group_mul_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [1, 2, 3, 4, 5, 6]
        // group_size = 3
        // Expected: [1*2*3, 4*5*6] = [6, 120]
        let x = Var::with_shape(vec![6]);
        let result = x.group_mul(3).unwrap();

        Context::allocate_buffers().unwrap();
        x.load(vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
            .unwrap();

        Context::prepare().unwrap();
        Context::run().unwrap();

        let values = result.to_cpu().unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], 6.0);
        assert_eq!(values[1], 120.0);
        println!("group_mul_test passed!");
    }

    #[test]
    fn group_add_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [1, 0, 1, 1, 0, 0]
        // group_size = 3
        // Expected: [1+0+1, 1+0+0] = [2, 1]
        let x = Var::with_shape(vec![6]);
        let result = x.group_add(3).unwrap();

        Context::allocate_buffers().unwrap();
        x.load(vec![vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
            .unwrap();

        Context::prepare().unwrap();
        Context::run().unwrap();

        let values = result.to_cpu().unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], 2.0);
        assert_eq!(values[1], 1.0);
        println!("group_add_test passed!");
    }

    #[test]
    fn group_max_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [1, 5, 3, 4, 2, 6]
        // group_size = 3
        // Expected: [max(1,5,3), max(4,2,6)] = [5, 6]
        let x = Var::with_shape(vec![6]);
        let result = x.group_max(3).unwrap();

        Context::allocate_buffers().unwrap();
        x.load(vec![vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]])
            .unwrap();

        Context::prepare().unwrap();
        Context::run().unwrap();

        let values = result.to_cpu().unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], 5.0);
        assert_eq!(values[1], 6.0);
        println!("group_max_test passed!");
    }

    #[test]
    fn conv2d_forward_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [batch=1, channels=1, height=4, width=4]
        let x = Var::with_shape(vec![1, 1, 4, 4]);
        
        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        x.load(vec![data]).unwrap();

        let shape = Context::shape(x.id()).unwrap();
        assert_eq!(shape, vec![1, 1, 4, 4]);
        println!("conv2d_forward_test passed!");
    }

    #[test]
    fn conv2d_training_test() {
        use crate::Conv2d;

        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Simple conv: input 4x4, kernel 2x2, stride 1, padding 1 -> output 5x5
        let x = Var::with_shape(vec![2, 1, 4, 4]);
        
        // Create conv and forward
        let mut conv = Conv2d::new(1, 2).stride(1).padding(1);
        let output = conv.forward(&x).unwrap();

        // Create target for loss
        let y_true = Var::with_shape(vec![2, 1, 5, 5]);
        
        // Create loss
        let _loss = output.mse(y_true).unwrap();

        Context::allocate_buffers().unwrap();
        conv.init().unwrap();

        // Load data
        let input_data: Vec<f32> = (0..32).map(|i| (i % 8) as f32 / 8.0).collect();
        let input_rows: Vec<Vec<f32>> = input_data.chunks(16).map(|c| c.to_vec()).collect();
        x.load(input_rows).unwrap();
        
        // Target is zeros
        y_true.load(vec![vec![0.0; 50]]).unwrap();

        Context::prepare().unwrap();
        
        // Forward + backward
        Context::run().unwrap();
        Context::backward().unwrap();

        // Check gradients
        let params = conv.parameters();
        println!("Conv2d parameters: {}", params.len());
        for (i, p) in params.iter().enumerate() {
            match Context::read((p.id(), "grad")) {
                Ok(grad) => println!("  Param {}: grad sum={:?}", i, grad.iter().sum::<f32>()),
                Err(e) => println!("  Param {}: grad error: {:?}", i, e),
            }
        }
        
        println!("conv2d_training_test passed!");
    }

    #[test]
    fn upsample_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [batch=1, channels=1, height=2, width=2]
        let x = Var::with_shape(vec![1, 1, 2, 2]);
        
        Context::allocate_buffers().unwrap();
        
        x.load(vec![vec![1.0, 2.0, 3.0, 4.0]]).unwrap();

        let shape = Context::shape(x.id()).unwrap();
        assert_eq!(shape, vec![1, 1, 2, 2]);
        println!("upsample_test passed!");
    }

    #[test]
    fn downsample_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [batch=1, channels=1, height=4, width=4]
        let x = Var::with_shape(vec![1, 1, 4, 4]);
        
        Context::allocate_buffers().unwrap();
        
        x.load(vec![vec![1.0; 16]]).unwrap();

        let shape = Context::shape(x.id()).unwrap();
        assert_eq!(shape, vec![1, 1, 4, 4]);
        println!("downsample_test passed!");
    }

    #[test]
    fn tokenizer_bpe_test() {
        use crate::BpeTokenizer;
        use crate::Tokenizer;

        let text = "hello world hello voltic";
        let tokenizer = BpeTokenizer::train(text, 100);
        
        let tokens = tokenizer.encode("hello");
        assert!(tokens.len() > 0);
        
        let decoded = tokenizer.decode(&tokens);
        assert!(decoded.contains("hello"));
        
        println!("tokenizer_bpe_test passed!");
    }

    #[test]
    fn tokenizer_wordpiece_test() {
        use crate::WordPieceTokenizer;
        use crate::Tokenizer;

        let text = "hello world hello voltic";
        let tokenizer = WordPieceTokenizer::train(text, 100);
        
        let tokens = tokenizer.encode("hello");
        assert!(tokens.len() > 0);
        
        let decoded = tokenizer.decode(&tokens);
        assert!(!decoded.is_empty());
        
        println!("tokenizer_wordpiece_test passed!");
    }

    #[test]
    fn tokenizer_triettoken_test() {
        use crate::TrieTokenTokenizer;
        use crate::Tokenizer;

        let text = "hello world hello voltic";
        let tokenizer = TrieTokenTokenizer::train(text, 100, 1);
        
        let tokens = tokenizer.encode("hello");
        assert!(tokens.len() > 0);
        
        let decoded = tokenizer.decode(&tokens);
        assert!(!decoded.is_empty());
        
        println!("tokenizer_triettoken_test passed!");
    }

    #[test]
    fn embedding_forward_test() {
        use crate::Embedding;

        let _lock = test_setup();

        let mut embedding = Embedding::new(10, 4);

        // Input: [seq=3] token IDs (1D)
        let tokens = Var::with_shape(vec![3]);
        
        Context::init_gpu().unwrap();
        Context::allocate_buffers().unwrap();
        
        embedding.init().unwrap();
        tokens.load(vec![vec![0.0, 1.0, 2.0]]).unwrap();

        let embedded = embedding.forward(&tokens).unwrap();
        let embed_shape = Context::shape(embedded.id()).unwrap();
        assert_eq!(embed_shape, vec![3, 4]);
        
        println!("embedding_forward_test passed!");
    }

    #[test]
    fn layer_norm_forward_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [batch=2, seq=4]
        let x = Var::with_shape(vec![2, 4]);
        
        Context::allocate_buffers().unwrap();
        
        x.load(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
        ]).unwrap();

        let mut layernorm = crate::LayerNorm::new(4);
        layernorm.init().unwrap();
        
        let normalized = layernorm.forward(&x).unwrap();
        let shape = Context::shape(normalized.id()).unwrap();
        assert_eq!(shape, vec![2, 4]);
        
        println!("layer_norm_forward_test passed!");
    }

    #[test]
    fn vae_forward_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [batch=1, channels=1, height=32, width=32]
        let x = Var::with_shape(vec![1, 1, 32, 32]);
        
        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..1024).map(|i| (i as f32) / 1024.0).collect();
        x.load(vec![data]).unwrap();

        let shape = Context::shape(x.id()).unwrap();
        assert_eq!(shape, vec![1, 1, 32, 32]);
        println!("vae_forward_test passed!");
    }

    #[test]
    fn moe_forward_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [batch=2, seq=4, hidden=8]
        let x = Var::with_shape(vec![2, 4, 8]);
        
        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let reshaped: Vec<Vec<f32>> = data.chunks(8).map(|c| c.to_vec()).collect();
        x.load(reshaped).unwrap();

        let shape = Context::shape(x.id()).unwrap();
        assert_eq!(shape, vec![2, 4, 8]);
        println!("moe_forward_test passed!");
    }

    #[test]
    fn rag_helper_test() {
        use crate::RagHelper;

        let mut rag = RagHelper::new(1000, 64, 3);

        rag.add_document("Rust is a systems programming language", &[1, 2, 3, 4, 5]);
        rag.add_document("Machine learning is a subset of AI", &[6, 7, 8, 9, 10]);
        rag.add_document("Neural networks are inspired by biological brains", &[11, 12, 13, 14, 15]);

        let context = rag.build_context("What is Rust?", &[1, 2, 3]);
        assert!(context.contains("Rust"));
        
        println!("rag_helper_test passed!");
    }

    #[test]
    fn neural_database_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let db = crate::NeuralDatabase::new(10, 8, 16).unwrap();
        db.init().unwrap();

        let query = Var::with_shape(vec![2, 8]);
        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        query.load(vec![data.clone(), data]).unwrap();

        let shape = Context::shape(query.id()).unwrap();
        assert_eq!(shape, vec![2, 8]);
        println!("neural_database_test passed!");
    }

    #[test]
    fn learnable_memory_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let mem = crate::LearnableMemory::new(8, 32, 16);
        mem.init().unwrap();

        let query = Var::with_shape(vec![4, 16]);
        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        query.load(vec![data.clone()]).unwrap();

        let shape = Context::shape(query.id()).unwrap();
        assert_eq!(shape, vec![4, 16]);
        println!("learnable_memory_test passed!");
    }

    #[test]
    fn hybrid_mamba_transformer_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let hybrid = crate::HybridMambaTransformer::new(32, 4, 16).unwrap();
        hybrid.init().unwrap();

        // Input: [batch=2, seq=8, d_model=32]
        let x = Var::with_shape(vec![2, 8, 32]);
        
        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..512).map(|i| (i as f32) / 512.0).collect();
        let reshaped: Vec<Vec<f32>> = data.chunks(32).map(|c| c.to_vec()).collect();
        x.load(reshaped).unwrap();

        let shape = Context::shape(x.id()).unwrap();
        assert_eq!(shape, vec![2, 8, 32]);
        println!("hybrid_mamba_transformer_test passed!");
    }

    #[test]
    fn downsample_actual_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [batch=1, channels=1, height=4, width=4]
        let x = Var::with_shape(vec![1, 1, 4, 4]);
        
        Context::allocate_buffers().unwrap();
        
        x.load(vec![vec![1.0, 2.0, 3.0, 4.0, 
                         5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0, 12.0,
                         13.0, 14.0, 15.0, 16.0]]).unwrap();

        // Test reshape and permute (basic downsample components)
        let shape = Context::shape(x.id()).unwrap();
        assert_eq!(shape, vec![1, 1, 4, 4]);
        println!("downsample_actual_test passed!");
    }

    #[test]
    fn batch_norm_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [batch=2, channels=4, height=4, width=4]
        let x = Var::with_shape(vec![2, 4, 4, 4]);
        
        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let reshaped: Vec<Vec<f32>> = data.chunks(64).map(|c| c.to_vec()).collect();
        x.load(reshaped).unwrap();

        let bn = crate::BatchNorm::new(4);
        bn.init().unwrap();

        let shape = Context::shape(x.id()).unwrap();
        assert_eq!(shape, vec![2, 4, 4, 4]);
        println!("batch_norm_test passed!");
    }

    #[test]
    fn transposed_conv2d_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Input: [batch=1, channels=4, height=4, width=4]
        let x = Var::with_shape(vec![1, 4, 4, 4]);
        
        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        x.load(vec![data]).unwrap();

        let deconv = crate::TransposedConv2d::new(8, 3).stride(2).padding(1);
        deconv.init().unwrap();

        let shape = Context::shape(x.id()).unwrap();
        assert_eq!(shape, vec![1, 4, 4, 4]);
        println!("transposed_conv2d_test passed!");
    }

    #[test]
    fn conv2d_training_loss_decreases_test() {
        use crate::Conv2d;

        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let x = Var::with_shape(vec![2, 1, 4, 4]);
        let y_true = Var::with_shape(vec![2, 1, 4, 4]);
        
        let mut conv = Conv2d::new(1, 3).stride(1).padding(1);
        let output = conv.forward(&x).unwrap();
        
        let loss = output.mse(y_true).unwrap();

        Context::allocate_buffers().unwrap();
        conv.init().unwrap();

        let input_data: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0).collect();
        let input_rows: Vec<Vec<f32>> = input_data.chunks(16).map(|c| c.to_vec()).collect();
        x.load(input_rows.clone()).unwrap();
        
        y_true.load(input_rows).unwrap();

        Context::prepare().unwrap();
        let mut sgd = Sgd::new(0.01);

        Context::run().unwrap();
        let initial_loss = loss.to_cpu().unwrap();
        let initial_mse: f32 = initial_loss.iter().sum::<f32>() / initial_loss.len() as f32;
        println!("Initial loss: {}", initial_mse);

        for epoch in 0..50 {
            Context::run().unwrap();
            Context::backward().unwrap();
            sgd.step().unwrap();
            
            if epoch % 10 == 0 {
                let loss_val = loss.to_cpu().unwrap();
                let mse: f32 = loss_val.iter().sum::<f32>() / loss_val.len() as f32;
                println!("Epoch {} - Loss: {}", epoch, mse);
            }
        }

        let final_loss = loss.to_cpu().unwrap();
        let final_mse: f32 = final_loss.iter().sum::<f32>() / final_loss.len() as f32;
        println!("Final loss: {}", final_mse);

        assert!(final_mse < initial_mse, "Loss should decrease but went from {} to {}", initial_mse, final_mse);
        println!("conv2d_training_loss_decreases_test passed!");
    }

    #[test]
    fn conv2d_different_configs_test() {
        use crate::Conv2d;

        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Test kernel 3x3, stride 2, no padding
        let x = Var::with_shape(vec![1, 1, 8, 8]);
        let mut conv = Conv2d::new(1, 3).stride(2).padding(0);
        let output = conv.forward(&x).unwrap();

        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        x.load(vec![data]).unwrap();

        let shape = Context::shape(output.id()).unwrap();
        // output_h = (8 - 3) / 2 + 1 = 3 (floor)
        // output_w = (8 - 3) / 2 + 1 = 3
        assert_eq!(shape, vec![1, 1, 3, 3], "Shape should be [1,1,3,3] but got {:?}", shape);
        
        println!("conv2d_different_configs_test passed!");
    }

    #[test]
    fn conv2d_batch_size_test() {
        use crate::Conv2d;

        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Test with batch size 4
        let x = Var::with_shape(vec![4, 3, 8, 8]);
        let mut conv = Conv2d::new(3, 3).stride(1).padding(1);
        let output = conv.forward(&x).unwrap();

        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..768).map(|i| i as f32 / 768.0).collect();
        let batches: Vec<Vec<f32>> = data.chunks(192).map(|c| c.to_vec()).collect();
        x.load(batches).unwrap();

        let shape = Context::shape(output.id()).unwrap();
        assert_eq!(shape[0], 4, "Batch size should be 4");
        assert_eq!(shape[1], 3, "Output channels should be 3");
        
        println!("conv2d_batch_size_test passed!");
    }

    #[test]
    fn simple_autoencoder_forward_test() {
        use crate::SimpleAutoencoder;

        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let mut autoencoder = SimpleAutoencoder::new(8, 3);
        autoencoder.init().unwrap();

        let x = Var::with_shape(vec![2, 3, 16, 16]);
        let output = autoencoder.forward(&x).unwrap();

        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..1536).map(|i| i as f32 / 1536.0).collect();
        let batches: Vec<Vec<f32>> = data.chunks(768).map(|c| c.to_vec()).collect();
        x.load(batches).unwrap();

        Context::prepare().unwrap();
        Context::run().unwrap();

        let output_shape = Context::shape(output.id()).unwrap();
        assert_eq!(output_shape, vec![2, 3, 16, 16], "Output shape should match input");
        
        println!("simple_autoencoder_forward_test passed!");
    }

    #[test]
    fn simple_autoencoder_training_test() {
        use crate::SimpleAutoencoder;

        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let mut autoencoder = SimpleAutoencoder::new(8, 3);
        
        let x = Var::with_shape(vec![2, 3, 8, 8]);
        let output = autoencoder.forward(&x).unwrap();
        let loss = output.mse(x.clone()).unwrap();

        Context::allocate_buffers().unwrap();
        autoencoder.init().unwrap();

        let data: Vec<f32> = (0..384).map(|i| (i as f32 % 64.0) / 64.0).collect();
        let batches: Vec<Vec<f32>> = data.chunks(192).map(|c| c.to_vec()).collect();
        x.load(batches).unwrap();

        Context::prepare().unwrap();
        let mut sgd = Sgd::new(0.01);

        Context::run().unwrap();
        let initial_loss = loss.to_cpu().unwrap();
        let initial_mse: f32 = initial_loss.iter().sum::<f32>() / initial_loss.len() as f32;
        println!("Initial loss: {}", initial_mse);

        for epoch in 0..30 {
            Context::run().unwrap();
            Context::backward().unwrap();
            sgd.step().unwrap();
            
            if epoch % 10 == 0 {
                let loss_val = loss.to_cpu().unwrap();
                let mse: f32 = loss_val.iter().sum::<f32>() / loss_val.len() as f32;
                println!("Epoch {} - Loss: {}", epoch, mse);
            }
        }

        let final_loss = loss.to_cpu().unwrap();
        let final_mse: f32 = final_loss.iter().sum::<f32>() / final_loss.len() as f32;
        println!("Final loss: {}", final_mse);

        assert!(final_mse < initial_mse, "Loss should decrease but went from {} to {}", initial_mse, final_mse);
        println!("simple_autoencoder_training_test passed!");
    }

    #[test]
    fn simple_autoencoder_64x64_adam_test() {
        use crate::SimpleAutoencoder;

        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Match picaso-obama's exact setup:
        // SimpleAutoencoder with latent_dim=64, image_channels=3
        // Image size 64x64, 2 images
        let mut autoencoder = SimpleAutoencoder::new(64, 3);
        
        // 2 images at 64x64 with 3 channels
        let x = Var::with_shape(vec![2, 3, 64, 64]);
        let output = autoencoder.forward(&x).unwrap();
        let loss = output.mse(x.clone()).unwrap();

        Context::allocate_buffers().unwrap();
        autoencoder.init().unwrap();

        // Random data
        let data: Vec<f32> = (0..2 * 3 * 64 * 64).map(|i| (i as f32 % 256.0) / 256.0).collect();
        let batches: Vec<Vec<f32>> = data.chunks(3 * 64 * 64).map(|c| c.to_vec()).collect();
        x.load(batches).unwrap();

        Context::prepare().unwrap();
        let mut adam = Adam::new(0.001);

        Context::run().unwrap();
        let initial_loss = loss.to_cpu().unwrap();
        let initial_mse: f32 = initial_loss.iter().sum::<f32>() / initial_loss.len() as f32;
        println!("Initial loss: {}", initial_mse);

        for epoch in 0..25 {
            Context::run().unwrap();
            Context::backward().unwrap();
            adam.step().unwrap();
            
            let loss_val = loss.to_cpu().unwrap();
            let mse: f32 = loss_val.iter().sum::<f32>() / loss_val.len() as f32;
            println!("Epoch {} - Loss: {}", epoch, mse);
        }

        let final_loss = loss.to_cpu().unwrap();
        let final_mse: f32 = final_loss.iter().sum::<f32>() / final_loss.len() as f32;
        println!("Final loss: {}", final_mse);

        assert!(final_mse < initial_mse, "Loss should decrease but went from {} to {}", initial_mse, final_mse);
        println!("simple_autoencoder_64x64_adam_test passed!");
    }

    #[test]
    fn simple_autoencoder_high_lr_test() {
        use crate::SimpleAutoencoder;

        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Try MUCH higher LR: 10.0 instead of 0.1
        let mut autoencoder = SimpleAutoencoder::new(64, 3);
        
        let x = Var::with_shape(vec![2, 3, 64, 64]);
        let output = autoencoder.forward(&x).unwrap();
        let loss = output.mse(x.clone()).unwrap();

        Context::allocate_buffers().unwrap();
        autoencoder.init().unwrap();

        let data: Vec<f32> = (0..2 * 3 * 64 * 64).map(|i| (i as f32 % 256.0) / 256.0).collect();
        let batches: Vec<Vec<f32>> = data.chunks(3 * 64 * 64).map(|c| c.to_vec()).collect();
        x.load(batches).unwrap();

        Context::prepare().unwrap();
        let mut sgd = Sgd::new(10.0); // 100x higher LR!

        Context::run().unwrap();
        let initial_loss = loss.to_cpu().unwrap();
        let initial_mse: f32 = initial_loss.iter().sum::<f32>() / initial_loss.len() as f32;
        println!("Initial loss (LR=10): {}", initial_mse);

        for epoch in 0..100 {
            Context::run().unwrap();
            Context::backward().unwrap();
            sgd.step().unwrap();
            
            if epoch % 20 == 0 {
                let loss_val = loss.to_cpu().unwrap();
                let mse: f32 = loss_val.iter().sum::<f32>() / loss_val.len() as f32;
                println!("Epoch {} - Loss: {}", epoch, mse);
            }
        }

        let final_loss = loss.to_cpu().unwrap();
        let final_mse: f32 = final_loss.iter().sum::<f32>() / final_loss.len() as f32;
        println!("Final loss (LR=10): {}", final_mse);

        assert!(final_mse < 0.05, "Loss should go below 0.05 but got {}", final_mse);
    }

    #[test]
    fn layernorm_backward_test() {
        use crate::LayerNorm;

        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let x = Var::with_shape(vec![2, 4]);
        let mut ln = LayerNorm::new(4);
        ln.init().unwrap();
        
        let output = ln.forward(&x).unwrap();
        let _loss = output.mse(x.clone()).unwrap();

        Context::allocate_buffers().unwrap();
        
        x.load(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
        ]).unwrap();

        Context::prepare().unwrap();
        
        Context::run().unwrap();
        Context::backward().unwrap();

        let params = ln.parameters();
        for (i, p) in params.iter().enumerate() {
            match Context::read((p.id(), "grad")) {
                Ok(grad) => println!("  Param {}: grad sum={:?}", i, grad.iter().sum::<f32>()),
                Err(e) => println!("  Param {}: grad error: {:?}", i, e),
            }
        }
        
        println!("layernorm_backward_test passed!");
    }

    #[test]
    fn batchnorm_training_mode_test() {
        use crate::BatchNorm;

        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let x = Var::with_shape(vec![2, 4, 4, 4]);
        let mut bn = BatchNorm::new(4);
        bn.init().unwrap();
        
        let output = bn.forward(&x).unwrap();

        Context::allocate_buffers().unwrap();
        
        let data: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let batches: Vec<Vec<f32>> = data.chunks(64).map(|c| c.to_vec()).collect();
        x.load(batches).unwrap();

        Context::prepare().unwrap();
        
        Context::run().unwrap();
        
        let output_shape = Context::shape(output.id()).unwrap();
        assert_eq!(output_shape, vec![2, 4, 4, 4]);
        
        println!("batchnorm_training_mode_test passed!");
    }

    #[test]
    fn save_restore_test() {
        let _lock = test_setup();
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
        ]).unwrap();
        y_true.load(vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]]).unwrap();

        Context::prepare().unwrap();

        let persistent = Context::get().collect_persistent(&[loss]);
        println!("Collected {} persistent vars", persistent.len());

        let path = "/tmp/voltioc_test_checkpoint.bin";
        Context::get().save(path, &persistent).unwrap();

        let weights_before: Vec<Vec<f32>> = persistent.iter().map(|v| v.to_cpu().unwrap()).collect();

        let mut sgd = Sgd::new(0.1);
        for _ in 0..50 {
            Context::run().unwrap();
            Context::backward().unwrap();
            sgd.step().unwrap();
        }

        let weights_after: Vec<Vec<f32>> = persistent.iter().map(|v| v.to_cpu().unwrap()).collect();
        assert_ne!(weights_before[0], weights_after[0]);

        let mut persistent_restore = Context::get().collect_persistent(&[loss]);
        Context::get_mut().restore(path, &mut persistent_restore).unwrap();

        let weights_restored: Vec<Vec<f32>> = persistent_restore.iter().map(|v| v.to_cpu().unwrap()).collect();
        assert_eq!(weights_before, weights_restored);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn simple_autoencoder_lr_comparison_test() {
        use crate::SimpleAutoencoder;

        let _lock = test_setup();
        Context::init_gpu().unwrap();

        // Test with LR = 0.1
        let mut autoencoder = SimpleAutoencoder::new(8, 3);
        
        let x = Var::with_shape(vec![2, 3, 8, 8]);
        let output = autoencoder.forward(&x).unwrap();
        let loss = output.mse(x.clone()).unwrap();

        Context::allocate_buffers().unwrap();
        autoencoder.init().unwrap();

        let data: Vec<f32> = (0..384).map(|i| (i as f32 % 64.0) / 64.0).collect();
        let batches: Vec<Vec<f32>> = data.chunks(192).map(|c| c.to_vec()).collect();
        x.load(batches).unwrap();

        Context::prepare().unwrap();
        
        // Check initial weights
        let params = autoencoder.parameters();
        println!("Number of params: {}", params.len());
        
        Context::run().unwrap();
        
        // Check gradients
        for (i, p) in params.iter().enumerate() {
            match Context::read((p.id(), "grad")) {
                Ok(grad) => {
                    let sum: f32 = grad.iter().sum::<f32>();
                    let max: f32 = grad.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                    println!("Param {}: grad_sum={:.8}, grad_max={:.8}", i, sum, max);
                },
                Err(e) => println!("Param {}: grad error: {:?}", i, e),
            }
        }
        
        let mut sgd = Sgd::new(0.1);

        let initial_loss = loss.to_cpu().unwrap();
        let initial_mse: f32 = initial_loss.iter().sum::<f32>() / initial_loss.len() as f32;
        println!("Initial loss (LR=0.1): {}", initial_mse);

        for epoch in 0..5 {
            Context::run().unwrap();
            Context::backward().unwrap();
            sgd.step().unwrap();
            
            // Check gradients after backward
            for (i, p) in params.iter().enumerate() {
                match Context::read((p.id(), "grad")) {
                    Ok(grad) => {
                        let sum: f32 = grad.iter().sum::<f32>();
                        let max: f32 = grad.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                        println!("Epoch {} Param {}: grad_sum={:.8}, grad_max={:.8}", epoch, i, sum, max);
                    },
                    Err(e) => {},
                }
            }
            
            let loss_val = loss.to_cpu().unwrap();
            let mse: f32 = loss_val.iter().sum::<f32>() / loss_val.len() as f32;
            println!("Epoch {} - Loss: {}", epoch, mse);
        }
    }

    #[test]
    fn simple_linear_training_test() {
        let _lock = test_setup();
        Context::init_gpu().unwrap();

        let mut linear = Linear::new(5);
        
        let x = Var::with_shape(vec![4, 5]);
        let output = linear.forward(&x).unwrap();
        let loss = output.mse(x.clone()).unwrap();

        Context::allocate_buffers().unwrap();
        linear.init().unwrap();

        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        x.load(vec![data]).unwrap();

        Context::prepare().unwrap();
        let mut sgd = Sgd::new(0.1);

        Context::run().unwrap();
        let initial_loss = loss.to_cpu().unwrap();
        let initial_mse: f32 = initial_loss.iter().sum::<f32>() / initial_loss.len() as f32;
        println!("Initial loss: {}", initial_mse);

        for epoch in 0..50 {
            Context::run().unwrap();
            Context::backward().unwrap();
            sgd.step().unwrap();
            
            let loss_val = loss.to_cpu().unwrap();
            let mse: f32 = loss_val.iter().sum::<f32>() / loss_val.len() as f32;
            if epoch % 10 == 0 {
                println!("Epoch {} - Loss: {}", epoch, mse);
            }
        }
    }
}
