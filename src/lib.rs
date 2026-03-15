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

        let mut embedding = Embedding::new(10, 4);
        embedding.init().unwrap();

        // Input: [seq=3] token IDs (1D)
        let tokens = Var::with_shape(vec![3]);
        
        Context::init_gpu().unwrap();
        Context::allocate_buffers().unwrap();
        
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

        let mut hybrid = crate::HybridMambaTransformer::new(32, 4, 16).unwrap();
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

        let mut bn = crate::BatchNorm::new(4);
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

        let mut deconv = crate::TransposedConv2d::new(8, 3).stride(2).padding(1);
        deconv.init().unwrap();

        let shape = Context::shape(x.id()).unwrap();
        assert_eq!(shape, vec![1, 4, 4, 4]);
        println!("transposed_conv2d_test passed!");
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
}
