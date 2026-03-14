use crate::{Context, Embedding, GptBlock, LayerNorm, Linear, Result, Var, VolticError};

// ─── Config ─────────────────────────────────────────────────────────────────────

pub struct GptConfig {
    pub vocab_size: u32,
    pub seq_len: u32,
    pub d_model: u32,
    pub num_heads: u32,
    pub num_layers: u32,
    pub ff_dim: Option<u32>, // defaults to 4 * d_model
}

impl GptConfig {
    pub fn ff_dim(&self) -> u32 {
        self.ff_dim.unwrap_or(self.d_model * 4)
    }
}

// ─── Gpt Model ───────────────────────────────────────────────────────────────────

pub struct Gpt {
    config: GptConfig,
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<GptBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
    pos_ids: Option<Var>, // [B*T] — tiled 0..T repeated B times
    mask: Option<Var>,    // [T, T] — causal mask
}

impl Gpt {
    pub fn new(config: GptConfig) -> Result<Self> {
        let mut blocks = Vec::with_capacity(config.num_layers as usize);
        for _ in 0..config.num_layers {
            blocks.push(GptBlock::new(config.d_model, config.num_heads)?);
        }

        Ok(Self {
            tok_emb: Embedding::new(config.vocab_size, config.d_model),
            pos_emb: Embedding::new(config.seq_len, config.d_model),
            ln_f: LayerNorm::new(config.d_model),
            lm_head: Linear::new(config.vocab_size),
            blocks,
            pos_ids: None,
            mask: None,
            config,
        })
    }

    /// Build the static computation graph for one forward pass.
    /// Call once before `Context::allocate_buffers()`.
    ///
    /// `tokens` — Var with shape `[B, T]` containing f32-cast token IDs.
    pub fn forward(&mut self, tokens: &Var) -> Result<Var> {
        let shape = Context::shape(tokens.id()).ok_or(VolticError::EmptyShape)?;
        if shape.len() != 2 {
            return Err(VolticError::InvalidDimension {
                dim: shape.len(),
                ndim: 2,
            });
        }

        let b = shape[0];
        let t = shape[1];
        let d = self.config.d_model;

        if t > self.config.seq_len {
            return Err(VolticError::Internal(format!(
                "sequence length {} exceeds max seq_len {}",
                t, self.config.seq_len
            )));
        }

        // ── causal mask [T, T] — values loaded in init() ──────────────────────
        let mask = Var::with_shape(vec![t, t]);
        self.mask = Some(mask.clone());

        // ── token embeddings ──────────────────────────────────────────────────
        // Embedding.forward expects 1-D input, so flatten [B, T] -> [B*T].
        let tokens_flat = tokens.reshape(vec![b * t])?; // [B*T]
        let tok_emb = self.tok_emb.forward(&tokens_flat)?; // [B*T, d]
        let tok_emb = tok_emb.reshape(vec![b, t, d])?; // [B, T, d]

        // ── positional embeddings ─────────────────────────────────────────────
        // Use a [B*T] tiled pos_ids placeholder (values 0,1,...,T-1 repeated B
        // times) so the embedding output is [B*T, d], which we reshape to
        // [B, T, d].  Then a plain elementwise add gives [B, T, d].
        let pos_ids_tiled = Var::with_shape(vec![b * t]);
        self.pos_ids = Some(pos_ids_tiled.clone());

        let pos_emb = self.pos_emb.forward(&pos_ids_tiled)?; // [B*T, d]
        let pos_emb = pos_emb.reshape(vec![b, t, d])?; // [B, T, d]

        // ── combine: both [B, T, d] — plain elementwise add ───────────────────
        let mut x = tok_emb.add_safe(pos_emb)?; // [B, T, d]

        // ── transformer blocks ────────────────────────────────────────────────
        for block in &mut self.blocks {
            x = block.forward(&x, &mask)?;
        }

        // ── final layer norm + LM head ────────────────────────────────────────
        x = self.ln_f.forward(&x)?; // [B, T, d]
        x = self.lm_head.forward(&x)?; // [B, T, vocab]
        x = x.softmax(2)?; // [B, T, vocab]

        Ok(x)
    }

    /// Call after `Context::allocate_buffers()` to initialise weights and
    /// write constant data (positional IDs, causal mask) into GPU buffers.
    pub fn init(&self) -> Result<()> {
        // Layer-norm gamma=1 / beta=0
        self.ln_f.init()?;
        for block in &self.blocks {
            block.init()?;
        }

        // Tiled positional IDs: 0, 1, ..., T-1, 0, 1, ..., T-1, ...
        if let Some(pos_ids) = &self.pos_ids {
            let shape = Context::shape(pos_ids.id()).ok_or(VolticError::EmptyShape)?;
            let len = shape[0] as usize; // B*T
            let t = self.config.seq_len as usize;
            let data: Vec<f32> = (0..len).map(|i| (i % t) as f32).collect();
            pos_ids.load(vec![data])?;
        }

        // Causal mask
        if let Some(mask) = &self.mask {
            mask.load_causal_mask()?;
        }

        Ok(())
    }

    pub fn parameters(&self) -> Vec<&Var> {
        let mut p = vec![];
        p.extend(self.tok_emb.parameters());
        p.extend(self.pos_emb.parameters());
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.ln_f.parameters());
        p.extend(self.lm_head.parameters());
        p
    }
}
