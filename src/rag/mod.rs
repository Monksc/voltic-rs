use crate::{Context, Embedding, Result, Var, VolticError};

pub struct RagConfig {
    pub embedding_dim: u32,
    pub top_k: usize,
}

pub struct DocumentChunk {
    pub text: String,
    pub embedding: Vec<f32>,
}

pub struct RagHelper {
    embedding_layer: Embedding,
    config: RagConfig,
    document_chunks: Vec<DocumentChunk>,
}

impl RagHelper {
    pub fn new(vocab_size: u32, embedding_dim: u32, top_k: usize) -> Self {
        Self {
            embedding_layer: Embedding::new(vocab_size, embedding_dim),
            config: RagConfig {
                embedding_dim,
                top_k,
            },
            document_chunks: Vec::new(),
        }
    }

    pub fn add_document(&mut self, text: &str, tokens: &[u32]) {
        let mut chunk = DocumentChunk {
            text: text.to_string(),
            embedding: Vec::with_capacity(self.config.embedding_dim as usize),
        };

        for &token_id in tokens {
            chunk.embedding.push(token_id as f32);
        }

        while chunk.embedding.len() < self.config.embedding_dim as usize {
            chunk.embedding.push(0.0);
        }
        chunk.embedding.truncate(self.config.embedding_dim as usize);

        self.document_chunks.push(chunk);
    }

    pub fn retrieve(&self, query_embedding: &[f32], top_k: usize) -> Vec<(&str, f32)> {
        let k = top_k.min(self.document_chunks.len());
        if k == 0 {
            return Vec::new();
        }

        let mut scores: Vec<(usize, f32)> = self
            .document_chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let similarity = self.cosine_similarity(query_embedding, &chunk.embedding);
                (i, similarity)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .take(k)
            .map(|(i, score)| (self.document_chunks[i].text.as_str(), score))
            .collect()
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    pub fn build_context(&self, query: &str, query_tokens: &[u32]) -> String {
        let mut query_embedding = Vec::with_capacity(self.config.embedding_dim as usize);
        for &token_id in query_tokens {
            query_embedding.push(token_id as f32);
        }
        while query_embedding.len() < self.config.embedding_dim as usize {
            query_embedding.push(0.0);
        }
        query_embedding.truncate(self.config.embedding_dim as usize);

        let results = self.retrieve(&query_embedding, self.config.top_k);

        let mut context = format!("Query: {}\n\nRelevant Context:\n", query);
        for (i, (text, score)) in results.iter().enumerate() {
            context.push_str(&format!("[{}] (score: {:.3}) {}\n\n", i + 1, score, text));
        }

        context
    }

    pub fn forward(&mut self, query_tokens: Var) -> Result<Var> {
        self.embedding_layer.forward(&query_tokens)
    }

    pub fn parameters(&self) -> Vec<&Var> {
        self.embedding_layer.parameters()
    }
}
