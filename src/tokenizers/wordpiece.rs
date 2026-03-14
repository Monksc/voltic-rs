use crate::tokenizers::{pre_tokenize_words, Tokenizer};
use std::collections::HashMap;

pub struct WordPieceTokenizer {
    vocab: HashMap<String, u32>,
    vocab_inv: HashMap<u32, String>,
    unk_token: String,
    max_input_chars: usize,
}

impl WordPieceTokenizer {
    pub fn train(text: &str, vocab_size: usize) -> Self {
        let words = pre_tokenize_words(text);

        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for word in &words {
            let lower = word.to_lowercase();
            *word_freqs.entry(lower).or_insert(0) += 1;
        }

        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);

        let mut char_freqs: HashMap<char, usize> = HashMap::new();
        for word in &words {
            for c in word.chars() {
                let freq = word_freqs.get(&word.to_lowercase()).copied().unwrap_or(0);
                *char_freqs.entry(c).or_insert(0) += freq;
            }
        }

        let mut chars: Vec<(char, usize)> = char_freqs.into_iter().collect();
        chars.sort_by(|a, b| b.1.cmp(&a.1));

        for (c, _) in chars.iter().take(vocab_size.saturating_sub(1)) {
            let token = c.to_string();
            if !vocab.contains_key(&token) {
                vocab.insert(token, vocab.len() as u32);
            }
        }

        let max_input_chars = 100usize;
        let unk_token = "[UNK]".to_string();

        let vocab_inv: HashMap<u32, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        Self {
            vocab,
            vocab_inv,
            unk_token,
            max_input_chars,
        }
    }

    fn tokenize_word(&self, word: &str) -> Vec<String> {
        if word.len() > self.max_input_chars {
            return vec![self.unk_token.clone()];
        }

        let chars: Vec<char> = word.chars().collect();
        let mut start = 0;
        let mut result = Vec::new();

        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;

            while start < end {
                let subword: String = if start == 0 {
                    chars[start..end].iter().collect()
                } else {
                    format!("##{}", chars[start..end].iter().collect::<String>())
                };

                if self.vocab.contains_key(&subword) {
                    result.push(subword);
                    found = true;
                    break;
                }
                end -= 1;
            }

            if !found {
                return vec![self.unk_token.clone()];
            }

            start = end;
        }

        result
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let words = pre_tokenize_words(text);
        let mut result = Vec::new();

        for word in words {
            let tokens = self.tokenize_word(&word.to_lowercase());
            result.extend(tokens);
        }

        result
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let tokens = self.tokenize(text);

        tokens
            .iter()
            .filter_map(|t| self.vocab.get(t).copied())
            .collect()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();

        for (i, token_id) in tokens.iter().enumerate() {
            if let Some(token) = self.vocab_inv.get(token_id) {
                let decoded = token.trim_start_matches("##");
                if i > 0 && !token.starts_with("##") && !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(decoded);
            }
        }

        result
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
