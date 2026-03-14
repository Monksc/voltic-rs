use super::{pre_tokenize, Tokenizer};
use std::collections::HashMap;

pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    vocab_inv: HashMap<u32, String>,
    merges: Vec<(String, String)>,
}

impl BpeTokenizer {
    pub fn train(text: &str, vocab_size: usize) -> Self {
        let mut vocab: HashMap<String, u32> = HashMap::new();

        for byte in 0..=255u8 {
            let s = (byte as char).to_string();
            vocab.insert(s, byte as u32);
        }

        let mut words = pre_tokenize(text);
        for word in &mut words {
            let chars: Vec<char> = word.chars().collect();
            let mut new_word = String::new();
            for (i, c) in chars.iter().enumerate() {
                new_word.push(*c);
                if i < chars.len() - 1 {
                    new_word.push_str("</c>");
                }
            }
            *word = new_word;
        }

        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for word in &words {
            *word_freqs.entry(word.clone()).or_insert(0) += 1;
        }

        let mut merges = Vec::new();

        while vocab.len() < vocab_size {
            let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();

            for (word, freq) in &word_freqs {
                let chars: Vec<&str> = word.split("</c>").collect();
                for i in 0..chars.len().saturating_sub(1) {
                    let pair = (chars[i].to_string(), chars[i + 1].to_string());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            let best_pair = pair_freqs
                .iter()
                .max_by_key(|&(_, &count)| count)
                .map(|(pair, _)| pair.clone());

            let Some((first, second)) = best_pair else {
                break;
            };

            merges.push((first.clone(), second.clone()));

            let mut new_word_freqs: HashMap<String, usize> = HashMap::new();
            let pair_str = format!("{}</c>{}", first, second);

            for (word, freq) in &word_freqs {
                let chars: Vec<&str> = word.split("</c>").collect();
                let mut new_chars: Vec<String> = Vec::new();
                let mut i = 0;
                while i < chars.len() {
                    if i < chars.len() - 1 && chars[i] == first && chars[i + 1] == second {
                        new_chars.push(pair_str.clone());
                        i += 2;
                    } else {
                        new_chars.push(chars[i].to_string());
                        i += 1;
                    }
                }
                let new_word = new_chars.join("</c>");
                new_word_freqs.insert(new_word, *freq);
            }

            word_freqs = new_word_freqs;

            let new_token = format!("{}</c>{}", first, second);
            vocab.insert(new_token.clone(), vocab.len() as u32);
        }

        let vocab_inv: HashMap<u32, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        Self {
            vocab,
            vocab_inv,
            merges,
        }
    }

    pub fn from_files(vocab_file: &str, merges_file: &str) -> std::io::Result<Self> {
        let vocab_content = std::fs::read_to_string(vocab_file)?;
        let mut vocab: HashMap<String, u32> = HashMap::new();

        for (i, line) in vocab_content.lines().enumerate() {
            if !line.is_empty() {
                vocab.insert(line.to_string(), i as u32);
            }
        }

        let merges_content = std::fs::read_to_string(merges_file)?;
        let merges: Vec<(String, String)> = merges_content
            .lines()
            .filter_map(|line| {
                let parts: Vec<&str> = line.split(' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        let vocab_inv: HashMap<u32, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        Ok(Self {
            vocab,
            vocab_inv,
            merges,
        })
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let tokens = pre_tokenize(text);

        let mut result: Vec<String> = Vec::new();

        for token in &tokens {
            let chars: Vec<&str> = token.split("").filter(|s| !s.is_empty()).collect();
            let mut word: Vec<String> = chars.iter().map(|s| s.to_string()).collect();

            for (first, second) in &self.merges {
                let mut new_word: Vec<String> = Vec::new();
                let mut i = 0;
                while i < word.len() {
                    if i < word.len() - 1 {
                        let pair = format!("{}{}", word[i], word[i + 1]);
                        if pair == format!("{}{}", first, second) {
                            new_word.push(format!("{}{}", word[i], word[i + 1]));
                            i += 2;
                            continue;
                        }
                    }
                    new_word.push(word[i].clone());
                    i += 1;
                }
                word = new_word;
            }

            result.extend(word);
        }

        result
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let tokens = self.tokenize(text);

        tokens
            .iter()
            .filter_map(|t| self.vocab.get(t).copied())
            .collect()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();

        for token_id in tokens {
            if let Some(token) = self.vocab_inv.get(token_id) {
                let decoded = token.replace("</c>", "");
                result.push_str(&decoded);
            }
        }

        result
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
