use super::{pre_tokenize_words, Tokenizer};
use std::collections::HashMap;

struct TrieNode {
    children: HashMap<char, usize>,
    freq: usize,
    is_word: bool,
}

struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    fn new() -> Self {
        let mut nodes = Vec::new();
        nodes.push(TrieNode {
            children: HashMap::new(),
            freq: 0,
            is_word: false,
        });
        Self { nodes }
    }

    fn insert(&mut self, word: &str) {
        let mut current = 0;
        for c in word.chars() {
            let idx = if let Some(&child) = self.nodes[current].children.get(&c) {
                child
            } else {
                let new_idx = self.nodes.len();
                self.nodes.push(TrieNode {
                    children: HashMap::new(),
                    freq: 0,
                    is_word: false,
                });
                self.nodes[current].children.insert(c, new_idx);
                new_idx
            };
            current = idx;
        }
        self.nodes[current].is_word = true;
        self.nodes[current].freq += 1;

        let mut node = current;
        while node != 0 {
            node = self.find_parent(node);
            if node != 0 {
                self.nodes[node].freq += 1;
            }
        }
    }

    fn find_parent(&self, child: usize) -> usize {
        for (i, node) in self.nodes.iter().enumerate() {
            if node.children.values().any(|&v| v == child) {
                return i;
            }
        }
        0
    }

    fn get_frequent_substrings(&self, word: &str, min_freq: usize) -> Vec<(String, usize)> {
        let mut substrings: Vec<(String, usize)> = Vec::new();
        let chars: Vec<char> = word.chars().collect();

        for start in 0..chars.len() {
            let mut current = 0;
            let mut valid = true;

            for i in start..chars.len() {
                if let Some(&next) = self.nodes[current].children.get(&chars[i]) {
                    current = next;

                    let substring: String = chars[start..=i].iter().collect();
                    if self.nodes[current].freq >= min_freq && i - start >= 1 {
                        substrings.push((substring, self.nodes[current].freq));
                    }
                } else {
                    valid = false;
                    break;
                }
            }
        }

        substrings.sort_by(|a, b| b.1.cmp(&a.1));
        substrings
    }
}

pub struct TrieTokenTokenizer {
    vocab: HashMap<String, u32>,
    vocab_inv: HashMap<u32, String>,
    merges: Vec<(String, String)>,
    trie: Trie,
    min_freq: usize,
}

impl TrieTokenTokenizer {
    pub fn train(text: &str, vocab_size: usize, min_frequency: usize) -> Self {
        let words = pre_tokenize_words(text);

        let mut trie = Trie::new();
        for word in &words {
            trie.insert(word);
        }

        let mut vocab: HashMap<String, u32> = HashMap::new();
        for byte in 0..=255u8 {
            let s = (byte as char).to_string();
            vocab.insert(s, byte as u32);
        }

        let mut words_with_space: Vec<String> = words.iter().map(|w| format!(" {}", w)).collect();

        let mut merges = Vec::new();

        for word in &mut words_with_space {
            let substrings = trie.get_frequent_substrings(word, min_frequency);
            for (substr, _) in substrings {
                if !vocab.contains_key(&substr) && vocab.len() < vocab_size {
                    vocab.insert(substr.clone(), vocab.len() as u32);
                }
            }
        }

        let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();

        for word in &words_with_space {
            let chars: Vec<String> = word
                .split("")
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();

            for i in 0..chars.len().saturating_sub(1) {
                let pair = (chars[i].clone(), chars[i + 1].clone());

                let substr = format!("{}{}", pair.0, pair.1);
                if vocab.contains_key(&substr) || (vocab.len() < vocab_size && trie.nodes.len() > 1)
                {
                    *pair_freqs.entry(pair).or_insert(0) += 1;
                }
            }
        }

        while vocab.len() < vocab_size {
            let best = pair_freqs
                .iter()
                .filter(|&(&(ref _a, ref _b), &freq)| freq >= min_frequency)
                .max_by_key(|&(_, &freq)| freq)
                .map(|(pair, &freq)| (pair.clone(), freq));

            let Some(((first, second), _)) = best else {
                break;
            };

            merges.push((first.clone(), second.clone()));

            let merged = format!("{}{}", first, second);
            vocab.insert(merged.clone(), vocab.len() as u32);

            pair_freqs.retain(|(a, b), _| !(a == &first && b == &second));
        }

        let vocab_inv: HashMap<u32, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        Self {
            vocab,
            vocab_inv,
            merges,
            trie,
            min_freq: min_frequency,
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let words = pre_tokenize_words(text);
        let mut result = Vec::new();

        for word in words {
            let with_space = format!(" {}", word);
            let mut tokens = self.tokenize_with_merges(&with_space);
            result.append(&mut tokens);
        }

        result
    }

    fn tokenize_with_merges(&self, text: &str) -> Vec<String> {
        let chars: Vec<String> = text
            .split("")
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        let mut tokens: Vec<String> = chars.clone();

        for (first, second) in &self.merges {
            let mut new_tokens = Vec::new();
            let mut i = 0;

            while i < tokens.len() {
                if i < tokens.len() - 1 && &tokens[i] == first && &tokens[i + 1] == second {
                    new_tokens.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }

            tokens = new_tokens;
        }

        tokens
    }
}

impl Tokenizer for TrieTokenTokenizer {
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
                let decoded = token.trim_start_matches(' ');
                result.push_str(decoded);
            }
        }

        result
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
