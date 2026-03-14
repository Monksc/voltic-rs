pub mod bpe;
pub mod trietoken;
pub mod wordpiece;

pub use bpe::BpeTokenizer;
pub use trietoken::TrieTokenTokenizer;
pub use wordpiece::WordPieceTokenizer;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
    fn vocab_size(&self) -> usize;
}

fn pre_tokenize(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if c.is_whitespace() {
            if !current.is_empty() {
                result.push(current.clone());
                current.clear();
            }
            result.push(c.to_string());
        } else if !c.is_alphanumeric() && c != '\'' {
            if !current.is_empty() {
                result.push(current.clone());
                current.clear();
            }
            result.push(c.to_string());
        } else {
            current.push(c);
        }
    }

    if !current.is_empty() {
        result.push(current);
    }

    result
}

fn pre_tokenize_words(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if c.is_whitespace() {
            if !current.is_empty() {
                result.push(current.clone());
                current.clear();
            }
        } else {
            current.push(c);
        }
    }

    if !current.is_empty() {
        result.push(current);
    }

    result
}
