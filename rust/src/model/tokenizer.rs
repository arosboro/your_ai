//! Tokenizer integration using HuggingFace tokenizers

use std::path::Path;
use tokenizers::Tokenizer;

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
}

impl TokenizerWrapper {
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }

    pub fn from_pretrained(_model_id: &str) -> anyhow::Result<Self> {
        // Placeholder - would download from HuggingFace Hub
        anyhow::bail!(
            "from_pretrained not yet implemented - use from_file with a local tokenizer.json"
        )
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))
    }

    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Batch tokenization error: {}", e))?;
        Ok(encodings.iter().map(|e| e.get_ids().to_vec()).collect())
    }
}
