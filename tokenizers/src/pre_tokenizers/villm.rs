use serde::{Deserialize, Serialize};

use crate::tokenizer::{PreTokenizer, Result};

/// A minimal ViLLM pre-tokenizer that performs no splitting.
///
/// The ViLLMModel handles its own pre-tokenization internally within
/// `Model::tokenize()`, so this pre-tokenizer acts as a passthrough.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct ViLLMPreTokenizer;

impl PreTokenizer for ViLLMPreTokenizer {
    fn pre_tokenize(&self, _pretokenized: &mut crate::tokenizer::PreTokenizedString) -> Result<()> {
        Ok(())
    }
}
