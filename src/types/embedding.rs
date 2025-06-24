//! Embedding generation types

use std::collections::HashMap;

/// Embedding response
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// List of embedding vectors
    pub embeddings: Vec<Vec<f32>>,
    /// Model used for embeddings
    pub model: String,
    /// Usage information
    pub usage: Option<EmbeddingUsage>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Embedding usage information
#[derive(Debug, Clone)]
pub struct EmbeddingUsage {
    /// Number of prompt tokens
    pub prompt_tokens: u32,
    /// Total tokens processed
    pub total_tokens: u32,
}
