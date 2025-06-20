//! OpenAI Embeddings Implementation
//!
//! This module provides the OpenAI implementation of the EmbeddingCapability trait.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::traits::EmbeddingCapability;
use crate::types::{EmbeddingResponse, EmbeddingUsage};

use super::config::OpenAiConfig;

/// OpenAI embeddings API request structure
#[derive(Debug, Clone, Serialize)]
struct OpenAiEmbeddingRequest {
    /// Input text(s) to embed
    input: Vec<String>,
    /// Model to use for embeddings
    model: String,
    /// Encoding format (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    /// Number of dimensions (optional, for newer models)
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    /// User identifier (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

/// OpenAI embeddings API response structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAiEmbeddingResponse {
    /// List of embedding objects
    data: Vec<OpenAiEmbeddingObject>,
    /// Model used
    model: String,
    /// Usage information
    usage: OpenAiEmbeddingUsage,
}

/// Individual embedding object
#[derive(Debug, Clone, Deserialize)]
struct OpenAiEmbeddingObject {
    /// Embedding vector
    embedding: Vec<f32>,
    /// Index in the input array
    index: usize,
}

/// Usage information for embeddings
#[derive(Debug, Clone, Deserialize)]
struct OpenAiEmbeddingUsage {
    /// Number of prompt tokens
    prompt_tokens: u32,
    /// Total tokens
    total_tokens: u32,
}

/// OpenAI embeddings capability implementation.
///
/// This struct provides the OpenAI-specific implementation of text embeddings
/// using the OpenAI Embeddings API.
///
/// # Supported Models
/// - text-embedding-3-small (1536 dimensions)
/// - text-embedding-3-large (3072 dimensions, configurable)
/// - text-embedding-ada-002 (1536 dimensions, legacy)
///
/// # API Reference
/// https://platform.openai.com/docs/api-reference/embeddings
#[derive(Debug, Clone)]
pub struct OpenAiEmbeddings {
    /// OpenAI configuration
    config: OpenAiConfig,
    /// HTTP client
    http_client: reqwest::Client,
}

impl OpenAiEmbeddings {
    /// Create a new OpenAI embeddings instance.
    ///
    /// # Arguments
    /// * `config` - OpenAI configuration
    /// * `http_client` - HTTP client for making requests
    pub fn new(config: OpenAiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Get the default embedding model.
    fn default_model(&self) -> String {
        "text-embedding-3-small".to_string()
    }

    /// Make an embeddings API request.
    async fn make_request(
        &self,
        request: OpenAiEmbeddingRequest,
    ) -> Result<OpenAiEmbeddingResponse, LlmError> {
        let url = format!("{}/embeddings", self.config.base_url);

        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.get_headers() {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {}", e)))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {}", e)))?;
            headers.insert(header_name, header_value);
        }

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("OpenAI API error {}: {}", status, error_text),
                details: None,
            });
        }

        let openai_response: OpenAiEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {}", e)))?;

        Ok(openai_response)
    }

    /// Convert OpenAI response to our standard format.
    fn convert_response(&self, openai_response: OpenAiEmbeddingResponse) -> EmbeddingResponse {
        // Sort embeddings by index to maintain order
        let mut embeddings_with_index: Vec<_> = openai_response.data.into_iter().collect();
        embeddings_with_index.sort_by_key(|obj| obj.index);

        let embeddings: Vec<Vec<f32>> = embeddings_with_index
            .into_iter()
            .map(|obj| obj.embedding)
            .collect();

        EmbeddingResponse {
            embeddings,
            model: openai_response.model,
            usage: Some(EmbeddingUsage {
                prompt_tokens: openai_response.usage.prompt_tokens,
                total_tokens: openai_response.usage.total_tokens,
            }),
            metadata: HashMap::new(),
        }
    }
}

#[async_trait]
impl EmbeddingCapability for OpenAiEmbeddings {
    /// Generate embeddings for the given input texts.
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        if input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        // Use model from common params or default
        let model = if !self.config.common_params.model.is_empty() {
            self.config.common_params.model.clone()
        } else {
            self.default_model()
        };

        let request = OpenAiEmbeddingRequest {
            input,
            model,
            encoding_format: Some("float".to_string()),
            dimensions: None, // Let OpenAI use default dimensions
            user: self.config.openai_params.user.clone(),
        };

        let openai_response = self.make_request(request).await?;
        Ok(self.convert_response(openai_response))
    }

    /// Get the dimension of embeddings produced by this provider.
    fn embedding_dimension(&self) -> usize {
        // Return dimension based on model
        let model = if !self.config.common_params.model.is_empty() {
            &self.config.common_params.model
        } else {
            "text-embedding-3-small"
        };

        match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536, // Default fallback
        }
    }

    /// Get the maximum number of tokens that can be embedded at once.
    fn max_tokens_per_embedding(&self) -> usize {
        8192 // OpenAI's current limit
    }

    /// Get supported embedding models for OpenAI.
    fn supported_embedding_models(&self) -> Vec<String> {
        vec![
            "text-embedding-3-small".to_string(),
            "text-embedding-3-large".to_string(),
            "text-embedding-ada-002".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::OpenAiConfig;

    #[test]
    fn test_embedding_dimensions() {
        let config = OpenAiConfig::new("test-key");
        let client = reqwest::Client::new();
        let embeddings = OpenAiEmbeddings::new(config, client);

        assert_eq!(embeddings.embedding_dimension(), 1536);
        assert_eq!(embeddings.max_tokens_per_embedding(), 8192);
    }

    #[test]
    fn test_supported_models() {
        let config = OpenAiConfig::new("test-key");
        let client = reqwest::Client::new();
        let embeddings = OpenAiEmbeddings::new(config, client);

        let models = embeddings.supported_embedding_models();
        assert!(models.contains(&"text-embedding-3-small".to_string()));
        assert!(models.contains(&"text-embedding-3-large".to_string()));
        assert!(models.contains(&"text-embedding-ada-002".to_string()));
    }
}
