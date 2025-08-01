//! OpenAI Embeddings Implementation
//!
//! This module provides the OpenAI implementation of embedding capabilities,
//! supporting all features including custom dimensions, encoding formats, and batch processing.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::LlmError;
use crate::traits::{EmbeddingCapability, EmbeddingExtensions, OpenAiEmbeddingCapability};
use crate::types::{
    EmbeddingFormat, EmbeddingModelInfo, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
};

use super::config::OpenAiConfig;

/// `OpenAI` embeddings API request structure
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

/// `OpenAI` embeddings API response structure
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
/// This struct provides a comprehensive implementation of OpenAI's embedding capabilities,
/// including support for custom dimensions, encoding formats, and advanced features.
///
/// # Supported Models
/// - text-embedding-3-small (1536 dimensions, configurable)
/// - text-embedding-3-large (3072 dimensions, configurable)
/// - text-embedding-ada-002 (1536 dimensions, legacy)
///
/// # API Reference
/// <https://platform.openai.com/docs/api-reference/embeddings>
#[derive(Debug, Clone)]
pub struct OpenAiEmbeddings {
    /// OpenAI configuration
    config: OpenAiConfig,
    /// HTTP client
    http_client: reqwest::Client,
}

impl OpenAiEmbeddings {
    /// Create a new OpenAI embeddings instance
    pub const fn new(config: OpenAiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Get the default embedding model
    fn default_model(&self) -> String {
        "text-embedding-3-small".to_string()
    }

    /// Build the request body for OpenAI API
    fn build_request(&self, request: &EmbeddingRequest) -> OpenAiEmbeddingRequest {
        let model = request
            .model
            .clone()
            .or_else(|| {
                if !self.config.common_params.model.is_empty() {
                    Some(self.config.common_params.model.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| self.default_model());

        let encoding_format = request.encoding_format.as_ref().map(|f| match f {
            EmbeddingFormat::Float => "float".to_string(),
            EmbeddingFormat::Base64 => "base64".to_string(),
        });

        OpenAiEmbeddingRequest {
            input: request.input.clone(),
            model,
            encoding_format,
            dimensions: request.dimensions,
            user: request
                .user
                .clone()
                .or_else(|| self.config.openai_params.user.clone()),
        }
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
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
            headers.insert(header_name, header_value);
        }

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("OpenAI API error {status}: {error_text}"),
                details: None,
            });
        }

        let openai_response: OpenAiEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {e}")))?;

        Ok(openai_response)
    }

    /// Convert OpenAI response to our standard format
    fn convert_response(&self, openai_response: OpenAiEmbeddingResponse) -> EmbeddingResponse {
        // Sort embeddings by index to maintain order
        let mut embeddings_with_index: Vec<_> = openai_response.data.into_iter().collect();
        embeddings_with_index.sort_by_key(|obj| obj.index);

        let embeddings: Vec<Vec<f32>> = embeddings_with_index
            .into_iter()
            .map(|obj| obj.embedding)
            .collect();

        EmbeddingResponse::new(embeddings, openai_response.model).with_usage(EmbeddingUsage::new(
            openai_response.usage.prompt_tokens,
            openai_response.usage.total_tokens,
        ))
    }

    /// Get model information for OpenAI embedding models
    fn get_model_info(&self, model_id: &str) -> EmbeddingModelInfo {
        match model_id {
            "text-embedding-3-small" => EmbeddingModelInfo::new(
                model_id.to_string(),
                "Text Embedding 3 Small".to_string(),
                1536,
                8192,
            )
            .with_custom_dimensions(),

            "text-embedding-3-large" => EmbeddingModelInfo::new(
                model_id.to_string(),
                "Text Embedding 3 Large".to_string(),
                3072,
                8192,
            )
            .with_custom_dimensions(),

            "text-embedding-ada-002" => EmbeddingModelInfo::new(
                model_id.to_string(),
                "Text Embedding Ada 002 (Legacy)".to_string(),
                1536,
                8192,
            ),

            _ => EmbeddingModelInfo::new(model_id.to_string(), model_id.to_string(), 1536, 8192),
        }
    }
}

#[async_trait]
impl EmbeddingCapability for OpenAiEmbeddings {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        if input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        let request = EmbeddingRequest::new(input);
        self.embed_with_config(request).await
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

    fn supported_embedding_models(&self) -> Vec<String> {
        vec![
            "text-embedding-3-small".to_string(),
            "text-embedding-3-large".to_string(),
            "text-embedding-ada-002".to_string(),
        ]
    }
}

#[async_trait]
impl EmbeddingExtensions for OpenAiEmbeddings {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        if request.input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        let openai_request = self.build_request(&request);
        let openai_response = self.make_request(openai_request).await?;
        Ok(self.convert_response(openai_response))
    }

    async fn list_embedding_models(&self) -> Result<Vec<EmbeddingModelInfo>, LlmError> {
        let models = self.supported_embedding_models();
        let model_infos = models
            .into_iter()
            .map(|id| self.get_model_info(&id))
            .collect();
        Ok(model_infos)
    }
}

#[async_trait]
impl OpenAiEmbeddingCapability for OpenAiEmbeddings {
    async fn embed_with_dimensions(
        &self,
        input: Vec<String>,
        dimensions: u32,
    ) -> Result<EmbeddingResponse, LlmError> {
        let request = EmbeddingRequest::new(input).with_dimensions(dimensions);
        self.embed_with_config(request).await
    }

    async fn embed_with_format(
        &self,
        input: Vec<String>,
        format: EmbeddingFormat,
    ) -> Result<EmbeddingResponse, LlmError> {
        let request = EmbeddingRequest::new(input).with_encoding_format(format);
        self.embed_with_config(request).await
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

    #[test]
    fn test_model_info() {
        let config = OpenAiConfig::new("test-key");
        let client = reqwest::Client::new();
        let embeddings = OpenAiEmbeddings::new(config, client);

        let info = embeddings.get_model_info("text-embedding-3-small");
        assert_eq!(info.id, "text-embedding-3-small");
        assert_eq!(info.dimension, 1536);
        assert!(info.supports_custom_dimensions);
    }

    #[test]
    fn test_build_request() {
        let config = OpenAiConfig::new("test-key");
        let client = reqwest::Client::new();
        let embeddings = OpenAiEmbeddings::new(config, client);

        let request = EmbeddingRequest::new(vec!["test".to_string()])
            .with_model("text-embedding-3-large")
            .with_dimensions(2048)
            .with_encoding_format(EmbeddingFormat::Float);

        let openai_request = embeddings.build_request(&request);
        assert_eq!(openai_request.model, "text-embedding-3-large");
        assert_eq!(openai_request.dimensions, Some(2048));
        assert_eq!(openai_request.encoding_format, Some("float".to_string()));
    }
}
