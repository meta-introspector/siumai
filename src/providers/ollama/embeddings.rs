//! Ollama Embeddings Capability Implementation
//!
//! Implements the `EmbeddingCapability` trait for Ollama using the /api/embed endpoint.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::traits::EmbeddingCapability;
use crate::types::*;

use super::config::OllamaParams;
use super::types::*;
use super::utils::*;

/// Ollama Embedding Capability Implementation
pub struct OllamaEmbeddingCapability {
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
    pub ollama_params: OllamaParams,
}

impl OllamaEmbeddingCapability {
    /// Creates a new Ollama embedding capability
    pub const fn new(
        base_url: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
        ollama_params: OllamaParams,
    ) -> Self {
        Self {
            base_url,
            http_client,
            http_config,
            ollama_params,
        }
    }

    /// Build embedding request body
    fn build_embedding_request_body(
        &self,
        texts: &[String],
        model: Option<&str>,
    ) -> Result<OllamaEmbeddingRequest, LlmError> {
        let model = model
            .unwrap_or("nomic-embed-text") // Default embedding model
            .to_string();

        validate_model_name(&model)?;

        // Convert input to appropriate format
        let input = if texts.len() == 1 {
            serde_json::Value::String(texts[0].clone())
        } else {
            serde_json::Value::Array(
                texts
                    .iter()
                    .map(|t| serde_json::Value::String(t.clone()))
                    .collect(),
            )
        };

        // Build model options
        let options = build_model_options(
            None, // temperature (not applicable for embeddings)
            None, // max_tokens (not applicable for embeddings)
            None, // top_p (not applicable for embeddings)
            None, // frequency_penalty (not applicable for embeddings)
            None, // presence_penalty (not applicable for embeddings)
            self.ollama_params.options.as_ref(),
        );

        Ok(OllamaEmbeddingRequest {
            model,
            input,
            truncate: Some(true), // Default to true for safety
            options: if options.is_empty() {
                None
            } else {
                Some(options)
            },
            keep_alive: self.ollama_params.keep_alive.clone(),
        })
    }

    /// Parse embedding response
    fn parse_embedding_response(&self, response: OllamaEmbeddingResponse) -> EmbeddingResponse {
        // Convert f64 to f32
        let embeddings: Vec<Vec<f32>> = response
            .embeddings
            .into_iter()
            .map(|embedding| embedding.into_iter().map(|x| x as f32).collect())
            .collect();

        EmbeddingResponse {
            embeddings,
            model: response.model,
            usage: Some(crate::types::EmbeddingUsage {
                prompt_tokens: response.prompt_eval_count.unwrap_or(0),
                total_tokens: response.prompt_eval_count.unwrap_or(0),
            }),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Embed single text
    pub async fn embed_single(
        &self,
        text: String,
        model: Option<String>,
    ) -> Result<Vec<f64>, LlmError> {
        let headers = build_headers(&self.http_config.headers)?;
        let body = self.build_embedding_request_body(&[text], model.as_deref())?;
        let url = format!("{}/api/embed", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpError(format!(
                "Embedding request failed: {status} - {error_text}"
            )));
        }

        let ollama_response: OllamaEmbeddingResponse = response.json().await?;

        if ollama_response.embeddings.is_empty() {
            return Err(LlmError::ParseError("No embeddings returned".to_string()));
        }

        Ok(ollama_response.embeddings[0].clone())
    }

    /// Embed multiple texts with custom model
    pub async fn embed_with_model(
        &self,
        texts: Vec<String>,
        model: String,
    ) -> Result<EmbeddingResponse, LlmError> {
        let headers = build_headers(&self.http_config.headers)?;
        let body = self.build_embedding_request_body(&texts, Some(&model))?;
        let url = format!("{}/api/embed", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpError(format!(
                "Embedding request failed: {status} - {error_text}"
            )));
        }

        let ollama_response: OllamaEmbeddingResponse = response.json().await?;
        Ok(self.parse_embedding_response(ollama_response))
    }

    /// Get available embedding models
    pub fn get_embedding_models() -> Vec<String> {
        vec![
            "nomic-embed-text:latest".to_string(),
            "all-minilm:latest".to_string(),
            "mxbai-embed-large:latest".to_string(),
            "snowflake-arctic-embed:latest".to_string(),
        ]
    }
}

#[async_trait]
impl EmbeddingCapability for OllamaEmbeddingCapability {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        let headers = build_headers(&self.http_config.headers)?;
        let body = self.build_embedding_request_body(&texts, None)?;
        let url = format!("{}/api/embed", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpError(format!(
                "Embedding request failed: {status} - {error_text}"
            )));
        }

        let ollama_response: OllamaEmbeddingResponse = response.json().await?;
        Ok(self.parse_embedding_response(ollama_response))
    }

    fn embedding_dimension(&self) -> usize {
        // Return a reasonable default dimension for Ollama embedding models
        // Note: The actual dimension depends on the specific model being used:
        // - nomic-embed-text: 768
        // - all-minilm: 384
        // - mxbai-embed-large: 1024
        // - snowflake-arctic-embed: 1024
        // Since we can't determine the exact model at this point, we use a common default
        384
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_embedding_request_body() {
        let capability = OllamaEmbeddingCapability::new(
            "http://localhost:11434".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
            OllamaParams::default(),
        );

        let texts = vec!["Hello world".to_string()];
        let body = capability
            .build_embedding_request_body(&texts, Some("nomic-embed-text"))
            .unwrap();

        assert_eq!(body.model, "nomic-embed-text");
        assert_eq!(body.truncate, Some(true));

        if let serde_json::Value::String(input_text) = body.input {
            assert_eq!(input_text, "Hello world");
        } else {
            panic!("Expected string input for single text");
        }
    }

    #[test]
    fn test_build_embedding_request_body_multiple() {
        let capability = OllamaEmbeddingCapability::new(
            "http://localhost:11434".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
            OllamaParams::default(),
        );

        let texts = vec!["Hello".to_string(), "World".to_string()];
        let body = capability
            .build_embedding_request_body(&texts, Some("nomic-embed-text"))
            .unwrap();

        assert_eq!(body.model, "nomic-embed-text");

        if let serde_json::Value::Array(input_array) = body.input {
            assert_eq!(input_array.len(), 2);
        } else {
            panic!("Expected array input for multiple texts");
        }
    }

    #[test]
    fn test_parse_embedding_response() {
        let capability = OllamaEmbeddingCapability::new(
            "http://localhost:11434".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
            OllamaParams::default(),
        );

        let ollama_response = OllamaEmbeddingResponse {
            model: "nomic-embed-text".to_string(),
            embeddings: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            total_duration: Some(1_000_000_000),
            load_duration: Some(100_000_000),
            prompt_eval_count: Some(10),
        };

        let response = capability.parse_embedding_response(ollama_response);
        assert_eq!(response.model, "nomic-embed-text".to_string());
        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(response.embeddings[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(response.embeddings[1], vec![0.4, 0.5, 0.6]);
        assert_eq!(response.usage.unwrap().prompt_tokens, 10);
    }

    #[test]
    fn test_get_embedding_models() {
        let models = OllamaEmbeddingCapability::get_embedding_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"nomic-embed-text:latest".to_string()));
        assert!(models.contains(&"all-minilm:latest".to_string()));
    }
}
