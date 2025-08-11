//! Gemini Embeddings Implementation
//!
//! This module provides the Gemini implementation of the EmbeddingCapability trait,
//! supporting task-type optimization, title context, and custom dimensions.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::LlmError;
use crate::traits::{EmbeddingCapability, EmbeddingExtensions, GeminiEmbeddingCapability};
use crate::types::{EmbeddingModelInfo, EmbeddingRequest, EmbeddingResponse, EmbeddingTaskType};

use super::types::GeminiConfig;

/// Gemini embedding request structure
#[derive(Debug, Clone, Serialize)]
struct GeminiEmbeddingRequest {
    /// Model name (required for batch requests)
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    /// Content to embed (single content object)
    content: GeminiContent,
    /// Embedding configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_config: Option<GeminiEmbeddingConfig>,
}

/// Gemini embedding configuration
#[derive(Debug, Clone, Serialize)]
struct GeminiEmbeddingConfig {
    /// Task type for optimization
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<String>,
    /// Title for context
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    /// Output dimensions
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<u32>,
}

/// Gemini content structure for embeddings
#[derive(Debug, Clone, Serialize)]
struct GeminiContent {
    /// Content parts
    parts: Vec<GeminiPart>,
}

/// Gemini content part
#[derive(Debug, Clone, Serialize)]
struct GeminiPart {
    /// Text content
    text: String,
}

/// Gemini embedding response structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiEmbeddingResponse {
    /// Embedding data
    embedding: GeminiEmbeddingData,
}

/// Gemini embedding data
#[derive(Debug, Clone, Deserialize)]
struct GeminiEmbeddingData {
    /// Embedding values
    values: Vec<f32>,
}

/// Gemini batch embedding request for multiple contents
#[derive(Debug, Clone, Serialize)]
struct GeminiBatchEmbeddingRequest {
    /// Multiple embedding requests
    requests: Vec<GeminiEmbeddingRequest>,
}

/// Gemini batch embedding response
#[derive(Debug, Clone, Deserialize)]
struct GeminiBatchEmbeddingResponse {
    /// List of embeddings
    embeddings: Vec<GeminiEmbeddingData>,
}

/// Gemini embeddings capability implementation.
///
/// This struct provides the Gemini-specific implementation of text embeddings
/// using the Gemini Embedding API with support for task types and context.
///
/// # Supported Models
/// - text-embedding-004 (768 dimensions)
/// - text-multilingual-embedding-002 (768 dimensions)
///
/// # API Reference
/// <https://ai.google.dev/api/embed-content>
#[derive(Debug, Clone)]
pub struct GeminiEmbeddings {
    /// Gemini configuration
    config: GeminiConfig,
    /// HTTP client
    http_client: reqwest::Client,
}

impl GeminiEmbeddings {
    /// Create a new Gemini embeddings instance
    pub fn new(config: GeminiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Convert task type to Gemini format
    fn convert_task_type(task_type: &EmbeddingTaskType) -> String {
        match task_type {
            EmbeddingTaskType::RetrievalQuery => "RETRIEVAL_QUERY".to_string(),
            EmbeddingTaskType::RetrievalDocument => "RETRIEVAL_DOCUMENT".to_string(),
            EmbeddingTaskType::SemanticSimilarity => "SEMANTIC_SIMILARITY".to_string(),
            EmbeddingTaskType::Classification => "CLASSIFICATION".to_string(),
            EmbeddingTaskType::Clustering => "CLUSTERING".to_string(),
            EmbeddingTaskType::QuestionAnswering => "QUESTION_ANSWERING".to_string(),
            EmbeddingTaskType::FactVerification => "FACT_VERIFICATION".to_string(),
            EmbeddingTaskType::Unspecified => "TASK_TYPE_UNSPECIFIED".to_string(),
        }
    }

    /// Build the request body for Gemini API (single content)
    fn build_request(
        &self,
        text: &str,
        task_type: Option<&EmbeddingTaskType>,
        title: Option<&str>,
        output_dimensionality: Option<u32>,
    ) -> GeminiEmbeddingRequest {
        let content = GeminiContent {
            parts: vec![GeminiPart {
                text: text.to_string(),
            }],
        };

        let embedding_config =
            if task_type.is_some() || title.is_some() || output_dimensionality.is_some() {
                Some(GeminiEmbeddingConfig {
                    task_type: task_type.map(Self::convert_task_type),
                    title: title.map(|s| s.to_string()),
                    output_dimensionality,
                })
            } else {
                None
            };

        GeminiEmbeddingRequest {
            model: None, // Single requests don't need model field
            content,
            embedding_config,
        }
    }

    /// Build batch request for multiple contents
    fn build_batch_request(
        &self,
        texts: &[String],
        task_type: Option<&EmbeddingTaskType>,
        title: Option<&str>,
        output_dimensionality: Option<u32>,
    ) -> GeminiBatchEmbeddingRequest {
        let requests: Vec<GeminiEmbeddingRequest> = texts
            .iter()
            .map(|text| {
                let content = GeminiContent {
                    parts: vec![GeminiPart { text: text.clone() }],
                };

                let embedding_config =
                    if task_type.is_some() || title.is_some() || output_dimensionality.is_some() {
                        Some(GeminiEmbeddingConfig {
                            task_type: task_type.map(Self::convert_task_type),
                            title: title.map(|s| s.to_string()),
                            output_dimensionality,
                        })
                    } else {
                        None
                    };

                GeminiEmbeddingRequest {
                    model: Some(format!("models/{}", self.config.model)),
                    content,
                    embedding_config,
                }
            })
            .collect();

        GeminiBatchEmbeddingRequest { requests }
    }

    /// Make HTTP request to Gemini API for single embedding
    async fn make_request(
        &self,
        request: GeminiEmbeddingRequest,
    ) -> Result<GeminiEmbeddingResponse, LlmError> {
        let model = if !self.config.model.is_empty() {
            &self.config.model
        } else {
            "gemini-embedding-001"
        };

        let url = crate::utils::url::join_url(
            &self.config.base_url,
            &format!("models/{model}:embedContent"),
        );

        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &self.config.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("Gemini API error: {status_code} - {error_text}"),
            ));
        }

        let gemini_response: GeminiEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse Gemini response: {e}")))?;

        Ok(gemini_response)
    }

    /// Make batch request to Gemini API
    async fn make_batch_request(
        &self,
        request: GeminiBatchEmbeddingRequest,
    ) -> Result<GeminiBatchEmbeddingResponse, LlmError> {
        let model = if !self.config.model.is_empty() {
            &self.config.model
        } else {
            "gemini-embedding-001"
        };

        let url = crate::utils::url::join_url(
            &self.config.base_url,
            &format!("models/{model}:batchEmbedContents"),
        );

        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &self.config.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("Gemini batch API error: {status_code} - {error_text}"),
            ));
        }

        let gemini_response: GeminiBatchEmbeddingResponse = response.json().await.map_err(|e| {
            LlmError::ParseError(format!("Failed to parse Gemini batch response: {e}"))
        })?;

        Ok(gemini_response)
    }

    /// Convert Gemini response to our standard format
    fn convert_response(&self, gemini_response: GeminiEmbeddingResponse) -> EmbeddingResponse {
        let embeddings = vec![gemini_response.embedding.values];
        let model = if !self.config.model.is_empty() {
            self.config.model.clone()
        } else {
            "gemini-embedding-001".to_string()
        };

        EmbeddingResponse::new(embeddings, model)
    }

    /// Convert Gemini batch response to our standard format
    fn convert_batch_response(
        &self,
        gemini_response: GeminiBatchEmbeddingResponse,
    ) -> EmbeddingResponse {
        let embeddings: Vec<Vec<f32>> = gemini_response
            .embeddings
            .into_iter()
            .map(|embedding| embedding.values)
            .collect();

        let model = if !self.config.model.is_empty() {
            self.config.model.clone()
        } else {
            "gemini-embedding-001".to_string()
        };

        EmbeddingResponse::new(embeddings, model)
    }

    /// Get model information for Gemini embedding models
    fn get_model_info(&self, model_id: &str) -> EmbeddingModelInfo {
        match model_id {
            "gemini-embedding-001" => EmbeddingModelInfo::new(
                model_id.to_string(),
                "Gemini Embedding 001".to_string(),
                3072, // Default dimension, can be customized
                2048,
            )
            .with_task(EmbeddingTaskType::RetrievalQuery)
            .with_task(EmbeddingTaskType::RetrievalDocument)
            .with_task(EmbeddingTaskType::SemanticSimilarity)
            .with_task(EmbeddingTaskType::Classification)
            .with_task(EmbeddingTaskType::Clustering)
            .with_task(EmbeddingTaskType::QuestionAnswering)
            .with_task(EmbeddingTaskType::FactVerification),

            _ => EmbeddingModelInfo::new(model_id.to_string(), model_id.to_string(), 3072, 2048),
        }
    }
}

#[async_trait]
impl EmbeddingCapability for GeminiEmbeddings {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        if input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        if input.len() == 1 {
            // Single embedding request
            let request = self.build_request(&input[0], None, None, None);
            let response = self.make_request(request).await?;
            Ok(self.convert_response(response))
        } else {
            // Batch embedding request
            let batch_request = self.build_batch_request(&input, None, None, None);
            let batch_response = self.make_batch_request(batch_request).await?;
            Ok(self.convert_batch_response(batch_response))
        }
    }

    fn embedding_dimension(&self) -> usize {
        3072 // Default dimension for Gemini embedding models (can be customized)
    }

    fn max_tokens_per_embedding(&self) -> usize {
        2048 // Gemini's current limit
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        vec!["gemini-embedding-001".to_string()]
    }
}

#[async_trait]
impl EmbeddingExtensions for GeminiEmbeddings {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        if request.input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        // Extract Gemini-specific parameters
        let task_type = request
            .provider_params
            .get("task_type")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "RETRIEVAL_QUERY" => EmbeddingTaskType::RetrievalQuery,
                "RETRIEVAL_DOCUMENT" => EmbeddingTaskType::RetrievalDocument,
                "SEMANTIC_SIMILARITY" => EmbeddingTaskType::SemanticSimilarity,
                "CLASSIFICATION" => EmbeddingTaskType::Classification,
                "CLUSTERING" => EmbeddingTaskType::Clustering,
                "QUESTION_ANSWERING" => EmbeddingTaskType::QuestionAnswering,
                "FACT_VERIFICATION" => EmbeddingTaskType::FactVerification,
                _ => EmbeddingTaskType::Unspecified,
            });

        let title = request
            .provider_params
            .get("title")
            .and_then(|v| v.as_str());

        if request.input.len() == 1 {
            // Single embedding request
            let gemini_request = self.build_request(
                &request.input[0],
                task_type.as_ref(),
                title,
                request.dimensions,
            );

            let response = self.make_request(gemini_request).await?;
            Ok(self.convert_response(response))
        } else {
            // Batch embedding request
            let batch_request = self.build_batch_request(
                &request.input,
                task_type.as_ref(),
                title,
                request.dimensions,
            );

            let batch_response = self.make_batch_request(batch_request).await?;
            Ok(self.convert_batch_response(batch_response))
        }
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
impl GeminiEmbeddingCapability for GeminiEmbeddings {
    async fn embed_with_task_type(
        &self,
        input: Vec<String>,
        task_type: EmbeddingTaskType,
    ) -> Result<EmbeddingResponse, LlmError> {
        if input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        if input.len() == 1 {
            let request = self.build_request(&input[0], Some(&task_type), None, None);
            let response = self.make_request(request).await?;
            Ok(self.convert_response(response))
        } else {
            let batch_request = self.build_batch_request(&input, Some(&task_type), None, None);
            let batch_response = self.make_batch_request(batch_request).await?;
            Ok(self.convert_batch_response(batch_response))
        }
    }

    async fn embed_with_title(
        &self,
        input: Vec<String>,
        title: String,
    ) -> Result<EmbeddingResponse, LlmError> {
        if input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        if input.len() == 1 {
            let request = self.build_request(&input[0], None, Some(&title), None);
            let response = self.make_request(request).await?;
            Ok(self.convert_response(response))
        } else {
            let batch_request = self.build_batch_request(&input, None, Some(&title), None);
            let batch_response = self.make_batch_request(batch_request).await?;
            Ok(self.convert_batch_response(batch_response))
        }
    }

    async fn embed_with_output_dimensionality(
        &self,
        input: Vec<String>,
        output_dimensionality: u32,
    ) -> Result<EmbeddingResponse, LlmError> {
        if input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        if input.len() == 1 {
            let request = self.build_request(&input[0], None, None, Some(output_dimensionality));
            let response = self.make_request(request).await?;
            Ok(self.convert_response(response))
        } else {
            let batch_request =
                self.build_batch_request(&input, None, None, Some(output_dimensionality));
            let batch_response = self.make_batch_request(batch_request).await?;
            Ok(self.convert_batch_response(batch_response))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_task_type() {
        assert_eq!(
            GeminiEmbeddings::convert_task_type(&EmbeddingTaskType::RetrievalQuery),
            "RETRIEVAL_QUERY"
        );
        assert_eq!(
            GeminiEmbeddings::convert_task_type(&EmbeddingTaskType::SemanticSimilarity),
            "SEMANTIC_SIMILARITY"
        );
        assert_eq!(
            GeminiEmbeddings::convert_task_type(&EmbeddingTaskType::Unspecified),
            "TASK_TYPE_UNSPECIFIED"
        );
    }

    #[test]
    fn test_embedding_dimensions() {
        let config = GeminiConfig {
            api_key: "test-key".to_string(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            model: "gemini-embedding-001".to_string(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
        };
        let client = reqwest::Client::new();
        let embeddings = GeminiEmbeddings::new(config, client);

        assert_eq!(embeddings.embedding_dimension(), 3072);
        assert_eq!(embeddings.max_tokens_per_embedding(), 2048);
    }

    #[test]
    fn test_supported_models() {
        let config = GeminiConfig {
            api_key: "test-key".to_string(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            model: "gemini-embedding-001".to_string(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
        };
        let client = reqwest::Client::new();
        let embeddings = GeminiEmbeddings::new(config, client);

        let models = embeddings.supported_embedding_models();
        assert!(models.contains(&"gemini-embedding-001".to_string()));
    }

    #[test]
    fn test_model_info() {
        let config = GeminiConfig {
            api_key: "test-key".to_string(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            model: "gemini-embedding-001".to_string(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
        };
        let client = reqwest::Client::new();
        let embeddings = GeminiEmbeddings::new(config, client);

        let info = embeddings.get_model_info("gemini-embedding-001");
        assert_eq!(info.id, "gemini-embedding-001");
        assert_eq!(info.dimension, 3072);
        assert!(
            info.supported_tasks
                .contains(&EmbeddingTaskType::RetrievalQuery)
        );
    }
}
