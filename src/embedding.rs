//! Unified Embedding Interface
//!
//! This module provides a unified interface for embedding functionality across all providers,
//! with builder patterns for easy configuration and provider-specific optimizations.

use std::collections::HashMap;

use crate::error::LlmError;
use crate::traits::EmbeddingExtensions;
use crate::types::{
    BatchEmbeddingRequest, BatchOptions, EmbeddingFormat, EmbeddingModelInfo, EmbeddingRequest,
    EmbeddingResponse, EmbeddingTaskType,
};

/// Unified embedding client that works with any provider
pub struct EmbeddingClient {
    capability: Box<dyn EmbeddingExtensions>,
}

impl EmbeddingClient {
    /// Create a new embedding client with the given capability
    pub fn new(capability: Box<dyn EmbeddingExtensions>) -> Self {
        Self { capability }
    }

    /// Generate embeddings for the given texts
    pub async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        self.capability.embed(texts).await
    }

    /// Generate embeddings with advanced configuration
    pub async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        self.capability.embed_with_config(request).await
    }

    /// Process multiple embedding requests in batch
    pub async fn embed_batch(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<crate::types::BatchEmbeddingResponse, LlmError> {
        self.capability.embed_batch(requests).await
    }

    /// Get information about available embedding models
    pub async fn list_models(&self) -> Result<Vec<EmbeddingModelInfo>, LlmError> {
        self.capability.list_embedding_models().await
    }

    /// Calculate similarity between two embedding vectors
    pub fn calculate_similarity(
        &self,
        embedding1: &[f32],
        embedding2: &[f32],
    ) -> Result<f32, LlmError> {
        self.capability.calculate_similarity(embedding1, embedding2)
    }

    /// Get the dimension of embeddings produced by this provider
    pub fn embedding_dimension(&self) -> usize {
        self.capability.embedding_dimension()
    }

    /// Get the maximum number of tokens that can be embedded at once
    pub fn max_tokens_per_embedding(&self) -> usize {
        self.capability.max_tokens_per_embedding()
    }

    /// Get supported embedding models for this provider
    pub fn supported_models(&self) -> Vec<String> {
        self.capability.supported_embedding_models()
    }
}

/// Builder for creating embedding requests with fluent API
#[derive(Debug, Clone, Default)]
pub struct EmbeddingRequestBuilder {
    input: Vec<String>,
    model: Option<String>,
    dimensions: Option<u32>,
    encoding_format: Option<EmbeddingFormat>,
    user: Option<String>,
    provider_params: HashMap<String, serde_json::Value>,
}

impl EmbeddingRequestBuilder {
    /// Create a new embedding request builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder with input texts
    pub fn with_input(texts: Vec<String>) -> Self {
        Self {
            input: texts,
            ..Default::default()
        }
    }

    /// Add input texts
    pub fn input(mut self, texts: Vec<String>) -> Self {
        self.input = texts;
        self
    }

    /// Add a single input text
    pub fn add_text(mut self, text: impl Into<String>) -> Self {
        self.input.push(text.into());
        self
    }

    /// Set the model to use
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set custom dimensions
    pub fn dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set encoding format
    pub fn encoding_format(mut self, format: EmbeddingFormat) -> Self {
        self.encoding_format = Some(format);
        self
    }

    /// Set user identifier
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Add provider-specific parameter
    pub fn provider_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.provider_params.insert(key.into(), value);
        self
    }

    /// Set task type for Gemini (provider-specific)
    pub fn task_type(mut self, task_type: EmbeddingTaskType) -> Self {
        let task_str = match task_type {
            EmbeddingTaskType::RetrievalQuery => "RETRIEVAL_QUERY",
            EmbeddingTaskType::RetrievalDocument => "RETRIEVAL_DOCUMENT",
            EmbeddingTaskType::SemanticSimilarity => "SEMANTIC_SIMILARITY",
            EmbeddingTaskType::Classification => "CLASSIFICATION",
            EmbeddingTaskType::Clustering => "CLUSTERING",
            EmbeddingTaskType::QuestionAnswering => "QUESTION_ANSWERING",
            EmbeddingTaskType::FactVerification => "FACT_VERIFICATION",
            EmbeddingTaskType::Unspecified => "TASK_TYPE_UNSPECIFIED",
        };
        self.provider_params.insert(
            "task_type".to_string(),
            serde_json::Value::String(task_str.to_string()),
        );
        self
    }

    /// Set title for Gemini (provider-specific)
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.provider_params
            .insert("title".to_string(), serde_json::Value::String(title.into()));
        self
    }

    /// Set truncation for Ollama (provider-specific)
    pub fn truncate(mut self, truncate: bool) -> Self {
        self.provider_params
            .insert("truncate".to_string(), serde_json::Value::Bool(truncate));
        self
    }

    /// Set keep-alive for Ollama (provider-specific)
    pub fn keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.provider_params.insert(
            "keep_alive".to_string(),
            serde_json::Value::String(keep_alive.into()),
        );
        self
    }

    /// Build the embedding request
    pub fn build(self) -> Result<EmbeddingRequest, LlmError> {
        if self.input.is_empty() {
            return Err(LlmError::InvalidInput("Input cannot be empty".to_string()));
        }

        Ok(EmbeddingRequest {
            input: self.input,
            model: self.model,
            dimensions: self.dimensions,
            encoding_format: self.encoding_format,
            user: self.user,
            provider_params: self.provider_params,
        })
    }
}

/// Builder for creating batch embedding requests
#[derive(Debug, Clone, Default)]
pub struct BatchEmbeddingRequestBuilder {
    requests: Vec<EmbeddingRequest>,
    max_concurrency: Option<usize>,
    request_timeout: Option<std::time::Duration>,
    fail_fast: bool,
}

impl BatchEmbeddingRequestBuilder {
    /// Create a new batch embedding request builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an embedding request to the batch
    pub fn add_request(mut self, request: EmbeddingRequest) -> Self {
        self.requests.push(request);
        self
    }

    /// Add multiple embedding requests to the batch
    pub fn add_requests(mut self, requests: Vec<EmbeddingRequest>) -> Self {
        self.requests.extend(requests);
        self
    }

    /// Set maximum concurrent requests
    pub fn max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = Some(max_concurrency);
        self
    }

    /// Set timeout for each request
    pub fn request_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// Enable fail-fast mode (stop on first error)
    pub fn fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }

    /// Build the batch embedding request
    pub fn build(self) -> Result<BatchEmbeddingRequest, LlmError> {
        if self.requests.is_empty() {
            return Err(LlmError::InvalidInput("Batch cannot be empty".to_string()));
        }

        Ok(BatchEmbeddingRequest {
            requests: self.requests,
            batch_options: BatchOptions {
                max_concurrency: self.max_concurrency,
                request_timeout: self.request_timeout,
                fail_fast: self.fail_fast,
            },
        })
    }
}

/// Convenience functions for common embedding operations
pub mod convenience {
    use super::*;

    /// Create a simple embedding request for a single text
    pub fn embed_text(text: impl Into<String>) -> EmbeddingRequestBuilder {
        EmbeddingRequestBuilder::with_input(vec![text.into()])
    }

    /// Create a simple embedding request for multiple texts
    pub fn embed_texts(texts: Vec<String>) -> EmbeddingRequestBuilder {
        EmbeddingRequestBuilder::with_input(texts)
    }

    /// Create an embedding request optimized for retrieval queries
    pub fn embed_query(query: impl Into<String>) -> EmbeddingRequestBuilder {
        EmbeddingRequestBuilder::with_input(vec![query.into()])
            .task_type(EmbeddingTaskType::RetrievalQuery)
    }

    /// Create an embedding request optimized for retrieval documents
    pub fn embed_document(document: impl Into<String>) -> EmbeddingRequestBuilder {
        EmbeddingRequestBuilder::with_input(vec![document.into()])
            .task_type(EmbeddingTaskType::RetrievalDocument)
    }

    /// Create an embedding request for semantic similarity
    pub fn embed_for_similarity(text: impl Into<String>) -> EmbeddingRequestBuilder {
        EmbeddingRequestBuilder::with_input(vec![text.into()])
            .task_type(EmbeddingTaskType::SemanticSimilarity)
    }

    /// Create an embedding request for classification
    pub fn embed_for_classification(text: impl Into<String>) -> EmbeddingRequestBuilder {
        EmbeddingRequestBuilder::with_input(vec![text.into()])
            .task_type(EmbeddingTaskType::Classification)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_request_builder() {
        let request = EmbeddingRequestBuilder::new()
            .add_text("Hello, world!")
            .model("text-embedding-3-small")
            .dimensions(1024)
            .encoding_format(EmbeddingFormat::Float)
            .user("test-user")
            .task_type(EmbeddingTaskType::RetrievalQuery)
            .build()
            .unwrap();

        assert_eq!(request.input, vec!["Hello, world!"]);
        assert_eq!(request.model, Some("text-embedding-3-small".to_string()));
        assert_eq!(request.dimensions, Some(1024));
        assert_eq!(request.user, Some("test-user".to_string()));
        assert!(request.provider_params.contains_key("task_type"));
    }

    #[test]
    fn test_batch_request_builder() {
        let request1 = EmbeddingRequestBuilder::with_input(vec!["text1".to_string()])
            .build()
            .unwrap();
        let request2 = EmbeddingRequestBuilder::with_input(vec!["text2".to_string()])
            .build()
            .unwrap();

        let batch = BatchEmbeddingRequestBuilder::new()
            .add_request(request1)
            .add_request(request2)
            .max_concurrency(2)
            .fail_fast(true)
            .build()
            .unwrap();

        assert_eq!(batch.requests.len(), 2);
        assert_eq!(batch.batch_options.max_concurrency, Some(2));
        assert!(batch.batch_options.fail_fast);
    }

    #[test]
    fn test_convenience_functions() {
        let query_request = convenience::embed_query("search query").build().unwrap();
        assert!(
            query_request
                .provider_params
                .get("task_type")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("RETRIEVAL_QUERY")
        );

        let doc_request = convenience::embed_document("document content")
            .build()
            .unwrap();
        assert!(
            doc_request
                .provider_params
                .get("task_type")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("RETRIEVAL_DOCUMENT")
        );
    }
}
