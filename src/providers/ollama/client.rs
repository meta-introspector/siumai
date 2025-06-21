//! Ollama Client Implementation
//!
//! Main client that aggregates all Ollama capabilities.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::LlmClient;
use crate::stream::ChatStream;
use crate::traits::{ChatCapability, EmbeddingCapability, LlmProvider, ModelListingCapability, ProviderCapabilities};
use crate::types::*;

use super::chat::OllamaChatCapability;
use super::completion::OllamaCompletionCapability;
use super::config::{OllamaConfig, OllamaParams};
use super::embeddings::OllamaEmbeddingCapability;
use super::models::OllamaModelsCapability;
use super::get_default_models;

/// Ollama Client
#[allow(dead_code)]
pub struct OllamaClient {
    /// Chat capability implementation
    chat_capability: OllamaChatCapability,
    /// Completion capability implementation
    completion_capability: OllamaCompletionCapability,
    /// Embedding capability implementation
    embedding_capability: OllamaEmbeddingCapability,
    /// Models capability implementation
    models_capability: OllamaModelsCapability,
    /// Common parameters
    common_params: CommonParams,
    /// Ollama-specific parameters
    ollama_params: OllamaParams,
    /// HTTP client for making requests
    http_client: reqwest::Client,
    /// Base URL for Ollama API
    base_url: String,
}

impl OllamaClient {
    /// Creates a new Ollama client with configuration and HTTP client
    pub fn new(config: OllamaConfig, http_client: reqwest::Client) -> Self {
        let chat_capability = OllamaChatCapability::new(
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
            config.ollama_params.clone(),
        );

        let completion_capability = OllamaCompletionCapability::new(
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
            config.ollama_params.clone(),
        );

        let embedding_capability = OllamaEmbeddingCapability::new(
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
            config.ollama_params.clone(),
        );

        let models_capability = OllamaModelsCapability::new(
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
        );

        Self {
            chat_capability,
            completion_capability,
            embedding_capability,
            models_capability,
            common_params: config.common_params,
            ollama_params: config.ollama_params,
            http_client,
            base_url: config.base_url,
        }
    }

    /// Creates a new Ollama client with configuration
    pub fn new_with_config(config: OllamaConfig) -> Self {
        let http_client = reqwest::Client::new();
        Self::new(config, http_client)
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get common parameters
    pub fn common_params(&self) -> &CommonParams {
        &self.common_params
    }

    /// Get Ollama-specific parameters
    pub fn ollama_params(&self) -> &OllamaParams {
        &self.ollama_params
    }

    /// Update common parameters
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    /// Update Ollama-specific parameters
    pub fn with_ollama_params(mut self, params: OllamaParams) -> Self {
        self.ollama_params = params;
        self
    }

    /// Set model
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set keep alive duration
    pub fn with_keep_alive<S: Into<String>>(mut self, duration: S) -> Self {
        self.ollama_params.keep_alive = Some(duration.into());
        self
    }

    /// Enable raw mode
    pub fn with_raw(mut self, raw: bool) -> Self {
        self.ollama_params.raw = Some(raw);
        self
    }

    /// Set output format
    pub fn with_format<S: Into<String>>(mut self, format: S) -> Self {
        self.ollama_params.format = Some(format.into());
        self
    }

    /// Add model option
    pub fn with_option<K: Into<String>>(mut self, key: K, value: serde_json::Value) -> Self {
        let mut options = self.ollama_params.options.unwrap_or_default();
        options.insert(key.into(), value);
        self.ollama_params.options = Some(options);
        self
    }

    /// Generate text completion (using /api/generate endpoint)
    pub async fn generate(&self, prompt: String) -> Result<String, LlmError> {
        self.completion_capability.generate(prompt).await
    }

    /// Generate text completion with streaming
    pub async fn generate_stream(&self, prompt: String) -> Result<ChatStream, LlmError> {
        self.completion_capability.generate_stream(prompt).await
    }

    /// Check if Ollama server is running
    pub async fn health_check(&self) -> Result<bool, LlmError> {
        let url = format!("{}/api/version", self.base_url);
        
        match self.http_client.get(&url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Get Ollama version
    pub async fn version(&self) -> Result<String, LlmError> {
        let url = format!("{}/api/version", self.base_url);
        
        let response = self.http_client
            .get(&url)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(LlmError::HttpError(format!(
                "Failed to get version: {}",
                response.status()
            )));
        }

        let version_response: super::types::OllamaVersionResponse = response.json().await?;
        Ok(version_response.version)
    }
}

#[async_trait]
impl ChatCapability for OllamaClient {
    /// Chat with tools implementation
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Create a ChatRequest from messages and tools
        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: false,
        };
        self.chat_capability.chat(request).await
    }

    /// Streaming chat with tools
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Create a ChatRequest with proper common_params
        let mut request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: true,
        };
        request.stream = true;

        let headers = crate::providers::ollama::utils::build_headers(&self.chat_capability.http_config.headers)?;
        let body = self.chat_capability.build_chat_request_body(&request)?;
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(crate::error::LlmError::HttpError(format!(
                "Chat request failed: {} - {}",
                status,
                error_text
            )));
        }

        // Create stream from response
        use futures_util::StreamExt;
        let stream = response.bytes_stream();
        let mapped_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    for line in chunk_str.lines() {
                        if let Ok(Some(json_value)) = crate::providers::ollama::utils::parse_streaming_line(line) {
                            if let Ok(ollama_response) = serde_json::from_value::<crate::providers::ollama::types::OllamaChatResponse>(json_value) {
                                let content_delta = ollama_response.message.content.clone();
                                return Ok(crate::types::ChatStreamEvent::ContentDelta {
                                    delta: content_delta,
                                    index: Some(0),
                                });
                            }
                        }
                    }
                    Ok(crate::types::ChatStreamEvent::ContentDelta {
                        delta: String::new(),
                        index: Some(0),
                    })
                }
                Err(e) => Err(crate::error::LlmError::StreamError(format!("Stream error: {}", e))),
            }
        });

        Ok(Box::pin(mapped_stream))
    }
}

#[async_trait]
impl EmbeddingCapability for OllamaClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        self.embedding_capability.embed(texts).await
    }

    fn embedding_dimension(&self) -> usize {
        self.embedding_capability.embedding_dimension()
    }
}

#[async_trait]
impl ModelListingCapability for OllamaClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}

impl LlmClient for OllamaClient {
    fn provider_name(&self) -> &'static str {
        "ollama"
    }

    fn supported_models(&self) -> Vec<String> {
        get_default_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_custom_feature("completion", true)
            .with_custom_feature("model_management", true)
            .with_custom_feature("local_models", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl LlmProvider for OllamaClient {
    fn provider_name(&self) -> &'static str {
        "ollama"
    }

    fn supported_models(&self) -> Vec<String> {
        get_default_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_custom_feature("completion", true)
            .with_custom_feature("model_management", true)
            .with_custom_feature("local_models", true)
    }

    fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new_with_config(config);
        
        assert_eq!(LlmProvider::provider_name(&client), "ollama");
        assert_eq!(client.base_url(), "http://localhost:11434");
    }

    #[test]
    fn test_client_builder_pattern() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new_with_config(config)
            .with_model("llama3.2")
            .with_temperature(0.7)
            .with_max_tokens(1000)
            .with_keep_alive("10m")
            .with_raw(true)
            .with_format("json")
            .with_option("top_p", serde_json::Value::Number(serde_json::Number::from_f64(0.9).unwrap()));

        assert_eq!(client.common_params().model, "llama3.2".to_string());
        assert_eq!(client.common_params().temperature, Some(0.7));
        assert_eq!(client.common_params().max_tokens, Some(1000));
        assert_eq!(client.ollama_params().keep_alive, Some("10m".to_string()));
        assert_eq!(client.ollama_params().raw, Some(true));
        assert_eq!(client.ollama_params().format, Some("json".to_string()));
    }
}
