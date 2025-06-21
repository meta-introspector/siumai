//! Gemini Client Implementation
//!
//! Main client structure that aggregates all Gemini capabilities.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use std::time::Duration;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::*;
use crate::types::*;

use super::chat::GeminiChatCapability;
use super::files::GeminiFiles;
use super::models::GeminiModels;
use super::types::{GeminiConfig, GenerationConfig, SafetySetting};

/// Gemini client that implements the `LlmClient` trait
#[derive(Debug, Clone)]
pub struct GeminiClient {
    /// HTTP client for making requests
    pub http_client: HttpClient,
    /// Gemini configuration
    pub config: GeminiConfig,
    /// Chat capability implementation
    pub chat_capability: GeminiChatCapability,
    /// Models capability implementation
    pub models_capability: GeminiModels,
    /// Files capability implementation
    pub files_capability: GeminiFiles,
}

impl GeminiClient {
    /// Create a new Gemini client with the given configuration
    pub fn new(config: GeminiConfig) -> Result<Self, LlmError> {
        let timeout = Duration::from_secs(config.timeout.unwrap_or(30));

        let http_client = HttpClient::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| {
                LlmError::ConfigurationError(format!("Failed to create HTTP client: {e}"))
            })?;

        let chat_capability = GeminiChatCapability::new(config.clone(), http_client.clone());

        let models_capability = GeminiModels::new(config.clone(), http_client.clone());

        let files_capability = GeminiFiles::new(config.clone(), http_client.clone());

        Ok(Self {
            http_client,
            config,
            chat_capability,
            models_capability,
            files_capability,
        })
    }

    /// Create a new Gemini client with API key
    pub fn with_api_key(api_key: String) -> Result<Self, LlmError> {
        let config = GeminiConfig::new(api_key);
        Self::new(config)
    }

    /// Set the model to use
    pub fn with_model(mut self, model: String) -> Self {
        self.config.model = model;
        self
    }

    /// Set the base URL
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.config.base_url = base_url;
        self
    }

    /// Set generation configuration
    pub fn with_generation_config(mut self, config: GenerationConfig) -> Self {
        self.config.generation_config = Some(config);
        self
    }

    /// Set safety settings
    pub fn with_safety_settings(mut self, settings: Vec<SafetySetting>) -> Self {
        self.config.safety_settings = Some(settings);
        self
    }

    /// Set HTTP timeout
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = Some(timeout.as_secs());
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.temperature = Some(temperature);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set max output tokens
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.max_output_tokens = Some(max_tokens);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set top-p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.top_p = Some(top_p);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set top-k
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.top_k = Some(top_k);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set stop sequences
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.stop_sequences = Some(stop_sequences);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set candidate count
    pub fn with_candidate_count(mut self, count: i32) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.candidate_count = Some(count);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Enable structured output with JSON schema
    pub fn with_json_schema(mut self, schema: serde_json::Value) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.response_mime_type = Some("application/json".to_string());
        generation_config.response_schema = Some(schema);
        self.config.generation_config = Some(generation_config);
        self
    }



    /// Get the API key
    pub fn api_key(&self) -> &str {
        &self.config.api_key
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Get the model
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Get the generation configuration
    pub const fn generation_config(&self) -> Option<&GenerationConfig> {
        self.config.generation_config.as_ref()
    }

    /// Get the safety settings
    pub const fn safety_settings(&self) -> Option<&Vec<SafetySetting>> {
        self.config.safety_settings.as_ref()
    }
}

#[async_trait]
impl ChatCapability for GeminiClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.chat_capability.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.chat_capability.chat_stream(messages, tools).await
    }
}

#[async_trait]
impl ModelListingCapability for GeminiClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}

#[async_trait]
impl FileManagementCapability for GeminiClient {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        self.files_capability.upload_file(request).await
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        self.files_capability.list_files(query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        self.files_capability.retrieve_file(file_id).await
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        self.files_capability.delete_file(file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        self.files_capability.get_file_content(file_id).await
    }
}

impl LlmClient for GeminiClient {
    fn provider_name(&self) -> &'static str {
        "gemini"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "gemini-1.5-flash".to_string(),
            "gemini-1.5-flash-8b".to_string(),
            "gemini-1.5-pro".to_string(),
            "gemini-2.0-flash-exp".to_string(),
            "gemini-exp-1114".to_string(),
            "gemini-exp-1121".to_string(),
            "gemini-exp-1206".to_string(),
        ]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_file_management()
            .with_custom_feature("code_execution", true)
            .with_custom_feature("thinking_mode", true)
            .with_custom_feature("safety_settings", true)
            .with_custom_feature("cached_content", true)
            .with_custom_feature("json_schema", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Builder for creating Gemini clients
#[derive(Debug, Clone)]
pub struct GeminiBuilder {
    config: GeminiConfig,
}

impl GeminiBuilder {
    /// Create a new Gemini builder
    pub fn new() -> Self {
        Self {
            config: GeminiConfig::default(),
        }
    }

    /// Set the API key
    pub fn api_key(mut self, api_key: String) -> Self {
        self.config.api_key = api_key;
        self
    }

    /// Set the model
    pub fn model(mut self, model: String) -> Self {
        self.config.model = model;
        self
    }

    /// Set the base URL
    pub fn base_url(mut self, base_url: String) -> Self {
        self.config.base_url = base_url;
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.temperature = Some(temperature);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, max_tokens: i32) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.max_output_tokens = Some(max_tokens);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set top-p
    pub fn top_p(mut self, top_p: f32) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.top_p = Some(top_p);
        self.config.generation_config = Some(generation_config);
        self
    }

    /// Set top-k
    pub fn top_k(mut self, top_k: i32) -> Self {
        let mut generation_config = self.config.generation_config.unwrap_or_default();
        generation_config.top_k = Some(top_k);
        self.config.generation_config = Some(generation_config);
        self
    }



    /// Build the Gemini client
    pub fn build(self) -> Result<GeminiClient, LlmError> {
        if self.config.api_key.is_empty() {
            return Err(LlmError::ConfigurationError(
                "API key is required".to_string(),
            ));
        }

        GeminiClient::new(self.config)
    }
}

impl Default for GeminiBuilder {
    fn default() -> Self {
        Self::new()
    }
}
