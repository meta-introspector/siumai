//! `xAI` Client Implementation
//!
//! Main client for the `xAI` provider that aggregates all capabilities.

use async_trait::async_trait;
use std::time::Duration;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::{ChatCapability, ProviderCapabilities};
use crate::types::*;

use super::chat::XaiChatCapability;
use super::config::XaiConfig;

/// `xAI` Client
///
/// Main client that provides access to all `xAI` capabilities.
/// This client implements the `LlmClient` trait for unified access
/// and also provides `xAI`-specific functionality.
#[derive(Debug)]
pub struct XaiClient {
    /// Chat capability
    pub chat_capability: XaiChatCapability,
    /// Common parameters
    pub common_params: CommonParams,
    /// HTTP client
    pub http_client: reqwest::Client,
    /// Tracing configuration
    tracing_config: Option<crate::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active
    _tracing_guard: Option<Option<tracing_appender::non_blocking::WorkerGuard>>,
}

impl XaiClient {
    /// Create a new `xAI` client
    pub async fn new(config: XaiConfig) -> Result<Self, LlmError> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| LlmError::InvalidInput(format!("Invalid xAI configuration: {e}")))?;

        // Create HTTP client with timeout
        let http_client = reqwest::Client::builder()
            .timeout(
                config
                    .http_config
                    .timeout
                    .unwrap_or(Duration::from_secs(30)),
            )
            .build()
            .map_err(|e| LlmError::HttpError(format!("Failed to create HTTP client: {e}")))?;

        Self::with_http_client(config, http_client).await
    }

    /// Create a new `xAI` client with a custom HTTP client
    pub async fn with_http_client(
        config: XaiConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| LlmError::InvalidInput(format!("Invalid xAI configuration: {e}")))?;

        // Create chat capability
        let chat_capability = XaiChatCapability::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.http_config.clone(),
        );

        Ok(Self {
            chat_capability,
            common_params: config.common_params,
            http_client,
            tracing_config: None,
            _tracing_guard: None,
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> XaiConfig {
        XaiConfig {
            api_key: self.chat_capability.api_key.clone(),
            base_url: self.chat_capability.base_url.clone(),
            common_params: self.common_params.clone(),
            http_config: self.chat_capability.http_config.clone(),
            web_search_config: WebSearchConfig::default(),
        }
    }

    /// Update common parameters
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    /// Update model
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Update temperature
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Update max tokens
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }
}

#[async_trait]
impl LlmClient for XaiClient {
    fn provider_name(&self) -> &'static str {
        "xai"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "grok-3-latest".to_string(),
            "grok-3".to_string(),
            "grok-3-fast".to_string(),
            "grok-3-mini".to_string(),
            "grok-3-mini-fast".to_string(),
            "grok-2-vision-1212".to_string(),
            "grok-2-1212".to_string(),
            "grok-beta".to_string(),
            "grok-vision-beta".to_string(),
        ]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_custom_feature("reasoning", true)
            .with_custom_feature("deferred_completion", true)
            .with_custom_feature("structured_outputs", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl ChatCapability for XaiClient {
    /// Chat with tools implementation
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Create a ChatRequest from messages and tools, using client's configuration
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

    /// Chat stream implementation
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.chat_capability.chat_stream(messages, tools).await
    }
}

/// `xAI`-specific methods
impl XaiClient {
    /// Chat with reasoning effort (for thinking models)
    pub async fn chat_with_reasoning(
        &self,
        messages: Vec<ChatMessage>,
        reasoning_effort: &str,
    ) -> Result<ChatResponse, LlmError> {
        let mut provider_params = std::collections::HashMap::new();
        provider_params.insert(
            "reasoning_effort".to_string(),
            serde_json::Value::String(reasoning_effort.to_string()),
        );

        let request = ChatRequest {
            messages,
            tools: None,
            common_params: self.common_params.clone(),
            provider_params: Some(ProviderParams {
                params: provider_params,
            }),
            http_config: None,
            web_search: None,
            stream: false,
        };

        self.chat_capability.chat(request).await
    }

    /// Create a deferred completion
    pub async fn create_deferred_completion(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<String, LlmError> {
        let mut provider_params = std::collections::HashMap::new();
        provider_params.insert("deferred".to_string(), serde_json::Value::Bool(true));

        let request = ChatRequest {
            messages,
            tools: None,
            common_params: self.common_params.clone(),
            provider_params: Some(ProviderParams {
                params: provider_params,
            }),
            http_config: None,
            web_search: None,
            stream: false,
        };

        // This would return a request_id instead of a full response
        // Implementation would need to handle the deferred response format
        let _response = self.chat_capability.chat(request).await?;

        // For now, return a placeholder - this would need proper implementation
        // to handle xAI's deferred completion API response format
        Err(LlmError::UnsupportedOperation(
            "Deferred completion not implemented yet".to_string(),
        ))
    }

    /// Get a deferred completion result
    pub async fn get_deferred_completion(
        &self,
        request_id: &str,
    ) -> Result<ChatResponse, LlmError> {
        let url = format!(
            "{}/chat/deferred-completion/{}",
            self.chat_capability.base_url, request_id
        );
        let headers = super::utils::build_headers(
            &self.chat_capability.api_key,
            &self.chat_capability.http_config.headers,
        )?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        match response.status().as_u16() {
            200 => {
                let _xai_response: super::types::XaiChatResponse = response.json().await?;
                // We need to make parse_chat_response public or create a wrapper
                Err(LlmError::UnsupportedOperation(
                    "Get deferred completion not implemented yet".to_string(),
                ))
            }
            202 => Err(LlmError::ApiError {
                code: 202,
                message: "Deferred completion not ready yet".to_string(),
                details: None,
            }),
            _ => {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                Err(LlmError::ApiError {
                    code: status.as_u16(),
                    message: format!("xAI API error: {error_text}"),
                    details: serde_json::from_str(&error_text).ok(),
                })
            }
        }
    }

    /// Set the tracing guard to keep tracing system active
    pub(crate) fn set_tracing_guard(
        &mut self,
        guard: Option<Option<tracing_appender::non_blocking::WorkerGuard>>,
    ) {
        self._tracing_guard = guard;
    }

    /// Set the tracing configuration
    pub(crate) fn set_tracing_config(&mut self, config: Option<crate::tracing::TracingConfig>) {
        self.tracing_config = config;
    }
}
