//! Custom Provider Framework
//!
//! This module provides a framework for implementing custom AI providers,
//! allowing users to easily integrate new AI services into the library.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::*;
use crate::types::*;

pub mod guide;

/// Custom provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomProviderConfig {
    /// Provider name
    pub name: String,
    /// Base URL for the API
    pub base_url: String,
    /// API key or authentication token
    pub api_key: String,
    /// Custom headers
    pub headers: HashMap<String, String>,
    /// Request timeout in seconds
    pub timeout: Option<u64>,
    /// Custom parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl CustomProviderConfig {
    /// Create a new custom provider configuration
    pub fn new<S: Into<String>>(name: S, base_url: S, api_key: S) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            headers: HashMap::new(),
            timeout: Some(30),
            custom_params: HashMap::new(),
        }
    }

    /// Add a custom header
    pub fn with_header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout = Some(timeout_seconds);
        self
    }

    /// Add a custom parameter
    pub fn with_param<K: Into<String>, V: Serialize>(mut self, key: K, value: V) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.custom_params.insert(key.into(), json_value);
        }
        self
    }
}

/// Custom provider trait
///
/// Implement this trait to create a custom AI provider
#[async_trait]
pub trait CustomProvider: Send + Sync {
    /// Get provider name
    fn name(&self) -> &str;

    /// Get supported models
    fn supported_models(&self) -> Vec<String>;

    /// Get provider capabilities
    fn capabilities(&self) -> ProviderCapabilities;

    /// Send a chat request
    async fn chat(&self, request: CustomChatRequest) -> Result<CustomChatResponse, LlmError>;

    /// Send a streaming chat request
    async fn chat_stream(&self, request: CustomChatRequest) -> Result<ChatStream, LlmError>;

    /// Validate configuration
    fn validate_config(&self, config: &CustomProviderConfig) -> Result<(), LlmError> {
        if config.name.is_empty() {
            return Err(LlmError::InvalidParameter("Provider name cannot be empty".to_string()));
        }
        if config.base_url.is_empty() {
            return Err(LlmError::InvalidParameter("Base URL cannot be empty".to_string()));
        }
        if config.api_key.is_empty() {
            return Err(LlmError::InvalidParameter("API key cannot be empty".to_string()));
        }
        Ok(())
    }

    /// Transform request before sending
    fn transform_request(&self, request: &mut CustomChatRequest) -> Result<(), LlmError> {
        // Default implementation does nothing
        Ok(())
    }

    /// Transform response after receiving
    fn transform_response(&self, response: &mut CustomChatResponse) -> Result<(), LlmError> {
        // Default implementation does nothing
        Ok(())
    }
}

/// Custom chat request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomChatRequest {
    /// Messages
    pub messages: Vec<ChatMessage>,
    /// Model name
    pub model: String,
    /// Tools (if supported)
    pub tools: Option<Vec<Tool>>,
    /// Stream flag
    pub stream: bool,
    /// Custom parameters
    pub params: HashMap<String, serde_json::Value>,
}

impl CustomChatRequest {
    /// Create a new custom chat request
    pub fn new(messages: Vec<ChatMessage>, model: String) -> Self {
        Self {
            messages,
            model,
            tools: None,
            stream: false,
            params: HashMap::new(),
        }
    }

    /// Add tools
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Enable streaming
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Add a parameter
    pub fn with_param<K: Into<String>, V: Serialize>(mut self, key: K, value: V) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.params.insert(key.into(), json_value);
        }
        self
    }
}

/// Custom chat response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomChatResponse {
    /// Response content
    pub content: String,
    /// Tool calls (if any)
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Usage information
    pub usage: Option<Usage>,
    /// Finish reason
    pub finish_reason: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl CustomChatResponse {
    /// Create a new custom chat response
    pub fn new(content: String) -> Self {
        Self {
            content,
            tool_calls: None,
            usage: None,
            finish_reason: None,
            metadata: HashMap::new(),
        }
    }

    /// Add tool calls
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// Add usage information
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set finish reason
    pub fn with_finish_reason<S: Into<String>>(mut self, reason: S) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Add metadata
    pub fn with_metadata<K: Into<String>, V: Serialize>(mut self, key: K, value: V) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), json_value);
        }
        self
    }

    /// Convert to standard ChatResponse
    pub fn to_chat_response(&self, provider_name: &str) -> ChatResponse {
        ChatResponse {
            content: MessageContent::Text(self.content.clone()),
            tool_calls: self.tool_calls.clone(),
            usage: self.usage.clone(),
            finish_reason: self.finish_reason.as_ref().map(|r| match r.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "tool_calls" => FinishReason::ToolCalls,
                "content_filter" => FinishReason::ContentFilter,
                _ => FinishReason::Other(r.clone()),
            }),
            metadata: ResponseMetadata {
                id: None,
                model: None,
                created: Some(chrono::Utc::now()),
                provider: provider_name.to_string(),
                request_id: None,
            },
            provider_data: self.metadata.clone(),
        }
    }
}

/// Custom provider client wrapper
pub struct CustomProviderClient {
    /// Custom provider implementation
    provider: Box<dyn CustomProvider>,
    /// Configuration
    config: CustomProviderConfig,
    /// HTTP client
    http_client: reqwest::Client,
}

impl CustomProviderClient {
    /// Create a new custom provider client
    pub fn new(
        provider: Box<dyn CustomProvider>,
        config: CustomProviderConfig,
    ) -> Result<Self, LlmError> {
        // Validate configuration
        provider.validate_config(&config)?;

        // Create HTTP client with custom configuration
        let mut client_builder = reqwest::Client::builder();

        if let Some(timeout) = config.timeout {
            client_builder = client_builder.timeout(std::time::Duration::from_secs(timeout));
        }

        let http_client = client_builder
            .build()
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            provider,
            config,
            http_client,
        })
    }

    /// Get the underlying provider
    pub fn provider(&self) -> &dyn CustomProvider {
        self.provider.as_ref()
    }

    /// Get the configuration
    pub fn config(&self) -> &CustomProviderConfig {
        &self.config
    }

    /// Get the HTTP client
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }
}

#[async_trait]
impl ChatCapability for CustomProviderClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut request = CustomChatRequest::new(messages, "default".to_string());

        if let Some(tools) = tools {
            request = request.with_tools(tools);
        }

        // Add custom parameters from config
        for (key, value) in &self.config.custom_params {
            request.params.insert(key.clone(), value.clone());
        }

        let mut response = self.provider.chat(request).await?;
        self.provider.transform_response(&mut response)?;

        Ok(response.to_chat_response(&self.config.name))
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let mut request = CustomChatRequest::new(messages, "default".to_string())
            .with_stream(true);

        if let Some(tools) = tools {
            request = request.with_tools(tools);
        }

        // Add custom parameters from config
        for (key, value) in &self.config.custom_params {
            request.params.insert(key.clone(), value.clone());
        }

        self.provider.chat_stream(request).await
    }
}

impl LlmClient for CustomProviderClient {
    fn provider_name(&self) -> &'static str {
        // Note: This returns a static str, but the actual name is dynamic
        // In a real implementation, you might want to use a different approach
        "custom"
    }

    fn supported_models(&self) -> Vec<String> {
        self.provider.supported_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.provider.capabilities()
    }
}

/// Helper trait for building custom providers
pub trait CustomProviderBuilder {
    /// Build the custom provider
    fn build(self) -> Result<Box<dyn CustomProvider>, LlmError>;
}
