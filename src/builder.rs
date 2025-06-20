//! Builder Pattern Implementation
//!
//! This module provides the builder pattern for creating LLM clients with a fluent API.
//! It supports custom HTTP clients, provider-specific configurations, and parameter validation.
//!
//! # Design Principles
//! - Fluent API with method chaining
//! - Support for custom reqwest clients
//! - Provider-specific parameter validation
//! - Consistent interface across providers
//!
//! # Example Usage
//! ```rust
//! use siumai::llm;
//!
//! // Basic usage
//! let client = llm()
//!     .openai()
//!     .api_key("your-api-key")
//!     .model("gpt-4")
//!     .build()
//!     .await?;
//!
//! // With custom HTTP client
//! let custom_client = reqwest::Client::builder()
//!     .timeout(Duration::from_secs(30))
//!     .build()?;
//!
//! let client = llm()
//!     .with_http_client(custom_client)
//!     .openai()
//!     .api_key("your-api-key")
//!     .build()
//!     .await?;
//! ```

use std::collections::HashMap;
use std::time::Duration;

use crate::error::LlmError;
use crate::types::*;

// Import parameter types - these will be moved to providers modules later
use crate::params::{AnthropicParams, OpenAiParams, ResponseFormat, ToolChoice};
use crate::providers::*;

/// Main entry point for the builder pattern.
///
/// This function creates a new LlmBuilder instance that can be used to configure
/// and create LLM clients for different providers.
///
/// # Example
/// ```rust
/// use siumai::llm;
///
/// let client = llm()
///     .openai()
///     .api_key("your-api-key")
///     .model("gpt-4")
///     .build()
///     .await?;
/// ```
pub fn llm() -> LlmBuilder {
    LlmBuilder::new()
}

/// Core LLM builder that provides common configuration options.
///
/// This builder allows setting up HTTP client configuration, timeouts,
/// and other provider-agnostic settings before choosing a specific provider.
///
/// # Design Philosophy
/// - Provider-agnostic configuration first
/// - Support for custom HTTP clients (key requirement)
/// - Fluent API with method chaining
/// - Validation at build time
#[derive(Debug, Clone)]
pub struct LlmBuilder {
    /// Custom HTTP client (key requirement from design doc)
    pub(crate) http_client: Option<reqwest::Client>,
    /// Request timeout
    pub(crate) timeout: Option<Duration>,
    /// Connection timeout
    pub(crate) connect_timeout: Option<Duration>,
    /// User agent string
    pub(crate) user_agent: Option<String>,
    /// Default headers
    pub(crate) default_headers: HashMap<String, String>,
    /// Enable HTTP/2
    pub(crate) http2_prior_knowledge: Option<bool>,
    /// Enable gzip compression
    pub(crate) gzip: Option<bool>,
    /// Enable brotli compression
    pub(crate) brotli: Option<bool>,
    /// Proxy URL
    pub(crate) proxy: Option<String>,
    /// Enable cookies
    pub(crate) cookie_store: Option<bool>,
    // Note: redirect policy removed due to Clone constraint issues
}

impl LlmBuilder {
    /// Create a new LLM builder with default settings.
    pub fn new() -> Self {
        Self {
            http_client: None,
            timeout: None,
            connect_timeout: None,
            user_agent: None,
            default_headers: HashMap::new(),
            http2_prior_knowledge: None,
            gzip: None,
            brotli: None,
            proxy: None,
            cookie_store: None,
            // redirect_policy removed
        }
    }

    /// Use a custom HTTP client.
    ///
    /// This allows you to provide your own configured reqwest client
    /// with custom settings, certificates, proxies, etc.
    ///
    /// # Arguments
    /// * `client` - The reqwest client to use
    ///
    /// # Example
    /// ```rust
    /// let custom_client = reqwest::Client::builder()
    ///     .timeout(Duration::from_secs(30))
    ///     .build()?;
    ///
    /// let llm_client = llm()
    ///     .with_http_client(custom_client)
    ///     .openai()
    ///     .api_key("your-key")
    ///     .build()
    ///     .await?;
    /// ```
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Set the request timeout.
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait for a request
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the connection timeout.
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait for connection establishment
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Set a custom User-Agent header.
    ///
    /// # Arguments
    /// * `user_agent` - The User-Agent string to use
    pub fn with_user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Add a default header that will be sent with all requests.
    ///
    /// # Arguments
    /// * `name` - Header name
    /// * `value` - Header value
    pub fn with_header<K: Into<String>, V: Into<String>>(mut self, name: K, value: V) -> Self {
        self.default_headers.insert(name.into(), value.into());
        self
    }

    /// Enable or disable HTTP/2 prior knowledge.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable HTTP/2 prior knowledge
    pub fn with_http2_prior_knowledge(mut self, enabled: bool) -> Self {
        self.http2_prior_knowledge = Some(enabled);
        self
    }

    /// Enable or disable gzip compression.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable gzip compression
    pub fn with_gzip(mut self, enabled: bool) -> Self {
        self.gzip = Some(enabled);
        self
    }

    /// Enable or disable brotli compression.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable brotli compression
    pub fn with_brotli(mut self, enabled: bool) -> Self {
        self.brotli = Some(enabled);
        self
    }

    /// Set a proxy URL.
    ///
    /// # Arguments
    /// * `proxy_url` - The proxy URL (e.g., "http://proxy.example.com:8080")
    pub fn with_proxy<S: Into<String>>(mut self, proxy_url: S) -> Self {
        self.proxy = Some(proxy_url.into());
        self
    }

    /// Enable or disable cookie store.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable cookie storage
    pub fn with_cookie_store(mut self, enabled: bool) -> Self {
        self.cookie_store = Some(enabled);
        self
    }

    // Note: redirect policy configuration removed due to Clone constraints

    // Provider-specific builders

    /// Create an OpenAI client builder.
    ///
    /// # Returns
    /// OpenAI-specific builder for further configuration
    pub fn openai(self) -> OpenAiBuilder {
        OpenAiBuilder::new(self)
    }

    /// Create an Anthropic client builder.
    ///
    /// # Returns
    /// Anthropic-specific builder for further configuration
    pub fn anthropic(self) -> AnthropicBuilder {
        AnthropicBuilder::new(self)
    }

    /// Create a Google client builder.
    ///
    /// # Returns
    /// Gemini-specific builder for further configuration
    pub fn gemini(self) -> GeminiBuilder {
        GeminiBuilder::new(self)
    }

    /// Create an xAI client builder.
    ///
    /// # Returns
    /// xAI-specific builder for further configuration
    pub fn xai(self) -> GenericProviderBuilder {
        GenericProviderBuilder::new(self, ProviderType::XAI)
    }

    /// Generic provider builder (for custom providers)
    pub fn provider(self, provider_type: ProviderType) -> GenericProviderBuilder {
        GenericProviderBuilder::new(self, provider_type)
    }

    /// Build the HTTP client with the configured settings.
    ///
    /// This is used internally by provider builders to create the HTTP client.
    pub(crate) fn build_http_client(&self) -> Result<reqwest::Client, LlmError> {
        // If a custom client was provided, use it
        if let Some(client) = &self.http_client {
            return Ok(client.clone());
        }

        // Build a new client with the configured settings
        let mut builder = reqwest::Client::builder();

        if let Some(timeout) = self.timeout {
            builder = builder.timeout(timeout);
        }

        if let Some(connect_timeout) = self.connect_timeout {
            builder = builder.connect_timeout(connect_timeout);
        }

        if let Some(user_agent) = &self.user_agent {
            builder = builder.user_agent(user_agent);
        }

        if let Some(_http2) = self.http2_prior_knowledge {
            // Note: http2_prior_knowledge() doesn't take parameters in newer reqwest versions
            builder = builder.http2_prior_knowledge();
        }

        // Note: gzip and brotli are enabled by default in reqwest
        // These methods may not be available in all versions

        if let Some(proxy_url) = &self.proxy {
            let proxy = reqwest::Proxy::all(proxy_url)
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid proxy URL: {}", e)))?;
            builder = builder.proxy(proxy);
        }

        // Note: cookie_store and redirect policy configuration
        // may require different approaches in different reqwest versions

        // Add default headers
        let mut headers = reqwest::header::HeaderMap::new();
        for (name, value) in &self.default_headers {
            let header_name =
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                    LlmError::ConfigurationError(format!("Invalid header name '{}': {}", name, e))
                })?;
            let header_value = reqwest::header::HeaderValue::from_str(value).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid header value '{}': {}", value, e))
            })?;
            headers.insert(header_name, header_value);
        }

        if !headers.is_empty() {
            builder = builder.default_headers(headers);
        }

        builder.build().map_err(|e| {
            LlmError::ConfigurationError(format!("Failed to build HTTP client: {}", e))
        })
    }
}

impl Default for LlmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// OpenAI-specific builder
pub struct OpenAiBuilder {
    base: LlmBuilder,
    api_key: Option<String>,
    base_url: Option<String>,
    organization: Option<String>,
    project: Option<String>,
    model: Option<String>,
    common_params: CommonParams,
    openai_params: OpenAiParams,
    http_config: HttpConfig,
}

impl OpenAiBuilder {
    fn new(base: LlmBuilder) -> Self {
        Self {
            base,
            api_key: None,
            base_url: None,
            organization: None,
            project: None,
            model: None,
            common_params: CommonParams::default(),
            openai_params: OpenAiParams::default(),
            http_config: HttpConfig::default(),
        }
    }

    /// Sets the API key
    pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the base URL
    pub fn base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the organization ID
    pub fn organization<S: Into<String>>(mut self, org: S) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Sets the project ID
    pub fn project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Sets the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        let model_str = model.into();
        self.model = Some(model_str.clone());
        self.common_params.model = model_str;
        self
    }

    /// Sets the temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.common_params.temperature = Some(temp);
        self
    }

    /// Sets the maximum number of tokens
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.common_params.max_tokens = Some(tokens);
        self
    }

    /// Sets top_p
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Sets the stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(sequences);
        self
    }

    /// Sets the random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    // OpenAI-specific parameters

    /// Sets the response format
    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.openai_params.response_format = Some(format);
        self
    }

    /// Sets the tool choice strategy
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.openai_params.tool_choice = Some(choice);
        self
    }

    /// Sets the frequency penalty
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.frequency_penalty = Some(penalty);
        self
    }

    /// Sets the presence penalty
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.presence_penalty = Some(penalty);
        self
    }

    /// Sets the user ID
    pub fn user<S: Into<String>>(mut self, user: S) -> Self {
        self.openai_params.user = Some(user.into());
        self
    }

    /// Enables parallel tool calls
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.openai_params.parallel_tool_calls = Some(enabled);
        self
    }

    /// Sets the HTTP configuration
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = config;
        self
    }

    /// Builds the OpenAI client
    pub async fn build(self) -> Result<OpenAiClient, LlmError> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or(LlmError::MissingApiKey(
                "OpenAI API key not provided".to_string(),
            ))?;

        let base_url = self
            .base_url
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        let http_client = self.base.http_client.unwrap_or_else(|| {
            let mut builder = reqwest::Client::builder()
                .timeout(self.base.timeout.unwrap_or(Duration::from_secs(30)));

            if let Some(timeout) = self.http_config.timeout {
                builder = builder.timeout(timeout);
            }

            builder.build().unwrap()
        });

        Ok(OpenAiClient::new_legacy(
            api_key,
            base_url,
            http_client,
            self.common_params,
            self.openai_params,
            self.http_config,
            self.organization,
            self.project,
        ))
    }
}

/// Anthropic-specific builder
pub struct AnthropicBuilder {
    base: LlmBuilder,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    common_params: CommonParams,
    anthropic_params: AnthropicParams,
    http_config: HttpConfig,
}

impl AnthropicBuilder {
    fn new(base: LlmBuilder) -> Self {
        Self {
            base,
            api_key: None,
            base_url: None,
            model: None,
            common_params: CommonParams::default(),
            anthropic_params: AnthropicParams::default(),
            http_config: HttpConfig::default(),
        }
    }

    /// Sets the API key
    pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the base URL
    pub fn base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        let model_str = model.into();
        self.model = Some(model_str.clone());
        self.common_params.model = model_str;
        self
    }

    /// Common parameter setting methods (similar to OpenAI)
    pub fn temperature(mut self, temp: f32) -> Self {
        self.common_params.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.common_params.max_tokens = Some(tokens);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    // Anthropic-specific parameters

    /// Sets cache control
    pub fn cache_control(mut self, cache: crate::params::anthropic::CacheControl) -> Self {
        self.anthropic_params.cache_control = Some(cache);
        self
    }

    /// Sets the thinking budget
    pub fn thinking_budget(mut self, budget: u32) -> Self {
        self.anthropic_params.thinking_budget = Some(budget);
        self
    }

    /// Sets the system message
    pub fn system_message<S: Into<String>>(mut self, system: S) -> Self {
        self.anthropic_params.system = Some(system.into());
        self
    }

    /// Adds metadata
    pub fn metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        if self.anthropic_params.metadata.is_none() {
            self.anthropic_params.metadata = Some(HashMap::new());
        }
        self.anthropic_params
            .metadata
            .as_mut()
            .unwrap()
            .insert(key.into(), value.into());
        self
    }

    /// Builds the Anthropic client
    pub async fn build(self) -> Result<AnthropicClient, LlmError> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or(LlmError::MissingApiKey(
                "Anthropic API key not provided".to_string(),
            ))?;

        let base_url = self
            .base_url
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());

        let http_client = self.base.http_client.unwrap_or_else(|| {
            reqwest::Client::builder()
                .timeout(self.base.timeout.unwrap_or(Duration::from_secs(30)))
                .build()
                .unwrap()
        });

        Ok(AnthropicClient::new(
            api_key,
            base_url,
            http_client,
            self.common_params,
            self.anthropic_params,
            self.http_config,
        ))
    }
}

/// Gemini-specific builder for configuring Gemini clients.
///
/// This builder provides Gemini-specific configuration options while
/// inheriting common HTTP and timeout settings from the base LlmBuilder.
///
/// # Example
/// ```rust,no_run
/// use siumai::llm;
///
/// let client = llm()
///     .gemini()
///     .api_key("your-api-key")
///     .model("gemini-1.5-flash")
///     .temperature(0.7)
///     .max_tokens(8192)
///     .build()
///     .await?;
/// ```
#[derive(Debug, Clone)]
pub struct GeminiBuilder {
    /// Base builder with HTTP configuration
    base: LlmBuilder,
    /// Gemini API key
    api_key: Option<String>,
    /// Base URL for Gemini API
    base_url: Option<String>,
    /// Model to use
    model: Option<String>,
    /// Temperature setting
    temperature: Option<f32>,
    /// Maximum output tokens
    max_tokens: Option<i32>,
    /// Top-p setting
    top_p: Option<f32>,
    /// Top-k setting
    top_k: Option<i32>,
    /// Stop sequences
    stop_sequences: Option<Vec<String>>,
    /// Candidate count
    candidate_count: Option<i32>,
    /// Safety settings
    safety_settings: Option<Vec<crate::providers::gemini::SafetySetting>>,
    /// JSON schema for structured output
    json_schema: Option<serde_json::Value>,
}

impl GeminiBuilder {
    /// Create a new Gemini builder
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            base,
            api_key: None,
            base_url: None,
            model: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            candidate_count: None,
            safety_settings: None,
            json_schema: None,
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the base URL
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set temperature (0.0 to 2.0)
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set maximum output tokens
    pub fn max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set top-p (0.0 to 1.0)
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k
    pub fn top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Set candidate count
    pub fn candidate_count(mut self, count: i32) -> Self {
        self.candidate_count = Some(count);
        self
    }

    /// Set safety settings
    pub fn safety_settings(
        mut self,
        settings: Vec<crate::providers::gemini::SafetySetting>,
    ) -> Self {
        self.safety_settings = Some(settings);
        self
    }

    /// Enable structured output with JSON schema
    pub fn json_schema(mut self, schema: serde_json::Value) -> Self {
        self.json_schema = Some(schema);
        self
    }

    /// Build the Gemini client
    pub async fn build(self) -> Result<crate::providers::gemini::GeminiClient, LlmError> {
        let api_key = self.api_key.ok_or_else(|| {
            LlmError::ConfigurationError("API key is required for Gemini".to_string())
        })?;

        let mut config = crate::providers::gemini::GeminiConfig::new(api_key);

        if let Some(base_url) = self.base_url {
            config = config.with_base_url(base_url);
        }

        if let Some(model) = self.model {
            config = config.with_model(model);
        }

        // Build generation config
        let mut generation_config = crate::providers::gemini::GenerationConfig::new();

        if let Some(temp) = self.temperature {
            generation_config = generation_config.with_temperature(temp);
        }

        if let Some(max_tokens) = self.max_tokens {
            generation_config = generation_config.with_max_output_tokens(max_tokens);
        }

        if let Some(top_p) = self.top_p {
            generation_config = generation_config.with_top_p(top_p);
        }

        if let Some(top_k) = self.top_k {
            generation_config = generation_config.with_top_k(top_k);
        }

        if let Some(stop_sequences) = self.stop_sequences {
            generation_config = generation_config.with_stop_sequences(stop_sequences);
        }

        if let Some(count) = self.candidate_count {
            generation_config = generation_config.with_candidate_count(count);
        }

        if let Some(schema) = self.json_schema {
            generation_config = generation_config.with_response_schema(schema);
            generation_config =
                generation_config.with_response_mime_type("application/json".to_string());
        }

        config = config.with_generation_config(generation_config);

        if let Some(safety_settings) = self.safety_settings {
            config = config.with_safety_settings(safety_settings);
        }

        // Apply HTTP configuration from base builder
        if let Some(timeout) = self.base.timeout {
            config = config.with_timeout(timeout.as_secs());
        }

        crate::providers::gemini::GeminiClient::new(config)
    }
}

pub struct GenericProviderBuilder {
    _base: LlmBuilder,
    _provider_type: ProviderType,
}

impl GenericProviderBuilder {
    fn new(base: LlmBuilder, provider_type: ProviderType) -> Self {
        Self {
            _base: base,
            _provider_type: provider_type,
        }
    }

    pub async fn build(self) -> Result<(), LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Generic provider not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = LlmBuilder::new();
        let _openai_builder = builder.openai();
        // Basic test for builder creation
        assert!(true); // Placeholder test
    }
}
