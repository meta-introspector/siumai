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
//! ```rust,no_run
//! use siumai::builder::LlmBuilder;
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Basic usage
//!     let client = LlmBuilder::new()
//!         .openai()
//!         .api_key("your-api-key")
//!         .model("gpt-4")
//!         .build()
//!         .await?;
//!
//!     // With custom HTTP client
//!     let custom_client = reqwest::Client::builder()
//!         .timeout(Duration::from_secs(30))
//!         .build()?;
//!
//!     let client = LlmBuilder::new()
//!         .with_http_client(custom_client)
//!         .openai()
//!         .api_key("your-api-key")
//!         .build()
//!         .await?;
//!
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::time::Duration;

use crate::error::LlmError;
use crate::types::*;

// Import parameter types - these will be moved to providers modules later
use crate::params::{AnthropicParams, OpenAiParams, ResponseFormat, ToolChoice};
use crate::providers::ollama::config::OllamaParams;
use crate::providers::*;

/// Quick `OpenAI` client creation with minimal configuration.
///
/// Uses environment variable `OPENAI_API_KEY` and default settings.
///
/// # Example
/// ```rust,no_run
/// use siumai::{quick_openai, quick_openai_with_model};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Uses OPENAI_API_KEY env var and gpt-4o-mini model
///     let client = quick_openai().await?;
///
///     // With custom model
///     let client = quick_openai_with_model("gpt-4").await?;
///
///     Ok(())
/// }
/// ```
pub async fn quick_openai() -> Result<crate::providers::openai::OpenAiClient, LlmError> {
    quick_openai_with_model("gpt-4o-mini").await
}

/// Quick `OpenAI` client creation with custom model.
pub async fn quick_openai_with_model(
    model: &str,
) -> Result<crate::providers::openai::OpenAiClient, LlmError> {
    LlmBuilder::new().openai().model(model).build().await
}

/// Quick Anthropic client creation with minimal configuration.
///
/// Uses environment variable `ANTHROPIC_API_KEY` and default settings.
pub async fn quick_anthropic() -> Result<crate::providers::anthropic::AnthropicClient, LlmError> {
    quick_anthropic_with_model("claude-3-5-sonnet-20241022").await
}

/// Quick Anthropic client creation with custom model.
pub async fn quick_anthropic_with_model(
    model: &str,
) -> Result<crate::providers::anthropic::AnthropicClient, LlmError> {
    LlmBuilder::new().anthropic().model(model).build().await
}

/// Quick Gemini client creation with minimal configuration.
///
/// Uses environment variable `GEMINI_API_KEY` and default settings.
pub async fn quick_gemini() -> Result<crate::providers::gemini::GeminiClient, LlmError> {
    quick_gemini_with_model("gemini-1.5-flash").await
}

/// Quick Gemini client creation with custom model.
pub async fn quick_gemini_with_model(
    model: &str,
) -> Result<crate::providers::gemini::GeminiClient, LlmError> {
    LlmBuilder::new().gemini().model(model).build().await
}

/// Quick Ollama client creation with minimal configuration.
///
/// Uses default Ollama settings (<http://localhost:11434>) and llama3.2 model.
pub async fn quick_ollama() -> Result<crate::providers::ollama::OllamaClient, LlmError> {
    quick_ollama_with_model("llama3.2").await
}

/// Quick Ollama client creation with custom model.
pub async fn quick_ollama_with_model(
    model: &str,
) -> Result<crate::providers::ollama::OllamaClient, LlmError> {
    LlmBuilder::new().ollama().model(model).build().await
}

/// Quick Groq client creation with minimal configuration.
///
/// Uses environment variable `GROQ_API_KEY` and default settings.
pub async fn quick_groq() -> Result<crate::providers::groq::GroqClient, LlmError> {
    quick_groq_with_model("llama-3.3-70b-versatile").await
}

/// Quick Groq client creation with custom model.
pub async fn quick_groq_with_model(
    model: &str,
) -> Result<crate::providers::groq::GroqClient, LlmError> {
    LlmBuilder::new().groq().model(model).build().await
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

    /// Create a builder with sensible defaults for production use.
    ///
    /// Sets reasonable timeouts, compression, and other production-ready settings.
    pub fn with_defaults() -> Self {
        Self::new()
            .with_timeout(Duration::from_secs(60))
            .with_connect_timeout(Duration::from_secs(10))
            .with_user_agent("siumai/0.1.0")
            .with_gzip(true)
            .with_brotli(true)
    }

    /// Create a builder optimized for fast responses.
    ///
    /// Uses shorter timeouts suitable for interactive applications.
    pub fn fast() -> Self {
        Self::new()
            .with_timeout(Duration::from_secs(30))
            .with_connect_timeout(Duration::from_secs(5))
            .with_user_agent("siumai/0.1.0")
    }

    /// Create a builder optimized for long-running operations.
    ///
    /// Uses longer timeouts suitable for batch processing or complex tasks.
    pub fn long_running() -> Self {
        Self::new()
            .with_timeout(Duration::from_secs(300))
            .with_connect_timeout(Duration::from_secs(30))
            .with_user_agent("siumai/0.1.0")
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
    /// ```rust,no_run
    /// use std::time::Duration;
    /// use siumai::builder::LlmBuilder;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let custom_client = reqwest::Client::builder()
    ///         .timeout(Duration::from_secs(30))
    ///         .build()?;
    ///
    ///     let llm_client = LlmBuilder::new()
    ///         .with_http_client(custom_client)
    ///         .openai()
    ///         .api_key("your-key")
    ///         .build()
    ///         .await?;
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Set the request timeout.
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait for a request
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the connection timeout.
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait for connection establishment
    pub const fn with_connect_timeout(mut self, timeout: Duration) -> Self {
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
    pub const fn with_http2_prior_knowledge(mut self, enabled: bool) -> Self {
        self.http2_prior_knowledge = Some(enabled);
        self
    }

    /// Enable or disable gzip compression.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable gzip compression
    pub const fn with_gzip(mut self, enabled: bool) -> Self {
        self.gzip = Some(enabled);
        self
    }

    /// Enable or disable brotli compression.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable brotli compression
    pub const fn with_brotli(mut self, enabled: bool) -> Self {
        self.brotli = Some(enabled);
        self
    }

    /// Set a proxy URL.
    ///
    /// # Arguments
    /// * `proxy_url` - The proxy URL (e.g., "<http://proxy.example.com:8080>")
    pub fn with_proxy<S: Into<String>>(mut self, proxy_url: S) -> Self {
        self.proxy = Some(proxy_url.into());
        self
    }

    /// Enable or disable cookie store.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable cookie storage
    pub const fn with_cookie_store(mut self, enabled: bool) -> Self {
        self.cookie_store = Some(enabled);
        self
    }

    // Note: redirect policy configuration removed due to Clone constraints

    // Provider-specific builders

    /// Create an `OpenAI` client builder.
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
    pub const fn gemini(self) -> GeminiBuilder {
        GeminiBuilder::new(self)
    }

    /// Create an Ollama client builder.
    ///
    /// # Returns
    /// Ollama-specific builder for further configuration
    pub fn ollama(self) -> OllamaBuilder {
        OllamaBuilder::new(self)
    }

    /// Create an xAI client builder.
    ///
    /// # Returns
    /// xAI-specific builder for further configuration
    pub fn xai(self) -> crate::providers::xai::XaiBuilder {
        crate::providers::xai::XaiBuilder::new()
    }

    /// Create a Groq client builder.
    ///
    /// # Returns
    /// Groq-specific builder for further configuration
    pub fn groq(self) -> crate::providers::groq::GroqBuilder {
        crate::providers::groq::GroqBuilder::new()
    }

    // OpenAI-Compatible Providers

    /// Create a `DeepSeek` client builder (OpenAI-compatible).
    ///
    /// `DeepSeek` provides cost-effective AI with reasoning capabilities.
    /// Uses OpenAI-compatible API with provider-specific optimizations.
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::builder::LlmBuilder;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .deepseek()
    ///         .api_key("your-deepseek-api-key")
    ///         .model("deepseek-reasoner")
    ///         .reasoning(true)?
    ///         .temperature(0.1)
    ///         .build()
    ///         .await?;
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn deepseek(
        self,
    ) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder<
        crate::providers::openai_compatible::DeepSeekProvider,
    > {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self)
    }

    /// Create an `OpenRouter` client builder (OpenAI-compatible).
    ///
    /// `OpenRouter` provides access to multiple AI models through a unified API.
    /// Supports model routing and fallback strategies.
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::builder::LlmBuilder;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .openrouter()
    ///         .api_key("your-openrouter-api-key")
    ///         .model("openai/gpt-4")
    ///         .site_url("https://myapp.com")?
    ///         .app_name("My App")?
    ///         .temperature(0.7)
    ///         .build()
    ///         .await?;
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn openrouter(
        self,
    ) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder<
        crate::providers::openai_compatible::OpenRouterProvider,
    > {
        crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self)
    }

    /// Generic provider builder (for custom providers)
    pub const fn provider(self, provider_type: ProviderType) -> GenericProviderBuilder {
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
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid proxy URL: {e}")))?;
            builder = builder.proxy(proxy);
        }

        // Note: cookie_store and redirect policy configuration
        // may require different approaches in different reqwest versions

        // Add default headers
        let mut headers = reqwest::header::HeaderMap::new();
        for (name, value) in &self.default_headers {
            let header_name =
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                    LlmError::ConfigurationError(format!("Invalid header name '{name}': {e}"))
                })?;
            let header_value = reqwest::header::HeaderValue::from_str(value).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid header value '{value}': {e}"))
            })?;
            headers.insert(header_name, header_value);
        }

        if !headers.is_empty() {
            builder = builder.default_headers(headers);
        }

        builder
            .build()
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to build HTTP client: {e}")))
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
    tracing_config: Option<crate::tracing::TracingConfig>,
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
            tracing_config: None,
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
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.common_params.temperature = Some(temp);
        self
    }

    /// Sets the maximum number of tokens
    pub const fn max_tokens(mut self, tokens: u32) -> Self {
        self.common_params.max_tokens = Some(tokens);
        self
    }

    /// Sets `top_p`
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Sets the stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(sequences);
        self
    }

    /// Sets the random seed
    pub const fn seed(mut self, seed: u64) -> Self {
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
    pub const fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.frequency_penalty = Some(penalty);
        self
    }

    /// Sets the presence penalty
    pub const fn presence_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.presence_penalty = Some(penalty);
        self
    }

    /// Sets the user ID
    pub fn user<S: Into<String>>(mut self, user: S) -> Self {
        self.openai_params.user = Some(user.into());
        self
    }

    /// Enables parallel tool calls
    pub const fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.openai_params.parallel_tool_calls = Some(enabled);
        self
    }

    /// Sets the HTTP configuration
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = config;
        self
    }

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    pub fn debug_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::development())
    }

    /// Enable minimal tracing (info level, LLM only)
    pub fn minimal_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::minimal())
    }

    /// Enable production-ready JSON tracing
    pub fn json_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::json_production())
    }

    /// Enable simple tracing (uses debug configuration)
    pub fn enable_tracing(self) -> Self {
        self.debug_tracing()
    }

    /// Disable tracing explicitly
    pub fn disable_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::disabled())
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing
    ///
    /// This enables multi-line, indented JSON formatting and organized header display
    /// in debug logs, making them more human-readable for debugging purposes.
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Provider::openai()
    ///     .api_key("your-key")
    ///     .model("gpt-4o-mini")
    ///     .debug_tracing()
    ///     .pretty_json(true)  // Enable pretty formatting
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development);

        let updated_config = crate::tracing::TracingConfigBuilder::from_config(config)
            .pretty_json(pretty)
            .build();

        self.tracing_config = Some(updated_config);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs
    ///
    /// When enabled (default), sensitive values like API keys and authorization tokens
    /// are automatically masked in logs for security. Only the first and last few
    /// characters are shown.
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Provider::openai()
    ///     .api_key("your-key")
    ///     .model("gpt-4o-mini")
    ///     .debug_tracing()
    ///     .mask_sensitive_values(false)  // Disable masking (not recommended for production)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development);

        let updated_config = crate::tracing::TracingConfigBuilder::from_config(config)
            .mask_sensitive_values(mask)
            .build();

        self.tracing_config = Some(updated_config);
        self
    }

    /// Builds the `OpenAI` client
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

        // Initialize tracing if configured
        let _tracing_guard = if let Some(tracing_config) = self.tracing_config {
            Some(crate::tracing::init_tracing(tracing_config)?)
        } else {
            None
        };

        let http_client = self.base.http_client.unwrap_or_else(|| {
            let mut builder = reqwest::Client::builder()
                .timeout(self.base.timeout.unwrap_or(Duration::from_secs(30)));

            if let Some(timeout) = self.http_config.timeout {
                builder = builder.timeout(timeout);
            }

            builder.build().unwrap()
        });

        let mut client = OpenAiClient::new_legacy(
            api_key,
            base_url,
            http_client,
            self.common_params,
            self.openai_params,
            self.http_config,
            self.organization,
            self.project,
        );

        // Set tracing guard to keep tracing system active
        client.set_tracing_guard(_tracing_guard);

        Ok(client)
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
    tracing_config: Option<crate::tracing::TracingConfig>,
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
            tracing_config: None,
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

    /// Common parameter setting methods (similar to `OpenAI`)
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.common_params.temperature = Some(temp);
        self
    }

    pub const fn max_tokens(mut self, tokens: u32) -> Self {
        self.common_params.max_tokens = Some(tokens);
        self
    }

    pub const fn top_p(mut self, top_p: f32) -> Self {
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
    pub const fn thinking_budget(mut self, budget: u32) -> Self {
        self.anthropic_params.thinking_budget = Some(budget);
        self
    }

    /// Enable thinking mode with default budget (10k tokens)
    pub const fn with_thinking_enabled(mut self) -> Self {
        self.anthropic_params.thinking_budget = Some(10000);
        self
    }

    /// Enable thinking mode with specified budget tokens
    pub const fn with_thinking_mode(mut self, budget_tokens: Option<u32>) -> Self {
        self.anthropic_params.thinking_budget = budget_tokens;
        self
    }

    /// Sets the system message
    pub fn system_message<S: Into<String>>(mut self, system: S) -> Self {
        self.anthropic_params.system = Some(system.into());
        self
    }

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    pub fn debug_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::development())
    }

    /// Enable minimal tracing (info level, LLM only)
    pub fn minimal_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::minimal())
    }

    /// Enable production-ready JSON tracing
    pub fn json_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::json_production())
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development)
            .with_pretty_json(pretty);
        self.tracing_config = Some(config);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development)
            .with_mask_sensitive_values(mask);
        self.tracing_config = Some(config);
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

        // Initialize tracing if configured
        let _tracing_guard = if let Some(ref tracing_config) = self.tracing_config {
            Some(crate::tracing::init_tracing(tracing_config.clone())?)
        } else {
            None
        };

        let http_client = self.base.http_client.unwrap_or_else(|| {
            reqwest::Client::builder()
                .timeout(self.base.timeout.unwrap_or(Duration::from_secs(30)))
                .build()
                .unwrap()
        });

        // Convert AnthropicParams to AnthropicSpecificParams
        let specific_params = crate::providers::anthropic::types::AnthropicSpecificParams {
            beta_features: self
                .anthropic_params
                .beta_features
                .clone()
                .unwrap_or_default(),
            cache_control: self.anthropic_params.cache_control.as_ref().map(|_cc| {
                crate::providers::anthropic::cache::CacheControl::ephemeral() // Convert from params::CacheControl
            }),
            thinking_config: self.anthropic_params.thinking_budget.map(|budget| {
                crate::providers::anthropic::thinking::ThinkingConfig::enabled(budget)
            }),
            metadata: self.anthropic_params.metadata.as_ref().map(|m| {
                // Convert HashMap<String, String> to serde_json::Value
                let mut json_map = serde_json::Map::new();
                for (k, v) in m {
                    json_map.insert(k.clone(), serde_json::Value::String(v.clone()));
                }
                serde_json::Value::Object(json_map)
            }),
        };

        // Create AnthropicClient with the converted specific_params
        let mut client = AnthropicClient::new(
            api_key,
            base_url,
            http_client,
            self.common_params,
            self.anthropic_params,
            self.http_config,
        );

        // Update the client with the specific params and tracing
        client = client.with_specific_params(specific_params);
        client.set_tracing_guard(_tracing_guard);
        client.set_tracing_config(self.tracing_config);

        Ok(client)
    }
}

/// Gemini-specific builder for configuring Gemini clients.
///
/// This builder provides Gemini-specific configuration options while
/// inheriting common HTTP and timeout settings from the base `LlmBuilder`.
///
/// # Example
/// ```rust,no_run
/// use siumai::builder::LlmBuilder;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = LlmBuilder::new()
///         .gemini()
///         .api_key("your-api-key")
///         .model("gemini-1.5-flash")
///         .temperature(0.7)
///         .max_tokens(8192)
///         .build()
///         .await?;
///
///     Ok(())
/// }
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
    /// Thinking configuration
    thinking_config: Option<crate::providers::gemini::ThinkingConfig>,
    /// Tracing configuration
    tracing_config: Option<crate::tracing::TracingConfig>,
}

impl GeminiBuilder {
    /// Create a new Gemini builder
    pub const fn new(base: LlmBuilder) -> Self {
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
            thinking_config: None,
            tracing_config: None,
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
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set maximum output tokens
    pub const fn max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set top-p (0.0 to 1.0)
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k
    pub const fn top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Set candidate count
    pub const fn candidate_count(mut self, count: i32) -> Self {
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

    /// Set thinking budget in tokens
    ///
    /// - Use -1 for dynamic thinking (model decides)
    /// - Use 0 to attempt to disable thinking (may not work on all models)
    /// - Use positive values to set a specific token budget
    ///
    /// The actual supported range depends on the model being used.
    pub const fn thinking_budget(mut self, budget: i32) -> Self {
        if self.thinking_config.is_none() {
            self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::new());
        }
        if let Some(ref mut config) = self.thinking_config {
            config.thinking_budget = Some(budget);
        }
        self
    }

    /// Enable or disable thought summaries in response
    ///
    /// This controls whether thinking summaries are included in the response,
    /// not whether the model thinks internally.
    pub const fn thought_summaries(mut self, include: bool) -> Self {
        if self.thinking_config.is_none() {
            self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::new());
        }
        if let Some(ref mut config) = self.thinking_config {
            config.include_thoughts = Some(include);
        }
        self
    }

    /// Enable dynamic thinking (model decides when and how much to think)
    pub const fn thinking(mut self) -> Self {
        self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::dynamic());
        self
    }

    /// Attempt to disable thinking
    ///
    /// Note: Not all models support disabling thinking. If the model doesn't
    /// support it, the API will return an appropriate error.
    pub const fn disable_thinking(mut self) -> Self {
        self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::disabled());
        self
    }

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    pub fn debug_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::development())
    }

    /// Enable minimal tracing (info level, LLM only)
    pub fn minimal_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::minimal())
    }

    /// Enable production-ready JSON tracing
    pub fn json_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::json_production())
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development)
            .with_pretty_json(pretty);
        self.tracing_config = Some(config);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development)
            .with_mask_sensitive_values(mask);
        self.tracing_config = Some(config);
        self
    }

    /// Build the Gemini client
    pub async fn build(self) -> Result<crate::providers::gemini::GeminiClient, LlmError> {
        let api_key = self.api_key.ok_or_else(|| {
            LlmError::ConfigurationError("API key is required for Gemini".to_string())
        })?;

        // Initialize tracing if configured
        let _tracing_guard = if let Some(ref tracing_config) = self.tracing_config {
            Some(crate::tracing::init_tracing(tracing_config.clone())?)
        } else {
            None
        };

        let mut config = crate::providers::gemini::GeminiConfig::new(api_key);

        if let Some(base_url) = self.base_url {
            config = config.with_base_url(base_url);
        }

        // Basic validation of thinking configuration
        if let Some(thinking_config) = &self.thinking_config {
            thinking_config.validate().map_err(|e| {
                crate::error::LlmError::ConfigurationError(format!(
                    "Invalid thinking configuration: {e}"
                ))
            })?;
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

        // Apply thinking configuration to generation config
        if let Some(thinking_config) = &self.thinking_config {
            generation_config = generation_config.with_thinking_config(thinking_config.clone());
        }

        config = config.with_generation_config(generation_config);

        if let Some(safety_settings) = self.safety_settings {
            config = config.with_safety_settings(safety_settings);
        }

        // Apply HTTP configuration from base builder
        if let Some(timeout) = self.base.timeout {
            config = config.with_timeout(timeout.as_secs());
        }

        let mut client = crate::providers::gemini::GeminiClient::new(config)?;
        client.set_tracing_guard(_tracing_guard);
        client.set_tracing_config(self.tracing_config);

        Ok(client)
    }
}

/// Ollama-specific builder
pub struct OllamaBuilder {
    base: LlmBuilder,
    base_url: Option<String>,
    model: Option<String>,
    common_params: CommonParams,
    ollama_params: OllamaParams,
    http_config: HttpConfig,
    tracing_config: Option<crate::tracing::TracingConfig>,
}

impl OllamaBuilder {
    /// Create a new Ollama builder
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            base,
            base_url: None,
            model: None,
            common_params: CommonParams::default(),
            ollama_params: OllamaParams::default(),
            http_config: HttpConfig::default(),
            tracing_config: None,
        }
    }

    /// Set the base URL for Ollama API
    ///
    /// # Arguments
    /// * `url` - The base URL (e.g., "<http://localhost:11434>")
    pub fn base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the model to use
    ///
    /// # Arguments
    /// * `model` - The model name (e.g., "llama3.2", "mistral:7b")
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the temperature for generation
    ///
    /// # Arguments
    /// * `temperature` - Temperature value (0.0 to 2.0)
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tokens to generate
    ///
    /// # Arguments
    /// * `max_tokens` - Maximum tokens to generate
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top-p value for nucleus sampling
    ///
    /// # Arguments
    /// * `top_p` - Top-p value (0.0 to 1.0)
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set how long to keep the model loaded in memory
    ///
    /// # Arguments
    /// * `duration` - Duration string (e.g., "5m", "1h", "30s")
    pub fn keep_alive<S: Into<String>>(mut self, duration: S) -> Self {
        self.ollama_params.keep_alive = Some(duration.into());
        self
    }

    /// Enable or disable raw mode (bypass templating)
    ///
    /// # Arguments
    /// * `raw` - Whether to enable raw mode
    pub const fn raw(mut self, raw: bool) -> Self {
        self.ollama_params.raw = Some(raw);
        self
    }

    /// Set the output format
    ///
    /// # Arguments
    /// * `format` - Format string ("json" or JSON schema)
    pub fn format<S: Into<String>>(mut self, format: S) -> Self {
        self.ollama_params.format = Some(format.into());
        self
    }

    /// Add a model option
    ///
    /// # Arguments
    /// * `key` - Option key
    /// * `value` - Option value
    pub fn option<K: Into<String>>(mut self, key: K, value: serde_json::Value) -> Self {
        let mut options = self.ollama_params.options.unwrap_or_default();
        options.insert(key.into(), value);
        self.ollama_params.options = Some(options);
        self
    }

    /// Set multiple model options at once
    ///
    /// # Arguments
    /// * `options` - `HashMap` of options
    pub fn options(
        mut self,
        options: std::collections::HashMap<String, serde_json::Value>,
    ) -> Self {
        self.ollama_params.options = Some(options);
        self
    }

    /// Enable or disable NUMA support
    ///
    /// # Arguments
    /// * `numa` - Whether to enable NUMA support
    pub const fn numa(mut self, numa: bool) -> Self {
        self.ollama_params.numa = Some(numa);
        self
    }

    /// Set the context window size
    ///
    /// # Arguments
    /// * `num_ctx` - Context window size
    pub const fn num_ctx(mut self, num_ctx: u32) -> Self {
        self.ollama_params.num_ctx = Some(num_ctx);
        self
    }

    /// Set the number of GPU layers to use
    ///
    /// # Arguments
    /// * `num_gpu` - Number of GPU layers
    pub const fn num_gpu(mut self, num_gpu: u32) -> Self {
        self.ollama_params.num_gpu = Some(num_gpu);
        self
    }

    /// Set the batch size for processing
    ///
    /// # Arguments
    /// * `num_batch` - Batch size
    pub const fn num_batch(mut self, num_batch: u32) -> Self {
        self.ollama_params.num_batch = Some(num_batch);
        self
    }

    /// Set the main GPU to use
    ///
    /// # Arguments
    /// * `main_gpu` - Main GPU index
    pub const fn main_gpu(mut self, main_gpu: u32) -> Self {
        self.ollama_params.main_gpu = Some(main_gpu);
        self
    }

    /// Enable or disable memory mapping
    ///
    /// # Arguments
    /// * `use_mmap` - Whether to use memory mapping
    pub const fn use_mmap(mut self, use_mmap: bool) -> Self {
        self.ollama_params.use_mmap = Some(use_mmap);
        self
    }

    /// Set the number of threads to use
    ///
    /// # Arguments
    /// * `num_thread` - Number of threads
    pub const fn num_thread(mut self, num_thread: u32) -> Self {
        self.ollama_params.num_thread = Some(num_thread);
        self
    }

    /// Enable thinking mode for thinking models
    ///
    /// # Arguments
    /// * `think` - Whether to enable thinking mode
    pub const fn think(mut self, think: bool) -> Self {
        self.ollama_params.think = Some(think);
        self
    }

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    pub fn debug_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::development())
    }

    /// Enable minimal tracing (info level, LLM only)
    pub fn minimal_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::minimal())
    }

    /// Enable production-ready JSON tracing
    pub fn json_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::json_production())
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development)
            .with_pretty_json(pretty);
        self.tracing_config = Some(config);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development)
            .with_mask_sensitive_values(mask);
        self.tracing_config = Some(config);
        self
    }

    /// Build the Ollama client
    pub async fn build(self) -> Result<crate::providers::ollama::OllamaClient, LlmError> {
        let base_url = self
            .base_url
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        // Initialize tracing if configured
        let _tracing_guard = if let Some(ref tracing_config) = self.tracing_config {
            Some(crate::tracing::init_tracing(tracing_config.clone())?)
        } else {
            None
        };

        let mut config = crate::providers::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .common_params(self.common_params)
            .http_config(self.http_config)
            .ollama_params(self.ollama_params);

        if let Some(model) = self.model {
            config = config.model(model);
        }

        let config = config.build()?;
        let http_client = self.base.build_http_client()?;

        let mut client = crate::providers::ollama::OllamaClient::new(config, http_client);
        client.set_tracing_guard(_tracing_guard);
        client.set_tracing_config(self.tracing_config);

        Ok(client)
    }
}

pub struct GenericProviderBuilder {
    _base: LlmBuilder,
    _provider_type: ProviderType,
}

impl GenericProviderBuilder {
    const fn new(base: LlmBuilder, provider_type: ProviderType) -> Self {
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
        // Placeholder test
    }
}
