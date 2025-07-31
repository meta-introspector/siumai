//! Siumai LLM Interface
//!
//! This module provides the main siumai interface for calling different provider functionality,
//! similar to `llm_dart`'s approach. It uses dynamic dispatch to route calls to the
//! appropriate provider implementation.

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::*;
use crate::types::*;
use std::any::Any;
use std::collections::HashMap;

/// The main siumai LLM provider that can dynamically dispatch to different capabilities
///
/// This is inspired by `llm_dart`'s unified interface design, allowing you to
/// call different provider functionality through a single interface.
pub struct Siumai {
    /// The underlying provider client
    client: Box<dyn LlmClient>,
    /// Capability registry for dynamic dispatch
    #[allow(dead_code)]
    capabilities: HashMap<String, Box<dyn Any + Send + Sync>>,
    /// Provider-specific metadata
    metadata: ProviderMetadata,
}

/// Metadata about the provider
#[derive(Debug, Clone)]
pub struct ProviderMetadata {
    pub provider_type: ProviderType,
    pub provider_name: String,
    pub supported_models: Vec<String>,
    pub capabilities: ProviderCapabilities,
}

impl Siumai {
    /// Create a new siumai provider
    pub fn new(client: Box<dyn LlmClient>) -> Self {
        let metadata = ProviderMetadata {
            provider_type: match client.provider_name() {
                "openai" => ProviderType::OpenAi,
                "anthropic" => ProviderType::Anthropic,
                "gemini" => ProviderType::Gemini,
                "ollama" => ProviderType::Ollama,
                "xai" => ProviderType::XAI,
                name => ProviderType::Custom(name.to_string()),
            },
            provider_name: client.provider_name().to_string(),
            supported_models: client.supported_models(),
            capabilities: client.capabilities(),
        };

        Self {
            client,
            capabilities: HashMap::new(),
            metadata,
        }
    }

    /// Check if a capability is supported
    pub fn supports(&self, capability: &str) -> bool {
        self.metadata.capabilities.supports(capability)
    }

    /// Get provider metadata
    pub const fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    /// Get the underlying client
    pub fn client(&self) -> &dyn LlmClient {
        self.client.as_ref()
    }

    /// Type-safe audio capability access
    ///
    /// Note: This method provides access regardless of reported capability support.
    /// Actual support depends on the specific model being used.
    pub fn audio_capability(&self) -> AudioCapabilityProxy {
        AudioCapabilityProxy::new(self, self.supports("audio"))
    }

    /// Type-safe embedding capability access
    ///
    /// Note: This method provides access regardless of reported capability support.
    /// Actual support depends on the specific model being used.
    pub fn embedding_capability(&self) -> EmbeddingCapabilityProxy {
        EmbeddingCapabilityProxy::new(self, self.supports("embedding"))
    }

    /// Type-safe vision capability access
    ///
    /// Note: This method provides access regardless of reported capability support.
    /// Actual support depends on the specific model being used.
    pub fn vision_capability(&self) -> VisionCapabilityProxy {
        VisionCapabilityProxy::new(self, self.supports("vision"))
    }

    /// Generate embeddings for the given input texts
    ///
    /// This is a convenience method that directly calls the embedding functionality
    /// without requiring the user to go through the capability proxy.
    ///
    /// # Arguments
    /// * `texts` - List of strings to generate embeddings for
    ///
    /// # Returns
    /// List of embedding vectors (one per input text)
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Siumai::builder()
    ///     .openai()
    ///     .api_key("your-api-key")
    ///     .build()
    ///     .await?;
    ///
    /// let texts = vec!["Hello, world!".to_string()];
    /// let response = client.embed(texts).await?;
    /// println!("Got {} embeddings", response.embeddings.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        EmbeddingCapability::embed(self, texts).await
    }
}

#[async_trait::async_trait]
impl ChatCapability for Siumai {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.client.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.client.chat_stream(messages, tools).await
    }
}

#[async_trait::async_trait]
impl EmbeddingCapability for Siumai {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        // Dynamic dispatch based on provider type
        match self.client.provider_name() {
            "openai" => {
                // Try to get OpenAI client and call embed
                if let Some(openai_client) = self.client.as_any().downcast_ref::<crate::providers::openai::OpenAiClient>() {
                    openai_client.embed(texts).await
                } else {
                    Err(LlmError::UnsupportedOperation(
                        "OpenAI provider does not implement embedding functionality".to_string()
                    ))
                }
            }
            "ollama" => {
                // Try to get Ollama client and call embed
                if let Some(ollama_client) = self.client.as_any().downcast_ref::<crate::providers::ollama::OllamaClient>() {
                    ollama_client.embed(texts).await
                } else {
                    Err(LlmError::UnsupportedOperation(
                        "Ollama provider does not implement embedding functionality".to_string()
                    ))
                }
            }
            provider_name => {
                Err(LlmError::UnsupportedOperation(
                    format!("Provider {provider_name} does not support embedding functionality. Consider using OpenAI or Ollama for embeddings.")
                ))
            }
        }
    }

    fn embedding_dimension(&self) -> usize {
        // Try to get dimension from the underlying client
        match self.client.provider_name() {
            "openai" => {
                if let Some(openai_client) = self.client.as_any().downcast_ref::<crate::providers::openai::OpenAiClient>() {
                    openai_client.embedding_dimension()
                } else {
                    1536 // Default OpenAI dimension
                }
            }
            "ollama" => {
                if let Some(ollama_client) = self.client.as_any().downcast_ref::<crate::providers::ollama::OllamaClient>() {
                    ollama_client.embedding_dimension()
                } else {
                    384 // Default Ollama dimension
                }
            }
            _ => 1536 // Default fallback
        }
    }

    fn max_tokens_per_embedding(&self) -> usize {
        // Try to get max tokens from the underlying client
        match self.client.provider_name() {
            "openai" => {
                if let Some(openai_client) = self.client.as_any().downcast_ref::<crate::providers::openai::OpenAiClient>() {
                    openai_client.max_tokens_per_embedding()
                } else {
                    8192 // Default OpenAI limit
                }
            }
            "ollama" => {
                if let Some(ollama_client) = self.client.as_any().downcast_ref::<crate::providers::ollama::OllamaClient>() {
                    ollama_client.max_tokens_per_embedding()
                } else {
                    8192 // Default limit
                }
            }
            _ => 8192 // Default fallback
        }
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        // Try to get supported models from the underlying client
        match self.client.provider_name() {
            "openai" => {
                if let Some(openai_client) = self.client.as_any().downcast_ref::<crate::providers::openai::OpenAiClient>() {
                    openai_client.supported_embedding_models()
                } else {
                    vec![
                        "text-embedding-3-small".to_string(),
                        "text-embedding-3-large".to_string(),
                        "text-embedding-ada-002".to_string(),
                    ]
                }
            }
            "ollama" => {
                if let Some(ollama_client) = self.client.as_any().downcast_ref::<crate::providers::ollama::OllamaClient>() {
                    ollama_client.supported_embedding_models()
                } else {
                    vec!["nomic-embed-text".to_string()]
                }
            }
            _ => vec!["default".to_string()]
        }
    }
}

impl LlmClient for Siumai {
    fn provider_name(&self) -> &'static str {
        // We need to return a static str, so we'll use a match
        match self.metadata.provider_type {
            ProviderType::OpenAi => "openai",
            ProviderType::Anthropic => "anthropic",
            ProviderType::Gemini => "gemini",
            ProviderType::XAI => "xai",
            ProviderType::Ollama => "ollama",
            ProviderType::Custom(_) => "custom",
            ProviderType::Groq => "groq",
        }
    }

    fn supported_models(&self) -> Vec<String> {
        self.metadata.supported_models.clone()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.metadata.capabilities.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Builder for creating siumai providers with specific capabilities
pub struct SiumaiBuilder {
    provider_type: Option<ProviderType>,
    provider_name: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    capabilities: Vec<String>,
    common_params: CommonParams,
    http_config: HttpConfig,
    organization: Option<String>,
    project: Option<String>,
    tracing_config: Option<crate::tracing::TracingConfig>,
}

impl SiumaiBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            provider_type: None,
            provider_name: None,
            api_key: None,
            base_url: None,
            model: None,
            capabilities: Vec::new(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            organization: None,
            project: None,
            tracing_config: None,
        }
    }

    /// Set the provider type
    pub fn provider(mut self, provider_type: ProviderType) -> Self {
        self.provider_type = Some(provider_type);
        self
    }

    /// Set the provider by name (dynamic dispatch)
    /// This provides the llm_dart-style ai().provider('name') interface
    pub fn provider_name<S: Into<String>>(mut self, name: S) -> Self {
        let name = name.into();
        self.provider_name = Some(name.clone());

        // Map provider name to type
        self.provider_type = Some(match name.as_str() {
            "openai" => ProviderType::OpenAi,
            "anthropic" => ProviderType::Anthropic,
            "gemini" => ProviderType::Gemini,
            "ollama" => ProviderType::Ollama,
            "xai" => ProviderType::XAI,
            "deepseek" => ProviderType::Custom("deepseek".to_string()),
            "openrouter" => ProviderType::Custom("openrouter".to_string()),
            "groq" => ProviderType::Custom("groq".to_string()),
            _ => ProviderType::Custom(name),
        });
        self
    }

    // Convenience methods for specific providers (llm_dart style)

    /// Create an `OpenAI` provider (convenience method)
    pub fn openai(mut self) -> Self {
        self.provider_type = Some(ProviderType::OpenAi);
        self.provider_name = Some("openai".to_string());
        self
    }

    /// Create an Anthropic provider (convenience method)
    pub fn anthropic(mut self) -> Self {
        self.provider_type = Some(ProviderType::Anthropic);
        self.provider_name = Some("anthropic".to_string());
        self
    }

    /// Create a Gemini provider (convenience method)
    pub fn gemini(mut self) -> Self {
        self.provider_type = Some(ProviderType::Gemini);
        self.provider_name = Some("gemini".to_string());
        self
    }

    /// Create an Ollama provider (convenience method)
    pub fn ollama(mut self) -> Self {
        self.provider_type = Some(ProviderType::Ollama);
        self.provider_name = Some("ollama".to_string());
        self
    }

    /// Create a `DeepSeek` provider (convenience method)
    pub fn deepseek(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("deepseek".to_string()));
        self.provider_name = Some("deepseek".to_string());
        self
    }

    /// Create an `OpenRouter` provider (convenience method)
    pub fn openrouter(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("openrouter".to_string()));
        self.provider_name = Some("openrouter".to_string());
        self
    }

    /// Create a Groq provider (convenience method)
    pub fn groq(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("groq".to_string()));
        self.provider_name = Some("groq".to_string());
        self
    }

    /// Create an xAI provider (convenience method)
    pub fn xai(mut self) -> Self {
        self.provider_type = Some(ProviderType::XAI);
        self.provider_name = Some("xai".to_string());
        self
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

    /// Set temperature
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set organization (for `OpenAI`)
    pub fn organization<S: Into<String>>(mut self, organization: S) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Set project (for `OpenAI`)
    pub fn project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Enable a specific capability
    pub fn with_capability<S: Into<String>>(mut self, capability: S) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    /// Enable audio capability
    pub fn with_audio(self) -> Self {
        self.with_capability("audio")
    }

    /// Enable vision capability
    pub fn with_vision(self) -> Self {
        self.with_capability("vision")
    }

    /// Enable embedding capability
    pub fn with_embedding(self) -> Self {
        self.with_capability("embedding")
    }

    /// Enable image generation capability
    pub fn with_image_generation(self) -> Self {
        self.with_capability("image_generation")
    }

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    ///
    /// This allows you to configure detailed tracing and monitoring for this client.
    /// The tracing configuration will override any global tracing settings.
    ///
    /// # Arguments
    /// * `config` - The tracing configuration to use
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Siumai::builder()
    ///     .openai()
    ///     .api_key("your-key")
    ///     .model("gpt-4o-mini")
    ///     .tracing(TracingConfig::debug())
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
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

    /// Build the siumai provider
    pub async fn build(self) -> Result<Siumai, LlmError> {
        let provider_type = self.provider_type.ok_or_else(|| {
            LlmError::ConfigurationError("Provider type not specified".to_string())
        })?;

        let api_key = self
            .api_key
            .ok_or_else(|| LlmError::ConfigurationError("API key not specified".to_string()))?;

        // Create the appropriate client based on provider type
        let client: Box<dyn LlmClient> = match provider_type {
            ProviderType::OpenAi => {
                let mut config = crate::providers::openai::OpenAiConfig::new(api_key)
                    .with_base_url(
                        self.base_url
                            .unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
                    )
                    .with_model(self.model.unwrap_or_else(|| "gpt-4o-mini".to_string()));

                // Set common parameters
                if let Some(temp) = self.common_params.temperature {
                    config = config.with_temperature(temp);
                }
                if let Some(max_tokens) = self.common_params.max_tokens {
                    config = config.with_max_tokens(max_tokens);
                }

                // Set organization and project if provided
                if let Some(org) = self.organization {
                    config = config.with_organization(org);
                }
                if let Some(proj) = self.project {
                    config = config.with_project(proj);
                }

                let http_client = reqwest::Client::new();
                Box::new(crate::providers::openai::OpenAiClient::new(
                    config,
                    http_client,
                ))
            }
            ProviderType::Anthropic => {
                let base_url = self
                    .base_url
                    .unwrap_or_else(|| "https://api.anthropic.com".to_string());
                let model = self
                    .model
                    .unwrap_or_else(|| "claude-3-5-sonnet-20241022".to_string());

                // Set model in common params
                let mut common_params = self.common_params;
                common_params.model = model;

                let http_client = reqwest::Client::new();
                Box::new(crate::providers::anthropic::AnthropicClient::new(
                    api_key,
                    base_url,
                    http_client,
                    common_params,
                    crate::params::AnthropicParams::default(),
                    self.http_config,
                ))
            }
            ProviderType::Gemini => {
                return Err(LlmError::UnsupportedOperation(
                    "Gemini provider not yet implemented in unified interface".to_string(),
                ));
            }
            ProviderType::XAI => {
                return Err(LlmError::UnsupportedOperation(
                    "xAI provider not yet implemented in unified interface".to_string(),
                ));
            }
            ProviderType::Ollama => {
                let base_url = self
                    .base_url
                    .unwrap_or_else(|| "http://localhost:11434".to_string());
                let model = self.model.unwrap_or_else(|| "llama3.2:latest".to_string());

                // Set model in common params
                let mut common_params = self.common_params;
                common_params.model = model.clone();

                let config = crate::providers::ollama::config::OllamaConfig {
                    base_url,
                    model: Some(model),
                    common_params,
                    ollama_params: crate::providers::ollama::config::OllamaParams::default(),
                    http_config: self.http_config,
                };

                let http_client = reqwest::Client::new();
                Box::new(crate::providers::ollama::OllamaClient::new(
                    config,
                    http_client,
                ))
            }
            ProviderType::Groq => {
                let base_url = self
                    .base_url
                    .unwrap_or_else(|| "https://api.groq.com/openai/v1".to_string());
                let model = self
                    .model
                    .unwrap_or_else(|| "llama-3.3-70b-versatile".to_string());

                let mut config = crate::providers::groq::GroqConfig::new(api_key)
                    .with_base_url(base_url)
                    .with_model(model);

                // Set common parameters
                if let Some(temp) = self.common_params.temperature {
                    config = config.with_temperature(temp);
                }
                if let Some(max_tokens) = self.common_params.max_tokens {
                    config = config.with_max_tokens(max_tokens);
                }

                let http_client = reqwest::Client::new();
                Box::new(crate::providers::groq::GroqClient::new(config, http_client))
            }
            ProviderType::Custom(name) => {
                match name.as_str() {
                    "deepseek" => {
                        // Use OpenAI-compatible client for DeepSeek
                        let mut config = crate::providers::openai::OpenAiConfig::new(api_key)
                            .with_base_url(
                                self.base_url
                                    .unwrap_or_else(|| "https://api.deepseek.com".to_string()),
                            )
                            .with_model(self.model.unwrap_or_else(|| "deepseek-chat".to_string()));

                        // Set common parameters
                        if let Some(temp) = self.common_params.temperature {
                            config = config.with_temperature(temp);
                        }
                        if let Some(max_tokens) = self.common_params.max_tokens {
                            config = config.with_max_tokens(max_tokens);
                        }

                        let http_client = reqwest::Client::new();
                        Box::new(crate::providers::openai::OpenAiClient::new(
                            config,
                            http_client,
                        ))
                    }
                    "openrouter" => {
                        // Use OpenAI-compatible client for OpenRouter
                        let mut config = crate::providers::openai::OpenAiConfig::new(api_key)
                            .with_base_url(
                                self.base_url
                                    .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string()),
                            )
                            .with_model(
                                self.model
                                    .unwrap_or_else(|| "openai/gpt-3.5-turbo".to_string()),
                            );

                        // Set common parameters
                        if let Some(temp) = self.common_params.temperature {
                            config = config.with_temperature(temp);
                        }
                        if let Some(max_tokens) = self.common_params.max_tokens {
                            config = config.with_max_tokens(max_tokens);
                        }

                        let http_client = reqwest::Client::new();
                        Box::new(crate::providers::openai::OpenAiClient::new(
                            config,
                            http_client,
                        ))
                    }
                    "groq" => {
                        // Use OpenAI-compatible client for Groq
                        let mut config =
                            crate::providers::openai::OpenAiConfig::new(api_key)
                                .with_base_url(self.base_url.unwrap_or_else(|| {
                                    "https://api.groq.com/openai/v1".to_string()
                                }))
                                .with_model(
                                    self.model
                                        .unwrap_or_else(|| "llama-3.3-70b-versatile".to_string()),
                                );

                        // Set common parameters
                        if let Some(temp) = self.common_params.temperature {
                            config = config.with_temperature(temp);
                        }
                        if let Some(max_tokens) = self.common_params.max_tokens {
                            config = config.with_max_tokens(max_tokens);
                        }

                        let http_client = reqwest::Client::new();
                        Box::new(crate::providers::openai::OpenAiClient::new(
                            config,
                            http_client,
                        ))
                    }
                    _ => {
                        return Err(LlmError::UnsupportedOperation(format!(
                            "Custom provider '{name}' not yet implemented"
                        )));
                    }
                }
            }
        };

        Ok(Siumai::new(client))
    }
}

/// Type-safe proxy for audio capabilities
pub struct AudioCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

impl<'a> AudioCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports audio support (for reference only)
    ///
    /// Note: This is based on static capability information and may not reflect
    /// the actual capabilities of the current model. Use as a hint, not a restriction.
    /// The library will never block operations based on this information.
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider name for debugging
    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }

    /// Get a support status message (optional, for user-controlled warnings)
    ///
    /// Returns a message about support status that you can choose to display or ignore.
    /// The library itself will not automatically warn or log anything.
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports audio support", self.provider_name())
        } else {
            format!(
                "Provider {} does not report audio support, but this may still work depending on the model",
                self.provider_name()
            )
        }
    }

    /// Placeholder for future audio operations
    ///
    /// This will attempt the operation regardless of reported support.
    /// Actual errors will come from the API if the model doesn't support it.
    pub async fn placeholder_operation(&self) -> Result<String, LlmError> {
        // No automatic warnings - let the user decide if they want to check support
        Err(LlmError::UnsupportedOperation(
            "Audio operations not yet implemented. Use provider-specific client.".to_string(),
        ))
    }
}

/// Type-safe proxy for embedding capabilities
pub struct EmbeddingCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

impl<'a> EmbeddingCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports embedding support (for reference only)
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider name for debugging
    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }

    /// Get a support status message (optional, for user-controlled information)
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!(
                "Provider {} reports embedding support",
                self.provider_name()
            )
        } else {
            format!(
                "Provider {} does not report embedding support, but this may still work depending on the model",
                self.provider_name()
            )
        }
    }

    /// Generate embeddings for the given input texts
    pub async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        self.provider.embed(texts).await
    }

    /// Get the dimension of embeddings produced by this provider
    pub fn embedding_dimension(&self) -> usize {
        self.provider.embedding_dimension()
    }

    /// Get the maximum number of tokens that can be embedded at once
    pub fn max_tokens_per_embedding(&self) -> usize {
        self.provider.max_tokens_per_embedding()
    }

    /// Get supported embedding models for this provider
    pub fn supported_embedding_models(&self) -> Vec<String> {
        self.provider.supported_embedding_models()
    }

    /// Placeholder for future embedding operations (deprecated, use embed() instead)
    #[deprecated(note = "Use embed() method instead")]
    pub async fn placeholder_operation(&self) -> Result<String, LlmError> {
        // No automatic warnings - let the user decide if they want to check support
        Err(LlmError::UnsupportedOperation(
            "Use embed() method instead of placeholder_operation()".to_string(),
        ))
    }
}

/// Type-safe proxy for vision capabilities
pub struct VisionCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

impl<'a> VisionCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports vision support (for reference only)
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider name for debugging
    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }

    /// Get a support status message (optional, for user-controlled information)
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports vision support", self.provider_name())
        } else {
            format!(
                "Provider {} does not report vision support, but this may still work depending on the model",
                self.provider_name()
            )
        }
    }

    /// Placeholder for future vision operations
    pub async fn placeholder_operation(&self) -> Result<String, LlmError> {
        // No automatic warnings - let the user decide if they want to check support
        Err(LlmError::UnsupportedOperation(
            "Vision operations not yet implemented. Use provider-specific client.".to_string(),
        ))
    }
}

impl Default for SiumaiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Provider registry for dynamic provider creation
pub struct ProviderRegistry {
    factories: HashMap<String, Box<dyn ProviderFactory>>,
}

/// Factory trait for creating providers
pub trait ProviderFactory: Send + Sync {
    fn create_provider(&self, config: ProviderConfig) -> Result<Box<dyn LlmClient>, LlmError>;
    fn supported_capabilities(&self) -> Vec<String>;
}

/// Configuration for provider creation
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub capabilities: Vec<String>,
}

impl ProviderRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a provider factory
    pub fn register<S: Into<String>>(&mut self, name: S, factory: Box<dyn ProviderFactory>) {
        self.factories.insert(name.into(), factory);
    }

    /// Create a provider by name
    pub fn create_provider(&self, name: &str, config: ProviderConfig) -> Result<Siumai, LlmError> {
        let factory = self
            .factories
            .get(name)
            .ok_or_else(|| LlmError::ConfigurationError(format!("Unknown provider: {name}")))?;

        let client = factory.create_provider(config)?;
        Ok(Siumai::new(client))
    }

    /// Get supported providers
    pub fn supported_providers(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    // Mock provider for testing that doesn't support embedding
    #[derive(Debug)]
    struct MockProvider;

    #[async_trait]
    impl ChatCapability for MockProvider {
        async fn chat_with_tools(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                id: Some("mock-123".to_string()),
                content: MessageContent::Text("Mock response".to_string()),
                model: Some("mock-model".to_string()),
                usage: None,
                finish_reason: Some(FinishReason::Stop),
                tool_calls: None,
                thinking: None,
                metadata: std::collections::HashMap::new(),
            })
        }

        async fn chat_stream(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation("Streaming not supported in mock".to_string()))
        }
    }

    impl LlmClient for MockProvider {
        fn provider_name(&self) -> &'static str {
            "mock"
        }

        fn supported_models(&self) -> Vec<String> {
            vec!["mock-model".to_string()]
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_chat()
            // Note: not adding .with_embedding() to test unsupported case
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[tokio::test]
    async fn test_siumai_embedding_unsupported_provider() {
        // Create a mock provider that doesn't support embedding
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Box::new(mock_provider));

        // Test that embedding returns an error for unsupported provider
        let result = siumai.embed(vec!["test".to_string()]).await;
        assert!(result.is_err());

        if let Err(LlmError::UnsupportedOperation(msg)) = result {
            assert!(msg.contains("does not support embedding functionality"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[test]
    fn test_embedding_capability_proxy() {
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Box::new(mock_provider));

        let proxy = siumai.embedding_capability();
        assert_eq!(proxy.provider_name(), "custom"); // MockProvider gets mapped to "custom" type
        assert!(!proxy.is_reported_as_supported()); // Mock provider doesn't report embedding support
    }

    #[tokio::test]
    async fn test_embedding_capability_proxy_embed() {
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Box::new(mock_provider));

        let proxy = siumai.embedding_capability();
        let result = proxy.embed(vec!["test".to_string()]).await;
        assert!(result.is_err());

        if let Err(LlmError::UnsupportedOperation(msg)) = result {
            assert!(msg.contains("does not support embedding functionality"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }
}
