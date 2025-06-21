//! OpenAI-Compatible Provider Builder
//!
//! This module provides the builder pattern for OpenAI-compatible providers.

use crate::builder::LlmBuilder;
use crate::error::LlmError;
use crate::providers::openai::OpenAiClient;
use std::marker::PhantomData;

use super::config::OpenAiCompatibleConfig;
use super::providers::{DeepSeekProvider, OpenAiCompatibleProvider, OpenRouterProvider};

/// OpenAI-compatible provider builder with type-safe provider selection.
///
/// This builder provides a type-safe way to configure OpenAI-compatible providers
/// with provider-specific methods and validation.
pub struct OpenAiCompatibleBuilder<P: OpenAiCompatibleProvider> {
    /// Base LLM builder for HTTP configuration
    base: LlmBuilder,
    /// Provider-specific configuration
    config: OpenAiCompatibleConfig,
    /// Type marker for the provider
    _provider: PhantomData<P>,
}

impl<P: OpenAiCompatibleProvider> OpenAiCompatibleBuilder<P> {
    /// Create a new OpenAI-compatible builder for the specified provider
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            base,
            config: OpenAiCompatibleConfig::new(P::PROVIDER_ID.to_string(), String::new()),
            _provider: PhantomData,
        }
    }

    /// Set the API key for the provider
    pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
        self.config.api_key = key.into();
        self
    }

    /// Set the model to use
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.config = self.config.with_model(model.into());
        self
    }

    /// Set the base URL (overrides provider default)
    pub fn base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.config = self.config.with_base_url(url.into());
        self
    }

    /// Set the temperature parameter
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.config.common_params.temperature = Some(temp);
        self
    }

    /// Set the maximum number of tokens to generate
    pub const fn max_tokens(mut self, tokens: u32) -> Self {
        self.config.common_params.max_tokens = Some(tokens);
        self
    }

    /// Set the `top_p` parameter for nucleus sampling
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.config.common_params.top_p = Some(top_p);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.config.common_params.stop_sequences = Some(sequences);
        self
    }

    /// Set the random seed for reproducible outputs
    pub const fn seed(mut self, seed: u64) -> Self {
        self.config.common_params.seed = Some(seed);
        self
    }

    /// Add a provider-specific parameter
    pub fn with_provider_param<T: serde::Serialize>(
        mut self,
        key: String,
        value: T,
    ) -> Result<Self, LlmError> {
        self.config = self.config.with_provider_param(key, value)?;
        Ok(self)
    }

    /// Build the OpenAI-compatible client
    pub async fn build(self) -> Result<OpenAiCompatibleClient<P>, LlmError> {
        // Validate configuration
        P::validate_config(&self.config)?;

        // Transform provider-specific parameters
        let mut provider_params = self.config.provider_params.clone();
        P::transform_params(&mut provider_params)?;

        // Convert to OpenAI configuration
        let openai_config = self
            .config
            .to_openai_config(P::DEFAULT_BASE_URL, P::DEFAULT_MODEL)?;

        // Build HTTP client
        let http_client = self.base.build_http_client()?;

        // Create OpenAI client with the configuration
        let openai_client = OpenAiClient::new(openai_config, http_client);

        // Wrap in compatible client
        Ok(OpenAiCompatibleClient::new(openai_client))
    }
}

/// OpenAI-compatible client wrapper that provides provider-specific metadata.
///
/// This struct wraps an `OpenAI` client but provides provider-specific information
/// and maintains type safety for the provider.
pub struct OpenAiCompatibleClient<P: OpenAiCompatibleProvider> {
    /// Underlying `OpenAI` client
    pub(crate) client: OpenAiClient,
    /// Provider type marker
    _provider: PhantomData<P>,
}

impl<P: OpenAiCompatibleProvider> OpenAiCompatibleClient<P> {
    /// Create a new compatible client
    pub const fn new(client: OpenAiClient) -> Self {
        Self {
            client,
            _provider: PhantomData,
        }
    }

    /// Get the provider ID
    pub const fn provider_id(&self) -> &'static str {
        P::PROVIDER_ID
    }

    /// Get the provider display name
    pub const fn display_name(&self) -> &'static str {
        P::DISPLAY_NAME
    }

    /// Get the provider description
    pub const fn description(&self) -> &'static str {
        P::DESCRIPTION
    }

    /// Get the underlying `OpenAI` client
    pub const fn inner(&self) -> &OpenAiClient {
        &self.client
    }

    /// Get the underlying `OpenAI` client mutably
    pub const fn inner_mut(&mut self) -> &mut OpenAiClient {
        &mut self.client
    }

    /// Convert into the underlying `OpenAI` client
    pub fn into_inner(self) -> OpenAiClient {
        self.client
    }
}

// Provider-specific builder extensions

/// DeepSeek-specific builder methods
impl OpenAiCompatibleBuilder<DeepSeekProvider> {
    /// Enable reasoning mode (switches to deepseek-reasoner model)
    pub fn reasoning(self, enabled: bool) -> Result<Self, LlmError> {
        self.with_provider_param("reasoning".to_string(), enabled)
    }

    /// Enable coding mode (switches to deepseek-coder model)
    pub fn coding(self, enabled: bool) -> Result<Self, LlmError> {
        self.with_provider_param("coding".to_string(), enabled)
    }

    /// Set thinking budget for reasoning (DeepSeek-specific parameter)
    pub fn thinking_budget(self, budget: u32) -> Result<Self, LlmError> {
        self.with_provider_param("thinking_budget".to_string(), budget)
    }
}

/// OpenRouter-specific builder methods
impl OpenAiCompatibleBuilder<OpenRouterProvider> {
    /// Set the site URL for `OpenRouter` (becomes HTTP-Referer header)
    pub fn site_url<S: Into<String>>(self, url: S) -> Result<Self, LlmError> {
        self.with_provider_param("site_url".to_string(), url.into())
    }

    /// Set the application name for `OpenRouter` (becomes X-Title header)
    pub fn app_name<S: Into<String>>(self, name: S) -> Result<Self, LlmError> {
        self.with_provider_param("app_name".to_string(), name.into())
    }

    /// Set fallback models for `OpenRouter` routing
    pub fn fallback_models(self, models: Vec<String>) -> Result<Self, LlmError> {
        self.with_provider_param("fallback_models".to_string(), models)
    }
}
