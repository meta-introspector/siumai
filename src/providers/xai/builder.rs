//! `xAI` Builder Implementation
//!
//! Provides a builder pattern for creating `xAI` clients.

use crate::error::LlmError;
use crate::types::{CommonParams, HttpConfig, WebSearchConfig};

use super::client::XaiClient;
use super::config::XaiConfig;

/// `xAI` Client Builder
#[derive(Debug, Clone)]
pub struct XaiBuilder {
    config: XaiConfig,
    tracing_config: Option<crate::tracing::TracingConfig>,
}

impl XaiBuilder {
    /// Create a new `xAI` builder
    pub fn new() -> Self {
        Self {
            config: XaiConfig::default(),
            tracing_config: None,
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.config.api_key = api_key.into();
        self
    }

    /// Set the base URL
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.config.base_url = base_url.into();
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.config.common_params.model = model.into();
        self
    }

    /// Set the temperature
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.config.common_params.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top-p value
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.config.common_params.top_p = Some(top_p);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, stop: Vec<String>) -> Self {
        self.config.common_params.stop_sequences = Some(stop);
        self
    }

    /// Set the seed
    pub const fn seed(mut self, seed: u64) -> Self {
        self.config.common_params.seed = Some(seed);
        self
    }

    /// Set common parameters
    pub fn common_params(mut self, params: CommonParams) -> Self {
        self.config.common_params = params;
        self
    }

    /// Set HTTP configuration
    pub fn http_config(mut self, config: HttpConfig) -> Self {
        self.config.http_config = config;
        self
    }

    /// Set web search configuration
    pub fn web_search_config(mut self, config: WebSearchConfig) -> Self {
        self.config.web_search_config = config;
        self
    }

    /// Enable web search with default settings
    pub const fn enable_web_search(mut self) -> Self {
        self.config.web_search_config.enabled = true;
        self
    }

    /// Set the entire configuration
    pub fn config(mut self, config: XaiConfig) -> Self {
        self.config = config;
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

    /// Build the `xAI` client
    pub async fn build(self) -> Result<XaiClient, LlmError> {
        // Validate configuration
        self.config
            .validate()
            .map_err(|e| LlmError::InvalidInput(format!("Invalid xAI configuration: {e}")))?;

        // Initialize tracing if configured
        let _tracing_guard = if let Some(ref tracing_config) = self.tracing_config {
            Some(crate::tracing::init_tracing(tracing_config.clone())?)
        } else {
            None
        };

        // Set default model if not specified
        let mut config = self.config;
        if config.common_params.model.is_empty() {
            config.common_params.model = "grok-3-latest".to_string();
        }

        let mut client = XaiClient::new(config).await?;
        client.set_tracing_guard(_tracing_guard);
        client.set_tracing_config(self.tracing_config);

        Ok(client)
    }

    /// Build the `xAI` client with a custom HTTP client
    pub async fn build_with_client(
        self,
        http_client: reqwest::Client,
    ) -> Result<XaiClient, LlmError> {
        // Validate configuration
        self.config
            .validate()
            .map_err(|e| LlmError::InvalidInput(format!("Invalid xAI configuration: {e}")))?;

        // Initialize tracing if configured
        let _tracing_guard = if let Some(ref tracing_config) = self.tracing_config {
            Some(crate::tracing::init_tracing(tracing_config.clone())?)
        } else {
            None
        };

        // Set default model if not specified
        let mut config = self.config;
        if config.common_params.model.is_empty() {
            config.common_params.model = "grok-3-latest".to_string();
        }

        let mut client = XaiClient::with_http_client(config, http_client).await?;
        client.set_tracing_guard(_tracing_guard);
        client.set_tracing_config(self.tracing_config);

        Ok(client)
    }
}

impl Default for XaiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = XaiBuilder::new();
        assert_eq!(builder.config.base_url, "https://api.x.ai/v1");
        assert!(builder.config.api_key.is_empty());
    }

    #[test]
    fn test_builder_configuration() {
        let builder = XaiBuilder::new()
            .api_key("test-key")
            .model("grok-3-latest")
            .temperature(0.7)
            .max_tokens(1000);

        assert_eq!(builder.config.api_key, "test-key");
        assert_eq!(builder.config.common_params.model, "grok-3-latest");
        assert_eq!(builder.config.common_params.temperature, Some(0.7));
        assert_eq!(builder.config.common_params.max_tokens, Some(1000));
    }

    #[tokio::test]
    async fn test_builder_validation() {
        let builder = XaiBuilder::new();

        // Should fail without API key
        let result = builder.build().await;
        assert!(result.is_err());

        // Should succeed with API key
        let builder = XaiBuilder::new().api_key("test-key");
        let result = builder.build().await;
        assert!(result.is_ok());
    }
}
