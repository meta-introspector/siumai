//! OpenAI-Compatible Provider Configuration
//!
//! This module defines the configuration structures and types for OpenAI-compatible providers.

use crate::error::LlmError;
use crate::types::CommonParams;
use std::collections::HashMap;

/// OpenAI-compatible provider configuration.
///
/// This struct holds the configuration for any OpenAI-compatible provider.
/// It uses the `OpenAI` client internally but with provider-specific settings.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleConfig {
    /// Provider identifier
    pub provider_id: String,
    /// API key
    pub api_key: String,
    /// Base URL (defaults to provider's default)
    pub base_url: Option<String>,
    /// Model name (defaults to provider's default)
    pub model: Option<String>,
    /// Common AI parameters
    pub common_params: CommonParams,
    /// Provider-specific parameters
    pub provider_params: HashMap<String, serde_json::Value>,
}

impl OpenAiCompatibleConfig {
    /// Create a new configuration for a specific provider
    pub fn new(provider_id: String, api_key: String) -> Self {
        Self {
            provider_id,
            api_key,
            base_url: None,
            model: None,
            common_params: CommonParams::default(),
            provider_params: HashMap::new(),
        }
    }

    /// Set the base URL
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = Some(base_url);
        self
    }

    /// Set the model
    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    /// Set common parameters
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    /// Add a provider-specific parameter
    pub fn with_provider_param<T: serde::Serialize>(
        mut self,
        key: String,
        value: T,
    ) -> Result<Self, LlmError> {
        let json_value = serde_json::to_value(value)
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid parameter value: {e}")))?;
        self.provider_params.insert(key, json_value);
        Ok(self)
    }

    /// Convert to `OpenAI` configuration
    pub fn to_openai_config(
        self,
        default_base_url: &str,
        default_model: &str,
    ) -> Result<crate::providers::openai::config::OpenAiConfig, LlmError> {
        // Get base URL and model with provider defaults
        let base_url = self
            .base_url
            .unwrap_or_else(|| default_base_url.to_string());
        let mut model = self.model.unwrap_or_else(|| default_model.to_string());

        // Handle provider-specific model selection
        if self.provider_id == "deepseek" {
            use crate::models::openai_compatible::deepseek;

            // Override model based on reasoning/coding parameters
            if let Some(reasoning) = self.provider_params.get("reasoning") {
                if reasoning.as_bool() == Some(true) {
                    model = deepseek::REASONER.to_string();
                }
            } else if let Some(coding) = self.provider_params.get("coding")
                && coding.as_bool() == Some(true)
            {
                model = deepseek::V3.to_string(); // Use V3 for coding
            }
        }

        // Create OpenAI configuration
        let mut openai_config = crate::providers::openai::config::OpenAiConfig::new(self.api_key)
            .with_base_url(base_url)
            .with_model(model);

        // Apply common parameters
        if let Some(temp) = self.common_params.temperature {
            openai_config = openai_config.with_temperature(temp);
        }
        if let Some(max_tokens) = self.common_params.max_tokens {
            openai_config = openai_config.with_max_tokens(max_tokens);
        }
        if let Some(top_p) = self.common_params.top_p {
            openai_config.common_params.top_p = Some(top_p);
        }
        if let Some(stop_sequences) = self.common_params.stop_sequences {
            openai_config.common_params.stop_sequences = Some(stop_sequences);
        }
        if let Some(seed) = self.common_params.seed {
            openai_config.common_params.seed = Some(seed);
        }

        Ok(openai_config)
    }
}
