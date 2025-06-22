//! `OpenAI` Provider Builder
//!
//! This module provides the OpenAI-specific builder implementation that follows
//! the design pattern established in the main builder module.

use crate::builder::LlmBuilder;
use crate::error::LlmError;
use crate::params::{OpenAiParams, ResponseFormat, ToolChoice};
use crate::types::*;

use super::{OpenAiClient, OpenAiConfig};

/// OpenAI-specific builder for configuring `OpenAI` clients.
///
/// This builder provides OpenAI-specific configuration options while
/// inheriting common HTTP and timeout settings from the base `LlmBuilder`.
///
/// # Example
/// ```rust,no_run
/// use siumai::builder::LlmBuilder;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = LlmBuilder::new()
///         .openai()
///         .api_key("your-api-key")
///         .model("gpt-4")
///         .temperature(0.7)
///         .max_tokens(1000)
///         .build()
///         .await?;
///
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct OpenAiBuilder {
    /// Base builder with HTTP configuration
    base: LlmBuilder,
    /// `OpenAI` API key
    api_key: Option<String>,
    /// Base URL for `OpenAI` API
    base_url: Option<String>,
    /// Organization ID
    organization: Option<String>,
    /// Project ID
    project: Option<String>,
    /// Model name
    model: Option<String>,
    /// Common parameters shared across providers
    common_params: CommonParams,
    /// OpenAI-specific parameters
    openai_params: OpenAiParams,
    /// HTTP configuration
    http_config: HttpConfig,
}

impl OpenAiBuilder {
    /// Create a new `OpenAI` builder from the base LLM builder.
    ///
    /// # Arguments
    /// * `base` - The base LLM builder with HTTP configuration
    pub fn new(base: LlmBuilder) -> Self {
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

    // === Authentication and Connection ===

    /// Set the `OpenAI` API key.
    ///
    /// If not provided, the builder will attempt to read from the
    /// `OPENAI_API_KEY` environment variable.
    ///
    /// # Arguments
    /// * `key` - The `OpenAI` API key
    pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set a custom base URL for the `OpenAI` API.
    ///
    /// This is useful for using OpenAI-compatible APIs or proxies.
    ///
    /// # Arguments
    /// * `url` - The base URL (e.g., "<https://api.openai.com/v1>")
    pub fn base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the `OpenAI` organization ID.
    ///
    /// # Arguments
    /// * `org` - The organization ID
    pub fn organization<S: Into<String>>(mut self, org: S) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Set the `OpenAI` project ID.
    ///
    /// # Arguments
    /// * `project` - The project ID
    pub fn project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    // === Model and Common Parameters ===

    /// Set the model to use.
    ///
    /// # Arguments
    /// * `model` - The model name (e.g., "gpt-4", "gpt-3.5-turbo")
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        let model_str = model.into();
        self.model = Some(model_str.clone());
        self.common_params.model = model_str;
        self
    }

    /// Set the temperature for randomness in responses.
    ///
    /// # Arguments
    /// * `temp` - Temperature value (0.0 to 2.0)
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.common_params.temperature = Some(temp);
        self
    }

    /// Set the maximum number of tokens to generate.
    ///
    /// # Arguments
    /// * `tokens` - Maximum number of tokens
    pub const fn max_tokens(mut self, tokens: u32) -> Self {
        self.common_params.max_tokens = Some(tokens);
        self
    }

    /// Set the `top_p` parameter for nucleus sampling.
    ///
    /// # Arguments
    /// * `top_p` - Top-p value (0.0 to 1.0)
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set stop sequences that will halt generation.
    ///
    /// # Arguments
    /// * `sequences` - List of stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(sequences);
        self
    }

    /// Set a random seed for reproducible outputs.
    ///
    /// # Arguments
    /// * `seed` - Random seed value
    pub const fn seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    // === OpenAI-Specific Parameters ===

    /// Set the response format.
    ///
    /// # Arguments
    /// * `format` - The response format (text, `json_object`, etc.)
    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.openai_params.response_format = Some(format);
        self
    }

    /// Set the tool choice strategy.
    ///
    /// # Arguments
    /// * `choice` - The tool choice strategy
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.openai_params.tool_choice = Some(choice);
        self
    }

    /// Set the frequency penalty.
    ///
    /// # Arguments
    /// * `penalty` - Frequency penalty (-2.0 to 2.0)
    pub const fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.frequency_penalty = Some(penalty);
        self
    }

    /// Set the presence penalty.
    ///
    /// # Arguments
    /// * `penalty` - Presence penalty (-2.0 to 2.0)
    pub const fn presence_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.presence_penalty = Some(penalty);
        self
    }

    /// Set the user ID for tracking purposes.
    ///
    /// # Arguments
    /// * `user` - User identifier
    pub fn user<S: Into<String>>(mut self, user: S) -> Self {
        self.openai_params.user = Some(user.into());
        self
    }

    /// Enable or disable parallel tool calls.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable parallel tool calls
    pub const fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.openai_params.parallel_tool_calls = Some(enabled);
        self
    }

    // === HTTP Configuration ===

    /// Set HTTP configuration options.
    ///
    /// # Arguments
    /// * `config` - HTTP configuration
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = config;
        self
    }

    // === Build Method ===

    /// Build the `OpenAI` client with the configured settings.
    ///
    /// # Returns
    /// A configured `OpenAI` client ready for use
    ///
    /// # Errors
    /// Returns an error if:
    /// - API key is not provided and not found in environment
    /// - HTTP client configuration is invalid
    /// - Required parameters are missing
    pub async fn build(self) -> Result<OpenAiClient, LlmError> {
        // Get API key from builder or environment
        let api_key = self
            .api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or(LlmError::MissingApiKey(
                "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
                    .to_string(),
            ))?;

        // Set default base URL if not provided
        let base_url = self
            .base_url
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        // Build HTTP client using the base builder
        let http_client = self.base.build_http_client()?;

        // Create OpenAI configuration
        let config = OpenAiConfig {
            api_key,
            base_url,
            organization: self.organization,
            project: self.project,
            common_params: self.common_params,
            openai_params: self.openai_params,
            http_config: self.http_config,
            web_search_config: crate::types::WebSearchConfig::default(),
            use_responses_api: false,
            previous_response_id: None,
            built_in_tools: Vec::new(),
        };

        // Create and return the client
        Ok(OpenAiClient::new(config, http_client))
    }
}
