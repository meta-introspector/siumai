//! `Groq` Builder Implementation
//!
//! Builder pattern implementation for creating Groq clients.

use std::time::Duration;

use crate::error::LlmError;
use crate::types::HttpConfig;

use super::client::GroqClient;
use super::config::GroqConfig;

/// `Groq` client builder
#[derive(Debug, Clone)]
pub struct GroqBuilder {
    config: GroqConfig,
}

impl GroqBuilder {
    /// Create a new `Groq` builder
    pub fn new() -> Self {
        Self {
            config: GroqConfig::default(),
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
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.common_params.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top_p parameter
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.config.common_params.top_p = Some(top_p);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.config.common_params.stop_sequences = Some(stop_sequences);
        self
    }

    /// Set the seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.common_params.seed = Some(seed);
        self
    }

    /// Set request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.http_config.timeout = Some(timeout);
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, connect_timeout: Duration) -> Self {
        self.config.http_config.connect_timeout = Some(connect_timeout);
        self
    }

    /// Add a custom header
    pub fn header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.config
            .http_config
            .headers
            .insert(key.into(), value.into());
        self
    }

    /// Set proxy URL
    pub fn proxy<S: Into<String>>(mut self, proxy: S) -> Self {
        self.config.http_config.proxy = Some(proxy.into());
        self
    }

    /// Set user agent
    pub fn user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.config.http_config.user_agent = Some(user_agent.into());
        self
    }

    /// Add a built-in tool
    pub fn tool(mut self, tool: crate::types::Tool) -> Self {
        self.config.built_in_tools.push(tool);
        self
    }

    /// Add multiple built-in tools
    pub fn tools(mut self, tools: Vec<crate::types::Tool>) -> Self {
        self.config.built_in_tools.extend(tools);
        self
    }

    /// Build the `Groq` client
    pub async fn build(mut self) -> Result<GroqClient, LlmError> {
        // Try to get API key from environment if not set
        if self.config.api_key.is_empty() {
            if let Ok(api_key) = std::env::var("GROQ_API_KEY") {
                self.config.api_key = api_key;
            }
        }

        // Validate configuration
        self.config.validate()?;

        // Create HTTP client
        let mut client_builder = reqwest::Client::builder();

        // Set timeouts
        if let Some(timeout) = self.config.http_config.timeout {
            client_builder = client_builder.timeout(timeout);
        }
        if let Some(connect_timeout) = self.config.http_config.connect_timeout {
            client_builder = client_builder.connect_timeout(connect_timeout);
        }

        // Set proxy
        if let Some(proxy_url) = &self.config.http_config.proxy {
            let proxy = reqwest::Proxy::all(proxy_url)
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid proxy URL: {e}")))?;
            client_builder = client_builder.proxy(proxy);
        }

        // Set user agent
        if let Some(user_agent) = &self.config.http_config.user_agent {
            client_builder = client_builder.user_agent(user_agent);
        }

        let http_client = client_builder.build().map_err(|e| {
            LlmError::ConfigurationError(format!("Failed to create HTTP client: {e}"))
        })?;

        Ok(GroqClient::new(self.config, http_client))
    }

    /// Get the current configuration (for inspection)
    pub fn config(&self) -> &GroqConfig {
        &self.config
    }

    /// Set the entire HTTP configuration
    pub fn http_config(mut self, http_config: HttpConfig) -> Self {
        self.config.http_config = http_config;
        self
    }

    /// Set the entire configuration
    pub fn with_config(mut self, config: GroqConfig) -> Self {
        self.config = config;
        self
    }
}

impl Default for GroqBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groq_builder() {
        let builder = GroqBuilder::new()
            .api_key("test-key")
            .model("llama-3.3-70b-versatile")
            .temperature(0.7)
            .max_tokens(1000)
            .timeout(Duration::from_secs(30));

        let config = builder.config();
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.common_params.model, "llama-3.3-70b-versatile");
        assert_eq!(config.common_params.temperature, Some(0.7));
        assert_eq!(config.common_params.max_tokens, Some(1000));
        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_groq_builder_default() {
        let builder = GroqBuilder::default();
        let config = builder.config();
        assert_eq!(config.base_url, GroqConfig::DEFAULT_BASE_URL);
        assert_eq!(config.common_params.model, GroqConfig::default_model());
    }

    #[test]
    fn test_groq_builder_validation() {
        let builder = GroqBuilder::new()
            .api_key("") // Empty API key should fail validation
            .model("llama-3.3-70b-versatile");

        // This should fail during build due to empty API key
        assert!(builder.config.validate().is_err());
    }

    #[test]
    fn test_groq_builder_tools() {
        use crate::types::{Tool, ToolFunction};

        let tool = Tool {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: "test_function".to_string(),
                description: "A test function".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        };

        let builder = GroqBuilder::new().api_key("test-key").tool(tool.clone());

        let config = builder.config();
        assert_eq!(config.built_in_tools.len(), 1);
        assert_eq!(config.built_in_tools[0].function.name, "test_function");
    }

    #[test]
    fn test_groq_builder_headers() {
        let builder = GroqBuilder::new()
            .header("X-Custom-Header", "custom-value")
            .header("X-Another-Header", "another-value");

        let config = builder.config();
        assert_eq!(
            config.http_config.headers.get("X-Custom-Header"),
            Some(&"custom-value".to_string())
        );
        assert_eq!(
            config.http_config.headers.get("X-Another-Header"),
            Some(&"another-value".to_string())
        );
    }
}
