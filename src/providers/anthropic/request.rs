//! Anthropic Request Building Module
//!
//! Provides Anthropic-specific request building logic with consistent
//! parameter handling and validation.

use crate::error::LlmError;
use crate::params::anthropic::AnthropicParams;
use crate::request_factory::{RequestBuilder, RequestBuilderConfig};
use crate::types::{ChatMessage, ChatRequest, CommonParams, ProviderParams, Tool};

/// Anthropic parameter mapping trait
///
/// Handles parameter mapping and validation specific to Anthropic
pub trait AnthropicParameterMapper {
    /// Map common parameters to Anthropic format
    fn map_common_to_anthropic(&self, params: &CommonParams) -> serde_json::Value;

    /// Merge Anthropic-specific parameters
    fn merge_anthropic_params(
        &self,
        base: serde_json::Value,
        anthropic_params: &AnthropicParams,
    ) -> serde_json::Value;

    /// Validate Anthropic parameters
    fn validate_anthropic_params(&self, params: &serde_json::Value) -> Result<(), LlmError>;
}

/// Anthropic-specific request builder
///
/// Handles the construction of ChatRequest objects with proper
/// Anthropic parameter mapping and validation.
#[derive(Clone)]
pub struct AnthropicRequestBuilder {
    /// Common parameters shared across providers
    common_params: CommonParams,
    /// Anthropic-specific parameters
    anthropic_params: AnthropicParams,
}

impl AnthropicParameterMapper for AnthropicRequestBuilder {
    fn map_common_to_anthropic(&self, params: &CommonParams) -> serde_json::Value {
        let mut json = serde_json::json!({
            "model": params.model
        });

        // Map common parameters to Anthropic format
        if let Some(temp) = params.temperature {
            json["temperature"] = temp.into();
        }

        if let Some(max_tokens) = params.max_tokens {
            json["max_tokens"] = max_tokens.into();
        } else {
            // Anthropic requires max_tokens - set default if not provided
            json["max_tokens"] = 4096.into();
        }

        if let Some(top_p) = params.top_p {
            json["top_p"] = top_p.into();
        }

        // Handle Anthropic-specific stop sequences format
        if let Some(stop) = &params.stop_sequences {
            json["stop_sequences"] = stop.clone().into();
        }

        // Anthropic doesn't support seed parameter - don't include it

        json
    }

    fn merge_anthropic_params(
        &self,
        mut base: serde_json::Value,
        anthropic_params: &AnthropicParams,
    ) -> serde_json::Value {
        // Serialize Anthropic params and merge
        if let Ok(anthropic_json) = serde_json::to_value(anthropic_params)
            && let Some(anthropic_obj) = anthropic_json.as_object()
            && let Some(base_obj) = base.as_object_mut()
        {
            for (key, value) in anthropic_obj {
                if !value.is_null() {
                    base_obj.insert(key.clone(), value.clone());
                }
            }
        }
        base
    }

    fn validate_anthropic_params(&self, params: &serde_json::Value) -> Result<(), LlmError> {
        self.validate_anthropic_params_with_config(params, &RequestBuilderConfig::default())
    }
}

impl AnthropicRequestBuilder {
    /// Create a new Anthropic request builder
    ///
    /// # Arguments
    /// * `common_params` - Common parameters for the request
    /// * `anthropic_params` - Anthropic-specific parameters
    pub fn new(common_params: CommonParams, anthropic_params: AnthropicParams) -> Self {
        Self {
            common_params,
            anthropic_params,
        }
    }

    /// Create provider params from Anthropic params
    fn create_provider_params(&self) -> ProviderParams {
        ProviderParams::from_anthropic(self.anthropic_params.clone())
    }

    /// Validate Anthropic parameters with configuration based on official API spec
    fn validate_anthropic_params_with_config(
        &self,
        params: &serde_json::Value,
        config: &RequestBuilderConfig,
    ) -> Result<(), LlmError> {
        if !config.provider_validation {
            return Ok(()); // Skip validation if disabled
        }

        // Validate temperature range (Official spec: 0 <= x <= 1, default: 1)
        if let Some(temp) = params.get("temperature").and_then(|v| v.as_f64())
            && !(0.0..=1.0).contains(&temp)
        {
            return Err(LlmError::InvalidParameter(
                "Anthropic temperature must be between 0.0 and 1.0 per official API spec (validation can be disabled)".to_string(),
            ));
        }

        // Validate top_p range (Official spec: nucleus sampling, typically 0-1)
        if let Some(top_p) = params.get("top_p").and_then(|v| v.as_f64())
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err(LlmError::InvalidParameter(
                "Anthropic top_p must be between 0.0 and 1.0 per official API spec (validation can be disabled)".to_string(),
            ));
        }

        // Validate max_tokens (Official spec: required, x >= 1)
        if config.strict_validation
            && let Some(max_tokens) = params.get("max_tokens").and_then(|v| v.as_i64())
            && max_tokens <= 0
        {
            return Err(LlmError::InvalidParameter(
                "Anthropic max_tokens must be positive per official API spec (strict validation enabled)".to_string(),
            ));
        }

        // Validate thinking budget if present (Official spec: x >= 1024)
        if let Some(thinking_budget) = params.get("thinking_budget").and_then(|v| v.as_i64())
            && thinking_budget < 1024
        {
            return Err(LlmError::InvalidParameter(
                "Anthropic thinking budget must be at least 1024 tokens per official API spec (validation can be disabled)".to_string(),
            ));
            // Note: Upper limit may vary by model and change over time
            // We avoid hard-coding specific upper limits here
        }

        // Validate stop_sequences format and limits
        if let Some(stop_sequences) = params.get("stop_sequences")
            && let Some(stop_array) = stop_sequences.as_array()
        {
            // Anthropic typically supports multiple stop sequences, but exact limits may vary
            if config.strict_validation && stop_array.len() > 10 {
                return Err(LlmError::InvalidParameter(
                    "Anthropic stop_sequences should be reasonable in number (strict validation enabled)".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Validate Anthropic-specific requirements
    fn validate_anthropic_request(&self, request: &ChatRequest) -> Result<(), LlmError> {
        // Anthropic-specific validation
        let model = &request.common_params.model;
        if model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model name is required for Anthropic".to_string(),
            ));
        }

        // Validate model name format for Anthropic
        if !model.starts_with("claude-") {
            return Err(LlmError::InvalidParameter(
                "Anthropic model names should start with 'claude-'".to_string(),
            ));
        }

        // Check max_tokens requirement for Anthropic
        if request.common_params.max_tokens.is_none() {
            // Anthropic requires max_tokens, but we'll set a default in the parameter mapper
            // This is just a warning validation
        }

        // Validate thinking budget if present
        if let Some(thinking_budget) = self.anthropic_params.thinking_budget {
            if thinking_budget < 1024 {
                return Err(LlmError::InvalidParameter(
                    "Anthropic thinking budget must be at least 1024 tokens".to_string(),
                ));
            }
            if thinking_budget > 60000 {
                return Err(LlmError::InvalidParameter(
                    "Anthropic thinking budget cannot exceed 60000 tokens".to_string(),
                ));
            }
        }

        // Validate temperature range for Anthropic (stricter than OpenAI)
        if let Some(temp) = request.common_params.temperature
            && !(0.0..=1.0).contains(&temp)
        {
            return Err(LlmError::InvalidParameter(
                "Anthropic temperature must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }
}

impl RequestBuilder for AnthropicRequestBuilder {
    fn build_chat_request(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        self.build_chat_request_with_config(
            messages,
            tools,
            stream,
            &RequestBuilderConfig::default(),
        )
    }

    fn build_chat_request_with_config(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        stream: bool,
        config: &RequestBuilderConfig,
    ) -> Result<ChatRequest, LlmError> {
        // Create base parameters using the new mapping system
        let mut params_json = self.map_common_to_anthropic(&self.common_params);
        params_json = self.merge_anthropic_params(params_json, &self.anthropic_params);

        // Validate the mapped parameters with configuration
        self.validate_anthropic_params_with_config(&params_json, config)?;

        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: Some(self.create_provider_params()),
            http_config: None,
            web_search: None,
            stream,
        };

        // Validate the request (basic validation always enabled)
        if config.strict_validation {
            self.validate_request(&request)?;
            self.validate_anthropic_request(&request)?;
        }

        Ok(request)
    }

    fn get_common_params(&self) -> &CommonParams {
        &self.common_params
    }

    fn get_provider_params(&self) -> Option<ProviderParams> {
        Some(self.create_provider_params())
    }

    fn validate_request(&self, request: &ChatRequest) -> Result<(), LlmError> {
        // Standard validation
        if request.messages.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Messages cannot be empty".to_string(),
            ));
        }

        if request.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        // Anthropic-specific validation
        self.validate_anthropic_request(request)?;

        Ok(())
    }
}

/// Helper function to create an Anthropic request builder from capability components
pub fn create_anthropic_request_builder(
    common_params: CommonParams,
    anthropic_params: AnthropicParams,
) -> AnthropicRequestBuilder {
    AnthropicRequestBuilder::new(common_params, anthropic_params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MessageContent, MessageRole};

    #[test]
    fn test_anthropic_request_builder() {
        let common_params = CommonParams {
            model: "claude-3-5-sonnet-20241022".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            ..Default::default()
        };

        let anthropic_params = AnthropicParams::default();
        let builder = AnthropicRequestBuilder::new(common_params, anthropic_params);

        let messages = vec![crate::types::ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: Default::default(),
            tool_calls: None,
            tool_call_id: None,
        }];

        let request = builder
            .build_chat_request(messages, None, false)
            .expect("Should build request successfully");

        assert_eq!(request.common_params.model, "claude-3-5-sonnet-20241022");
        assert!(!request.stream);
        assert!(request.provider_params.is_some());
    }

    #[test]
    fn test_invalid_model_name() {
        let common_params = CommonParams {
            model: "gpt-4".to_string(), // Invalid for Anthropic
            ..Default::default()
        };

        let anthropic_params = AnthropicParams::default();
        let builder = AnthropicRequestBuilder::new(common_params, anthropic_params);

        let messages = vec![crate::types::ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: Default::default(),
            tool_calls: None,
            tool_call_id: None,
        }];

        let config = RequestBuilderConfig {
            strict_validation: true,
            provider_validation: true,
        };
        let result = builder.build_chat_request_with_config(messages, None, false, &config);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("should start with 'claude-'")
        );
    }

    #[test]
    fn test_thinking_budget_validation() {
        let common_params = CommonParams {
            model: "claude-3-5-sonnet-20241022".to_string(),
            ..Default::default()
        };

        let anthropic_params = AnthropicParams {
            thinking_budget: Some(500), // Too low
            ..Default::default()
        };

        let builder = AnthropicRequestBuilder::new(common_params, anthropic_params);

        let messages = vec![crate::types::ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: Default::default(),
            tool_calls: None,
            tool_call_id: None,
        }];

        let config = RequestBuilderConfig {
            strict_validation: true,
            provider_validation: true,
        };
        let result = builder.build_chat_request_with_config(messages, None, false, &config);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("at least 1024 tokens")
        );
    }

    #[test]
    fn test_temperature_validation() {
        let common_params = CommonParams {
            model: "claude-3-5-sonnet-20241022".to_string(),
            temperature: Some(1.5), // Too high for Anthropic
            ..Default::default()
        };

        let anthropic_params = AnthropicParams::default();
        let builder = AnthropicRequestBuilder::new(common_params, anthropic_params);

        let messages = vec![crate::types::ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: Default::default(),
            tool_calls: None,
            tool_call_id: None,
        }];

        let config = RequestBuilderConfig {
            strict_validation: true,
            provider_validation: true,
        };
        let result = builder.build_chat_request_with_config(messages, None, false, &config);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("between 0.0 and 1.0")
        );
    }
}
