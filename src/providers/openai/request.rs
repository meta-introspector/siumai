//! OpenAI Request Building Module
//!
//! Provides OpenAI-specific request building logic with consistent
//! parameter handling and validation.

use crate::error::LlmError;
use crate::params::openai::OpenAiParams;
use crate::request_factory::{RequestBuilder, RequestBuilderConfig};
use crate::types::{ChatMessage, ChatRequest, CommonParams, ProviderParams, Tool};

/// OpenAI parameter mapping trait
///
/// Handles parameter mapping and validation specific to OpenAI
pub trait OpenAiParameterMapper {
    /// Map common parameters to OpenAI format
    fn map_common_to_openai(&self, params: &CommonParams) -> serde_json::Value;

    /// Merge OpenAI-specific parameters
    fn merge_openai_params(
        &self,
        base: serde_json::Value,
        openai_params: &OpenAiParams,
    ) -> serde_json::Value;

    /// Validate OpenAI parameters
    fn validate_openai_params(&self, params: &serde_json::Value) -> Result<(), LlmError>;
}

/// OpenAI-specific request builder
///
/// Handles the construction of ChatRequest objects with proper
/// OpenAI parameter mapping and validation.
pub struct OpenAiRequestBuilder {
    /// Common parameters shared across providers
    common_params: CommonParams,
    /// OpenAI-specific parameters
    openai_params: OpenAiParams,
}

impl OpenAiParameterMapper for OpenAiRequestBuilder {
    fn map_common_to_openai(&self, params: &CommonParams) -> serde_json::Value {
        let mut json = serde_json::json!({
            "model": params.model
        });

        // Map common parameters to OpenAI format
        if let Some(temp) = params.temperature {
            json["temperature"] = temp.into();
        }

        if let Some(max_tokens) = params.max_tokens {
            json["max_tokens"] = max_tokens.into();
        }

        if let Some(top_p) = params.top_p {
            json["top_p"] = top_p.into();
        }

        if let Some(seed) = params.seed {
            json["seed"] = seed.into();
        }

        // Handle OpenAI-specific stop sequences format
        if let Some(stop) = &params.stop_sequences {
            json["stop"] = stop.clone().into();
        }

        json
    }

    fn merge_openai_params(
        &self,
        mut base: serde_json::Value,
        openai_params: &OpenAiParams,
    ) -> serde_json::Value {
        // Serialize OpenAI params and merge
        if let Ok(openai_json) = serde_json::to_value(openai_params)
            && let Some(openai_obj) = openai_json.as_object()
            && let Some(base_obj) = base.as_object_mut()
        {
            for (key, value) in openai_obj {
                if !value.is_null() {
                    base_obj.insert(key.clone(), value.clone());
                }
            }
        }
        base
    }

    fn validate_openai_params(&self, params: &serde_json::Value) -> Result<(), LlmError> {
        self.validate_openai_params_with_config(params, &RequestBuilderConfig::default())
    }
}

impl OpenAiRequestBuilder {
    /// Create a new OpenAI request builder
    ///
    /// # Arguments
    /// * `common_params` - Common parameters for the request
    /// * `openai_params` - OpenAI-specific parameters
    pub fn new(common_params: CommonParams, openai_params: OpenAiParams) -> Self {
        Self {
            common_params,
            openai_params,
        }
    }

    /// Create provider params from OpenAI params
    fn create_provider_params(&self) -> ProviderParams {
        ProviderParams::from_openai(self.openai_params.clone())
    }

    /// Validate OpenAI parameters with configuration based on official API spec
    fn validate_openai_params_with_config(
        &self,
        params: &serde_json::Value,
        config: &RequestBuilderConfig,
    ) -> Result<(), LlmError> {
        if !config.provider_validation {
            return Ok(()); // Skip validation if disabled
        }

        // Validate temperature range (Official spec: minimum: 0, maximum: 2, default: 1)
        if let Some(temp) = params.get("temperature").and_then(|v| v.as_f64())
            && !(0.0..=2.0).contains(&temp)
        {
            return Err(LlmError::InvalidParameter(
                "OpenAI temperature must be between 0.0 and 2.0 per official API spec (validation can be disabled)".to_string(),
            ));
        }

        // Validate top_p range (Official spec: minimum: 0, maximum: 1, default: 1)
        if let Some(top_p) = params.get("top_p").and_then(|v| v.as_f64())
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err(LlmError::InvalidParameter(
                "OpenAI top_p must be between 0.0 and 1.0 per official API spec (validation can be disabled)".to_string(),
            ));
        }

        // Validate frequency_penalty range (Official spec: minimum: -2, maximum: 2, default: 0)
        if let Some(freq_penalty) = params.get("frequency_penalty").and_then(|v| v.as_f64())
            && !(-2.0..=2.0).contains(&freq_penalty)
        {
            return Err(LlmError::InvalidParameter(
                "OpenAI frequency_penalty must be between -2.0 and 2.0 per official API spec (validation can be disabled)".to_string(),
            ));
        }

        // Validate presence_penalty range (Official spec: minimum: -2, maximum: 2, default: 0)
        if let Some(pres_penalty) = params.get("presence_penalty").and_then(|v| v.as_f64())
            && !(-2.0..=2.0).contains(&pres_penalty)
        {
            return Err(LlmError::InvalidParameter(
                "OpenAI presence_penalty must be between -2.0 and 2.0 per official API spec (validation can be disabled)".to_string(),
            ));
        }

        // Validate n parameter (Official spec: minimum: 1, maximum: 128, default: 1)
        if let Some(n) = params.get("n").and_then(|v| v.as_i64())
            && !(1..=128).contains(&n)
        {
            return Err(LlmError::InvalidParameter(
                "OpenAI n parameter must be between 1 and 128 per official API spec (validation can be disabled)".to_string(),
            ));
        }

        // Validate top_logprobs (Official spec: minimum: 0, maximum: 20)
        if let Some(top_logprobs) = params.get("top_logprobs").and_then(|v| v.as_i64())
            && !(0..=20).contains(&top_logprobs)
        {
            return Err(LlmError::InvalidParameter(
                "OpenAI top_logprobs must be between 0 and 20 per official API spec (validation can be disabled)".to_string(),
            ));
        }

        // Validate stop sequences (Official spec: up to 4 sequences)
        if let Some(stop) = params.get("stop")
            && let Some(stop_array) = stop.as_array()
            && stop_array.len() > 4
        {
            return Err(LlmError::InvalidParameter(
                "OpenAI stop parameter supports up to 4 sequences per official API spec (validation can be disabled)".to_string(),
            ));
        }

        // Note: seed parameter validation is not needed as it's already constrained by i64 type
        // Official spec range (-9223372036854776000 to 9223372036854776000) is essentially i64 range

        // Validate max_tokens (only if strict validation is enabled)
        // Note: max_tokens is deprecated in favor of max_completion_tokens
        if config.strict_validation {
            if let Some(max_tokens) = params.get("max_tokens").and_then(|v| v.as_i64())
                && max_tokens <= 0
            {
                return Err(LlmError::InvalidParameter(
                    "OpenAI max_tokens must be positive (strict validation enabled, parameter is deprecated)".to_string(),
                ));
            }

            if let Some(max_completion_tokens) =
                params.get("max_completion_tokens").and_then(|v| v.as_i64())
                && max_completion_tokens <= 0
            {
                return Err(LlmError::InvalidParameter(
                    "OpenAI max_completion_tokens must be positive (strict validation enabled)"
                        .to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Validate OpenAI-specific requirements
    fn validate_openai_request(&self, request: &ChatRequest) -> Result<(), LlmError> {
        // OpenAI-specific validation
        if let Some(ref tools) = request.tools
            && tools.len() > 128
        {
            return Err(LlmError::InvalidParameter(
                "OpenAI supports maximum 128 tools per request".to_string(),
            ));
        }

        // Validate model name format for OpenAI
        let model = &request.common_params.model;
        if model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model name is required for OpenAI".to_string(),
            ));
        }

        // Check for o1 model specific requirements
        if model.starts_with("o1-") {
            // o1 models have specific parameter restrictions
            if request.common_params.temperature.is_some() {
                return Err(LlmError::InvalidParameter(
                    "o1 models do not support temperature parameter".to_string(),
                ));
            }
            if request.common_params.top_p.is_some() {
                return Err(LlmError::InvalidParameter(
                    "o1 models do not support top_p parameter".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl RequestBuilder for OpenAiRequestBuilder {
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
        let mut params_json = self.map_common_to_openai(&self.common_params);
        params_json = self.merge_openai_params(params_json, &self.openai_params);

        // Validate the mapped parameters with configuration
        self.validate_openai_params_with_config(&params_json, config)?;

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
            self.validate_openai_request(&request)?;
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

        // OpenAI-specific validation
        self.validate_openai_request(request)?;

        Ok(())
    }
}

/// Helper function to create an OpenAI request builder from capability components
pub fn create_openai_request_builder(
    common_params: CommonParams,
    openai_params: OpenAiParams,
) -> OpenAiRequestBuilder {
    OpenAiRequestBuilder::new(common_params, openai_params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MessageContent, MessageRole};

    #[test]
    fn test_openai_request_builder() {
        let common_params = CommonParams {
            model: "gpt-4".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            ..Default::default()
        };

        let openai_params = OpenAiParams::default();
        let builder = OpenAiRequestBuilder::new(common_params, openai_params);

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

        assert_eq!(request.common_params.model, "gpt-4");
        assert!(!request.stream);
        assert!(request.provider_params.is_some());
    }

    #[test]
    fn test_o1_model_validation() {
        let common_params = CommonParams {
            model: "o1-preview".to_string(),
            temperature: Some(0.7), // This should cause validation error
            ..Default::default()
        };

        let openai_params = OpenAiParams::default();
        let builder = OpenAiRequestBuilder::new(common_params, openai_params);

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
                .contains("o1 models do not support temperature")
        );
    }

    #[test]
    fn test_too_many_tools_validation() {
        let common_params = CommonParams {
            model: "gpt-4".to_string(),
            ..Default::default()
        };

        let openai_params = OpenAiParams::default();
        let builder = OpenAiRequestBuilder::new(common_params, openai_params);

        let messages = vec![crate::types::ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: Default::default(),
            tool_calls: None,
            tool_call_id: None,
        }];

        // Create 129 tools (exceeds OpenAI limit of 128)
        let tools: Vec<Tool> = (0..129)
            .map(|i| Tool {
                r#type: "function".to_string(),
                function: crate::types::ToolFunction {
                    name: format!("tool_{}", i),
                    description: "Test tool".to_string(),
                    parameters: serde_json::json!({}),
                },
            })
            .collect();

        let config = RequestBuilderConfig {
            strict_validation: true,
            provider_validation: true,
        };
        let result = builder.build_chat_request_with_config(messages, Some(tools), false, &config);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("maximum 128 tools")
        );
    }
}
