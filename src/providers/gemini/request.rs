//! Gemini Request Building Module
//!
//! Provides Gemini-specific request building logic with consistent
//! parameter handling and validation.

use crate::error::LlmError;
use crate::params::gemini::GeminiParams;
use crate::request_factory::{RequestBuilder, RequestBuilderConfig};
use crate::types::{ChatMessage, ChatRequest, CommonParams, ProviderParams, Tool};

/// Gemini parameter mapping trait
///
/// Handles parameter mapping and validation specific to Gemini
pub trait GeminiParameterMapper {
    /// Map common parameters to Gemini format
    fn map_common_to_gemini(&self, params: &CommonParams) -> serde_json::Value;

    /// Merge Gemini-specific parameters
    fn merge_gemini_params(
        &self,
        base: serde_json::Value,
        gemini_params: &GeminiParams,
    ) -> serde_json::Value;

    /// Validate Gemini parameters
    fn validate_gemini_params(&self, params: &serde_json::Value) -> Result<(), LlmError>;
}

/// Gemini-specific request builder
///
/// Handles the construction of ChatRequest objects with proper
/// Gemini parameter mapping and validation.
pub struct GeminiRequestBuilder {
    /// Common parameters shared across providers
    common_params: CommonParams,
    /// Gemini-specific parameters
    gemini_params: GeminiParams,
}

impl GeminiParameterMapper for GeminiRequestBuilder {
    fn map_common_to_gemini(&self, params: &CommonParams) -> serde_json::Value {
        let mut json = serde_json::json!({
            "model": params.model
        });

        // Map common parameters to Gemini format (different parameter names)
        if let Some(temp) = params.temperature {
            json["temperature"] = temp.into();
        }

        if let Some(max_tokens) = params.max_tokens {
            // Gemini uses "maxOutputTokens" instead of "max_tokens"
            json["maxOutputTokens"] = max_tokens.into();
        }

        if let Some(top_p) = params.top_p {
            // Gemini uses "topP" instead of "top_p"
            json["topP"] = top_p.into();
        }

        // Handle Gemini-specific stop sequences format
        if let Some(stop) = &params.stop_sequences {
            // Gemini uses "stopSequences" instead of "stop_sequences"
            json["stopSequences"] = stop.clone().into();
        }

        // Gemini doesn't support seed parameter - don't include it

        json
    }

    fn merge_gemini_params(
        &self,
        mut base: serde_json::Value,
        gemini_params: &GeminiParams,
    ) -> serde_json::Value {
        // Serialize Gemini params and merge
        if let Ok(gemini_json) = serde_json::to_value(gemini_params)
            && let Some(gemini_obj) = gemini_json.as_object()
            && let Some(base_obj) = base.as_object_mut()
        {
            for (key, value) in gemini_obj {
                if !value.is_null() {
                    base_obj.insert(key.clone(), value.clone());
                }
            }
        }
        base
    }

    fn validate_gemini_params(&self, params: &serde_json::Value) -> Result<(), LlmError> {
        self.validate_gemini_params_with_config(params, &RequestBuilderConfig::default())
    }
}

impl GeminiRequestBuilder {
    /// Validate Gemini parameters with configurable validation
    fn validate_gemini_params_with_config(
        &self,
        params: &serde_json::Value,
        config: &RequestBuilderConfig,
    ) -> Result<(), LlmError> {
        // Skip validation if provider validation is disabled
        if !config.provider_validation {
            return Ok(());
        }

        // Validate temperature range
        if let Some(temp) = params.get("temperature").and_then(|v| v.as_f64())
            && !(0.0..=2.0).contains(&temp)
        {
            return Err(LlmError::InvalidParameter(
                "Gemini temperature must be between 0.0 and 2.0 per official API spec (validation can be disabled)".to_string(),
            ));
        }

        // Validate topP range
        if let Some(top_p) = params.get("topP").and_then(|v| v.as_f64())
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err(LlmError::InvalidParameter(
                "Gemini topP must be between 0.0 and 1.0 per official API spec (validation can be disabled)".to_string(),
            ));
        }

        // Validate maxOutputTokens
        if let Some(max_tokens) = params.get("maxOutputTokens").and_then(|v| v.as_i64()) {
            if max_tokens <= 0 {
                return Err(LlmError::InvalidParameter(
                    "Gemini maxOutputTokens must be positive per official API spec (validation can be disabled)".to_string(),
                ));
            }
            if max_tokens > 8192 {
                return Err(LlmError::InvalidParameter(
                    "Gemini maxOutputTokens cannot exceed 8192 per official API spec (validation can be disabled)".to_string(),
                ));
            }
        }

        // Validate thinking budget if present
        if let Some(thinking_budget) = params.get("thinking_budget").and_then(|v| v.as_i64()) {
            if thinking_budget < 1024 {
                return Err(LlmError::InvalidParameter(
                    "Gemini thinking budget must be at least 1024 tokens per official API spec (validation can be disabled)".to_string(),
                ));
            }
            if thinking_budget > 32768 {
                return Err(LlmError::InvalidParameter(
                    "Gemini thinking budget cannot exceed 32768 tokens per official API spec (validation can be disabled)".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Create a new Gemini request builder
    ///
    /// # Arguments
    /// * `common_params` - Common parameters for the request
    /// * `gemini_params` - Gemini-specific parameters
    pub fn new(common_params: CommonParams, gemini_params: GeminiParams) -> Self {
        Self {
            common_params,
            gemini_params,
        }
    }

    /// Create provider params from Gemini params
    fn create_provider_params(&self) -> ProviderParams {
        ProviderParams::from_gemini(self.gemini_params.clone())
    }

    /// Validate Gemini-specific requirements
    fn validate_gemini_request(&self, request: &ChatRequest) -> Result<(), LlmError> {
        // Gemini-specific validation
        let model = &request.common_params.model;
        if model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model name is required for Gemini".to_string(),
            ));
        }

        // Validate model name format for Gemini
        if !model.starts_with("gemini-") {
            return Err(LlmError::InvalidParameter(
                "Gemini model names should start with 'gemini-'".to_string(),
            ));
        }

        Ok(())
    }
}

impl RequestBuilder for GeminiRequestBuilder {
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
        let mut params_json = self.map_common_to_gemini(&self.common_params);
        params_json = self.merge_gemini_params(params_json, &self.gemini_params);

        // Validate the mapped parameters with configuration
        self.validate_gemini_params_with_config(&params_json, config)?;

        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: Some(self.create_provider_params()),
            http_config: None,
            web_search: None,
            stream,
        };

        // Validate the request
        self.validate_request(&request)?;
        self.validate_gemini_request(&request)?;

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

        // Gemini-specific validation
        self.validate_gemini_request(request)?;

        Ok(())
    }
}

/// Helper function to create a Gemini request builder from capability components
pub fn create_gemini_request_builder(
    common_params: CommonParams,
    gemini_params: GeminiParams,
) -> GeminiRequestBuilder {
    GeminiRequestBuilder::new(common_params, gemini_params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MessageContent, MessageRole};

    #[test]
    fn test_gemini_parameter_mapping() {
        let common_params = CommonParams {
            model: "gemini-1.5-pro".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            top_p: Some(0.9),
            stop_sequences: Some(vec!["STOP".to_string()]),
            seed: Some(42), // Should be ignored by Gemini
        };

        let gemini_params = GeminiParams::default();
        let builder = GeminiRequestBuilder::new(common_params.clone(), gemini_params);

        let mapped = builder.map_common_to_gemini(&common_params);

        // Verify Gemini-specific parameter name mappings
        assert_eq!(mapped["model"], "gemini-1.5-pro");
        assert!((mapped["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
        assert_eq!(mapped["maxOutputTokens"], 1000); // Different name
        assert!((mapped["topP"].as_f64().unwrap() - 0.9).abs() < 0.001); // Different name
        assert_eq!(mapped["stopSequences"], serde_json::json!(["STOP"])); // Different name

        // Gemini doesn't support seed
        assert!(mapped.get("seed").is_none());
    }

    #[test]
    fn test_gemini_validation() {
        let common_params = CommonParams {
            model: "gemini-1.5-pro".to_string(),
            ..Default::default()
        };

        let gemini_params = GeminiParams::default();
        let _builder = GeminiRequestBuilder::new(common_params, gemini_params);

        // Test invalid model name
        let invalid_model_params = CommonParams {
            model: "gpt-4".to_string(), // Invalid for Gemini
            ..Default::default()
        };

        let messages = vec![crate::types::ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: Default::default(),
            tool_calls: None,
            tool_call_id: None,
        }];

        let invalid_builder =
            GeminiRequestBuilder::new(invalid_model_params, GeminiParams::default());
        let result = invalid_builder.build_chat_request(messages, None, false);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("should start with 'gemini-'")
        );
    }
}
