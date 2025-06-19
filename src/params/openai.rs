//! OpenAI Parameter Mapping
//!
//! Contains OpenAI-specific parameter mapping and validation logic.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::types::{CommonParams, ProviderParams, ProviderType};
use super::common::{ParameterValidator, ParameterMapper as CommonMapper};
use super::mapper::{ParameterMapper, ParameterConstraints};

/// OpenAI Parameter Mapper
pub struct OpenAiParameterMapper;

impl ParameterMapper for OpenAiParameterMapper {
    fn map_common_params(&self, params: &CommonParams) -> serde_json::Value {
        let mut json = CommonMapper::map_common_to_json(params);

        // Handle OpenAI-specific stop sequences format
        if let Some(stop) = &params.stop_sequences {
            json["stop"] = stop.clone().into();
        }

        json
    }

    fn merge_provider_params(
        &self,
        base: serde_json::Value,
        provider: &ProviderParams,
    ) -> serde_json::Value {
        CommonMapper::merge_provider_params(base, provider)
    }

    fn validate_params(&self, params: &serde_json::Value) -> Result<(), LlmError> {
        // Validate OpenAI-specific parameter constraints
        if let Some(temp) = params.get("temperature") {
            if let Some(temp_val) = temp.as_f64() {
                ParameterValidator::validate_temperature(temp_val, 0.0, 2.0, "OpenAI")?;
            }
        }

        if let Some(top_p) = params.get("top_p") {
            if let Some(top_p_val) = top_p.as_f64() {
                ParameterValidator::validate_top_p(top_p_val)?;
            }
        }

        if let Some(max_tokens) = params.get("max_tokens") {
            if let Some(max_tokens_val) = max_tokens.as_u64() {
                ParameterValidator::validate_max_tokens(max_tokens_val, 1, 128000, "OpenAI")?;
            }
        }

        // Validate OpenAI-specific parameters
        if let Some(frequency_penalty) = params.get("frequency_penalty") {
            if let Some(penalty_val) = frequency_penalty.as_f64() {
                ParameterValidator::validate_numeric_range(
                    penalty_val, -2.0, 2.0, "frequency_penalty", "OpenAI"
                )?;
            }
        }

        if let Some(presence_penalty) = params.get("presence_penalty") {
            if let Some(penalty_val) = presence_penalty.as_f64() {
                ParameterValidator::validate_numeric_range(
                    penalty_val, -2.0, 2.0, "presence_penalty", "OpenAI"
                )?;
            }
        }

        if let Some(n) = params.get("n") {
            if let Some(n_val) = n.as_u64() {
                if n_val == 0 || n_val > 128 {
                    return Err(LlmError::InvalidParameter(
                        "n must be between 1 and 128 for OpenAI".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAi
    }

    fn supported_params(&self) -> Vec<&'static str> {
        vec![
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "user",
            "n",
            "stream",
            "response_format",
            "tool_choice",
            "tools",
            "parallel_tool_calls",
        ]
    }

    fn get_param_constraints(&self) -> ParameterConstraints {
        ParameterConstraints {
            temperature_min: 0.0,
            temperature_max: 2.0,
            max_tokens_min: 1,
            max_tokens_max: 128000,
            top_p_min: 0.0,
            top_p_max: 1.0,
        }
    }
}

/// OpenAI-specific parameter extensions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAiParams {
    /// Response format
    pub response_format: Option<ResponseFormat>,
    /// Tool choice strategy
    pub tool_choice: Option<ToolChoice>,
    /// Parallel tool calls
    pub parallel_tool_calls: Option<bool>,
    /// User ID
    pub user: Option<String>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// Logit bias
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Number of choices to return
    pub n: Option<u32>,
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Logprobs configuration
    pub logprobs: Option<bool>,
    /// Top logprobs to return
    pub top_logprobs: Option<u32>,
}

impl super::common::ProviderParamsExt for OpenAiParams {
    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAi
    }
}

/// OpenAI Response Format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { schema: serde_json::Value },
}

/// OpenAI Tool Choice
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String), // "none", "auto", "required"
    Function { 
        #[serde(rename = "type")]
        choice_type: String, // "function"
        function: FunctionChoice 
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionChoice {
    pub name: String,
}

/// OpenAI parameter builder for convenient parameter construction
pub struct OpenAiParamsBuilder {
    params: OpenAiParams,
}

impl OpenAiParamsBuilder {
    pub fn new() -> Self {
        Self {
            params: OpenAiParams::default(),
        }
    }

    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.params.response_format = Some(format);
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.params.tool_choice = Some(choice);
        self
    }

    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.params.parallel_tool_calls = Some(enabled);
        self
    }

    pub fn user(mut self, user_id: String) -> Self {
        self.params.user = Some(user_id);
        self
    }

    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.params.frequency_penalty = Some(penalty);
        self
    }

    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.params.presence_penalty = Some(penalty);
        self
    }

    pub fn n(mut self, choices: u32) -> Self {
        self.params.n = Some(choices);
        self
    }

    pub fn stream(mut self, enabled: bool) -> Self {
        self.params.stream = Some(enabled);
        self
    }

    pub fn logprobs(mut self, enabled: bool, top_logprobs: Option<u32>) -> Self {
        self.params.logprobs = Some(enabled);
        self.params.top_logprobs = top_logprobs;
        self
    }

    pub fn build(self) -> OpenAiParams {
        self.params
    }
}

impl Default for OpenAiParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_parameter_mapping() {
        let mapper = OpenAiParameterMapper;
        let params = CommonParams {
            model: "gpt-4".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            top_p: Some(0.9),
            stop_sequences: Some(vec!["STOP".to_string()]),
            seed: Some(42),
        };

        let mapped = mapper.map_common_params(&params);
        assert_eq!(mapped["model"], "gpt-4");
        assert_eq!(mapped["max_tokens"], 1000);
        assert_eq!(mapped["seed"], 42);
        assert_eq!(mapped["stop"], serde_json::json!(["STOP"]));
    }

    #[test]
    fn test_openai_parameter_validation() {
        let mapper = OpenAiParameterMapper;

        // Valid parameters
        let valid_params = serde_json::json!({
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1000,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
            "n": 1
        });
        assert!(mapper.validate_params(&valid_params).is_ok());

        // Invalid temperature
        let invalid_temp = serde_json::json!({
            "temperature": 3.0
        });
        assert!(mapper.validate_params(&invalid_temp).is_err());

        // Invalid frequency penalty
        let invalid_penalty = serde_json::json!({
            "frequency_penalty": 3.0
        });
        assert!(mapper.validate_params(&invalid_penalty).is_err());
    }

    #[test]
    fn test_openai_params_builder() {
        let params = OpenAiParamsBuilder::new()
            .response_format(ResponseFormat::JsonObject)
            .tool_choice(ToolChoice::String("auto".to_string()))
            .parallel_tool_calls(true)
            .user("test-user".to_string())
            .frequency_penalty(0.5)
            .presence_penalty(-0.2)
            .n(2)
            .stream(false)
            .logprobs(true, Some(5))
            .build();

        assert!(params.response_format.is_some());
        assert!(params.tool_choice.is_some());
        assert_eq!(params.parallel_tool_calls, Some(true));
        assert_eq!(params.user, Some("test-user".to_string()));
        assert_eq!(params.frequency_penalty, Some(0.5));
        assert_eq!(params.presence_penalty, Some(-0.2));
        assert_eq!(params.n, Some(2));
        assert_eq!(params.stream, Some(false));
        assert_eq!(params.logprobs, Some(true));
        assert_eq!(params.top_logprobs, Some(5));
    }
}
