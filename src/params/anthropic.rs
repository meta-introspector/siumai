//! Anthropic Parameter Mapping
//!
//! Contains Anthropic-specific parameter mapping and validation logic.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use validator::Validate;

use super::common::{ParameterMapper as CommonMapper, ParameterValidator};
use super::mapper::{ParameterConstraints, ParameterMapper};
use crate::error::LlmError;
use crate::types::{CommonParams, ProviderParams, ProviderType};

/// Anthropic Parameter Mapper
pub struct AnthropicParameterMapper;

impl ParameterMapper for AnthropicParameterMapper {
    fn map_common_params(&self, params: &CommonParams) -> serde_json::Value {
        let mut json = CommonMapper::map_common_to_json(params);

        // Anthropic API requires max_tokens parameter - set default if not provided
        if json.get("max_tokens").is_none() {
            json["max_tokens"] = 4096.into();
        }

        // Handle Anthropic-specific stop sequences format
        if let Some(stop) = &params.stop_sequences {
            json["stop_sequences"] = stop.clone().into();
        }

        // Remove seed as Anthropic doesn't support it
        json.as_object_mut().unwrap().remove("seed");

        json
    }

    fn merge_provider_params(
        &self,
        mut base: serde_json::Value,
        provider: &ProviderParams,
    ) -> serde_json::Value {
        if let serde_json::Value::Object(ref mut base_obj) = base {
            for (key, value) in &provider.params {
                // Anthropic-specific parameter handling
                match key.as_str() {
                    "cache_control" => {
                        // Handle cache control parameters
                        base_obj.insert(key.clone(), value.clone());
                    }
                    "thinking_budget" => {
                        // Handle thinking budget parameters
                        base_obj.insert(key.clone(), value.clone());
                    }
                    "system" => {
                        // Handle system message
                        base_obj.insert(key.clone(), value.clone());
                    }
                    "metadata" => {
                        // Handle metadata
                        base_obj.insert(key.clone(), value.clone());
                    }
                    _ => {
                        base_obj.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        base
    }

    fn validate_params(&self, params: &serde_json::Value) -> Result<(), LlmError> {
        // Validate Anthropic-specific parameter constraints
        if let Some(temp) = params.get("temperature") {
            if let Some(temp_val) = temp.as_f64() {
                ParameterValidator::validate_temperature(temp_val, 0.0, 1.0, "Anthropic")?;
            }
        }

        if let Some(top_p) = params.get("top_p") {
            if let Some(top_p_val) = top_p.as_f64() {
                ParameterValidator::validate_top_p(top_p_val)?;
            }
        }

        // Anthropic API requires max_tokens parameter
        if let Some(max_tokens) = params.get("max_tokens") {
            if let Some(max_tokens_val) = max_tokens.as_u64() {
                ParameterValidator::validate_max_tokens(max_tokens_val, 1, 200_000, "Anthropic")?;
            }
        } else {
            return Err(LlmError::InvalidParameter(
                "max_tokens is required for Anthropic API".to_string(),
            ));
        }

        // Validate Anthropic-specific parameters
        if let Some(thinking_budget) = params.get("thinking_budget") {
            if let Some(budget_val) = thinking_budget.as_u64() {
                if budget_val > 60_000 {
                    return Err(LlmError::InvalidParameter(
                        "thinking_budget must not exceed 60_000 for Anthropic".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }

    fn supported_params(&self) -> Vec<&'static str> {
        vec![
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stop_sequences",
            "system",
            "metadata",
            "stream",
            "cache_control",
            "thinking_budget",
        ]
    }

    fn get_param_constraints(&self) -> ParameterConstraints {
        ParameterConstraints {
            temperature_min: 0.0,
            temperature_max: 1.0,
            max_tokens_min: 1,
            max_tokens_max: 200_000,
            top_p_min: 0.0,
            top_p_max: 1.0,
        }
    }
}

/// Anthropic Cache Control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub r#type: String,
}

/// Anthropic-specific parameter extensions
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
pub struct AnthropicParams {
    /// Cache control
    pub cache_control: Option<CacheControl>,

    /// Thinking budget
    pub thinking_budget: Option<u32>,

    /// System message
    pub system: Option<String>,

    /// Metadata
    pub metadata: Option<HashMap<String, String>>,

    /// Whether to stream the response
    pub stream: Option<bool>,

    /// Beta features
    pub beta_features: Option<Vec<String>>,
}

impl super::common::ProviderParamsExt for AnthropicParams {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }
}

impl AnthropicParams {
    /// Validate Anthropic-specific parameters
    pub fn validate_params(&self) -> Result<(), LlmError> {
        use validator::Validate;
        self.validate()
            .map_err(|e| LlmError::InvalidParameter(e.to_string()))?;
        Ok(())
    }

    /// Create a builder for Anthropic parameters
    pub fn builder() -> AnthropicParamsBuilder {
        AnthropicParamsBuilder::new()
    }
}

/// Anthropic parameter builder for convenient parameter construction
pub struct AnthropicParamsBuilder {
    params: AnthropicParams,
}

impl AnthropicParamsBuilder {
    pub fn new() -> Self {
        Self {
            params: AnthropicParams::default(),
        }
    }

    pub fn cache_control(mut self, cache_control: CacheControl) -> Self {
        self.params.cache_control = Some(cache_control);
        self
    }

    pub const fn thinking_budget(mut self, budget: u32) -> Self {
        self.params.thinking_budget = Some(budget);
        self
    }

    pub fn system(mut self, system_message: String) -> Self {
        self.params.system = Some(system_message);
        self
    }

    pub fn metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.params.metadata = Some(metadata);
        self
    }

    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        if self.params.metadata.is_none() {
            self.params.metadata = Some(HashMap::new());
        }
        self.params.metadata.as_mut().unwrap().insert(key, value);
        self
    }

    pub const fn stream(mut self, enabled: bool) -> Self {
        self.params.stream = Some(enabled);
        self
    }

    pub fn beta_features(mut self, features: Vec<String>) -> Self {
        self.params.beta_features = Some(features);
        self
    }

    pub fn add_beta_feature(mut self, feature: String) -> Self {
        if self.params.beta_features.is_none() {
            self.params.beta_features = Some(Vec::new());
        }
        self.params.beta_features.as_mut().unwrap().push(feature);
        self
    }

    pub fn build(self) -> AnthropicParams {
        self.params
    }
}

impl Default for AnthropicParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheControl {
    pub fn ephemeral() -> Self {
        Self {
            r#type: "ephemeral".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_parameter_mapping() {
        let mapper = AnthropicParameterMapper;
        let params = CommonParams {
            model: "claude-3-5-sonnet-20241022".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            top_p: Some(0.9),
            stop_sequences: Some(vec!["STOP".to_string()]),
            seed: Some(42), // Should be removed for Anthropic
        };

        let mapped_params = mapper.map_common_params(&params);
        assert_eq!(mapped_params["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(mapped_params["max_tokens"], 1000);
        assert_eq!(mapped_params["stop_sequences"], serde_json::json!(["STOP"]));
        // Seed should not be present for Anthropic
        assert!(mapped_params.get("seed").is_none());
    }

    #[test]
    fn test_anthropic_parameter_mapping_with_default_max_tokens() {
        let mapper = AnthropicParameterMapper;
        let params = CommonParams {
            model: "claude-3-5-sonnet-20241022".to_string(),
            temperature: Some(0.7),
            max_tokens: None, // No max_tokens provided
            top_p: Some(0.9),
            stop_sequences: None,
            seed: None,
        };

        let mapped_params = mapper.map_common_params(&params);
        assert_eq!(mapped_params["model"], "claude-3-5-sonnet-20241022");
        // Should have default max_tokens
        assert_eq!(mapped_params["max_tokens"], 4096);
        // Use approximate comparison for floating point values
        assert!((mapped_params["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
        assert!((mapped_params["top_p"].as_f64().unwrap() - 0.9).abs() < 0.001);
        // Seed should not be present for Anthropic
        assert!(mapped_params.get("seed").is_none());
    }

    #[test]
    fn test_anthropic_parameter_validation() {
        let mapper = AnthropicParameterMapper;

        // Valid parameters
        let valid_params = serde_json::json!({
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1000,
            "thinking_budget": 30_000
        });
        assert!(mapper.validate_params(&valid_params).is_ok());

        // Invalid temperature (too high for Anthropic)
        let invalid_temp = serde_json::json!({
            "temperature": 1.5
        });
        assert!(mapper.validate_params(&invalid_temp).is_err());

        // Invalid thinking budget
        let invalid_budget = serde_json::json!({
            "thinking_budget": 70_000
        });
        assert!(mapper.validate_params(&invalid_budget).is_err());
    }

    #[test]
    fn test_anthropic_params_builder() {
        let mut metadata = HashMap::new();
        metadata.insert("user_id".to_string(), "12345".to_string());

        let params = AnthropicParamsBuilder::new()
            .cache_control(CacheControl::ephemeral())
            .thinking_budget(30_000)
            .system("You are a helpful assistant".to_string())
            .metadata(metadata)
            .add_metadata("session_id".to_string(), "abc123".to_string())
            .stream(false)
            .add_beta_feature("computer-use-2024-10-22".to_string())
            .add_beta_feature("prompt-caching-2024-07-31".to_string())
            .build();

        assert!(params.cache_control.is_some());
        assert_eq!(params.thinking_budget, Some(30_000));
        assert_eq!(
            params.system,
            Some("You are a helpful assistant".to_string())
        );
        assert!(params.metadata.is_some());
        assert_eq!(params.metadata.as_ref().unwrap().len(), 2);
        assert_eq!(params.stream, Some(false));
        assert!(params.beta_features.is_some());
        assert_eq!(params.beta_features.as_ref().unwrap().len(), 2);
    }
}
