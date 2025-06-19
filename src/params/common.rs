//! Common Parameter Utilities
//!
//! Contains common utilities and helper functions for parameter processing.

use crate::error::LlmError;
use crate::types::{CommonParams, ProviderParams, ProviderType};

/// Type-safe extensions for provider-specific parameters
pub trait ProviderParamsExt {
    /// Gets the provider type
    fn provider_type(&self) -> ProviderType;
}

/// Common parameter validation utilities
pub struct ParameterValidator;

impl ParameterValidator {
    /// Validates temperature parameter
    pub fn validate_temperature(temp: f64, min: f64, max: f64, provider: &str) -> Result<(), LlmError> {
        if temp < min || temp > max {
            return Err(LlmError::InvalidParameter(
                format!("temperature must be between {} and {} for {}", min, max, provider),
            ));
        }
        Ok(())
    }

    /// Validates top_p parameter
    pub fn validate_top_p(top_p: f64) -> Result<(), LlmError> {
        if top_p < 0.0 || top_p > 1.0 {
            return Err(LlmError::InvalidParameter(
                "top_p must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }

    /// Validates max_tokens parameter
    pub fn validate_max_tokens(max_tokens: u64, min: u64, max: u64, provider: &str) -> Result<(), LlmError> {
        if max_tokens < min || max_tokens > max {
            return Err(LlmError::InvalidParameter(
                format!("max_tokens must be between {} and {} for {}", min, max, provider),
            ));
        }
        Ok(())
    }

    /// Validates a generic numeric parameter
    pub fn validate_numeric_range<T: PartialOrd + std::fmt::Display>(
        value: T,
        min: T,
        max: T,
        param_name: &str,
        provider: &str,
    ) -> Result<(), LlmError> {
        if value < min || value > max {
            return Err(LlmError::InvalidParameter(
                format!("{} must be between {} and {} for {}", param_name, min, max, provider),
            ));
        }
        Ok(())
    }
}

/// Common parameter mapping utilities
pub struct ParameterMapper;

impl ParameterMapper {
    /// Maps common parameters to a base JSON structure
    pub fn map_common_to_json(params: &CommonParams) -> serde_json::Value {
        let mut json = serde_json::json!({
            "model": params.model
        });

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

        json
    }

    /// Merges provider-specific parameters into base JSON
    pub fn merge_provider_params(
        mut base: serde_json::Value,
        provider: &ProviderParams,
    ) -> serde_json::Value {
        if let serde_json::Value::Object(ref mut base_obj) = base {
            for (key, value) in &provider.params {
                base_obj.insert(key.clone(), value.clone());
            }
        }
        base
    }

    /// Converts stop sequences to the appropriate format for a provider
    pub fn map_stop_sequences(
        stop_sequences: &Option<Vec<String>>,
        field_name: &str,
    ) -> Option<(String, serde_json::Value)> {
        stop_sequences.as_ref().map(|stop| {
            (field_name.to_string(), stop.clone().into())
        })
    }
}

/// Parameter conversion utilities
pub struct ParameterConverter;

impl ParameterConverter {
    /// Converts a parameter name from common format to provider-specific format
    pub fn convert_param_name(common_name: &str, provider_type: &ProviderType) -> String {
        match (common_name, provider_type) {
            ("max_tokens", ProviderType::Gemini) => "maxOutputTokens".to_string(),
            ("top_p", ProviderType::Gemini) => "topP".to_string(),
            ("stop_sequences", ProviderType::Gemini) => "stopSequences".to_string(),
            ("stop_sequences", ProviderType::Anthropic) => "stop_sequences".to_string(),
            ("stop_sequences", ProviderType::OpenAi) => "stop".to_string(),
            _ => common_name.to_string(),
        }
    }

    /// Converts parameter value based on provider requirements
    pub fn convert_param_value(
        value: &serde_json::Value,
        param_name: &str,
        provider_type: &ProviderType,
    ) -> serde_json::Value {
        match (param_name, provider_type) {
            // Add any provider-specific value conversions here
            _ => value.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_validator() {
        // Test temperature validation
        assert!(ParameterValidator::validate_temperature(0.7, 0.0, 2.0, "test").is_ok());
        assert!(ParameterValidator::validate_temperature(3.0, 0.0, 2.0, "test").is_err());

        // Test top_p validation
        assert!(ParameterValidator::validate_top_p(0.9).is_ok());
        assert!(ParameterValidator::validate_top_p(1.5).is_err());

        // Test max_tokens validation
        assert!(ParameterValidator::validate_max_tokens(1000, 1, 200000, "test").is_ok());
        assert!(ParameterValidator::validate_max_tokens(0, 1, 200000, "test").is_err());
    }

    #[test]
    fn test_parameter_converter() {
        // Test parameter name conversion
        assert_eq!(
            ParameterConverter::convert_param_name("max_tokens", &ProviderType::Gemini),
            "maxOutputTokens"
        );
        assert_eq!(
            ParameterConverter::convert_param_name("max_tokens", &ProviderType::OpenAi),
            "max_tokens"
        );

        // Test stop sequences conversion
        assert_eq!(
            ParameterConverter::convert_param_name("stop_sequences", &ProviderType::OpenAi),
            "stop"
        );
        assert_eq!(
            ParameterConverter::convert_param_name("stop_sequences", &ProviderType::Anthropic),
            "stop_sequences"
        );
    }

    #[test]
    fn test_common_parameter_mapping() {
        let params = CommonParams {
            model: "test-model".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            top_p: Some(0.9),
            stop_sequences: None,
            seed: Some(42),
        };

        let json = ParameterMapper::map_common_to_json(&params);
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["max_tokens"], 1000);
        assert_eq!(json["seed"], 42);
    }
}
