//! Gemini Parameter Mapping
//!
//! Contains Gemini-specific parameter mapping and validation logic.

use serde::{Deserialize, Serialize};

use super::common::{ParameterConverter, ParameterValidator};
use super::mapper::{ParameterConstraints, ParameterMapper};
use crate::error::LlmError;
use crate::types::{CommonParams, ProviderParams, ProviderType};

/// Gemini Parameter Mapper
pub struct GeminiParameterMapper;

impl ParameterMapper for GeminiParameterMapper {
    fn map_common_params(&self, params: &CommonParams) -> serde_json::Value {
        let mut json = serde_json::json!({
            "model": params.model
        });

        // Gemini uses different parameter names
        if let Some(temp) = params.temperature {
            json["temperature"] = temp.into();
        }

        if let Some(max_tokens) = params.max_tokens {
            json["maxOutputTokens"] = max_tokens.into();
        }

        if let Some(top_p) = params.top_p {
            json["topP"] = top_p.into();
        }

        if let Some(stop) = &params.stop_sequences {
            json["stopSequences"] = stop.clone().into();
        }

        // Gemini doesn't support seed parameter
        // json["seed"] is not included

        json
    }

    fn merge_provider_params(
        &self,
        mut base: serde_json::Value,
        provider: &ProviderParams,
    ) -> serde_json::Value {
        if let serde_json::Value::Object(ref mut base_obj) = base {
            for (key, value) in &provider.params {
                // Convert parameter names to Gemini format if needed
                let gemini_key = ParameterConverter::convert_param_name(key, &ProviderType::Gemini);
                let gemini_value =
                    ParameterConverter::convert_param_value(value, key, &ProviderType::Gemini);
                base_obj.insert(gemini_key, gemini_value);
            }
        }
        base
    }

    fn validate_params(&self, params: &serde_json::Value) -> Result<(), LlmError> {
        // Validate Gemini-specific parameter constraints
        if let Some(temp) = params.get("temperature") {
            if let Some(temp_val) = temp.as_f64() {
                ParameterValidator::validate_temperature(temp_val, 0.0, 2.0, "Gemini")?;
            }
        }

        if let Some(top_p) = params.get("topP") {
            if let Some(top_p_val) = top_p.as_f64() {
                ParameterValidator::validate_top_p(top_p_val)?;
            }
        }

        if let Some(max_tokens) = params.get("maxOutputTokens") {
            if let Some(max_tokens_val) = max_tokens.as_u64() {
                ParameterValidator::validate_max_tokens(max_tokens_val, 1, 8192, "Gemini")?;
            }
        }

        // Validate Gemini-specific parameters
        if let Some(top_k) = params.get("topK") {
            if let Some(top_k_val) = top_k.as_u64() {
                if top_k_val == 0 || top_k_val > 40 {
                    return Err(LlmError::InvalidParameter(
                        "topK must be between 1 and 40 for Gemini".to_string(),
                    ));
                }
            }
        }

        if let Some(candidate_count) = params.get("candidateCount") {
            if let Some(count_val) = candidate_count.as_u64() {
                if count_val == 0 || count_val > 8 {
                    return Err(LlmError::InvalidParameter(
                        "candidateCount must be between 1 and 8 for Gemini".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Gemini
    }

    fn supported_params(&self) -> Vec<&'static str> {
        vec![
            "model",
            "temperature",
            "maxOutputTokens",
            "topP",
            "topK",
            "stopSequences",
            "candidateCount",
            "stream",
            "safetySettings",
            "generationConfig",
        ]
    }

    fn get_param_constraints(&self) -> ParameterConstraints {
        ParameterConstraints {
            temperature_min: 0.0,
            temperature_max: 2.0,
            max_tokens_min: 1,
            max_tokens_max: 8192,
            top_p_min: 0.0,
            top_p_max: 1.0,
        }
    }
}

/// Gemini-specific parameter extensions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeminiParams {
    /// Top-K sampling parameter
    pub top_k: Option<u32>,
    /// Number of candidate responses to generate
    pub candidate_count: Option<u32>,
    /// Safety settings
    pub safety_settings: Option<Vec<SafetySetting>>,
    /// Generation configuration
    pub generation_config: Option<GenerationConfig>,
    /// Whether to stream the response
    pub stream: Option<bool>,
}

impl super::common::ProviderParamsExt for GeminiParams {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Gemini
    }
}

/// Gemini Safety Setting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    pub category: SafetyCategory,
    pub threshold: SafetyThreshold,
}

/// Gemini Safety Categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyCategory {
    #[serde(rename = "HARM_CATEGORY_HARASSMENT")]
    Harassment,
    #[serde(rename = "HARM_CATEGORY_HATE_SPEECH")]
    HateSpeech,
    #[serde(rename = "HARM_CATEGORY_SEXUALLY_EXPLICIT")]
    SexuallyExplicit,
    #[serde(rename = "HARM_CATEGORY_DANGEROUS_CONTENT")]
    DangerousContent,
}

/// Gemini Safety Thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyThreshold {
    #[serde(rename = "BLOCK_NONE")]
    BlockNone,
    #[serde(rename = "BLOCK_LOW_AND_ABOVE")]
    BlockLowAndAbove,
    #[serde(rename = "BLOCK_MEDIUM_AND_ABOVE")]
    BlockMediumAndAbove,
    #[serde(rename = "BLOCK_HIGH_AND_ABOVE")]
    BlockHighAndAbove,
}

/// Gemini Generation Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub candidate_count: Option<u32>,
}

/// Gemini parameter builder for convenient parameter construction
pub struct GeminiParamsBuilder {
    params: GeminiParams,
}

impl GeminiParamsBuilder {
    pub fn new() -> Self {
        Self {
            params: GeminiParams::default(),
        }
    }

    pub fn top_k(mut self, top_k: u32) -> Self {
        self.params.top_k = Some(top_k);
        self
    }

    pub fn candidate_count(mut self, count: u32) -> Self {
        self.params.candidate_count = Some(count);
        self
    }

    pub fn safety_settings(mut self, settings: Vec<SafetySetting>) -> Self {
        self.params.safety_settings = Some(settings);
        self
    }

    pub fn add_safety_setting(
        mut self,
        category: SafetyCategory,
        threshold: SafetyThreshold,
    ) -> Self {
        if self.params.safety_settings.is_none() {
            self.params.safety_settings = Some(Vec::new());
        }
        self.params
            .safety_settings
            .as_mut()
            .unwrap()
            .push(SafetySetting {
                category,
                threshold,
            });
        self
    }

    pub fn generation_config(mut self, config: GenerationConfig) -> Self {
        self.params.generation_config = Some(config);
        self
    }

    pub fn stream(mut self, enabled: bool) -> Self {
        self.params.stream = Some(enabled);
        self
    }

    pub fn build(self) -> GeminiParams {
        self.params
    }
}

impl Default for GeminiParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_parameter_mapping() {
        let mapper = GeminiParameterMapper;
        let params = CommonParams {
            model: "gemini-pro".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            top_p: Some(0.9),
            stop_sequences: Some(vec!["STOP".to_string()]),
            seed: Some(42), // Should be ignored for Gemini
        };

        let mapped = mapper.map_common_params(&params);
        assert_eq!(mapped["model"], "gemini-pro");
        assert_eq!(mapped["maxOutputTokens"], 1000);
        // Use approximate comparison for floating point values
        let top_p_val = mapped["topP"].as_f64().unwrap();
        assert!((top_p_val - 0.9).abs() < 1e-6);
        assert_eq!(mapped["stopSequences"], serde_json::json!(["STOP"]));
        // Seed should not be present for Gemini
        assert!(mapped.get("seed").is_none());
    }

    #[test]
    fn test_gemini_parameter_validation() {
        let mapper = GeminiParameterMapper;

        // Valid parameters
        let valid_params = serde_json::json!({
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 1000,
            "topK": 20,
            "candidateCount": 2
        });
        assert!(mapper.validate_params(&valid_params).is_ok());

        // Invalid topK
        let invalid_top_k = serde_json::json!({
            "topK": 50
        });
        assert!(mapper.validate_params(&invalid_top_k).is_err());

        // Invalid candidateCount
        let invalid_count = serde_json::json!({
            "candidateCount": 10
        });
        assert!(mapper.validate_params(&invalid_count).is_err());
    }

    #[test]
    fn test_gemini_params_builder() {
        let params = GeminiParamsBuilder::new()
            .top_k(20)
            .candidate_count(2)
            .add_safety_setting(
                SafetyCategory::Harassment,
                SafetyThreshold::BlockMediumAndAbove,
            )
            .add_safety_setting(
                SafetyCategory::HateSpeech,
                SafetyThreshold::BlockHighAndAbove,
            )
            .stream(false)
            .build();

        assert_eq!(params.top_k, Some(20));
        assert_eq!(params.candidate_count, Some(2));
        assert!(params.safety_settings.is_some());
        assert_eq!(params.safety_settings.as_ref().unwrap().len(), 2);
        assert_eq!(params.stream, Some(false));
    }
}
