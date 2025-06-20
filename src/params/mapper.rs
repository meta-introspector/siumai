//! Parameter Mapping Trait and Factory
//!
//! Defines the core parameter mapping trait and factory for creating mappers.

use crate::error::LlmError;
use crate::types::{CommonParams, ProviderParams, ProviderType};

/// Parameter mapping trait - maps common parameters to provider-specific format
pub trait ParameterMapper {
    /// Maps common parameters to the provider's format
    fn map_common_params(&self, params: &CommonParams) -> serde_json::Value;

    /// Merges provider-specific parameters
    fn merge_provider_params(
        &self,
        base: serde_json::Value,
        provider: &ProviderParams,
    ) -> serde_json::Value;

    /// Validates the validity of the parameters
    fn validate_params(&self, params: &serde_json::Value) -> Result<(), LlmError>;

    /// Gets the provider type this mapper is for
    fn provider_type(&self) -> ProviderType;

    /// Gets the supported parameter names for this provider
    fn supported_params(&self) -> Vec<&'static str> {
        vec![
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stop_sequences",
            "seed",
        ]
    }

    /// Checks if a parameter is supported by this provider
    fn is_param_supported(&self, param_name: &str) -> bool {
        self.supported_params().contains(&param_name)
    }

    /// Gets parameter constraints for this provider
    fn get_param_constraints(&self) -> ParameterConstraints {
        ParameterConstraints::default()
    }
}

/// Parameter constraints for validation
#[derive(Debug, Clone)]
pub struct ParameterConstraints {
    pub temperature_min: f64,
    pub temperature_max: f64,
    pub max_tokens_min: u64,
    pub max_tokens_max: u64,
    pub top_p_min: f64,
    pub top_p_max: f64,
}

impl Default for ParameterConstraints {
    fn default() -> Self {
        Self {
            temperature_min: 0.0,
            temperature_max: 2.0,
            max_tokens_min: 1,
            max_tokens_max: 100000,
            top_p_min: 0.0,
            top_p_max: 1.0,
        }
    }
}

/// Parameter Mapper Factory
pub struct ParameterMapperFactory;

impl ParameterMapperFactory {
    /// Creates a parameter mapper based on the provider type
    pub fn create_mapper(provider_type: &ProviderType) -> Box<dyn ParameterMapper> {
        match provider_type {
            ProviderType::OpenAi => Box::new(crate::params::openai::OpenAiParameterMapper),
            ProviderType::Anthropic => Box::new(crate::params::anthropic::AnthropicParameterMapper),
            ProviderType::Gemini => Box::new(crate::params::gemini::GeminiParameterMapper),
            ProviderType::Ollama => Box::new(crate::params::ollama::OllamaParameterMapper), // Ollama has its own specific format
            ProviderType::XAI => Box::new(crate::params::openai::OpenAiParameterMapper), // xAI uses OpenAI-compatible format
            ProviderType::Custom(_) => Box::new(crate::params::openai::OpenAiParameterMapper), // Default to OpenAI format
        }
    }

    /// Gets all available mapper types
    pub fn available_mappers() -> Vec<ProviderType> {
        vec![
            ProviderType::OpenAi,
            ProviderType::Anthropic,
            ProviderType::Gemini,
            ProviderType::Ollama,
            ProviderType::XAI,
        ]
    }

    /// Validates that a provider type has a mapper
    pub fn has_mapper(provider_type: &ProviderType) -> bool {
        matches!(
            provider_type,
            ProviderType::OpenAi
                | ProviderType::Anthropic
                | ProviderType::Gemini
                | ProviderType::Ollama
                | ProviderType::XAI
                | ProviderType::Custom(_)
        )
    }
}

/// Parameter mapping utilities
pub struct ParameterMappingUtils;

impl ParameterMappingUtils {
    /// Converts common parameters to provider-specific format using the appropriate mapper
    pub fn convert_params(
        common_params: &CommonParams,
        provider_params: Option<&ProviderParams>,
        provider_type: &ProviderType,
    ) -> Result<serde_json::Value, LlmError> {
        let mapper = ParameterMapperFactory::create_mapper(provider_type);

        let mut result = mapper.map_common_params(common_params);

        if let Some(provider_params) = provider_params {
            result = mapper.merge_provider_params(result, provider_params);
        }

        mapper.validate_params(&result)?;

        Ok(result)
    }

    /// Validates parameters for a specific provider
    pub fn validate_for_provider(
        params: &serde_json::Value,
        provider_type: &ProviderType,
    ) -> Result<(), LlmError> {
        let mapper = ParameterMapperFactory::create_mapper(provider_type);
        mapper.validate_params(params)
    }

    /// Gets parameter constraints for a provider
    pub fn get_constraints(provider_type: &ProviderType) -> ParameterConstraints {
        let mapper = ParameterMapperFactory::create_mapper(provider_type);
        mapper.get_param_constraints()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_mapper_factory() {
        // Test that factory can create mappers for all supported providers
        let openai_mapper = ParameterMapperFactory::create_mapper(&ProviderType::OpenAi);
        assert_eq!(openai_mapper.provider_type(), ProviderType::OpenAi);

        let anthropic_mapper = ParameterMapperFactory::create_mapper(&ProviderType::Anthropic);
        assert_eq!(anthropic_mapper.provider_type(), ProviderType::Anthropic);

        let gemini_mapper = ParameterMapperFactory::create_mapper(&ProviderType::Gemini);
        assert_eq!(gemini_mapper.provider_type(), ProviderType::Gemini);
    }

    #[test]
    fn test_available_mappers() {
        let mappers = ParameterMapperFactory::available_mappers();
        assert!(mappers.contains(&ProviderType::OpenAi));
        assert!(mappers.contains(&ProviderType::Anthropic));
        assert!(mappers.contains(&ProviderType::Gemini));
    }

    #[test]
    fn test_has_mapper() {
        assert!(ParameterMapperFactory::has_mapper(&ProviderType::OpenAi));
        assert!(ParameterMapperFactory::has_mapper(&ProviderType::Anthropic));
        assert!(ParameterMapperFactory::has_mapper(&ProviderType::Gemini));
        assert!(ParameterMapperFactory::has_mapper(&ProviderType::Custom(
            "test".to_string()
        )));
    }

    #[test]
    fn test_parameter_constraints() {
        let constraints = ParameterConstraints::default();
        assert_eq!(constraints.temperature_min, 0.0);
        assert_eq!(constraints.temperature_max, 2.0);
        assert_eq!(constraints.max_tokens_min, 1);
        assert_eq!(constraints.top_p_min, 0.0);
        assert_eq!(constraints.top_p_max, 1.0);
    }
}
