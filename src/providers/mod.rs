//! Provider Module
//!
//! Contains specific implementations for each LLM provider.

pub mod anthropic;
pub mod gemini;
pub mod groq;
pub mod ollama;
pub mod openai;
pub mod openai_compatible;
pub mod xai;

pub mod test_provider;

// Re-export main types
pub use anthropic::AnthropicClient;
pub use gemini::GeminiClient;
pub use ollama::OllamaClient;
pub use openai::{
    ListResponsesQuery, OpenAiClient, OpenAiResponses, ResponseMetadata, ResponseStatus,
    ResponsesApiCapability,
};
pub use openai_compatible::{
    DeepSeekProvider, OpenAiCompatibleBuilder, OpenAiCompatibleClient, OpenAiCompatibleProvider,
    OpenRouterProvider,
};
pub use xai::XaiClient;

use crate::traits::ProviderCapabilities;
use crate::types::ProviderType;

/// Provider Information
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Provider type
    pub provider_type: ProviderType,
    /// Provider name
    pub name: &'static str,
    /// Description
    pub description: &'static str,
    /// Supported capabilities
    pub capabilities: ProviderCapabilities,
    /// Default base URL
    pub default_base_url: &'static str,
    /// Supported models
    pub supported_models: Vec<&'static str>,
}

/// Get information for all supported providers
pub fn get_supported_providers() -> Vec<ProviderInfo> {
    vec![
        ProviderInfo {
            provider_type: ProviderType::OpenAi,
            name: "OpenAI",
            description: "OpenAI GPT models including GPT-4, GPT-3.5, and specialized models",
            capabilities: ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_audio()
                .with_embedding()
                .with_custom_feature("structured_output", true)
                .with_custom_feature("batch_processing", true),
            default_base_url: "https://api.openai.com/v1",
            supported_models: vec![
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "o1-preview",
                "o1-mini",
                "dall-e-3",
                "dall-e-2",
                "whisper-1",
                "tts-1",
                "tts-1-hd",
                "text-embedding-3-large",
                "text-embedding-3-small",
                "text-embedding-ada-002",
            ],
        },
        ProviderInfo {
            provider_type: ProviderType::Anthropic,
            name: "Anthropic",
            description: "Anthropic Claude models with advanced reasoning capabilities",
            capabilities: ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_custom_feature("prompt_caching", true)
                .with_custom_feature("thinking_mode", true),
            default_base_url: "https://api.anthropic.com",
            supported_models: vec![
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-20240620",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
        },
        ProviderInfo {
            provider_type: ProviderType::Gemini,
            name: "Google Gemini",
            description: "Google Gemini models with multimodal capabilities and code execution",
            capabilities: ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_custom_feature("code_execution", true)
                .with_custom_feature("thinking_mode", true)
                .with_custom_feature("safety_settings", true)
                .with_custom_feature("cached_content", true)
                .with_custom_feature("json_schema", true),
            default_base_url: "https://generativelanguage.googleapis.com/v1beta",
            supported_models: vec![
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
                "gemini-1.5-pro",
                "gemini-2.0-flash-exp",
                "gemini-exp-1114",
                "gemini-exp-1121",
                "gemini-exp-1206",
            ],
        },
        ProviderInfo {
            provider_type: ProviderType::Ollama,
            name: "Ollama",
            description: "Local Ollama models with full control and privacy",
            capabilities: ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_embedding()
                .with_custom_feature("completion", true)
                .with_custom_feature("model_management", true)
                .with_custom_feature("local_models", true),
            default_base_url: "http://localhost:11434",
            supported_models: vec![
                "llama3.2:latest",
                "llama3.2:3b",
                "llama3.2:1b",
                "llama3.1:latest",
                "llama3.1:8b",
                "llama3.1:70b",
                "mistral:latest",
                "mistral:7b",
                "codellama:latest",
                "codellama:7b",
                "codellama:13b",
                "codellama:34b",
                "phi3:latest",
                "phi3:mini",
                "phi3:medium",
                "gemma:latest",
                "gemma:2b",
                "gemma:7b",
                "qwen2:latest",
                "qwen2:0.5b",
                "qwen2:1.5b",
                "qwen2:7b",
                "qwen2:72b",
                "deepseek-coder:latest",
                "deepseek-coder:6.7b",
                "deepseek-coder:33b",
                "nomic-embed-text:latest",
                "all-minilm:latest",
            ],
        },
        ProviderInfo {
            provider_type: ProviderType::XAI,
            name: "xAI",
            description: "xAI Grok models with advanced reasoning capabilities",
            capabilities: ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_custom_feature("reasoning", true)
                .with_custom_feature("deferred_completion", true)
                .with_custom_feature("structured_outputs", true),
            default_base_url: "https://api.x.ai/v1",
            supported_models: vec![
                "grok-3-latest",
                "grok-3",
                "grok-3-fast",
                "grok-3-mini",
                "grok-3-mini-fast",
                "grok-2-vision-1212",
                "grok-2-1212",
                "grok-beta",
                "grok-vision-beta",
            ],
        },
    ]
}

/// Get provider information by provider type
pub fn get_provider_info(provider_type: &ProviderType) -> Option<ProviderInfo> {
    get_supported_providers()
        .into_iter()
        .find(|info| &info.provider_type == provider_type)
}

/// Check if a model is supported by the provider
pub fn is_model_supported(provider_type: &ProviderType, model: &str) -> bool {
    if let Some(info) = get_provider_info(provider_type) {
        info.supported_models.contains(&model)
    } else {
        false
    }
}

/// Get the default model for a provider
pub const fn get_default_model(provider_type: &ProviderType) -> Option<&'static str> {
    match provider_type {
        ProviderType::OpenAi => Some("gpt-4o"),
        ProviderType::Anthropic => Some("claude-3-5-sonnet-20241022"),
        ProviderType::Gemini => Some("gemini-1.5-flash"),
        ProviderType::Ollama => Some("llama3.2:latest"),
        ProviderType::XAI => Some("grok-3-latest"),
        ProviderType::Custom(_) => None,
        ProviderType::Groq => Some("llama-3.3-70b-versatile"),
    }
}

/// Provider Factory
pub struct ProviderFactory;

impl ProviderFactory {
    /// Validate provider configuration
    pub fn validate_config(
        provider_type: &ProviderType,
        api_key: &str,
        model: &str,
    ) -> Result<(), crate::error::LlmError> {
        // Check API key
        if api_key.is_empty() {
            return Err(crate::error::LlmError::MissingApiKey(format!(
                "API key is required for {provider_type}"
            )));
        }

        // Check model support
        if !is_model_supported(provider_type, model) {
            return Err(crate::error::LlmError::ModelNotSupported(format!(
                "Model '{model}' is not supported by {provider_type}"
            )));
        }

        Ok(())
    }

    /// Get the recommended configuration for a provider
    pub fn get_recommended_config(provider_type: &ProviderType) -> crate::types::CommonParams {
        match provider_type {
            ProviderType::OpenAi => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("gpt-4o")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(4096),
                top_p: Some(1.0),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::Anthropic => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("claude-3-5-sonnet-20241022")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(4096),
                top_p: Some(1.0),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::Gemini => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("gemini-1.5-flash")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(8192),
                top_p: Some(0.95),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::Ollama => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("llama3.2:latest")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(4096),
                top_p: Some(0.9),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::XAI => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("grok-3-latest")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(4096),
                top_p: Some(1.0),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::Groq => crate::types::CommonParams {
                model: get_default_model(provider_type)
                    .unwrap_or("llama-3.3-70b-versatile")
                    .to_string(),
                temperature: Some(0.7),
                max_tokens: Some(8192),
                top_p: Some(1.0),
                stop_sequences: None,
                seed: None,
            },
            ProviderType::Custom(_) => crate::types::CommonParams::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_supported_providers() {
        let providers = get_supported_providers();
        assert!(!providers.is_empty());

        let openai_provider = providers
            .iter()
            .find(|p| p.provider_type == ProviderType::OpenAi);
        assert!(openai_provider.is_some());

        let anthropic_provider = providers
            .iter()
            .find(|p| p.provider_type == ProviderType::Anthropic);
        assert!(anthropic_provider.is_some());
    }

    #[test]
    fn test_model_support() {
        assert!(is_model_supported(&ProviderType::OpenAi, "gpt-4"));
        assert!(is_model_supported(
            &ProviderType::Anthropic,
            "claude-3-5-sonnet-20241022"
        ));
        assert!(!is_model_supported(&ProviderType::OpenAi, "claude-3-opus"));
    }

    #[test]
    fn test_default_models() {
        assert_eq!(get_default_model(&ProviderType::OpenAi), Some("gpt-4o"));
        assert_eq!(
            get_default_model(&ProviderType::Anthropic),
            Some("claude-3-5-sonnet-20241022")
        );
    }

    #[test]
    fn test_config_validation() {
        let result = ProviderFactory::validate_config(&ProviderType::OpenAi, "test-key", "gpt-4");
        assert!(result.is_ok());

        let result = ProviderFactory::validate_config(&ProviderType::OpenAi, "", "gpt-4");
        assert!(result.is_err());

        let result =
            ProviderFactory::validate_config(&ProviderType::OpenAi, "test-key", "invalid-model");
        assert!(result.is_err());
    }
}
