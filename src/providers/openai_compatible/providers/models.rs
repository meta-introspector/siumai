//! OpenAI-Compatible Provider Model Definitions
//!
//! This module contains model definitions for various OpenAI-compatible providers.

/// DeepSeek model constants
pub mod deepseek {
    /// DeepSeek Chat model
    pub const CHAT: &str = "deepseek-chat";
    /// DeepSeek Coder model
    pub const CODER: &str = "deepseek-coder";
    /// DeepSeek Reasoner model (V3)
    pub const REASONER: &str = "deepseek-reasoner";
    /// DeepSeek V3 model
    pub const DEEPSEEK_V3: &str = "deepseek-v3";

    /// All DeepSeek models
    pub const ALL: &[&str] = &[CHAT, CODER, REASONER, DEEPSEEK_V3];

    /// Get all DeepSeek models
    pub fn all_models() -> Vec<String> {
        ALL.iter().map(|&s| s.to_string()).collect()
    }
}

/// OpenRouter model constants
pub mod openrouter {
    /// OpenAI models via OpenRouter
    pub mod openai {
        pub const GPT_4: &str = "openai/gpt-4";
        pub const GPT_4_TURBO: &str = "openai/gpt-4-turbo";
        pub const GPT_4O: &str = "openai/gpt-4o";
    }

    /// Anthropic models via OpenRouter
    pub mod anthropic {
        pub const CLAUDE_3_5_SONNET: &str = "anthropic/claude-3.5-sonnet";
    }

    /// Google models via OpenRouter
    pub mod google {
        pub const GEMINI_PRO: &str = "google/gemini-pro";
        pub const GEMINI_1_5_PRO: &str = "google/gemini-1.5-pro";
    }

    /// Popular models collection
    pub mod popular {
        pub const GPT_4: &str = super::openai::GPT_4;
        pub const GPT_4O: &str = super::openai::GPT_4O;
        pub const CLAUDE_3_5_SONNET: &str = super::anthropic::CLAUDE_3_5_SONNET;
        pub const GEMINI_PRO: &str = super::google::GEMINI_PRO;
    }

    /// Get all OpenRouter models
    pub fn all_models() -> Vec<String> {
        vec![
            openai::GPT_4_TURBO.to_string(),
            openai::GPT_4O.to_string(),
            anthropic::CLAUDE_3_5_SONNET.to_string(),
            google::GEMINI_PRO.to_string(),
            google::GEMINI_1_5_PRO.to_string(),
        ]
    }
}

/// xAI model constants
pub mod xai {
    /// Grok Beta model
    pub const GROK_BETA: &str = "grok-beta";
    /// Grok Vision Beta model
    pub const GROK_VISION_BETA: &str = "grok-vision-beta";

    /// Get all xAI models
    pub fn all_models() -> Vec<String> {
        vec![
            GROK_BETA.to_string(),
            GROK_VISION_BETA.to_string(),
        ]
    }
}

/// Groq model constants
pub mod groq {
    /// Llama 3.1 70B Versatile
    pub const LLAMA_3_1_70B: &str = "llama-3.1-70b-versatile";
    /// Llama 3.1 8B Instant
    pub const LLAMA_3_1_8B: &str = "llama-3.1-8b-instant";
    /// Mixtral 8x7B
    pub const MIXTRAL_8X7B: &str = "mixtral-8x7b-32768";

    /// Get all Groq models
    pub fn all_models() -> Vec<String> {
        vec![
            LLAMA_3_1_70B.to_string(),
            LLAMA_3_1_8B.to_string(),
            MIXTRAL_8X7B.to_string(),
        ]
    }
}

/// Get models for a specific provider
pub fn get_models_for_provider(provider: &str) -> Vec<String> {
    match provider.to_lowercase().as_str() {
        "deepseek" => deepseek::all_models(),
        "openrouter" => openrouter::all_models(),
        "xai" => xai::all_models(),
        "groq" => groq::all_models(),
        _ => vec![],
    }
}

/// Check if a model is supported by a provider
pub fn is_model_supported(provider: &str, model: &str) -> bool {
    get_models_for_provider(provider).contains(&model.to_string())
}

/// Model recommendations for different use cases
pub mod recommendations {
    use super::*;

    /// Recommended model for general chat
    pub fn for_chat() -> &'static str {
        openrouter::openai::GPT_4O
    }

    /// Recommended model for coding tasks
    pub fn for_coding() -> &'static str {
        deepseek::CODER
    }

    /// Recommended model for reasoning tasks
    pub fn for_reasoning() -> &'static str {
        deepseek::REASONER
    }

    /// Recommended model for fast responses
    pub fn for_fast_response() -> &'static str {
        groq::LLAMA_3_1_8B
    }

    /// Recommended model for cost-effective usage
    pub fn for_cost_effective() -> &'static str {
        deepseek::CHAT
    }

    /// Recommended model for vision tasks
    pub fn for_vision() -> &'static str {
        openrouter::openai::GPT_4O
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_models() {
        let models = deepseek::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"deepseek-chat".to_string()));
    }

    #[test]
    fn test_openrouter_models() {
        let models = openrouter::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"openai/gpt-4o".to_string()));
    }

    #[test]
    fn test_get_models_for_provider() {
        let deepseek_models = get_models_for_provider("deepseek");
        assert!(!deepseek_models.is_empty());

        let unknown_models = get_models_for_provider("unknown");
        assert!(unknown_models.is_empty());
    }

    #[test]
    fn test_is_model_supported() {
        assert!(is_model_supported("deepseek", "deepseek-chat"));
        assert!(!is_model_supported("deepseek", "unknown-model"));
        assert!(!is_model_supported("unknown", "any-model"));
    }
}
