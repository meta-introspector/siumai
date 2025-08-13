//! OpenAI-Compatible Provider Model Definitions
//!
//! This module contains model definitions for various OpenAI-compatible providers.

/// `DeepSeek` model constants
pub mod deepseek {
    /// `DeepSeek` Chat model (points to DeepSeek-V3-0324)
    pub const CHAT: &str = "deepseek-chat";
    /// `DeepSeek` Reasoner model (points to DeepSeek-R1-0528)
    pub const REASONER: &str = "deepseek-reasoner";

    // Specific model versions
    /// `DeepSeek` V3 (2024-03-24)
    pub const DEEPSEEK_V3_0324: &str = "deepseek-v3-0324";
    /// `DeepSeek` R1 (2025-05-28)
    pub const DEEPSEEK_R1_0528: &str = "deepseek-r1-0528";
    /// `DeepSeek` R1 (2025-01-20)
    pub const DEEPSEEK_R1_20250120: &str = "deepseek-r1-20250120";

    // Legacy models (deprecated)
    /// `DeepSeek` Coder model (legacy)
    pub const CODER: &str = "deepseek-coder";
    /// `DeepSeek` V3 model (legacy alias)
    pub const DEEPSEEK_V3: &str = "deepseek-v3";

    /// All `DeepSeek` models
    pub const ALL: &[&str] = &[
        CHAT,
        REASONER,
        DEEPSEEK_V3_0324,
        DEEPSEEK_R1_0528,
        DEEPSEEK_R1_20250120,
        CODER,
        DEEPSEEK_V3,
    ];

    /// Get all `DeepSeek` models
    pub fn all_models() -> Vec<String> {
        ALL.iter().map(|&s| s.to_string()).collect()
    }

    /// Get current active models (non-legacy)
    pub fn active_models() -> Vec<String> {
        vec![
            CHAT.to_string(),
            REASONER.to_string(),
            DEEPSEEK_V3_0324.to_string(),
            DEEPSEEK_R1_0528.to_string(),
            DEEPSEEK_R1_20250120.to_string(),
        ]
    }
}

/// `OpenRouter` model constants
pub mod openrouter {
    /// `OpenAI` models via `OpenRouter`
    pub mod openai {
        pub const GPT_4: &str = "openai/gpt-4";
        pub const GPT_4_TURBO: &str = "openai/gpt-4-turbo";
        pub const GPT_4O: &str = "openai/gpt-4o";
        pub const GPT_4O_MINI: &str = "openai/gpt-4o-mini";
        pub const GPT_4_1: &str = "openai/gpt-4.1";
        pub const GPT_4_1_MINI: &str = "openai/gpt-4.1-mini";
        pub const O1: &str = "openai/o1";
        pub const O1_MINI: &str = "openai/o1-mini";
        pub const O3_MINI: &str = "openai/o3-mini";
    }

    /// Anthropic models via `OpenRouter`
    pub mod anthropic {
        pub const CLAUDE_3_5_SONNET: &str = "anthropic/claude-3.5-sonnet";
        pub const CLAUDE_3_5_HAIKU: &str = "anthropic/claude-3.5-haiku";
        pub const CLAUDE_SONNET_4: &str = "anthropic/claude-sonnet-4";
        pub const CLAUDE_OPUS_4: &str = "anthropic/claude-opus-4";
        pub const CLAUDE_OPUS_4_1: &str = "anthropic/claude-opus-4.1";
    }

    /// Google models via `OpenRouter`
    pub mod google {
        pub const GEMINI_PRO: &str = "google/gemini-pro";
        pub const GEMINI_1_5_PRO: &str = "google/gemini-1.5-pro";
        pub const GEMINI_2_0_FLASH: &str = "google/gemini-2.0-flash";
        pub const GEMINI_2_5_FLASH: &str = "google/gemini-2.5-flash";
        pub const GEMINI_2_5_PRO: &str = "google/gemini-2.5-pro";
    }

    /// DeepSeek models via `OpenRouter`
    pub mod deepseek {
        pub const DEEPSEEK_CHAT: &str = "deepseek/deepseek-chat";
        pub const DEEPSEEK_REASONER: &str = "deepseek/deepseek-reasoner";
        pub const DEEPSEEK_V3: &str = "deepseek/deepseek-v3";
        pub const DEEPSEEK_R1: &str = "deepseek/deepseek-r1";
    }

    /// Meta models via `OpenRouter`
    pub mod meta {
        pub const LLAMA_3_1_8B: &str = "meta-llama/llama-3.1-8b-instruct";
        pub const LLAMA_3_1_70B: &str = "meta-llama/llama-3.1-70b-instruct";
        pub const LLAMA_3_1_405B: &str = "meta-llama/llama-3.1-405b-instruct";
        pub const LLAMA_3_2_1B: &str = "meta-llama/llama-3.2-1b-instruct";
        pub const LLAMA_3_2_3B: &str = "meta-llama/llama-3.2-3b-instruct";
    }

    /// Mistral models via `OpenRouter`
    pub mod mistral {
        pub const MISTRAL_7B: &str = "mistralai/mistral-7b-instruct";
        pub const MIXTRAL_8X7B: &str = "mistralai/mixtral-8x7b-instruct";
        pub const MIXTRAL_8X22B: &str = "mistralai/mixtral-8x22b-instruct";
        pub const MISTRAL_LARGE: &str = "mistralai/mistral-large";
    }

    /// Popular models collection
    pub mod popular {
        use super::*;

        pub const GPT_4O: &str = openai::GPT_4O;
        pub const GPT_4_1: &str = openai::GPT_4_1;
        pub const CLAUDE_OPUS_4_1: &str = anthropic::CLAUDE_OPUS_4_1;
        pub const CLAUDE_SONNET_4: &str = anthropic::CLAUDE_SONNET_4;
        pub const GEMINI_2_5_PRO: &str = google::GEMINI_2_5_PRO;
        pub const DEEPSEEK_REASONER: &str = deepseek::DEEPSEEK_REASONER;
        pub const LLAMA_3_1_405B: &str = meta::LLAMA_3_1_405B;
    }

    /// Get all `OpenRouter` models
    pub fn all_models() -> Vec<String> {
        let mut models = Vec::new();

        // OpenAI models
        models.extend_from_slice(&[
            openai::GPT_4.to_string(),
            openai::GPT_4_TURBO.to_string(),
            openai::GPT_4O.to_string(),
            openai::GPT_4O_MINI.to_string(),
            openai::GPT_4_1.to_string(),
            openai::GPT_4_1_MINI.to_string(),
            openai::O1.to_string(),
            openai::O1_MINI.to_string(),
            openai::O3_MINI.to_string(),
        ]);

        // Anthropic models
        models.extend_from_slice(&[
            anthropic::CLAUDE_3_5_SONNET.to_string(),
            anthropic::CLAUDE_3_5_HAIKU.to_string(),
            anthropic::CLAUDE_SONNET_4.to_string(),
            anthropic::CLAUDE_OPUS_4.to_string(),
            anthropic::CLAUDE_OPUS_4_1.to_string(),
        ]);

        // Google models
        models.extend_from_slice(&[
            google::GEMINI_PRO.to_string(),
            google::GEMINI_1_5_PRO.to_string(),
            google::GEMINI_2_0_FLASH.to_string(),
            google::GEMINI_2_5_FLASH.to_string(),
            google::GEMINI_2_5_PRO.to_string(),
        ]);

        // DeepSeek models
        models.extend_from_slice(&[
            deepseek::DEEPSEEK_CHAT.to_string(),
            deepseek::DEEPSEEK_REASONER.to_string(),
            deepseek::DEEPSEEK_V3.to_string(),
            deepseek::DEEPSEEK_R1.to_string(),
        ]);

        // Meta models
        models.extend_from_slice(&[
            meta::LLAMA_3_1_8B.to_string(),
            meta::LLAMA_3_1_70B.to_string(),
            meta::LLAMA_3_1_405B.to_string(),
            meta::LLAMA_3_2_1B.to_string(),
            meta::LLAMA_3_2_3B.to_string(),
        ]);

        // Mistral models
        models.extend_from_slice(&[
            mistral::MISTRAL_7B.to_string(),
            mistral::MIXTRAL_8X7B.to_string(),
            mistral::MIXTRAL_8X22B.to_string(),
            mistral::MISTRAL_LARGE.to_string(),
        ]);

        models
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
        vec![GROK_BETA.to_string(), GROK_VISION_BETA.to_string()]
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
    pub const fn for_chat() -> &'static str {
        openrouter::openai::GPT_4O
    }

    /// Recommended model for coding tasks
    pub const fn for_coding() -> &'static str {
        deepseek::DEEPSEEK_V3_0324 // Use latest V3 model for coding
    }

    /// Recommended model for reasoning tasks
    pub const fn for_reasoning() -> &'static str {
        deepseek::REASONER
    }

    /// Recommended model for fast responses
    pub const fn for_fast_response() -> &'static str {
        groq::LLAMA_3_1_8B
    }

    /// Recommended model for cost-effective usage
    pub const fn for_cost_effective() -> &'static str {
        deepseek::CHAT
    }

    /// Recommended model for vision tasks
    pub const fn for_vision() -> &'static str {
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
