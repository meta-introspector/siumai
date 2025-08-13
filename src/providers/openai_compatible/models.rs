//! Model Constants for OpenAI-Compatible Providers
//!
//! This module provides constants for commonly used models across different
//! OpenAI-compatible providers. Using constants instead of convenience builder
//! methods keeps the API surface area manageable and maintainable.

/// DeepSeek model constants
pub mod deepseek {
    /// General purpose chat model
    pub const CHAT: &str = "deepseek-chat";
    
    /// Specialized coding model
    pub const CODER: &str = "deepseek-coder";
    
    /// Enhanced reasoning model
    pub const REASONER: &str = "deepseek-reasoner";
    
    /// All available DeepSeek models
    pub const ALL: &[&str] = &[CHAT, CODER, REASONER];
}

/// OpenRouter model constants
pub mod openrouter {
    /// OpenAI models through OpenRouter
    pub mod openai {
        pub const GPT_4: &str = "openai/gpt-4";
        pub const GPT_4_TURBO: &str = "openai/gpt-4-turbo";
        pub const GPT_4O: &str = "openai/gpt-4o";
        pub const GPT_4O_MINI: &str = "openai/gpt-4o-mini";
        pub const GPT_3_5_TURBO: &str = "openai/gpt-3.5-turbo";
        pub const O1_PREVIEW: &str = "openai/o1-preview";
        pub const O1_MINI: &str = "openai/o1-mini";
    }
    
    /// Anthropic models through OpenRouter
    pub mod anthropic {
        pub const CLAUDE_3_5_SONNET: &str = "anthropic/claude-3.5-sonnet";
        pub const CLAUDE_3_5_HAIKU: &str = "anthropic/claude-3.5-haiku";
        pub const CLAUDE_3_OPUS: &str = "anthropic/claude-3-opus";
        pub const CLAUDE_3_SONNET: &str = "anthropic/claude-3-sonnet";
        pub const CLAUDE_3_HAIKU: &str = "anthropic/claude-3-haiku";
    }
    
    /// Google models through OpenRouter
    pub mod google {
        pub const GEMINI_PRO: &str = "google/gemini-pro";
        pub const GEMINI_PRO_VISION: &str = "google/gemini-pro-vision";
        pub const GEMINI_1_5_PRO: &str = "google/gemini-1.5-pro";
        pub const GEMINI_1_5_FLASH: &str = "google/gemini-1.5-flash";
    }
    
    /// Meta models through OpenRouter
    pub mod meta {
        pub const LLAMA_2_70B_CHAT: &str = "meta-llama/llama-2-70b-chat";
        pub const LLAMA_3_8B_INSTRUCT: &str = "meta-llama/llama-3-8b-instruct";
        pub const LLAMA_3_70B_INSTRUCT: &str = "meta-llama/llama-3-70b-instruct";
        pub const LLAMA_3_1_8B_INSTRUCT: &str = "meta-llama/llama-3.1-8b-instruct";
        pub const LLAMA_3_1_70B_INSTRUCT: &str = "meta-llama/llama-3.1-70b-instruct";
        pub const LLAMA_3_1_405B_INSTRUCT: &str = "meta-llama/llama-3.1-405b-instruct";
    }
    
    /// Mistral models through OpenRouter
    pub mod mistral {
        pub const MISTRAL_7B_INSTRUCT: &str = "mistralai/mistral-7b-instruct";
        pub const MIXTRAL_8X7B_INSTRUCT: &str = "mistralai/mixtral-8x7b-instruct";
        pub const MIXTRAL_8X22B_INSTRUCT: &str = "mistralai/mixtral-8x22b-instruct";
    }
    
    /// Popular models for quick access
    pub mod popular {
        pub const GPT_4: &str = super::openai::GPT_4;
        pub const GPT_4O: &str = super::openai::GPT_4O;
        pub const CLAUDE_3_5_SONNET: &str = super::anthropic::CLAUDE_3_5_SONNET;
        pub const GEMINI_PRO: &str = super::google::GEMINI_PRO;
        pub const LLAMA_3_1_70B: &str = super::meta::LLAMA_3_1_70B_INSTRUCT;
    }
}

/// xAI model constants (when supported)
pub mod xai {
    pub const GROK_BETA: &str = "grok-beta";
    pub const GROK_3: &str = "grok-3";

    /// All available xAI models
    pub const ALL: &[&str] = &[GROK_BETA, GROK_3];
}

/// Groq model constants (when supported)
pub mod groq {
    pub const LLAMA_3_8B_8192: &str = "llama3-8b-8192";
    pub const LLAMA_3_70B_8192: &str = "llama3-70b-8192";
    pub const LLAMA_3_3_70B_VERSATILE: &str = "llama-3.3-70b-versatile";
    pub const MIXTRAL_8X7B_32768: &str = "mixtral-8x7b-32768";
    pub const GEMMA_7B_IT: &str = "gemma-7b-it";
    
    /// All available Groq models
    pub const ALL: &[&str] = &[
        LLAMA_3_8B_8192,
        LLAMA_3_70B_8192,
        LLAMA_3_3_70B_VERSATILE,
        MIXTRAL_8X7B_32768,
        GEMMA_7B_IT,
    ];
}

/// Model recommendation helpers
pub mod recommendations {
    use super::*;
    
    /// Get recommended model for general chat
    pub fn for_chat() -> &'static str {
        openrouter::openai::GPT_4O
    }
    
    /// Get recommended model for coding tasks
    pub fn for_coding() -> &'static str {
        deepseek::CODER
    }
    
    /// Get recommended model for reasoning tasks
    pub fn for_reasoning() -> &'static str {
        deepseek::REASONER
    }
    
    /// Get recommended model for fast responses
    pub fn for_fast_response() -> &'static str {
        openrouter::openai::GPT_4O_MINI
    }
    
    /// Get recommended model for cost-effective usage
    pub fn for_cost_effective() -> &'static str {
        deepseek::CHAT
    }
    
    /// Get recommended model for vision tasks
    pub fn for_vision() -> &'static str {
        openrouter::google::GEMINI_PRO_VISION
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_deepseek_models() {
        assert_eq!(deepseek::CHAT, "deepseek-chat");
        assert_eq!(deepseek::CODER, "deepseek-coder");
        assert_eq!(deepseek::REASONER, "deepseek-reasoner");
        assert_eq!(deepseek::ALL.len(), 3);
    }
    
    #[test]
    fn test_openrouter_models() {
        assert_eq!(openrouter::openai::GPT_4, "openai/gpt-4");
        assert_eq!(openrouter::anthropic::CLAUDE_3_5_SONNET, "anthropic/claude-3.5-sonnet");
        assert_eq!(openrouter::google::GEMINI_PRO, "google/gemini-pro");
    }
    
    #[test]
    fn test_recommendations() {
        assert!(!recommendations::for_chat().is_empty());
        assert!(!recommendations::for_coding().is_empty());
        assert!(!recommendations::for_reasoning().is_empty());
    }
}
