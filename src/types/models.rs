//! Model information and capabilities
//!
//! This module provides model information structures and unified access to model constants
//! from all providers for convenient developer usage.

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Model name
    pub name: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Model owner/organization
    pub owned_by: String,
    /// Creation timestamp
    pub created: Option<u64>,
    /// Model capabilities
    pub capabilities: Vec<String>,
    /// Context window size
    pub context_window: Option<u32>,
    /// Maximum output tokens
    pub max_output_tokens: Option<u32>,
    /// Input cost per token
    pub input_cost_per_token: Option<f64>,
    /// Output cost per token
    pub output_cost_per_token: Option<f64>,
}

/// Unified model constants for easy access across all providers
///
/// This module provides convenient access to model constants from all supported providers,
/// making it easy for developers to reference specific models without hardcoding strings.
///
/// # Examples
///
/// ```rust
/// use siumai::models;
///
/// // Simple access to popular models
/// let openai_model = models::openai::GPT_4O;
/// let anthropic_model = models::anthropic::CLAUDE_OPUS_4_1;
/// let gemini_model = models::gemini::GEMINI_2_5_PRO;
///
/// // Access popular recommendations
/// let flagship_models = [
///     models::popular::OPENAI_FLAGSHIP,
///     models::popular::ANTHROPIC_FLAGSHIP,
///     models::popular::GEMINI_FLAGSHIP,
/// ];
/// ```
pub mod constants {
    /// Re-export OpenAI model constants (detailed structure)
    pub use crate::providers::openai::model_constants as openai;

    /// Re-export Anthropic model constants (detailed structure)
    pub use crate::providers::anthropic::model_constants as anthropic;

    /// Re-export Gemini model constants (detailed structure)
    pub use crate::providers::gemini::model_constants as gemini;

    /// Re-export OpenAI-compatible provider model constants
    pub use crate::providers::openai_compatible::providers::models as openai_compatible;

    /// Popular models across all providers
    ///
    /// This module provides curated selections of popular models from each provider,
    /// organized by use case (flagship, balanced, reasoning, etc.).
    pub mod popular {
        /// Most capable models from each provider
        pub mod flagship {
            /// OpenAI's most capable model
            pub const OPENAI: &str = super::super::openai::popular::FLAGSHIP;
            /// Anthropic's most capable model
            pub const ANTHROPIC: &str = super::super::anthropic::popular::FLAGSHIP;
            /// Google's most capable model
            pub const GEMINI: &str = super::super::gemini::popular::FLAGSHIP;
        }

        /// Best balanced models (capability vs cost)
        pub mod balanced {
            /// OpenAI's balanced model
            pub const OPENAI: &str = super::super::openai::popular::BALANCED;
            /// Anthropic's balanced model
            pub const ANTHROPIC: &str = super::super::anthropic::popular::BALANCED;
            /// Google's balanced model
            pub const GEMINI: &str = super::super::gemini::popular::BALANCED;
        }

        /// Best reasoning models
        pub mod reasoning {
            /// OpenAI's reasoning model
            pub const OPENAI: &str = super::super::openai::popular::REASONING;
            /// Anthropic's thinking model
            pub const ANTHROPIC: &str = super::super::anthropic::popular::THINKING;
            /// Google's flagship model (has thinking)
            pub const GEMINI: &str = super::super::gemini::popular::FLAGSHIP;
        }

        /// Most economical models
        pub mod economical {
            /// OpenAI's economical model
            pub const OPENAI: &str = super::super::openai::popular::ECONOMICAL;
            /// Anthropic's fast model
            pub const ANTHROPIC: &str = super::super::anthropic::popular::FAST;
            /// Google's economical model
            pub const GEMINI: &str = super::super::gemini::popular::ECONOMICAL;
        }

        /// Latest and most advanced models
        pub mod latest {
            /// OpenAI's latest model
            pub const OPENAI: &str = super::super::openai::popular::LATEST;
            /// Anthropic's latest model
            pub const ANTHROPIC: &str = super::super::anthropic::popular::LATEST;
            /// Google's latest model
            pub const GEMINI: &str = super::super::gemini::popular::LATEST;
        }
    }

    /// Get all available chat models from all providers
    pub fn all_chat_models() -> Vec<&'static str> {
        let mut models = Vec::new();
        models.extend_from_slice(&openai::all_chat_models());
        models.extend_from_slice(&anthropic::all_chat_models());
        models.extend_from_slice(&gemini::all_chat_models());
        models
    }

    /// Get all reasoning models from all providers
    pub fn all_reasoning_models() -> Vec<&'static str> {
        let mut models = Vec::new();
        models.extend_from_slice(&openai::all_reasoning_models());
        models.extend_from_slice(&anthropic::all_thinking_models());
        models.extend_from_slice(&gemini::all_thinking_models());
        models
    }

    /// Get all multimodal models from all providers
    pub fn all_multimodal_models() -> Vec<&'static str> {
        let mut models = Vec::new();
        models.extend_from_slice(&openai::all_multimodal_models());
        models.extend_from_slice(&anthropic::all_vision_models());
        models.extend_from_slice(&gemini::all_chat_models()); // Most Gemini models are multimodal
        models
    }

    /// Get all audio generation models from all providers
    pub fn all_audio_generation_models() -> Vec<&'static str> {
        let mut models = Vec::new();
        models.extend_from_slice(&openai::audio::ALL);
        models.extend_from_slice(&gemini::all_audio_generation_models());
        models
    }

    /// Get all image generation models from all providers
    pub fn all_image_generation_models() -> Vec<&'static str> {
        let mut models = Vec::new();
        models.extend_from_slice(&openai::images::ALL);
        models.extend_from_slice(&gemini::all_image_generation_models());
        models
    }

    /// Get all embedding models from all providers
    pub fn all_embedding_models() -> Vec<&'static str> {
        let mut models = Vec::new();
        models.extend_from_slice(&openai::embeddings::ALL);
        models
    }
}

/// Simplified model constants with shorter namespaces
///
/// This module provides a more concise way to access model constants
/// with shorter namespaces for better developer experience.
///
/// # Examples
///
/// ```rust
/// use siumai::models;
///
/// // Short and sweet access
/// let model = models::openai::GPT_4O;
/// let claude = models::anthropic::CLAUDE_OPUS_4_1;
/// let gemini = models::gemini::GEMINI_2_5_PRO;
///
/// // Popular recommendations
/// let flagship = models::popular::OPENAI_FLAGSHIP;
/// ```
pub mod models {
    /// OpenAI models with simplified access
    pub mod openai {
        use crate::providers::openai::model_constants as c;

        // GPT-4o family
        pub const GPT_4O: &str = c::gpt_4o::GPT_4O;
        pub const GPT_4O_MINI: &str = c::gpt_4o::GPT_4O_MINI;
        pub const GPT_4O_AUDIO: &str = c::gpt_4o::GPT_4O_AUDIO_PREVIEW;

        // GPT-4.1 family
        pub const GPT_4_1: &str = c::gpt_4_1::GPT_4_1;
        pub const GPT_4_1_MINI: &str = c::gpt_4_1::GPT_4_1_MINI;
        pub const GPT_4_1_NANO: &str = c::gpt_4_1::GPT_4_1_NANO;

        // GPT-4.5 family
        pub const GPT_4_5: &str = c::gpt_4_5::GPT_4_5;
        pub const GPT_4_5_PREVIEW: &str = c::gpt_4_5::GPT_4_5_PREVIEW;

        // GPT-5 family
        pub const GPT_5: &str = c::gpt_5::GPT_5;
        pub const GPT_5_MINI: &str = c::gpt_5::GPT_5_MINI;
        pub const GPT_5_NANO: &str = c::gpt_5::GPT_5_NANO;

        // GPT-4 Turbo
        pub const GPT_4_TURBO: &str = c::gpt_4_turbo::GPT_4_TURBO;

        // GPT-4 Classic
        pub const GPT_4: &str = c::gpt_4::GPT_4;
        pub const GPT_4_32K: &str = c::gpt_4::GPT_4_32K;

        // o1 reasoning models
        pub const O1: &str = c::o1::O1;
        pub const O1_PREVIEW: &str = c::o1::O1_PREVIEW;
        pub const O1_MINI: &str = c::o1::O1_MINI;

        // o3 reasoning models
        pub const O3: &str = c::o3::O3;
        pub const O3_MINI: &str = c::o3::O3_MINI;

        // o4 reasoning models
        pub const O4_MINI: &str = c::o4::O4_MINI;

        // GPT-3.5
        pub const GPT_3_5_TURBO: &str = c::gpt_3_5::GPT_3_5_TURBO;

        // Audio models
        pub const TTS_1: &str = c::audio::TTS_1;
        pub const TTS_1_HD: &str = c::audio::TTS_1_HD;
        pub const WHISPER_1: &str = c::audio::WHISPER_1;

        // Image models
        pub const DALL_E_2: &str = c::images::DALL_E_2;
        pub const DALL_E_3: &str = c::images::DALL_E_3;

        // Embedding models
        pub const TEXT_EMBEDDING_3_SMALL: &str = c::embeddings::TEXT_EMBEDDING_3_SMALL;
        pub const TEXT_EMBEDDING_3_LARGE: &str = c::embeddings::TEXT_EMBEDDING_3_LARGE;
    }

    /// Anthropic models with simplified access
    pub mod anthropic {
        use crate::providers::anthropic::model_constants as c;

        // Claude Opus 4.1 (latest flagship)
        pub const CLAUDE_OPUS_4_1: &str = c::claude_opus_4_1::CLAUDE_OPUS_4_1;

        // Claude Opus 4
        pub const CLAUDE_OPUS_4: &str = c::claude_opus_4::CLAUDE_OPUS_4_20250514;

        // Claude Sonnet 4
        pub const CLAUDE_SONNET_4: &str = c::claude_sonnet_4::CLAUDE_SONNET_4_20250514;

        // Claude Sonnet 3.7 (thinking)
        pub const CLAUDE_SONNET_3_7: &str = c::claude_sonnet_3_7::CLAUDE_3_7_SONNET_20250219;

        // Claude Sonnet 3.5
        pub const CLAUDE_SONNET_3_5: &str = c::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20241022;
        pub const CLAUDE_SONNET_3_5_LEGACY: &str = c::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20240620;

        // Claude Haiku 3.5
        pub const CLAUDE_HAIKU_3_5: &str = c::claude_haiku_3_5::CLAUDE_3_5_HAIKU_20241022;

        // Claude 3 (legacy)
        pub const CLAUDE_OPUS_3: &str = c::claude_opus_3::CLAUDE_3_OPUS_20240229;
        pub const CLAUDE_SONNET_3: &str = c::claude_sonnet_3::CLAUDE_3_SONNET_20240229;
        pub const CLAUDE_HAIKU_3: &str = c::claude_haiku_3::CLAUDE_3_HAIKU_20240307;
    }

    /// Gemini models with simplified access
    pub mod gemini {
        use crate::providers::gemini::model_constants as c;

        // Gemini 2.5 Pro (flagship)
        pub const GEMINI_2_5_PRO: &str = c::gemini_2_5_pro::GEMINI_2_5_PRO;

        // Gemini 2.5 Flash
        pub const GEMINI_2_5_FLASH: &str = c::gemini_2_5_flash::GEMINI_2_5_FLASH;

        // Gemini 2.5 Flash-Lite
        pub const GEMINI_2_5_FLASH_LITE: &str = c::gemini_2_5_flash_lite::GEMINI_2_5_FLASH_LITE;

        // Gemini 2.0 Flash
        pub const GEMINI_2_0_FLASH: &str = c::gemini_2_0_flash::GEMINI_2_0_FLASH;
        pub const GEMINI_2_0_FLASH_EXP: &str = c::gemini_2_0_flash::GEMINI_2_0_FLASH_EXP;

        // Gemini 2.0 Flash-Lite
        pub const GEMINI_2_0_FLASH_LITE: &str = c::gemini_2_0_flash_lite::GEMINI_2_0_FLASH_LITE;

        // Gemini 1.5 (legacy)
        pub const GEMINI_1_5_PRO: &str = c::gemini_1_5_pro::GEMINI_1_5_PRO;
        pub const GEMINI_1_5_FLASH: &str = c::gemini_1_5_flash::GEMINI_1_5_FLASH;
        pub const GEMINI_1_5_FLASH_8B: &str = c::gemini_1_5_flash_8b::GEMINI_1_5_FLASH_8B;

        // Live API models
        pub const GEMINI_LIVE_2_5_FLASH: &str =
            c::gemini_2_5_flash_live::GEMINI_LIVE_2_5_FLASH_PREVIEW;
        pub const GEMINI_LIVE_2_0_FLASH: &str = c::gemini_2_0_flash_live::GEMINI_2_0_FLASH_LIVE_001;
    }

    /// OpenAI-compatible provider models
    pub mod openai_compatible {
        use crate::providers::openai_compatible::providers::models as c;

        /// DeepSeek models
        pub mod deepseek {
            use super::c;

            pub const CHAT: &str = c::deepseek::CHAT;
            pub const REASONER: &str = c::deepseek::REASONER;
            pub const V3: &str = c::deepseek::DEEPSEEK_V3_0324;
            pub const R1: &str = c::deepseek::DEEPSEEK_R1_0528;
        }

        /// OpenRouter models (popular selections)
        pub mod openrouter {
            use super::c;

            pub const GPT_4O: &str = c::openrouter::openai::GPT_4O;
            pub const CLAUDE_OPUS_4_1: &str = c::openrouter::anthropic::CLAUDE_OPUS_4_1;
            pub const GEMINI_2_5_PRO: &str = c::openrouter::google::GEMINI_2_5_PRO;
            pub const DEEPSEEK_REASONER: &str = c::openrouter::deepseek::DEEPSEEK_REASONER;
            pub const LLAMA_3_1_405B: &str = c::openrouter::meta::LLAMA_3_1_405B;
        }
    }
}
