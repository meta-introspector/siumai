//! # Siumai - A Unified LLM Interface Library
//!
//! Siumai is a unified LLM interface library for Rust, supporting multiple AI providers.
//! It adopts a trait-separated architectural pattern and provides a type-safe API.
//!
//! ## Features
//!
//! - **Capability Separation**: Uses traits to distinguish different AI capabilities (chat, audio, vision, etc.)
//! - **Shared Parameters**: AI parameters are shared as much as possible, with extension points for provider-specific parameters.
//! - **Builder Pattern**: Supports a builder pattern for chained method calls.
//! - **Type Safety**: Leverages Rust's type system to ensure compile-time safety.
//! - **HTTP Customization**: Supports passing in a reqwest client and custom HTTP configurations.
//! - **Library First**: Focuses on core library functionality, avoiding application-layer features.
//! - **Flexible Capability Access**: Capability checks serve as hints rather than restrictions, allowing users to try new model features.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create an OpenAI client
//!     let client = LlmBuilder::new()
//!         .openai()
//!         .api_key("your-api-key")
//!         .model("gpt-4")
//!         .temperature(0.7)
//!         .build()
//!         .await?;
//!
//!     // Send a chat request
//!     let messages = vec![user!("Hello, world!")];
//!     let response = client.chat(messages).await?;
//!     if let Some(text) = response.content_text() {
//!         println!("Response: {}", text);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Capability Access Philosophy
//!
//! Siumai takes a **permissive and quiet approach** to capability access. It never blocks operations
//! based on static capability information, and doesn't generate noise with automatic warnings.
//! The actual API determines what's supported:
//!
//! ```rust,no_run
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Siumai::builder()
//!         .openai()
//!         .api_key("your-api-key")
//!         .model("gpt-4o")  // This model supports vision
//!         .build()
//!         .await?;
//!
//!     // Get vision capability - this always works, regardless of "official" support
//!     let vision = client.vision_capability();
//!
//!     // Optionally check support status if you want to (no automatic warnings)
//!     if !vision.is_reported_as_supported() {
//!         // You can choose to show a warning, or just proceed silently
//!         println!("Note: Vision not officially supported, but trying anyway!");
//!     }
//!
//!     // The actual operation will succeed or fail based on the model's real capabilities
//!     // No pre-emptive blocking, no automatic noise
//!     // vision.analyze_image(...).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod benchmarks;
pub mod builder;
pub mod client;
pub mod custom_provider;
pub mod error;
pub mod error_handling;
pub mod multimodal;
pub mod params;
pub mod performance;
pub mod provider;
pub mod provider_features;
pub mod providers;
pub mod retry;
pub mod retry_strategy;
pub mod stream;
pub mod traits;
pub mod types;
pub mod web_search;

// Re-export main types and traits
pub use benchmarks::*;
pub use builder::*;
pub use client::*;
pub use custom_provider::*;
pub use error::LlmError;
pub use multimodal::*;
pub use performance::*;
pub use provider_features::*;
pub use retry_strategy::*;
pub use stream::*;
pub use traits::*;
pub use types::*;
pub use web_search::*;

/// Convenient pre-import module
pub mod prelude {
    pub use crate::benchmarks::*;
    pub use crate::builder::*;
    pub use crate::client::*;
    pub use crate::custom_provider::*;
    pub use crate::error::LlmError;
    pub use crate::multimodal::*;
    pub use crate::performance::*;
    pub use crate::provider::Siumai;
    pub use crate::provider::*;
    pub use crate::provider_features::*;
    pub use crate::retry_strategy::*;
    pub use crate::stream::*;
    pub use crate::traits::*;
    pub use crate::types::*;
    pub use crate::web_search::*;
    pub use crate::{Provider, assistant, provider, system, tool, user};
    pub use crate::{
        quick_anthropic, quick_anthropic_with_model, quick_gemini, quick_gemini_with_model,
        quick_ollama, quick_ollama_with_model, quick_openai, quick_openai_with_model,
    };
}

/// Provider entry point for creating specific provider clients
///
/// This is the main entry point for creating provider-specific clients.
/// Use this when you need access to provider-specific features and APIs.
///
/// # Example
/// ```rust,no_run
/// use siumai::prelude::*;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Get a client specifically for OpenAI
///     let openai_client = Provider::openai()
///         .api_key("your-openai-key")
///         .model("gpt-4")
///         .build()
///         .await?;
///
///     // You can now call both standard and OpenAI-specific methods
///     let messages = vec![user!("Hello!")];
///     let response = openai_client.chat(messages).await?;
///     // let assistant = openai_client.create_assistant(...).await?; // Example of specific feature
///
///     Ok(())
/// }
/// ```
pub struct Provider;

impl Provider {
    /// Create an `OpenAI` client builder
    pub fn openai() -> crate::builder::OpenAiBuilder {
        crate::builder::LlmBuilder::new().openai()
    }

    /// Create an Anthropic client builder
    pub fn anthropic() -> crate::builder::AnthropicBuilder {
        crate::builder::LlmBuilder::new().anthropic()
    }

    /// Create a Gemini client builder
    pub fn gemini() -> crate::builder::GeminiBuilder {
        crate::builder::LlmBuilder::new().gemini()
    }

    /// Create an xAI client builder
    pub fn xai() -> crate::builder::GenericProviderBuilder {
        crate::builder::LlmBuilder::new().xai()
    }

    /// Create an `OpenRouter` client builder
    pub fn openrouter() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder<
        crate::providers::openai_compatible::OpenRouterProvider,
    > {
        crate::builder::LlmBuilder::new().openrouter()
    }

    /// Create a `DeepSeek` client builder
    pub fn deepseek() -> crate::providers::openai_compatible::OpenAiCompatibleBuilder<
        crate::providers::openai_compatible::DeepSeekProvider,
    > {
        crate::builder::LlmBuilder::new().deepseek()
    }
}

/// Siumai unified interface entry point
///
/// This creates a unified client that can work with multiple LLM providers
/// through a single interface. Use this when you want provider-agnostic code
/// or need to switch between providers dynamically.
///
/// # Example
/// ```rust,no_run
/// use siumai::prelude::*;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Build a unified client, backed by Anthropic
///     let client = Siumai::builder()
///         .anthropic()
///         .api_key("your-anthropic-key")
///         .model("claude-3-sonnet-20240229")
///         .build()
///         .await?;
///
///     // Your code uses the standard Siumai interface
///     let messages = vec![user!("What is the capital of France?")];
///     let response = client.chat(messages).await?;
///
///     // If you decide to switch to OpenAI, you only change the builder.
///     // The `.chat(request)` call remains identical.
///
///     Ok(())
/// }
/// ```
impl crate::provider::Siumai {
    /// Create a new Siumai builder for unified interface
    pub fn builder() -> crate::provider::SiumaiBuilder {
        crate::provider::SiumaiBuilder::new()
    }
}

// Re-export convenience functions and builder
pub use crate::builder::{
    LlmBuilder, quick_anthropic, quick_anthropic_with_model, quick_gemini, quick_gemini_with_model,
    quick_openai, quick_openai_with_model,
};

// Convenient macro definitions
/// Creates a user message
///
/// For simple text messages, returns `ChatMessage` directly.
/// For messages with additional parameters, returns `ChatMessageBuilder`.
#[macro_export]
macro_rules! user {
    // Simple text message - returns ChatMessage directly
    ($content:expr) => {
        $crate::types::ChatMessage {
            role: $crate::types::MessageRole::User,
            content: $crate::types::MessageContent::Text($content.into()),
            metadata: $crate::types::MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    };
    // Message with cache control - returns ChatMessageBuilder for chaining
    ($content:expr, cache: $cache:expr) => {
        $crate::types::ChatMessage::user($content).cache_control($cache)
    };
}

/// Creates a user message builder for complex messages
///
/// Use this when you need to add images, cache control, or other complex features.
#[macro_export]
macro_rules! user_builder {
    ($content:expr) => {
        $crate::types::ChatMessage::user($content)
    };
}

/// Creates a system message
///
/// For simple text messages, returns `ChatMessage` directly.
/// For messages with additional parameters, returns `ChatMessageBuilder`.
#[macro_export]
macro_rules! system {
    // Simple text message - returns ChatMessage directly
    ($content:expr) => {
        $crate::types::ChatMessage {
            role: $crate::types::MessageRole::System,
            content: $crate::types::MessageContent::Text($content.into()),
            metadata: $crate::types::MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    };
    // Message with cache control - returns ChatMessageBuilder for chaining
    ($content:expr, cache: $cache:expr) => {
        $crate::types::ChatMessage::system($content).cache_control($cache)
    };
}

/// Creates an assistant message
///
/// For simple text messages, returns `ChatMessage` directly.
/// For messages with additional parameters, returns `ChatMessageBuilder`.
#[macro_export]
macro_rules! assistant {
    // Simple text message - returns ChatMessage directly
    ($content:expr) => {
        $crate::types::ChatMessage {
            role: $crate::types::MessageRole::Assistant,
            content: $crate::types::MessageContent::Text($content.into()),
            metadata: $crate::types::MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    };
    // Message with tool calls - returns ChatMessageBuilder for chaining
    ($content:expr, tools: $tools:expr) => {
        $crate::types::ChatMessage::assistant($content).with_tool_calls($tools)
    };
}

/// Creates a tool message
///
/// Returns `ChatMessage` directly since tool messages are typically simple.
#[macro_export]
macro_rules! tool {
    ($content:expr, id: $id:expr) => {
        $crate::types::ChatMessage {
            role: $crate::types::MessageRole::Tool,
            content: $crate::types::MessageContent::Text($content.into()),
            metadata: $crate::types::MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: Some($id.into()),
        }
    };
}

/// Multimodal user message macro
#[macro_export]
macro_rules! user_with_image {
    ($text:expr, $image_url:expr) => {
        $crate::types::ChatMessage::user($text).with_image($image_url.to_string(), None)
    };
    ($text:expr, $image_url:expr, detail: $detail:expr) => {
        $crate::types::ChatMessage::user($text)
            .with_image($image_url.to_string(), Some($detail.to_string()))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Siumai;

    #[test]
    fn test_macros() {
        // Test simple macros that return ChatMessage directly
        let user_msg = user!("Hello");
        assert_eq!(user_msg.role, MessageRole::User);

        let system_msg = system!("You are helpful");
        assert_eq!(system_msg.role, MessageRole::System);

        let assistant_msg = assistant!("I can help");
        assert_eq!(assistant_msg.role, MessageRole::Assistant);

        // Test that content is correctly set
        match user_msg.content {
            MessageContent::Text(text) => assert_eq!(text, "Hello"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_provider_builder() {
        let _openai_builder = Provider::openai();
        let _anthropic_builder = Provider::anthropic();
        let _siumai_builder = Siumai::builder();
        // Basic test for builder creation
        // Placeholder test
    }
}
