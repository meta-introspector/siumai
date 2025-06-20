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
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create an OpenAI client
//!     let client = llm()
//!         .openai()
//!         .api_key("your-api-key")
//!         .model("gpt-4")
//!         .temperature(0.7)
//!         .build()
//!         .await?;
//!
//!     // Send a chat request
//!     let request = ChatRequest::builder()
//!         .message(user!("Hello, world!"))
//!         .build();
//!
//!     let response = client.chat(request).await?;
//!     println!("Response: {}", response.text().unwrap_or(""));
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
    pub use crate::provider::*;
    pub use crate::provider_features::*;
    pub use crate::retry_strategy::*;
    pub use crate::stream::*;
    pub use crate::traits::*;
    pub use crate::types::*;
    pub use crate::web_search::*;
    pub use crate::{assistant, llm, provider, system, tool, user};
}

/// Global entry point function - creates an LLM builder
///
/// This is the main entry point for the siumai library. It creates a new
/// LlmBuilder that can be used to configure and create LLM clients.
///
/// # Example
/// ```rust,no_run
/// use siumai::llm;
///
/// let client = llm()
///     .openai()
///     .api_key("your-api-key")
///     .model("gpt-4")
///     .build()
///     .await?;
/// ```
pub fn llm() -> crate::builder::LlmBuilder {
    crate::builder::llm()
}

/// Siumai unified interface entry point
///
/// This creates a new SiumaiBuilder for building unified providers
/// that can work with multiple LLM providers through a single interface.
///
/// # Example
/// ```rust,no_run
/// use siumai::siumai;
///
/// // Future API (when fully implemented)
/// let provider = siumai()
///     .provider(ProviderType::OpenAi)
///     .api_key("your-api-key")
///     .model("gpt-4")
///     .with_audio()
///     .build()
///     .await?;
/// ```
pub fn siumai() -> crate::provider::SiumaiBuilder {
    crate::provider::ai()
}

// Convenient macro definitions
/// Creates a user message
///
/// For simple text messages, returns ChatMessage directly.
/// For messages with additional parameters, returns ChatMessageBuilder.
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
/// For simple text messages, returns ChatMessage directly.
/// For messages with additional parameters, returns ChatMessageBuilder.
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
/// For simple text messages, returns ChatMessage directly.
/// For messages with additional parameters, returns ChatMessageBuilder.
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
/// Returns ChatMessage directly since tool messages are typically simple.
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
    fn test_llm_builder() {
        let _builder = llm();
        // Basic test for builder creation
        assert!(true); // Placeholder test
    }
}
