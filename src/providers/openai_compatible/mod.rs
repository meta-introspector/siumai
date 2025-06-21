//! OpenAI-Compatible Provider Interface
//!
//! This module provides a unified interface for OpenAI-compatible providers,
//! allowing seamless integration of providers like `DeepSeek`, `OpenRouter`, etc.
//!
//! # Design Principles
//! - **Type Safety**: Compile-time validation of provider configurations
//! - **Zero-Cost Abstractions**: No runtime overhead for provider selection
//! - **Ergonomic API**: Intuitive builder pattern with provider-specific methods
//! - **Extensibility**: Easy to add new compatible providers
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::prelude::*;
//! use siumai::providers::openai_compatible::{deepseek, openrouter, recommendations};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // DeepSeek with reasoning model using constants
//!     let deepseek = LlmBuilder::new()
//!         .deepseek()
//!         .api_key("your-api-key")
//!         .model(deepseek::REASONER)  // Using model constant
//!         .reasoning(true)?
//!         .build()
//!         .await?;
//!
//!     // OpenRouter with GPT-4 using constants
//!     let openrouter = LlmBuilder::new()
//!         .openrouter()
//!         .api_key("your-api-key")
//!         .model(openrouter::openai::GPT_4)  // Using model constant
//!         .site_url("https://myapp.com")?
//!         .app_name("My App")?
//!         .temperature(0.7)
//!         .build()
//!         .await?;
//!
//!     // Using recommendation helpers
//!     let coding_model = LlmBuilder::new()
//!         .deepseek()
//!         .api_key("your-api-key")
//!         .model(recommendations::for_coding())  // Gets deepseek::CODER
//!         .build()
//!         .await?;
//!
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod providers;
pub mod builder;

// Re-export main types
pub use config::*;
pub use providers::*;
pub use builder::*;

// Re-export model constants for easy access
pub use providers::models::{deepseek, openrouter, xai, groq};

use crate::traits::*;

// Implement capability traits for the OpenAI-compatible client
#[async_trait::async_trait]
impl<P: OpenAiCompatibleProvider> ChatCapability for OpenAiCompatibleClient<P> {
    async fn chat_with_tools(
        &self,
        messages: Vec<crate::types::ChatMessage>,
        tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, crate::error::LlmError> {
        self.client.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<crate::types::ChatMessage>,
        tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::stream::ChatStream, crate::error::LlmError> {
        self.client.chat_stream(messages, tools).await
    }
}

// Note: ChatCapability already includes streaming and tool functionality
// through chat_stream() and chat_with_tools() methods

impl<P: OpenAiCompatibleProvider> LlmProvider for OpenAiCompatibleClient<P> {
    fn provider_name(&self) -> &'static str {
        P::PROVIDER_ID
    }

    fn supported_models(&self) -> Vec<String> {
        // This could be provider-specific or delegated to the OpenAI client
        LlmProvider::supported_models(&self.client)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        P::supported_capabilities()
    }

    fn http_client(&self) -> &reqwest::Client {
        self.client.http_client()
    }
}
