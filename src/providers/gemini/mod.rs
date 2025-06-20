//! Google Gemini Provider
//!
//! This module provides integration with Google's Gemini AI models.
//!
//! # Architecture
//! - `client.rs` - Main Gemini client that aggregates all capabilities
//! - `types.rs` - Gemini-specific type definitions based on OpenAPI spec
//! - `chat.rs` - Chat completion capability implementation
//! - `models.rs` - Model listing capability implementation
//! - `code_execution.rs` - Code execution feature implementation
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::llm;
//!
//! let client = llm()
//!     .gemini()
//!     .api_key("your-api-key")
//!     .model("gemini-1.5-flash")
//!     .build()
//!     .await?;
//!
//! // Use chat capability
//! let messages = vec![user!("Hello, world!")];
//! let response = client.chat_with_tools(messages, None).await?;
//! ```

// Core modules
pub mod chat;
pub mod client;
pub mod models;
pub mod types;

// Feature modules
pub mod code_execution;

// Re-export main types for convenience
pub use chat::GeminiChatCapability;
pub use client::{GeminiBuilder, GeminiClient};
pub use models::GeminiModels;
pub use types::*;
