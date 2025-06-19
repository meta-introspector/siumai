//! Anthropic-specific Data Types
//!
//! Contains data structures specific to the Anthropic Claude API.

use serde::{Deserialize, Serialize};

/// Anthropic Message format
#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,
}

/// Anthropic Chat Response
#[derive(Debug, Deserialize)]
pub struct AnthropicChatResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Option<AnthropicUsage>,
}

/// Anthropic Content Block
#[derive(Debug, Deserialize)]
pub struct AnthropicContentBlock {
    pub r#type: String,
    pub text: Option<String>,
}

/// Anthropic Usage
#[derive(Debug, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_creation_input_tokens: Option<u32>,
    pub cache_read_input_tokens: Option<u32>,
}

/// Anthropic-specific parameters
#[derive(Debug, Clone, Default)]
pub struct AnthropicSpecificParams {
    /// Beta features to enable
    pub beta_features: Vec<String>,
    /// Prompt caching configuration
    pub cache_control: Option<CacheControl>,
    /// Thinking mode configuration
    pub thinking_mode: Option<bool>,
    /// Custom metadata
    pub metadata: Option<serde_json::Value>,
}

/// Cache Control configuration for Anthropic
#[derive(Debug, Clone)]
pub struct CacheControl {
    /// Cache type (e.g., "ephemeral")
    pub r#type: String,
}

/// Tool Use block for Anthropic
#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicToolUse {
    pub r#type: String,
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Tool Result block for Anthropic
#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicToolResult {
    pub r#type: String,
    pub tool_use_id: String,
    pub content: String,
    pub is_error: Option<bool>,
}
