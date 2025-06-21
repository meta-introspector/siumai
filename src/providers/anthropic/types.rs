//! Anthropic-specific Data Types
//!
//! Contains data structures specific to the Anthropic Claude API.

use serde::{Deserialize, Serialize};

use super::cache::CacheControl;
use super::thinking::ThinkingConfig;

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

/// Anthropic Content Block according to official API documentation
/// <https://docs.anthropic.com/en/api/messages>
#[derive(Debug, Deserialize)]
pub struct AnthropicContentBlock {
    pub r#type: String,
    pub text: Option<String>,
    // Thinking-related fields
    pub thinking: Option<String>,
    pub signature: Option<String>,
    // Tool use fields
    pub id: Option<String>,
    pub name: Option<String>,
    pub input: Option<serde_json::Value>,
    // Tool result fields
    pub tool_use_id: Option<String>,
    pub content: Option<serde_json::Value>,
    pub is_error: Option<bool>,
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
    pub thinking_config: Option<ThinkingConfig>,
    /// Custom metadata
    pub metadata: Option<serde_json::Value>,
}

/// Anthropic Models List Response according to official API documentation
/// <https://docs.anthropic.com/en/api/models-list>
#[derive(Debug, Deserialize)]
pub struct AnthropicModelsResponse {
    pub data: Vec<AnthropicModelInfo>,
    pub first_id: Option<String>,
    pub has_more: bool,
    pub last_id: Option<String>,
}

/// Anthropic Model Information
#[derive(Debug, Deserialize)]
pub struct AnthropicModelInfo {
    pub id: String,
    pub display_name: String,
    pub created_at: String,
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
