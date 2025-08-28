//! OpenAI-specific Data Types
//!
//! Contains data structures specific to the `OpenAI` API.

use serde::{Deserialize, Serialize};

/// `OpenAI` Message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    pub tool_call_id: Option<String>,
}

/// `OpenAI` Tool Call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Option<OpenAiFunction>,
}

/// `OpenAI` Function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunction {
    pub name: String,
    pub arguments: String,
}

/// `OpenAI` Chat Response
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
    pub usage: Option<OpenAiUsage>,
}

/// `OpenAI` Choice
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiChoice {
    pub index: u32,
    pub message: OpenAiMessage,
    pub finish_reason: Option<String>,
}

/// `OpenAI` Usage
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// `OpenAI` Model information
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiModel {
    pub id: String,
    pub object: String,
    pub created: Option<u64>,
    pub owned_by: String,
    pub permission: Option<Vec<serde_json::Value>>,
    pub root: Option<String>,
    pub parent: Option<String>,
}

/// `OpenAI` Models API response
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiModelsResponse {
    pub object: String,
    pub data: Vec<OpenAiModel>,
}

/// OpenAI-specific parameters
#[derive(Debug, Clone, Default)]
pub struct OpenAiSpecificParams {
    /// Organization ID
    pub organization: Option<String>,
    /// Project ID
    pub project: Option<String>,
    /// Response format for structured output
    pub response_format: Option<serde_json::Value>,
    /// Logit bias
    pub logit_bias: Option<serde_json::Value>,
    /// Whether to return logprobs
    pub logprobs: Option<bool>,
    /// Number of logprobs to return
    pub top_logprobs: Option<u32>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// User identifier
    pub user: Option<String>,
}
