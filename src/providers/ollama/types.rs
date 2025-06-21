//! Ollama-specific type definitions
//!
//! This module contains type definitions specific to the Ollama API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ollama chat request
#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatRequest {
    /// Model name
    pub model: String,
    /// Messages in the conversation
    pub messages: Vec<OllamaChatMessage>,
    /// Tools available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OllamaTool>>,
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Output format (json or schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    /// Additional model options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    /// Keep model loaded duration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    /// Should the model think before responding (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
}

/// Ollama generate request
#[derive(Debug, Clone, Serialize)]
pub struct OllamaGenerateRequest {
    /// Model name
    pub model: String,
    /// Prompt text
    pub prompt: String,
    /// Suffix text (for completion)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    /// Images for multimodal models (base64 encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Output format (json or schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    /// Additional model options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    /// System message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Prompt template
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// Raw mode (bypass templating)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,
    /// Keep model loaded duration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    /// Context from previous request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
    /// Should the model think before responding (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
}

/// Ollama chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatMessage {
    /// Role of the message sender
    pub role: String,
    /// Content of the message
    pub content: String,
    /// Images for multimodal models (base64 encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    /// Tool calls made by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
    /// The model's thinking process (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

/// Ollama tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTool {
    /// Type of tool (always "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition
    pub function: OllamaFunction,
}

/// Ollama function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunction {
    /// Function name
    pub name: String,
    /// Function description
    pub description: String,
    /// Function parameters schema
    pub parameters: serde_json::Value,
}

/// Ollama tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall {
    /// Function being called
    pub function: OllamaFunctionCall,
}

/// Ollama function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunctionCall {
    /// Function name
    pub name: String,
    /// Function arguments
    pub arguments: serde_json::Value,
}

/// Ollama chat response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaChatResponse {
    /// Model used
    pub model: String,
    /// Creation timestamp
    pub created_at: String,
    /// Response message
    pub message: OllamaChatMessage,
    /// Whether the response is complete
    pub done: bool,
    /// Reason for completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    /// Total duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    /// Load duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    /// Prompt evaluation count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    /// Prompt evaluation duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    /// Evaluation count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    /// Evaluation duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

/// Ollama generate response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaGenerateResponse {
    /// Model used
    pub model: String,
    /// Creation timestamp
    pub created_at: String,
    /// Generated response text
    pub response: String,
    /// Whether the response is complete
    pub done: bool,
    /// Context for next request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
    /// Total duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    /// Load duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    /// Prompt evaluation count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    /// Prompt evaluation duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    /// Evaluation count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    /// Evaluation duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

/// Ollama embedding request
#[derive(Debug, Clone, Serialize)]
pub struct OllamaEmbeddingRequest {
    /// Model name
    pub model: String,
    /// Input text or list of texts
    pub input: serde_json::Value,
    /// Truncate input to fit context length
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    /// Additional model options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    /// Keep model loaded duration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

/// Ollama embedding response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaEmbeddingResponse {
    /// Model used
    pub model: String,
    /// Generated embeddings
    pub embeddings: Vec<Vec<f64>>,
    /// Total duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    /// Load duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    /// Prompt evaluation count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
}

/// Ollama model information
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModel {
    /// Model name
    pub name: String,
    /// Model identifier
    pub model: String,
    /// Last modified timestamp
    pub modified_at: String,
    /// Model size in bytes
    pub size: u64,
    /// Model digest
    pub digest: String,
    /// Model details
    pub details: OllamaModelDetails,
}

/// Ollama model details
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelDetails {
    /// Parent model
    pub parent_model: String,
    /// Model format
    pub format: String,
    /// Model family
    pub family: String,
    /// Model families
    pub families: Vec<String>,
    /// Parameter size
    pub parameter_size: String,
    /// Quantization level
    pub quantization_level: String,
}

/// Ollama models list response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelsResponse {
    /// List of models
    pub models: Vec<OllamaModel>,
}

/// Ollama running models response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaRunningModelsResponse {
    /// List of running models
    pub models: Vec<OllamaRunningModel>,
}

/// Ollama running model information
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaRunningModel {
    /// Model name
    pub name: String,
    /// Model identifier
    pub model: String,
    /// Model size in bytes
    pub size: u64,
    /// Model digest
    pub digest: String,
    /// Model details
    pub details: OllamaModelDetails,
    /// Expiration time
    pub expires_at: String,
    /// VRAM size in bytes
    pub size_vram: u64,
}

/// Ollama version response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaVersionResponse {
    /// Ollama version
    pub version: String,
}
