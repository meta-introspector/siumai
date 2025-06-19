//! Anthropic Utility Functions
//!
//! Common utility functions for Anthropic Claude API interactions.

use crate::error::LlmError;
use crate::types::*;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use super::types::*;

/// Build HTTP headers for Anthropic API requests
pub fn build_headers(
    api_key: &str,
    custom_headers: &std::collections::HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();

    // Set the authentication header
    headers.insert(
        "x-api-key",
        HeaderValue::from_str(api_key)
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid API key: {}", e)))?,
    );

    // Set the content type
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    // Set the Anthropic version
    headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

    // Add custom headers
    for (key, value) in custom_headers {
        let header_name: reqwest::header::HeaderName = key.parse().map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid header key '{}': {}", key, e))
        })?;
        headers.insert(
            header_name,
            HeaderValue::from_str(value).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid header value '{}': {}", value, e))
            })?,
        );
    }

    Ok(headers)
}

/// Convert message content to Anthropic format
pub fn convert_message_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
    match content {
        MessageContent::Text(text) => Ok(serde_json::Value::String(text.clone())),
        MessageContent::MultiModal(parts) => {
            let mut content_parts = Vec::new();

            for part in parts {
                match part {
                    ContentPart::Text { text } => {
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": text
                        }));
                    }
                    ContentPart::Image { image_url, detail: _ } => {
                        // Anthropic uses a different image format
                        content_parts.push(serde_json::json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_url // Base64 encoding needs to be handled here
                            }
                        }));
                    }
                    ContentPart::Audio { audio_url, format: _ } => {
                        // Anthropic does not currently support audio, treating it as text
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": format!("[Audio: {}]", audio_url)
                        }));
                    }
                }
            }

            Ok(serde_json::Value::Array(content_parts))
        }
    }
}

/// Convert messages to Anthropic format
pub fn convert_messages(
    messages: &[ChatMessage],
) -> Result<(Vec<AnthropicMessage>, Option<String>), LlmError> {
    let mut anthropic_messages = Vec::new();
    let mut system_message = None;

    for message in messages {
        match message.role {
            MessageRole::System => {
                // Anthropic handles system messages separately
                if let MessageContent::Text(text) = &message.content {
                    system_message = Some(text.clone());
                }
            }
            MessageRole::User => {
                anthropic_messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: convert_message_content(&message.content)?,
                });
            }
            MessageRole::Assistant => {
                anthropic_messages.push(AnthropicMessage {
                    role: "assistant".to_string(),
                    content: convert_message_content(&message.content)?,
                });
            }
            MessageRole::Tool => {
                // Handling of tool results for Anthropic
                anthropic_messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: convert_message_content(&message.content)?,
                });
            }
        }
    }

    Ok((anthropic_messages, system_message))
}

/// Parse Anthropic finish reason
pub fn parse_finish_reason(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("end_turn") => Some(FinishReason::Stop),
        Some("max_tokens") => Some(FinishReason::Length),
        Some("tool_use") => Some(FinishReason::ToolCalls),
        Some(other) => Some(FinishReason::Other(other.to_string())),
        None => None,
    }
}

/// Get default models for Anthropic
pub fn get_default_models() -> Vec<String> {
    vec![
        "claude-3-5-sonnet-20241022".to_string(),
        "claude-3-5-sonnet-20240620".to_string(),
        "claude-3-5-haiku-20241022".to_string(),
        "claude-3-opus-20240229".to_string(),
        "claude-3-sonnet-20240229".to_string(),
        "claude-3-haiku-20240307".to_string(),
    ]
}

/// Parse Anthropic response content
pub fn parse_response_content(content_blocks: &[AnthropicContentBlock]) -> MessageContent {
    match content_blocks.first() {
        Some(content_block) => match content_block.r#type.as_str() {
            "text" => MessageContent::Text(content_block.text.clone().unwrap_or_default()),
            _ => MessageContent::Text(String::new()),
        },
        None => MessageContent::Text(String::new()),
    }
}

/// Create Anthropic usage from response
pub fn create_usage_from_response(usage: Option<AnthropicUsage>) -> Option<Usage> {
    usage.map(|u| Usage {
        prompt_tokens: Some(u.input_tokens),
        completion_tokens: Some(u.output_tokens),
        total_tokens: Some(u.input_tokens + u.output_tokens),
        reasoning_tokens: None,
        cache_hit_tokens: u.cache_read_input_tokens,
        cache_creation_tokens: u.cache_creation_input_tokens,
    })
}
