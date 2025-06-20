//! Anthropic Utility Functions
//!
//! Common utility functions for Anthropic Claude API interactions.

use super::types::*;
use crate::error::LlmError;
use crate::types::*;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};

/// Build HTTP headers for Anthropic API requests according to official documentation
/// https://docs.anthropic.com/en/api/messages
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

    // Set the Anthropic version (required)
    headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

    // Add beta features if needed (for thinking and other experimental features)
    let mut beta_features = Vec::new();

    // Check if thinking is being used (look for thinking-related headers)
    if custom_headers.contains_key("anthropic-beta") {
        if let Some(beta_value) = custom_headers.get("anthropic-beta") {
            beta_features.push(beta_value.clone());
        }
    } else {
        // Default beta features for extended thinking
        beta_features.push("thinking-2024-12-19".to_string());
    }

    if !beta_features.is_empty() {
        headers.insert(
            "anthropic-beta",
            HeaderValue::from_str(&beta_features.join(","))
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid beta header: {}", e)))?,
        );
    }

    // Add custom headers (excluding anthropic-beta which we handle above)
    for (key, value) in custom_headers {
        if key == "anthropic-beta" {
            continue; // Already handled above
        }
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
                    ContentPart::Image {
                        image_url,
                        detail: _,
                    } => {
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
                    ContentPart::Audio {
                        audio_url,
                        format: _,
                    } => {
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
            MessageRole::Developer => {
                // Developer messages are treated as system-level instructions in Anthropic
                // Since Anthropic handles system messages separately, we'll add it to the system message
                if let MessageContent::Text(text) = &message.content {
                    let developer_text = format!("Developer instructions: {}", text);
                    system_message = Some(match system_message {
                        Some(existing) => format!("{}\n\n{}", existing, developer_text),
                        None => developer_text,
                    });
                }
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

/// Parse Anthropic finish reason according to official API documentation
/// https://docs.anthropic.com/en/api/handling-stop-reasons
pub fn parse_finish_reason(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("end_turn") => Some(FinishReason::Stop),
        Some("max_tokens") => Some(FinishReason::Length),
        Some("tool_use") => Some(FinishReason::ToolCalls),
        Some("stop_sequence") => Some(FinishReason::Other("stop_sequence".to_string())),
        Some("pause_turn") => Some(FinishReason::Other("pause_turn".to_string())),
        Some("refusal") => Some(FinishReason::ContentFilter),
        Some(other) => Some(FinishReason::Other(other.to_string())),
        None => None,
    }
}

/// Get default models for Anthropic according to latest available models
pub fn get_default_models() -> Vec<String> {
    vec![
        // Claude 4 models (latest)
        "claude-sonnet-4-20250514".to_string(),
        "claude-opus-4-20250514".to_string(),
        // Claude 3.7 models
        "claude-3-7-sonnet-20250219".to_string(),
        // Claude 3.5 models
        "claude-3-5-sonnet-20241022".to_string(),
        "claude-3-5-sonnet-20240620".to_string(),
        "claude-3-5-haiku-20241022".to_string(),
        // Claude 3 models
        "claude-3-opus-20240229".to_string(),
        "claude-3-sonnet-20240229".to_string(),
        "claude-3-haiku-20240307".to_string(),
    ]
}

/// Parse Anthropic response content with support for thinking blocks
pub fn parse_response_content(content_blocks: &[AnthropicContentBlock]) -> MessageContent {
    // Find the first text block (skip thinking blocks for main content)
    for content_block in content_blocks {
        match content_block.r#type.as_str() {
            "text" => return MessageContent::Text(content_block.text.clone().unwrap_or_default()),
            _ => continue,
        }
    }
    MessageContent::Text(String::new())
}

/// Parse Anthropic response content and extract tool calls
pub fn parse_response_content_and_tools(content_blocks: &[AnthropicContentBlock]) -> (MessageContent, Option<Vec<crate::types::ToolCall>>) {
    let mut text_content = String::new();
    let mut tool_calls = Vec::new();

    for content_block in content_blocks {
        match content_block.r#type.as_str() {
            "text" => {
                if let Some(text) = &content_block.text {
                    if !text_content.is_empty() {
                        text_content.push('\n');
                    }
                    text_content.push_str(text);
                }
            }
            "tool_use" => {
                if let (Some(id), Some(name), Some(input)) = (&content_block.id, &content_block.name, &content_block.input) {
                    tool_calls.push(crate::types::ToolCall {
                        id: id.clone(),
                        r#type: "function".to_string(),
                        function: Some(crate::types::FunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        }),
                    });
                }
            }
            _ => continue,
        }
    }

    let content = if text_content.is_empty() {
        MessageContent::Text(String::new())
    } else {
        MessageContent::Text(text_content)
    };

    let tools = if tool_calls.is_empty() { None } else { Some(tool_calls) };
    (content, tools)
}

/// Extract thinking content from Anthropic response
pub fn extract_thinking_content(content_blocks: &[AnthropicContentBlock]) -> Option<String> {
    for content_block in content_blocks {
        if content_block.r#type == "thinking" {
            return content_block.thinking.clone();
        }
    }
    None
}

/// Create Anthropic usage from response
pub fn create_usage_from_response(usage: Option<AnthropicUsage>) -> Option<Usage> {
    usage.map(|u| Usage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: u.input_tokens + u.output_tokens,
        reasoning_tokens: None,
        cached_tokens: u.cache_read_input_tokens,
    })
}

/// Map Anthropic error types to LlmError according to official documentation
/// https://docs.anthropic.com/en/api/errors
pub fn map_anthropic_error(
    status_code: u16,
    error_type: &str,
    error_message: &str,
    error_details: serde_json::Value,
) -> LlmError {
    match error_type {
        "authentication_error" => LlmError::AuthenticationError(error_message.to_string()),
        "permission_error" => {
            LlmError::AuthenticationError(format!("Permission denied: {}", error_message))
        }
        "invalid_request_error" => LlmError::InvalidInput(error_message.to_string()),
        "not_found_error" => LlmError::NotFound(error_message.to_string()),
        "request_too_large" => {
            LlmError::InvalidInput(format!("Request too large: {}", error_message))
        }
        "rate_limit_error" => LlmError::RateLimitError(error_message.to_string()),
        "api_error" => LlmError::ProviderError {
            provider: "anthropic".to_string(),
            message: format!("Internal API error: {}", error_message),
            error_code: Some("api_error".to_string()),
        },
        "overloaded_error" => LlmError::ProviderError {
            provider: "anthropic".to_string(),
            message: format!("API temporarily overloaded: {}", error_message),
            error_code: Some("overloaded_error".to_string()),
        },
        _ => LlmError::ApiError {
            code: status_code,
            message: format!("Anthropic API error ({}): {}", error_type, error_message),
            details: Some(error_details),
        },
    }
}

/// Convert tools to Anthropic format
pub fn convert_tools_to_anthropic_format(tools: &[crate::types::Tool]) -> Result<Vec<serde_json::Value>, LlmError> {
    let mut anthropic_tools = Vec::new();

    for tool in tools {
        let anthropic_tool = serde_json::json!({
            "name": tool.function.name,
            "description": tool.function.description,
            "input_schema": tool.function.parameters
        });
        anthropic_tools.push(anthropic_tool);
    }

    Ok(anthropic_tools)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MessageContent;

    #[test]
    fn test_parse_response_content_and_tools() {
        let content_blocks = vec![
            AnthropicContentBlock {
                r#type: "text".to_string(),
                text: Some("I'll help you get the weather.".to_string()),
                thinking: None,
                signature: None,
                id: None,
                name: None,
                input: None,
                tool_use_id: None,
                content: None,
                is_error: None,
            },
            AnthropicContentBlock {
                r#type: "tool_use".to_string(),
                text: None,
                thinking: None,
                signature: None,
                id: Some("toolu_123".to_string()),
                name: Some("get_weather".to_string()),
                input: Some(serde_json::json!({"location": "San Francisco"})),
                tool_use_id: None,
                content: None,
                is_error: None,
            },
        ];

        let (content, tool_calls) = parse_response_content_and_tools(&content_blocks);

        // Check content
        match content {
            MessageContent::Text(text) => assert_eq!(text, "I'll help you get the weather."),
            _ => panic!("Expected text content"),
        }

        // Check tool calls
        assert!(tool_calls.is_some());
        let tools = tool_calls.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].id, "toolu_123");
        assert_eq!(tools[0].r#type, "function");
        assert!(tools[0].function.is_some());
        let function = tools[0].function.as_ref().unwrap();
        assert_eq!(function.name, "get_weather");
        assert_eq!(function.arguments, r#"{"location":"San Francisco"}"#);
    }

    #[test]
    fn test_parse_response_content_and_tools_text_only() {
        let content_blocks = vec![
            AnthropicContentBlock {
                r#type: "text".to_string(),
                text: Some("Hello world".to_string()),
                thinking: None,
                signature: None,
                id: None,
                name: None,
                input: None,
                tool_use_id: None,
                content: None,
                is_error: None,
            },
        ];

        let (content, tool_calls) = parse_response_content_and_tools(&content_blocks);

        // Check content
        match content {
            MessageContent::Text(text) => assert_eq!(text, "Hello world"),
            _ => panic!("Expected text content"),
        }

        // Check no tool calls
        assert!(tool_calls.is_none());
    }
}
