//! `Groq` Utility Functions
//!
//! Utility functions for the Groq provider.

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::types::{ChatMessage, FinishReason, MessageContent, MessageRole};

/// Build HTTP headers for Groq API requests
pub fn build_headers(
    api_key: &str,
    custom_headers: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();

    // Authorization header
    let auth_value = format!("Bearer {api_key}");
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&auth_value)
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid API key format: {e}")))?,
    );

    // Content-Type header
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    // User-Agent header
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static("siumai/0.1.0 (groq-provider)"),
    );

    // Add custom headers
    for (key, value) in custom_headers {
        let header_name: reqwest::header::HeaderName = key.parse().map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid header name '{key}': {e}"))
        })?;
        let header_value = HeaderValue::from_str(value).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid header value for '{key}': {e}"))
        })?;
        headers.insert(header_name, header_value);
    }

    Ok(headers)
}

/// Convert internal messages to Groq API format
pub fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<serde_json::Value>, LlmError> {
    let mut groq_messages = Vec::new();

    for message in messages {
        let role = match message.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Developer => "system", // Map developer to system for Groq
            MessageRole::Tool => "tool",
        };

        let content = match &message.content {
            MessageContent::Text(text) => serde_json::Value::String(text.clone()),
            MessageContent::MultiModal(parts) => {
                let mut content_parts = Vec::new();
                for part in parts {
                    match part {
                        crate::types::ContentPart::Text { text } => {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": text
                            }));
                        }
                        crate::types::ContentPart::Image { image_url, detail } => {
                            let mut image_part = serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            });
                            if let Some(detail) = detail {
                                image_part["image_url"]["detail"] =
                                    serde_json::Value::String(detail.clone());
                            }
                            content_parts.push(image_part);
                        }
                        crate::types::ContentPart::Audio { audio_url, format } => {
                            content_parts.push(serde_json::json!({
                                "type": "audio",
                                "audio_url": audio_url,
                                "format": format
                            }));
                        }
                    }
                }
                serde_json::Value::Array(content_parts)
            }
        };

        let mut groq_message = serde_json::json!({
            "role": role,
            "content": content
        });

        // Add tool calls if present
        if let Some(ref tool_calls) = message.tool_calls {
            groq_message["tool_calls"] = serde_json::to_value(tool_calls)?;
        }

        // Add tool call ID if present
        if let Some(ref tool_call_id) = message.tool_call_id {
            groq_message["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
        }

        groq_messages.push(groq_message);
    }

    Ok(groq_messages)
}

/// Parse finish reason from Groq API response
pub fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("content_filter") => FinishReason::ContentFilter,
        Some("function_call") => FinishReason::ToolCalls, // Legacy function_call maps to tool_calls
        _ => FinishReason::Other("unknown".to_string()),
    }
}

/// Extract error message from Groq API error response
pub fn extract_error_message(error_text: &str) -> String {
    // Try to parse as JSON error response
    if let Ok(error_response) = serde_json::from_str::<super::types::GroqErrorResponse>(error_text)
    {
        return error_response.error.message;
    }

    // Fallback to raw error text
    error_text.to_string()
}

/// Validate Groq-specific parameters
pub fn validate_groq_params(params: &serde_json::Value) -> Result<(), LlmError> {
    // Validate frequency_penalty
    if let Some(freq_penalty) = params.get("frequency_penalty") {
        if let Some(value) = freq_penalty.as_f64() {
            if !(-2.0..=2.0).contains(&value) {
                return Err(LlmError::InvalidParameter(
                    "frequency_penalty must be between -2.0 and 2.0".to_string(),
                ));
            }
        }
    }

    // Validate presence_penalty
    if let Some(pres_penalty) = params.get("presence_penalty") {
        if let Some(value) = pres_penalty.as_f64() {
            if !(-2.0..=2.0).contains(&value) {
                return Err(LlmError::InvalidParameter(
                    "presence_penalty must be between -2.0 and 2.0".to_string(),
                ));
            }
        }
    }

    // Validate temperature
    if let Some(temperature) = params.get("temperature") {
        if let Some(value) = temperature.as_f64() {
            if !(0.0..=2.0).contains(&value) {
                return Err(LlmError::InvalidParameter(
                    "temperature must be between 0.0 and 2.0".to_string(),
                ));
            }
        }
    }

    // Validate top_p
    if let Some(top_p) = params.get("top_p") {
        if let Some(value) = top_p.as_f64() {
            if !(0.0..=1.0).contains(&value) {
                return Err(LlmError::InvalidParameter(
                    "top_p must be between 0.0 and 1.0".to_string(),
                ));
            }
        }
    }

    // Validate n (number of choices)
    if let Some(n) = params.get("n") {
        if let Some(value) = n.as_u64() {
            if value != 1 {
                return Err(LlmError::InvalidParameter(
                    "Groq only supports n=1".to_string(),
                ));
            }
        }
    }

    // Validate service_tier
    if let Some(service_tier) = params.get("service_tier") {
        if let Some(value) = service_tier.as_str() {
            if !["auto", "on_demand", "flex"].contains(&value) {
                return Err(LlmError::InvalidParameter(
                    "service_tier must be one of: auto, on_demand, flex".to_string(),
                ));
            }
        }
    }

    // Validate reasoning_effort (for qwen3 models)
    if let Some(reasoning_effort) = params.get("reasoning_effort") {
        if let Some(value) = reasoning_effort.as_str() {
            if !["none", "default"].contains(&value) {
                return Err(LlmError::InvalidParameter(
                    "reasoning_effort must be one of: none, default".to_string(),
                ));
            }
        }
    }

    // Validate reasoning_format
    if let Some(reasoning_format) = params.get("reasoning_format") {
        if let Some(value) = reasoning_format.as_str() {
            if !["hidden", "raw", "parsed"].contains(&value) {
                return Err(LlmError::InvalidParameter(
                    "reasoning_format must be one of: hidden, raw, parsed".to_string(),
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    #[test]
    fn test_build_headers() {
        let custom_headers = HashMap::new();
        let headers = build_headers("test-api-key", &custom_headers).unwrap();

        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer test-api-key");
        assert_eq!(headers.get(CONTENT_TYPE).unwrap(), "application/json");
        assert!(headers.get(USER_AGENT).is_some());
    }

    #[test]
    fn test_convert_messages() {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant").build(),
            ChatMessage::user("Hello, world!").build(),
        ];

        let groq_messages = convert_messages(&messages).unwrap();
        assert_eq!(groq_messages.len(), 2);
        assert_eq!(groq_messages[0]["role"], "system");
        assert_eq!(groq_messages[1]["role"], "user");
    }

    #[test]
    fn test_parse_finish_reason() {
        assert_eq!(parse_finish_reason(Some("stop")), FinishReason::Stop);
        assert_eq!(parse_finish_reason(Some("length")), FinishReason::Length);
        assert_eq!(
            parse_finish_reason(Some("tool_calls")),
            FinishReason::ToolCalls
        );
        assert_eq!(
            parse_finish_reason(Some("unknown")),
            FinishReason::Other("unknown".to_string())
        );
        assert_eq!(
            parse_finish_reason(None),
            FinishReason::Other("unknown".to_string())
        );
    }

    #[test]
    fn test_validate_groq_params() {
        // Valid parameters
        let valid_params = serde_json::json!({
            "temperature": 0.7,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
            "service_tier": "auto"
        });
        assert!(validate_groq_params(&valid_params).is_ok());

        // Invalid temperature
        let invalid_temp = serde_json::json!({
            "temperature": 3.0
        });
        assert!(validate_groq_params(&invalid_temp).is_err());

        // Invalid service_tier
        let invalid_tier = serde_json::json!({
            "service_tier": "invalid"
        });
        assert!(validate_groq_params(&invalid_tier).is_err());
    }
}
