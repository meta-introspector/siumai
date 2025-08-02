//! Anthropic streaming implementation using eventsource-stream
//!
//! This module provides Anthropic streaming functionality using the
//! eventsource-stream infrastructure for reliable UTF-8 and SSE handling.

use crate::error::LlmError;
use crate::params::AnthropicParams;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatResponse, FinishReason, MessageContent, Usage};
use crate::utils::streaming::{SseEventConverter, StreamProcessor};
use eventsource_stream::Event;
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

/// Anthropic stream event structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct AnthropicStreamEvent {
    r#type: String,
    message: Option<AnthropicMessage>,
    delta: Option<AnthropicDelta>,
    usage: Option<AnthropicUsage>,
}

/// Anthropic message structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct AnthropicMessage {
    id: Option<String>,
    model: Option<String>,
    role: Option<String>,
    content: Option<Vec<AnthropicContent>>,
    stop_reason: Option<String>,
}

/// Anthropic content structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

/// Anthropic delta structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: String,
    text: Option<String>,
}

/// Anthropic usage structure
#[derive(Debug, Clone, Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

/// Anthropic event converter
#[derive(Clone)]
pub struct AnthropicEventConverter {
    #[allow(dead_code)]
    config: AnthropicParams,
}

impl AnthropicEventConverter {
    pub fn new(config: AnthropicParams) -> Self {
        Self { config }
    }

    /// Convert Anthropic stream event to ChatStreamEvent
    fn convert_anthropic_event(&self, event: AnthropicStreamEvent) -> Option<ChatStreamEvent> {
        match event.r#type.as_str() {
            "message_start" => {
                // Stream start event
                None // We don't emit stream start events for now
            }
            "content_block_delta" => {
                if let Some(delta) = event.delta {
                    if let Some(text) = delta.text {
                        return Some(ChatStreamEvent::ContentDelta {
                            delta: text,
                            index: None,
                        });
                    }
                }
                None
            }
            "message_delta" => {
                // Handle usage or finish reason
                if let Some(usage) = event.usage {
                    let usage_info = Usage {
                        prompt_tokens: usage.input_tokens.unwrap_or(0),
                        completion_tokens: usage.output_tokens.unwrap_or(0),
                        total_tokens: usage.input_tokens.unwrap_or(0)
                            + usage.output_tokens.unwrap_or(0),
                        cached_tokens: None,
                        reasoning_tokens: None,
                    };
                    return Some(ChatStreamEvent::UsageUpdate { usage: usage_info });
                }
                None
            }
            "message_stop" => {
                let response = ChatResponse {
                    id: None,
                    model: None,
                    content: MessageContent::Text("".to_string()),
                    usage: None,
                    finish_reason: Some(FinishReason::Stop),
                    tool_calls: None,
                    thinking: None,
                    metadata: HashMap::new(),
                };
                Some(ChatStreamEvent::StreamEnd { response })
            }
            _ => None,
        }
    }
}

impl SseEventConverter for AnthropicEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Option<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            match serde_json::from_str::<AnthropicStreamEvent>(&event.data) {
                Ok(anthropic_event) => self.convert_anthropic_event(anthropic_event).map(Ok),
                Err(e) => Some(Err(LlmError::ParseError(format!(
                    "Failed to parse Anthropic event: {e}"
                )))),
            }
        })
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        let response = ChatResponse {
            id: None,
            model: None,
            content: MessageContent::Text("".to_string()),
            usage: None,
            finish_reason: Some(FinishReason::Stop),
            tool_calls: None,
            thinking: None,
            metadata: HashMap::new(),
        };

        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }
}

/// Anthropic streaming client
#[derive(Clone)]
pub struct AnthropicStreaming {
    config: AnthropicParams,
    http_client: reqwest::Client,
}

impl AnthropicStreaming {
    /// Create a new Anthropic streaming client
    pub fn new(config: AnthropicParams, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Create a chat stream from ChatRequest
    pub async fn create_chat_stream(
        self,
        request: crate::types::ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        // Build request body
        let mut request_body = serde_json::json!({
            "model": request.common_params.model,
            "messages": self.convert_messages(&request.messages)?,
            "stream": true,
            "max_tokens": request.common_params.max_tokens.unwrap_or(1000)
        });

        // Add common parameters
        if let Some(temp) = request.common_params.temperature {
            request_body["temperature"] = temp.into();
        }
        if let Some(top_p) = request.common_params.top_p {
            request_body["top_p"] = top_p.into();
        }
        if let Some(stop) = &request.common_params.stop_sequences {
            request_body["stop_sequences"] = stop.clone().into();
        }

        // Add tools if provided
        if let Some(tools) = &request.tools {
            request_body["tools"] = self.convert_tools(tools)?;
        }

        // Create headers
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse().unwrap());
        headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
        // Note: API key should be provided via config

        // Make the request
        let response = self
            .http_client
            .post("https://api.anthropic.com/v1/messages")
            .headers(headers)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Anthropic API error {status}: {error_text}"),
                details: None,
            });
        }

        // Create the stream using our new infrastructure
        let converter = AnthropicEventConverter::new(self.config);
        StreamProcessor::create_sse_stream(response, converter).await
    }

    /// Convert messages to Anthropic format
    fn convert_messages(
        &self,
        messages: &[crate::types::ChatMessage],
    ) -> Result<serde_json::Value, LlmError> {
        let anthropic_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                serde_json::json!({
                    "role": format!("{:?}", msg.role).to_lowercase(),
                    "content": msg.content_text().unwrap_or("")
                })
            })
            .collect();

        Ok(serde_json::Value::Array(anthropic_messages))
    }

    /// Convert tools to Anthropic format
    fn convert_tools(&self, tools: &[crate::types::Tool]) -> Result<serde_json::Value, LlmError> {
        let anthropic_tools: Vec<serde_json::Value> = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "input_schema": tool.function.parameters
                })
            })
            .collect();

        Ok(serde_json::Value::Array(anthropic_tools))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::AnthropicParams;
    use eventsource_stream::Event;

    fn create_test_config() -> AnthropicParams {
        AnthropicParams::default()
    }

    #[tokio::test]
    async fn test_anthropic_streaming_conversion() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        // Test content delta conversion
        let event = Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        };

        let result = converter.convert_event(event).await;
        assert!(result.is_some());

        if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = result {
            assert_eq!(delta, "Hello");
        } else {
            panic!("Expected ContentDelta event");
        }
    }

    #[tokio::test]
    async fn test_anthropic_stream_end() {
        let config = create_test_config();
        let converter = AnthropicEventConverter::new(config);

        let result = converter.handle_stream_end();
        assert!(result.is_some());

        if let Some(Ok(ChatStreamEvent::StreamEnd { .. })) = result {
            // Success
        } else {
            panic!("Expected StreamEnd event");
        }
    }
}
