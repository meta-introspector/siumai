//! `Groq` Streaming Implementation
//!
//! This module provides Groq-specific streaming functionality for chat completions.

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatRequest, Usage};
use crate::types::{ChatResponse, FinishReason, MessageContent};
use crate::utils::streaming::{SseEventConverter, StreamFactory};
use eventsource_stream::Event;
use std::future::Future;
use std::pin::Pin;

use super::config::GroqConfig;
use super::types::*;
use super::utils::*;

/// Groq event converter for SSE events
#[derive(Clone)]
pub struct GroqEventConverter {
    #[allow(dead_code)] // May be used for future configuration
    config: GroqConfig,
}

impl GroqEventConverter {
    /// Create a new Groq event converter
    pub fn new(config: GroqConfig) -> Self {
        Self { config }
    }

    /// Convert Groq stream response to ChatStreamEvent
    fn convert_groq_response(&self, response: GroqChatStreamChunk) -> ChatStreamEvent {
        // Handle usage information (final chunk)
        if let Some(usage) = response.usage {
            return ChatStreamEvent::UsageUpdate {
                usage: Usage {
                    prompt_tokens: usage.prompt_tokens.unwrap_or(0),
                    completion_tokens: usage.completion_tokens.unwrap_or(0),
                    total_tokens: usage.total_tokens.unwrap_or(0),
                    reasoning_tokens: None, // Groq doesn't provide reasoning tokens
                    cached_tokens: None,
                },
            };
        }

        // Process choices
        for choice in response.choices {
            let delta = choice.delta;

            // Handle finish reason (stream end)
            if let Some(finish_reason) = choice.finish_reason {
                let finish_reason_enum = match finish_reason.as_str() {
                    "stop" => FinishReason::Stop,
                    "length" => FinishReason::Length,
                    "tool_calls" => FinishReason::ToolCalls,
                    "content_filter" => FinishReason::ContentFilter,
                    _ => FinishReason::Stop,
                };

                return ChatStreamEvent::StreamEnd {
                    response: ChatResponse {
                        id: Some(response.id),
                        content: MessageContent::Text(String::new()),
                        model: Some(response.model),
                        usage: None,
                        finish_reason: Some(finish_reason_enum),
                        tool_calls: None,
                        thinking: None,
                        metadata: std::collections::HashMap::new(),
                    },
                };
            }

            // Handle content delta
            if let Some(content) = delta.content {
                return ChatStreamEvent::ContentDelta {
                    delta: content,
                    index: Some(choice.index as usize),
                };
            }

            // Handle tool calls
            if let Some(tool_calls) = delta.tool_calls {
                for tool_call in tool_calls {
                    if let Some(function) = tool_call.function
                        && let Some(arguments) = function.arguments
                    {
                        return ChatStreamEvent::ToolCallDelta {
                            index: tool_call.index.map(|i| i as usize),
                            id: tool_call.id.unwrap_or_default(),
                            function_name: function.name,
                            arguments_delta: Some(arguments),
                        };
                    }
                }
            }
        }

        // Default: empty content delta
        ChatStreamEvent::ContentDelta {
            delta: String::new(),
            index: None,
        }
    }
}

impl SseEventConverter for GroqEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Option<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            match serde_json::from_str::<GroqChatStreamChunk>(&event.data) {
                Ok(groq_response) => Some(Ok(self.convert_groq_response(groq_response))),
                Err(e) => Some(Err(LlmError::ParseError(format!(
                    "Failed to parse Groq event: {e}"
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
            metadata: std::collections::HashMap::new(),
        };
        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }
}

/// `Groq` streaming client
#[derive(Clone)]
pub struct GroqStreaming {
    /// `Groq` configuration
    config: GroqConfig,
    /// HTTP client
    http_client: reqwest::Client,
}

impl GroqStreaming {
    /// Create a new `Groq` streaming client
    pub fn new(config: GroqConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Create a streaming chat completion request
    pub async fn create_chat_stream(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let url = format!("{}/chat/completions", self.config.base_url);

        // Use the same request building logic as non-streaming
        let chat_capability = super::chat::GroqChatCapability::new(
            self.config.api_key.clone(),
            self.config.base_url.clone(),
            self.http_client.clone(),
            self.config.http_config.clone(),
            self.config.common_params.clone(),
        );

        let mut request_body = chat_capability.build_chat_request_body(&request)?;

        // Override with streaming-specific settings
        request_body["stream"] = serde_json::Value::Bool(true);
        request_body["stream_options"] = serde_json::json!({
            "include_usage": true
        });

        // Validate parameters
        validate_groq_params(&request_body)?;

        // Create headers
        let headers = build_headers(&self.config.api_key, &self.config.http_config.headers)?;

        // Create the stream using reqwest_eventsource for enhanced reliability
        let request_builder = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request_body);

        let converter = GroqEventConverter::new(self.config.clone());
        StreamFactory::create_eventsource_stream(request_builder, converter).await
    }
}
