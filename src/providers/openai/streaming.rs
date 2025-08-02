//! OpenAI streaming implementation using eventsource-stream
//!
//! This module provides OpenAI streaming functionality using the unified
//! eventsource-stream infrastructure for reliable UTF-8 and SSE handling.

use crate::error::LlmError;
use crate::providers::openai::config::OpenAiConfig;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatResponse, FinishReason, MessageContent, Usage};
use crate::utils::streaming::{SseEventConverter, StreamProcessor};
use eventsource_stream::Event;
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

/// OpenAI stream event structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAiStreamEvent {
    id: Option<String>,
    model: Option<String>,
    choices: Option<Vec<OpenAiStreamChoice>>,
    usage: Option<OpenAiStreamUsage>,
}

/// OpenAI stream choice
#[derive(Debug, Clone, Deserialize)]
struct OpenAiStreamChoice {
    index: Option<usize>,
    delta: Option<OpenAiStreamDelta>,
    finish_reason: Option<String>,
}

/// OpenAI stream delta
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OpenAiStreamDelta {
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
    thinking: Option<String>,
}

/// OpenAI tool call delta
#[derive(Debug, Clone, Deserialize)]
struct OpenAiToolCallDelta {
    index: Option<usize>,
    id: Option<String>,
    function: Option<OpenAiFunctionCallDelta>,
}

/// OpenAI function call delta
#[derive(Debug, Clone, Deserialize)]
struct OpenAiFunctionCallDelta {
    name: Option<String>,
    arguments: Option<String>,
}

/// OpenAI usage information
#[derive(Debug, Clone, Deserialize)]
struct OpenAiStreamUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
    completion_tokens_details: Option<OpenAiCompletionTokensDetails>,
    prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

/// OpenAI completion tokens details
#[derive(Debug, Clone, Deserialize)]
struct OpenAiCompletionTokensDetails {
    reasoning_tokens: Option<u32>,
}

/// OpenAI prompt tokens details
#[derive(Debug, Clone, Deserialize)]
struct OpenAiPromptTokensDetails {
    cached_tokens: Option<u32>,
}

/// OpenAI event converter
#[derive(Clone)]
pub struct OpenAiEventConverter {
    #[allow(dead_code)]
    config: OpenAiConfig,
}

impl OpenAiEventConverter {
    pub fn new(config: OpenAiConfig) -> Self {
        Self { config }
    }

    /// Convert OpenAI stream event to ChatStreamEvent
    fn convert_openai_event(&self, event: OpenAiStreamEvent) -> ChatStreamEvent {
        // Handle usage information
        if let Some(usage) = event.usage {
            let usage_info = Usage {
                prompt_tokens: usage.prompt_tokens.unwrap_or(0),
                completion_tokens: usage.completion_tokens.unwrap_or(0),
                total_tokens: usage.total_tokens.unwrap_or(0),
                cached_tokens: usage
                    .prompt_tokens_details
                    .and_then(|details| details.cached_tokens),
                reasoning_tokens: usage
                    .completion_tokens_details
                    .and_then(|details| details.reasoning_tokens),
            };
            return ChatStreamEvent::UsageUpdate { usage: usage_info };
        }

        // Handle choices
        if let Some(choices) = event.choices {
            for choice in choices {
                if let Some(delta) = choice.delta {
                    // Handle content delta
                    if let Some(content) = delta.content {
                        return ChatStreamEvent::ContentDelta {
                            delta: content,
                            index: choice.index,
                        };
                    }

                    // Handle thinking content (for reasoning models)
                    if let Some(thinking) = delta.thinking {
                        return ChatStreamEvent::ThinkingDelta { delta: thinking };
                    }

                    // Handle tool calls
                    if let Some(tool_calls) = delta.tool_calls {
                        for tool_call in tool_calls {
                            if let Some(function) = tool_call.function {
                                if let Some(name) = function.name {
                                    return ChatStreamEvent::ToolCallDelta {
                                        id: tool_call.id.unwrap_or_default(),
                                        function_name: Some(name),
                                        arguments_delta: None,
                                        index: tool_call.index,
                                    };
                                }
                                if let Some(arguments) = function.arguments {
                                    return ChatStreamEvent::ToolCallDelta {
                                        id: tool_call.id.unwrap_or_default(),
                                        function_name: None,
                                        arguments_delta: Some(arguments),
                                        index: tool_call.index,
                                    };
                                }
                            }
                        }
                    }
                }

                // Handle finish reason
                if let Some(finish_reason) = choice.finish_reason {
                    let reason = match finish_reason.as_str() {
                        "stop" => FinishReason::Stop,
                        "length" => FinishReason::Length,
                        "tool_calls" => FinishReason::ToolCalls,
                        "content_filter" => FinishReason::ContentFilter,
                        _ => FinishReason::Other(finish_reason),
                    };

                    let response = ChatResponse {
                        id: event.id,
                        model: event.model,
                        content: MessageContent::Text("".to_string()),
                        usage: None,
                        finish_reason: Some(reason),
                        tool_calls: None,
                        thinking: None,
                        metadata: HashMap::new(),
                    };

                    return ChatStreamEvent::StreamEnd { response };
                }
            }
        }

        // Default: empty content delta
        ChatStreamEvent::ContentDelta {
            delta: "".to_string(),
            index: None,
        }
    }
}

impl SseEventConverter for OpenAiEventConverter {
    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<Box<dyn Future<Output = Option<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            match serde_json::from_str::<OpenAiStreamEvent>(&event.data) {
                Ok(openai_event) => Some(Ok(self.convert_openai_event(openai_event))),
                Err(e) => Some(Err(LlmError::ParseError(format!(
                    "Failed to parse OpenAI event: {e}"
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

/// OpenAI streaming client
#[derive(Clone)]
pub struct OpenAiStreaming {
    config: OpenAiConfig,
    http_client: reqwest::Client,
}

impl OpenAiStreaming {
    /// Create a new OpenAI streaming client
    pub fn new(config: OpenAiConfig, http_client: reqwest::Client) -> Self {
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
        let url = format!("{}/chat/completions", self.config.base_url);

        // Build request body
        let mut request_body = serde_json::json!({
            "model": request.common_params.model,
            "messages": self.convert_messages(&request.messages)?,
            "stream": true,
            "stream_options": {
                "include_usage": true
            }
        });

        // Add common parameters
        if let Some(temp) = request.common_params.temperature {
            request_body["temperature"] = temp.into();
        }
        if let Some(max_tokens) = request.common_params.max_tokens {
            request_body["max_tokens"] = max_tokens.into();
        }
        if let Some(top_p) = request.common_params.top_p {
            request_body["top_p"] = top_p.into();
        }
        if let Some(stop) = &request.common_params.stop_sequences {
            request_body["stop"] = stop.clone().into();
        }
        if let Some(seed) = request.common_params.seed {
            request_body["seed"] = seed.into();
        }

        // Add tools if provided
        if let Some(tools) = &request.tools {
            request_body["tools"] = self.convert_tools(tools)?;
        }

        // Create headers
        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.get_headers() {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
            headers.insert(header_name, header_value);
        }

        // Make the request
        let response = self
            .http_client
            .post(&url)
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
                message: format!("OpenAI API error {status}: {error_text}"),
                details: None,
            });
        }

        // Create the stream using our new infrastructure
        let converter = OpenAiEventConverter::new(self.config);
        StreamProcessor::create_sse_stream(response, converter).await
    }

    /// Convert messages to OpenAI format
    fn convert_messages(
        &self,
        messages: &[crate::types::ChatMessage],
    ) -> Result<serde_json::Value, LlmError> {
        let openai_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                serde_json::json!({
                    "role": format!("{:?}", msg.role).to_lowercase(),
                    "content": msg.content_text().unwrap_or("")
                })
            })
            .collect();

        Ok(serde_json::Value::Array(openai_messages))
    }

    /// Convert tools to OpenAI format
    fn convert_tools(&self, tools: &[crate::types::Tool]) -> Result<serde_json::Value, LlmError> {
        let openai_tools: Vec<serde_json::Value> = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": tool.r#type,
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters
                    }
                })
            })
            .collect();

        Ok(serde_json::Value::Array(openai_tools))
    }
}
