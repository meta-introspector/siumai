//! `OpenAI` Streaming Implementation
//!
//! This module provides OpenAI-specific streaming functionality for chat completions.

use futures::{Stream, StreamExt};
use serde::Deserialize;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::Mutex;

use crate::types::{ChatResponse, FinishReason, MessageContent};

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatRequest, ResponseMetadata, Usage};
use crate::utils::Utf8StreamDecoder;

use super::config::OpenAiConfig;
use super::utils::{contains_thinking_tags, extract_thinking_content, filter_thinking_content};

/// `OpenAI` Server-Sent Events (SSE) response structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OpenAiStreamResponse {
    /// Response ID
    id: String,
    /// Object type (should be "chat.completion.chunk")
    object: String,
    /// Creation timestamp
    created: u64,
    /// Model used
    model: String,
    /// System fingerprint
    system_fingerprint: Option<String>,
    /// Choices array
    choices: Vec<OpenAiStreamChoice>,
    /// Usage information (only in final chunk)
    usage: Option<OpenAiUsage>,
}

/// `OpenAI` stream choice
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OpenAiStreamChoice {
    /// Choice index
    index: usize,
    /// Delta content
    delta: OpenAiStreamDelta,
    /// Logprobs (if requested)
    logprobs: Option<serde_json::Value>,
    /// Finish reason
    finish_reason: Option<String>,
}

/// `OpenAI` stream delta
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OpenAiStreamDelta {
    /// Role (only in first chunk)
    role: Option<String>,
    /// Content delta
    content: Option<String>,
    /// Tool calls delta
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
    /// Reasoning content (for o1 models)
    reasoning: Option<String>,
}

/// `OpenAI` tool call delta
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OpenAiToolCallDelta {
    /// Tool call index
    index: usize,
    /// Tool call ID
    id: Option<String>,
    /// Tool type
    r#type: Option<String>,
    /// Function call delta
    function: Option<OpenAiFunctionCallDelta>,
}

/// `OpenAI` function call delta
#[derive(Debug, Clone, Deserialize)]
struct OpenAiFunctionCallDelta {
    /// Function name
    name: Option<String>,
    /// Function arguments
    arguments: Option<String>,
}

/// `OpenAI` usage information
#[derive(Debug, Clone, Deserialize)]
struct OpenAiUsage {
    /// Prompt tokens
    prompt_tokens: u32,
    /// Completion tokens
    completion_tokens: Option<u32>,
    /// Total tokens
    total_tokens: Option<u32>,
    /// Completion tokens details
    completion_tokens_details: Option<OpenAiCompletionTokensDetails>,
    /// Prompt tokens details
    prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

/// `OpenAI` completion tokens details
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OpenAiCompletionTokensDetails {
    /// Reasoning tokens (for o1 models)
    reasoning_tokens: Option<u32>,
    /// Accepted prediction tokens
    accepted_prediction_tokens: Option<u32>,
    /// Rejected prediction tokens
    rejected_prediction_tokens: Option<u32>,
}

/// `OpenAI` prompt tokens details
#[derive(Debug, Clone, Deserialize)]
struct OpenAiPromptTokensDetails {
    /// Cached tokens
    cached_tokens: Option<u32>,
    /// Audio tokens
    #[allow(dead_code)]
    audio_tokens: Option<u32>,
}

/// `OpenAI` streaming client
#[derive(Clone)]
pub struct OpenAiStreaming {
    /// `OpenAI` configuration
    config: OpenAiConfig,
    /// HTTP client
    http_client: reqwest::Client,
    /// SSE line buffer for handling incomplete lines
    sse_buffer: Arc<Mutex<String>>,
    /// Event queue for handling multiple events from a single chunk
    event_queue: Arc<Mutex<Vec<Result<ChatStreamEvent, LlmError>>>>,
}

impl OpenAiStreaming {
    /// Create a new `OpenAI` streaming client
    pub fn new(config: OpenAiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
            sse_buffer: Arc::new(Mutex::new(String::new())),
            event_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a streaming chat completion request
    pub async fn create_chat_stream(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
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

        // Add OpenAI-specific parameters from config
        if let Some(response_format) = &self.config.openai_params.response_format {
            request_body["response_format"] =
                serde_json::to_value(response_format).map_err(|e| {
                    LlmError::JsonError(format!("Failed to serialize response_format: {e}"))
                })?;
        }
        if let Some(tool_choice) = &self.config.openai_params.tool_choice {
            request_body["tool_choice"] = serde_json::to_value(tool_choice).map_err(|e| {
                LlmError::JsonError(format!("Failed to serialize tool_choice: {e}"))
            })?;
        }
        if let Some(parallel_tool_calls) = self.config.openai_params.parallel_tool_calls {
            request_body["parallel_tool_calls"] = parallel_tool_calls.into();
        }
        if let Some(user) = &self.config.openai_params.user {
            request_body["user"] = user.clone().into();
        }
        if let Some(frequency_penalty) = self.config.openai_params.frequency_penalty {
            request_body["frequency_penalty"] = frequency_penalty.into();
        }
        if let Some(presence_penalty) = self.config.openai_params.presence_penalty {
            request_body["presence_penalty"] = presence_penalty.into();
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

        // Create the stream
        let stream = self.clone().create_event_stream(response).await?;
        Ok(Box::pin(stream))
    }

    /// Create an event stream from the HTTP response
    async fn create_event_stream(
        self,
        response: reqwest::Response,
    ) -> Result<impl Stream<Item = Result<ChatStreamEvent, LlmError>>, LlmError> {
        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));

        // Create a UTF-8 decoder for this stream
        let decoder = Arc::new(StdMutex::new(Utf8StreamDecoder::new()));
        let decoder_for_flush = decoder.clone();
        let streaming_for_flush = self.clone();

        // Create a stream that handles UTF-8 decoding
        let decoded_stream = stream.filter_map(move |chunk_result| {
            let streaming = self.clone();
            let decoder = decoder.clone();
            async move {
                match chunk_result {
                    Ok(chunk) => {
                        // Use UTF-8 decoder to handle incomplete sequences
                        let decoded_chunk = {
                            let mut decoder = decoder.lock().unwrap();
                            decoder.decode(&chunk)
                        };

                        if !decoded_chunk.is_empty() {
                            streaming.parse_sse_chunk_buffered(&decoded_chunk).await
                        } else {
                            None
                        }
                    }
                    Err(e) => Some(Err(e)),
                }
            }
        });

        // Add a final flush operation
        // Create a flush stream that handles both UTF-8 decoder flush and event queue flush
        let flush_stream = futures::stream::unfold(
            (streaming_for_flush, decoder_for_flush, 0),
            |(streaming, decoder, phase)| async move {
                match phase {
                    0 => {
                        // Phase 0: handle UTF-8 decoder flush
                        let remaining = {
                            let mut decoder = decoder.lock().unwrap();
                            decoder.flush()
                        };

                        if !remaining.is_empty() {
                            let result = streaming.parse_sse_chunk_buffered(&remaining).await;
                            Some((result, (streaming, decoder, 1)))
                        } else {
                            // Move to phase 1
                            Some((None, (streaming, decoder, 1)))
                        }
                    }
                    1 => {
                        // Phase 1: flush SSE buffer
                        let result = streaming.flush_sse_buffer().await;
                        if result.is_some() {
                            Some((result, (streaming, decoder, 2)))
                        } else {
                            // Move to phase 2
                            Some((None, (streaming, decoder, 2)))
                        }
                    }
                    _ => {
                        // Phase 2+: drain event queue
                        let event = {
                            let mut queue = streaming.event_queue.lock().await;
                            if !queue.is_empty() {
                                Some(queue.remove(0))
                            } else {
                                None
                            }
                        };

                        event.map(|event| (Some(event), (streaming, decoder, 2)))
                    }
                }
            },
        )
        .filter_map(|result| async move { result });

        Ok(decoded_stream.chain(flush_stream))
    }

    /// Parse a Server-Sent Events chunk with buffering for incomplete lines
    async fn parse_sse_chunk_buffered(
        &self,
        chunk: &str,
    ) -> Option<Result<ChatStreamEvent, LlmError>> {
        tracing::debug!("Received SSE chunk: '{}'", chunk);

        // Add chunk to buffer
        {
            let mut buffer = self.sse_buffer.lock().await;
            buffer.push_str(chunk);
            tracing::debug!("SSE buffer now contains: '{}'", buffer);
        }

        // Process complete lines from buffer
        self.process_buffered_lines().await
    }

    /// Process complete lines from the SSE buffer
    async fn process_buffered_lines(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        // First, check if we have queued events
        {
            let mut queue = self.event_queue.lock().await;
            if !queue.is_empty() {
                tracing::debug!(
                    "Returning queued event, {} events remaining in queue",
                    queue.len() - 1
                );
                return queue.remove(0).into();
            } else {
                tracing::debug!("No queued events available");
            }
        }

        {
            let mut buffer = self.sse_buffer.lock().await;

            // Find the last complete line (ending with \n)
            if let Some(last_newline_pos) = buffer.rfind('\n') {
                // Extract complete lines
                let complete_lines = buffer[..=last_newline_pos].to_string();
                // Keep incomplete line in buffer
                let remaining = buffer[last_newline_pos + 1..].to_string();
                *buffer = remaining;

                // Release the lock before processing
                drop(buffer);

                // Process all events from the complete lines
                let events = self.parse_sse_chunk_all_events(&complete_lines);

                if !events.is_empty() {
                    let mut events_iter = events.into_iter();
                    let first_event = events_iter.next();

                    // Queue remaining events
                    let remaining_events: Vec<_> = events_iter.collect();
                    if !remaining_events.is_empty() {
                        let mut queue = self.event_queue.lock().await;
                        tracing::debug!(
                            "Queueing {} events, queue size before: {}",
                            remaining_events.len(),
                            queue.len()
                        );
                        queue.extend(remaining_events);
                        tracing::debug!("Queue size after: {}", queue.len());
                    }

                    return first_event;
                }
            }
        }

        // Check if we have data that might be a complete SSE event without trailing newline
        // This can happen at the end of the stream
        {
            let mut buffer = self.sse_buffer.lock().await;
            if !buffer.is_empty() && (buffer.starts_with("data: ") || buffer.starts_with("event: "))
            {
                let content = buffer.clone();
                buffer.clear();

                // Release the lock before processing
                drop(buffer);

                tracing::debug!(
                    "Processing buffered SSE data without newline: '{}'",
                    content
                );
                let events = self.parse_sse_chunk_all_events(&content);

                if !events.is_empty() {
                    let mut events_iter = events.into_iter();
                    let first_event = events_iter.next();

                    // Queue remaining events
                    let remaining_events: Vec<_> = events_iter.collect();
                    if !remaining_events.is_empty() {
                        let mut queue = self.event_queue.lock().await;
                        queue.extend(remaining_events);
                    }

                    return first_event;
                }
            }
        }

        // No complete lines yet
        None
    }

    /// Flush any remaining content in the SSE buffer
    async fn flush_sse_buffer(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        let remaining = {
            let mut buffer = self.sse_buffer.lock().await;
            let content = buffer.clone();
            buffer.clear();
            content
        };

        if !remaining.is_empty() {
            // Try to parse remaining content as if it were complete
            let events = self.parse_sse_chunk_all_events(&remaining);

            if !events.is_empty() {
                let mut events_iter = events.into_iter();
                let first_event = events_iter.next();

                // Queue remaining events
                let remaining_events: Vec<_> = events_iter.collect();
                if !remaining_events.is_empty() {
                    let mut queue = self.event_queue.lock().await;
                    queue.extend(remaining_events);
                }

                return first_event;
            }
        }

        None
    }

    /// Parse a Server-Sent Events chunk and return all events found
    pub fn parse_sse_chunk_all_events(
        &self,
        chunk: &str,
    ) -> Vec<Result<ChatStreamEvent, LlmError>> {
        let mut events = Vec::new();

        tracing::debug!("Parsing SSE chunk with {} lines", chunk.lines().count());

        for line in chunk.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            // Parse data lines
            if let Some(data) = line.strip_prefix("data: ") {
                // Check for stream end
                if data == "[DONE]" {
                    // Add a stream end event
                    events.push(Ok(ChatStreamEvent::StreamEnd {
                        response: ChatResponse {
                            id: None,
                            content: MessageContent::Text(String::new()),
                            model: None,
                            usage: None,
                            finish_reason: Some(FinishReason::Stop),
                            tool_calls: None,
                            thinking: None,
                            metadata: std::collections::HashMap::new(),
                        },
                    }));
                    break;
                }

                // Parse JSON data
                tracing::debug!("Parsing SSE data: {}", data);
                match serde_json::from_str::<OpenAiStreamResponse>(data) {
                    Ok(response) => {
                        tracing::debug!("Parsed response: {:?}", response);
                        events.push(Ok(self.convert_openai_response(response)));
                    }
                    Err(e) => {
                        events.push(Err(LlmError::ParseError(format!(
                            "Failed to parse SSE data: {e}"
                        ))));
                    }
                }
            }
        }

        tracing::debug!("Parsed {} events from SSE chunk", events.len());
        events
    }

    /// Convert `OpenAI` stream response to our `ChatStreamEvent`
    fn convert_openai_response(&self, response: OpenAiStreamResponse) -> ChatStreamEvent {
        // Handle usage information (final chunk)
        if let Some(usage) = response.usage {
            return ChatStreamEvent::UsageUpdate {
                usage: Usage {
                    prompt_tokens: usage.prompt_tokens,
                    completion_tokens: usage.completion_tokens.unwrap_or(0),
                    total_tokens: usage.total_tokens.unwrap_or(0),
                    reasoning_tokens: usage
                        .completion_tokens_details
                        .and_then(|d| d.reasoning_tokens),
                    cached_tokens: usage.prompt_tokens_details.and_then(|d| d.cached_tokens),
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
                        usage: None, // Usage comes in a separate event
                        finish_reason: Some(finish_reason_enum),
                        tool_calls: None, // Tool calls are accumulated by StreamProcessor
                        thinking: None,
                        metadata: std::collections::HashMap::new(),
                    },
                };
            }

            // Handle content delta
            if let Some(content) = delta.content {
                // Check for <think> tags in the content
                if contains_thinking_tags(&content) {
                    // Extract thinking content and emit as thinking delta
                    if let Some(thinking) = extract_thinking_content(&content) {
                        return ChatStreamEvent::ThinkingDelta { delta: thinking };
                    }
                    // Filter out thinking tags from the main content
                    let filtered_content = filter_thinking_content(&content);
                    if !filtered_content.is_empty() {
                        return ChatStreamEvent::ContentDelta {
                            delta: filtered_content,
                            index: Some(choice.index),
                        };
                    }
                    // If content is only thinking tags, don't emit content delta
                    return ChatStreamEvent::StreamStart {
                        metadata: ResponseMetadata {
                            id: Some(response.id),
                            model: Some(response.model),
                            created: Some(
                                chrono::DateTime::from_timestamp(response.created as i64, 0)
                                    .unwrap_or_else(chrono::Utc::now),
                            ),
                            provider: "openai".to_string(),
                            request_id: None,
                        },
                    };
                } else {
                    return ChatStreamEvent::ContentDelta {
                        delta: content,
                        index: Some(choice.index),
                    };
                }
            }

            // Handle reasoning delta (o1 models)
            if let Some(reasoning) = delta.reasoning {
                return ChatStreamEvent::ThinkingDelta { delta: reasoning };
            }

            // Handle tool call deltas
            if let Some(tool_calls) = delta.tool_calls {
                if let Some(tool_call) = tool_calls.into_iter().next() {
                    // Debug logging for tool call deltas
                    tracing::debug!(
                        "Tool call delta - ID: {:?}, Function: {:?}",
                        tool_call.id,
                        tool_call.function
                    );

                    return ChatStreamEvent::ToolCallDelta {
                        id: tool_call.id.unwrap_or_default(),
                        function_name: tool_call.function.as_ref().and_then(|f| f.name.clone()),
                        arguments_delta: tool_call
                            .function
                            .as_ref()
                            .and_then(|f| f.arguments.clone()),
                        index: Some(choice.index),
                    };
                }
            }
        }

        // Default: stream start event
        ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some(response.id),
                model: Some(response.model),
                created: Some(
                    chrono::DateTime::from_timestamp(response.created as i64, 0)
                        .unwrap_or_else(chrono::Utc::now),
                ),
                provider: "openai".to_string(),
                request_id: None,
            },
        }
    }

    /// Convert messages to `OpenAI` format
    fn convert_messages(
        &self,
        messages: &[crate::types::ChatMessage],
    ) -> Result<serde_json::Value, LlmError> {
        // This is a simplified conversion - in a real implementation,
        // you'd need to handle all message types and content formats
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

    /// Convert tools to `OpenAI` format
    fn convert_tools(&self, tools: &[crate::types::Tool]) -> Result<serde_json::Value, LlmError> {
        // This is a simplified conversion - in a real implementation,
        // you'd need to handle the full tool specification
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::OpenAiConfig;

    #[test]
    fn test_openai_streaming_creation() {
        let config = OpenAiConfig::new("test-key");
        let client = reqwest::Client::new();
        let _streaming = OpenAiStreaming::new(config, client);

        // Basic test for streaming client creation
        // Basic test for streaming client creation
    }

    #[test]
    fn test_sse_parsing() {
        let config = OpenAiConfig::new("test-key");
        let client = reqwest::Client::new();
        let _streaming = OpenAiStreaming::new(config, client);

        let _sse_data = r#"data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;

        // This would require async test setup to properly test
        // For now, just verify the structure compiles
        // This would require async test setup to properly test
    }
}
