//! `Groq` Streaming Implementation
//!
//! This module provides Groq-specific streaming functionality for chat completions.

use futures::{Stream, StreamExt};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::Mutex;

use crate::ResponseMetadata;
use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatRequest, Usage};
use crate::utils::Utf8StreamDecoder;

use super::config::GroqConfig;
use super::types::*;
use super::utils::*;

/// `Groq` streaming client
#[derive(Clone)]
pub struct GroqStreaming {
    /// `Groq` configuration
    config: GroqConfig,
    /// HTTP client
    http_client: reqwest::Client,
    /// SSE line buffer for handling incomplete lines
    sse_buffer: Arc<Mutex<String>>,
}

impl GroqStreaming {
    /// Create a new `Groq` streaming client
    pub fn new(config: GroqConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
            sse_buffer: Arc::new(Mutex::new(String::new())),
        }
    }

    /// Create a streaming chat completion request
    pub async fn create_chat_stream(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let url = format!("{}/chat/completions", self.config.base_url);

        // Build request body
        let mut request_body = serde_json::json!({
            "model": request.common_params.model,
            "messages": convert_messages(&request.messages)?,
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
            if !tools.is_empty() {
                request_body["tools"] = serde_json::to_value(tools)?;
            }
        }

        // Add provider-specific parameters if provided
        if let Some(provider_params) = &request.provider_params {
            for (key, value) in &provider_params.params {
                request_body[key] = value.clone();
            }
        }

        // Validate parameters
        validate_groq_params(&request_body)?;

        // Create headers
        let headers = build_headers(&self.config.api_key, &self.config.http_config.headers)?;

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
            let error_message = extract_error_message(&error_text);
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Groq API error {status}: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
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
        let flush_stream = futures::stream::once(async move {
            let remaining = {
                let mut decoder = decoder_for_flush.lock().unwrap();
                decoder.flush()
            };

            if !remaining.is_empty() {
                streaming_for_flush
                    .parse_sse_chunk_buffered(&remaining)
                    .await
            } else {
                // Also flush any remaining SSE buffer content
                streaming_for_flush.flush_sse_buffer().await
            }
        })
        .filter_map(|result| async move { result });

        Ok(decoded_stream.chain(flush_stream))
    }

    /// Parse a Server-Sent Events chunk with buffering for incomplete lines
    async fn parse_sse_chunk_buffered(
        &self,
        chunk: &str,
    ) -> Option<Result<ChatStreamEvent, LlmError>> {
        // Add chunk to buffer
        {
            let mut buffer = self.sse_buffer.lock().await;
            buffer.push_str(chunk);
        }

        // Process complete lines from buffer
        self.process_buffered_lines().await
    }

    /// Process complete lines from the SSE buffer
    async fn process_buffered_lines(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
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

            // Process the complete lines
            return self.parse_sse_chunk(&complete_lines).await;
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
            self.parse_sse_chunk(&remaining).await
        } else {
            None
        }
    }

    /// Parse a Server-Sent Events chunk (original method for complete lines)
    async fn parse_sse_chunk(&self, chunk: &str) -> Option<Result<ChatStreamEvent, LlmError>> {
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
                    return None;
                }

                // Parse JSON data
                match serde_json::from_str::<GroqChatStreamChunk>(data) {
                    Ok(response) => {
                        return Some(Ok(self.convert_groq_response(response)));
                    }
                    Err(e) => {
                        return Some(Err(LlmError::ParseError(format!(
                            "Failed to parse SSE data: {e}"
                        ))));
                    }
                }
            }
        }

        None
    }

    /// Convert `Groq` stream response to our `ChatStreamEvent`
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

            // Handle content delta
            if let Some(content) = delta.content {
                return ChatStreamEvent::ContentDelta {
                    delta: content,
                    index: Some(choice.index as usize),
                };
            }

            // Handle tool call deltas
            if let Some(tool_calls) = delta.tool_calls {
                if let Some(tool_call) = tool_calls.into_iter().next() {
                    return ChatStreamEvent::ToolCallDelta {
                        id: tool_call.id.unwrap_or_default(),
                        function_name: tool_call.function.as_ref().and_then(|f| f.name.clone()),
                        arguments_delta: tool_call
                            .function
                            .as_ref()
                            .and_then(|f| f.arguments.clone()),
                        index: Some(choice.index as usize),
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
                provider: "groq".to_string(),
                request_id: None,
            },
        }
    }
}
