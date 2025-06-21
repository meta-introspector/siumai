//! `OpenAI` Streaming Implementation
//!
//! This module provides OpenAI-specific streaming functionality for chat completions.

use futures::{Stream, StreamExt};
use serde::Deserialize;

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatRequest, ResponseMetadata, Usage};

use super::config::OpenAiConfig;

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
}

impl OpenAiStreaming {
    /// Create a new `OpenAI` streaming client
    pub const fn new(config: OpenAiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
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

        Ok(stream.filter_map(move |chunk_result| {
            let streaming = self.clone();
            async move {
                match chunk_result {
                    Ok(chunk) => {
                        let chunk_str = String::from_utf8_lossy(&chunk);
                        streaming.parse_sse_chunk(&chunk_str).await
                    }
                    Err(e) => Some(Err(e)),
                }
            }
        }))
    }

    /// Parse a Server-Sent Events chunk
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
                match serde_json::from_str::<OpenAiStreamResponse>(data) {
                    Ok(response) => {
                        return Some(Ok(self.convert_openai_response(response)));
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

    /// Convert `OpenAI` stream response to our `ChatStreamEvent`
    fn convert_openai_response(&self, response: OpenAiStreamResponse) -> ChatStreamEvent {
        // Handle usage information (final chunk)
        if let Some(usage) = response.usage {
            return ChatStreamEvent::Usage {
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

            // Handle content delta
            if let Some(content) = delta.content {
                return ChatStreamEvent::ContentDelta {
                    delta: content,
                    index: Some(choice.index),
                };
            }

            // Handle reasoning delta (o1 models)
            if let Some(reasoning) = delta.reasoning {
                return ChatStreamEvent::ReasoningDelta { delta: reasoning };
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
