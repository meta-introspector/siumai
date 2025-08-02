//! Ollama streaming implementation using eventsource-stream
//!
//! This module provides Ollama streaming functionality using the
//! eventsource-stream infrastructure for JSON streaming.

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatResponse, FinishReason, MessageContent, Usage};
use crate::utils::streaming::{JsonEventConverter, StreamProcessor};
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

/// Ollama stream response structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OllamaStreamResponse {
    model: Option<String>,
    message: Option<OllamaMessage>,
    done: Option<bool>,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
}

/// Ollama message structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OllamaMessage {
    role: Option<String>,
    content: Option<String>,
}

/// Ollama event converter
#[derive(Clone)]
pub struct OllamaEventConverter;

impl Default for OllamaEventConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaEventConverter {
    pub fn new() -> Self {
        Self
    }

    /// Convert Ollama stream response to ChatStreamEvent
    fn convert_ollama_response(&self, response: OllamaStreamResponse) -> Option<ChatStreamEvent> {
        // Handle completion
        if response.done == Some(true) {
            // Handle usage information
            if let (Some(prompt_tokens), Some(completion_tokens)) =
                (response.prompt_eval_count, response.eval_count)
            {
                let usage_info = Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    cached_tokens: None,
                    reasoning_tokens: None,
                };
                return Some(ChatStreamEvent::UsageUpdate { usage: usage_info });
            }

            // Stream end
            let response = ChatResponse {
                id: None,
                model: response.model,
                content: MessageContent::Text("".to_string()),
                usage: None,
                finish_reason: Some(FinishReason::Stop),
                tool_calls: None,
                thinking: None,
                metadata: HashMap::new(),
            };
            return Some(ChatStreamEvent::StreamEnd { response });
        }

        // Handle content delta
        if let Some(message) = response.message {
            if let Some(content) = message.content {
                return Some(ChatStreamEvent::ContentDelta {
                    delta: content,
                    index: None,
                });
            }
        }

        None
    }
}

impl JsonEventConverter for OllamaEventConverter {
    fn convert_json<'a>(
        &'a self,
        json_data: &'a str,
    ) -> Pin<Box<dyn Future<Output = Option<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>
    {
        Box::pin(async move {
            match serde_json::from_str::<OllamaStreamResponse>(json_data) {
                Ok(ollama_response) => self.convert_ollama_response(ollama_response).map(Ok),
                Err(e) => Some(Err(LlmError::ParseError(format!(
                    "Failed to parse Ollama JSON: {e}"
                )))),
            }
        })
    }
}

/// Ollama streaming client
#[derive(Clone)]
pub struct OllamaStreaming {
    http_client: reqwest::Client,
}

impl OllamaStreaming {
    /// Create a new Ollama streaming client
    pub fn new(http_client: reqwest::Client) -> Self {
        Self { http_client }
    }

    /// Create a chat stream from URL, headers, and body
    pub async fn create_chat_stream(
        self,
        url: String,
        headers: reqwest::header::HeaderMap,
        body: crate::providers::ollama::types::OllamaChatRequest,
    ) -> Result<ChatStream, LlmError> {
        // Make the HTTP request
        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
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
                message: format!("Ollama API error {status}: {error_text}"),
                details: None,
            });
        }

        // Create the stream using our new infrastructure
        let converter = OllamaEventConverter::new();
        StreamProcessor::create_json_stream(response, converter).await
    }

    /// Create a completion stream from URL, headers, and body
    pub async fn create_completion_stream(
        self,
        url: String,
        headers: reqwest::header::HeaderMap,
        body: crate::providers::ollama::types::OllamaGenerateRequest,
    ) -> Result<ChatStream, LlmError> {
        // Make the HTTP request
        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
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
                message: format!("Ollama API error {status}: {error_text}"),
                details: None,
            });
        }

        // Create the stream using our new infrastructure
        let converter = OllamaEventConverter::new();
        StreamProcessor::create_json_stream(response, converter).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ollama_streaming_conversion() {
        let converter = OllamaEventConverter::new();

        // Test content delta conversion
        let json_data =
            r#"{"model":"llama2","message":{"role":"assistant","content":"Hello"},"done":false}"#;

        let result = converter.convert_json(json_data).await;
        assert!(result.is_some());

        if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = result {
            assert_eq!(delta, "Hello");
        } else {
            panic!("Expected ContentDelta event");
        }
    }

    #[tokio::test]
    async fn test_ollama_stream_end() {
        let converter = OllamaEventConverter::new();

        // Test stream end conversion
        let json_data = r#"{"model":"llama2","done":true,"prompt_eval_count":10,"eval_count":20}"#;

        let result = converter.convert_json(json_data).await;
        assert!(result.is_some());

        if let Some(Ok(ChatStreamEvent::UsageUpdate { usage })) = result {
            assert_eq!(usage.prompt_tokens, 10);
            assert_eq!(usage.completion_tokens, 20);
        } else {
            panic!("Expected UsageUpdate event");
        }
    }
}
