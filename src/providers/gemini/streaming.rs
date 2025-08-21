//! Gemini streaming implementation using eventsource-stream
//!
//! This module provides Gemini streaming functionality using the
//! eventsource-stream infrastructure for JSON streaming.

use crate::error::LlmError;
use crate::providers::gemini::types::GeminiConfig;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatResponse, FinishReason, MessageContent, Usage};
use crate::utils::streaming::{SseEventConverter, StreamFactory};
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

/// Gemini stream response structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiStreamResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

/// Gemini candidate structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

/// Gemini content structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct GeminiContent {
    parts: Option<Vec<GeminiPart>>,
    role: Option<String>,
}

/// Gemini part structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiPart {
    text: Option<String>,
    /// Optional. Whether this is a thought summary (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
}

/// Gemini usage metadata
#[derive(Debug, Clone, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<u32>,
    /// Number of tokens used for thinking (only for thinking models)
    #[serde(rename = "thoughtsTokenCount")]
    thoughts_token_count: Option<u32>,
}

/// Gemini event converter
#[derive(Clone)]
pub struct GeminiEventConverter {
    #[allow(dead_code)]
    config: GeminiConfig,
}

impl GeminiEventConverter {
    pub fn new(config: GeminiConfig) -> Self {
        Self { config }
    }

    /// Convert Gemini stream response to ChatStreamEvent
    fn convert_gemini_response(&self, response: GeminiStreamResponse) -> Option<ChatStreamEvent> {
        // First, prioritize content over usage updates
        // Handle candidates for content and finish reasons
        if let Some(candidates) = response.candidates {
            for candidate in candidates {
                // Handle content first (most important)
                if let Some(content) = candidate.content
                    && let Some(parts) = content.parts
                {
                    for part in parts {
                        if let Some(text) = part.text {
                            // Check if this is thinking content
                            if part.thought.unwrap_or(false) {
                                return Some(ChatStreamEvent::ThinkingDelta { delta: text });
                            } else {
                                return Some(ChatStreamEvent::ContentDelta {
                                    delta: text,
                                    index: None,
                                });
                            }
                        }
                    }
                }

                // Handle finish reason
                if let Some(finish_reason) = candidate.finish_reason {
                    let reason = match finish_reason.as_str() {
                        "STOP" => FinishReason::Stop,
                        "MAX_TOKENS" => FinishReason::Length,
                        "SAFETY" => FinishReason::ContentFilter,
                        _ => FinishReason::Other(finish_reason),
                    };

                    let response = ChatResponse {
                        id: None,
                        model: None,
                        content: MessageContent::Text("".to_string()),
                        usage: None,
                        finish_reason: Some(reason),
                        tool_calls: None,
                        thinking: None,
                        metadata: HashMap::new(),
                    };

                    return Some(ChatStreamEvent::StreamEnd { response });
                }
            }
        }

        // Handle usage metadata only if no content was found
        if let Some(usage) = response.usage_metadata {
            let usage_info = Usage {
                prompt_tokens: usage.prompt_token_count.unwrap_or(0),
                completion_tokens: usage.candidates_token_count.unwrap_or(0),
                total_tokens: usage.total_token_count.unwrap_or(0),
                cached_tokens: None,
                reasoning_tokens: usage.thoughts_token_count,
            };
            return Some(ChatStreamEvent::UsageUpdate { usage: usage_info });
        }

        None
    }
}

impl SseEventConverter for GeminiEventConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Pin<Box<dyn Future<Output = Option<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            // Skip empty events
            if event.data.trim().is_empty() {
                return None;
            }

            // Parse the JSON data from the SSE event
            match serde_json::from_str::<GeminiStreamResponse>(&event.data) {
                Ok(gemini_response) => self.convert_gemini_response(gemini_response).map(Ok),
                Err(e) => Some(Err(LlmError::ParseError(format!(
                    "Failed to parse Gemini SSE JSON: {e}"
                )))),
            }
        })
    }
}

/// Gemini streaming client
#[derive(Debug, Clone)]
pub struct GeminiStreaming {
    config: GeminiConfig,
    http_client: reqwest::Client,
}

impl GeminiStreaming {
    /// Create a new Gemini streaming client
    pub fn new(http_client: reqwest::Client) -> Self {
        Self {
            config: GeminiConfig::default(),
            http_client,
        }
    }

    /// Create a chat stream from URL, API key, and request
    pub async fn create_chat_stream(
        self,
        url: String,
        api_key: String,
        request: crate::providers::gemini::types::GenerateContentRequest,
    ) -> Result<ChatStream, LlmError> {
        // Make the HTTP request
        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &api_key)
            .json(&request)
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
                message: format!("Gemini API error {status}: {error_text}"),
                details: None,
            });
        }

        // Create the stream using SSE infrastructure (Gemini uses SSE format)
        let mut config = self.config;
        config.api_key = api_key.clone();
        let converter = GeminiEventConverter::new(config);
        StreamFactory::create_eventsource_stream(
            self.http_client
                .post(&url)
                .header("Content-Type", "application/json")
                .header("x-goog-api-key", &api_key)
                .json(&request),
            converter,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::gemini::types::GeminiConfig;

    fn create_test_config() -> GeminiConfig {
        GeminiConfig {
            api_key: "test-key".to_string(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_gemini_streaming_conversion() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        // Test content delta conversion
        let json_data = r#"{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}"#;
        let event = eventsource_stream::Event {
            event: "".to_string(),
            data: json_data.to_string(),
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
    async fn test_gemini_finish_reason() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        // Test finish reason conversion
        let json_data = r#"{"candidates":[{"finishReason":"STOP"}]}"#;
        let event = eventsource_stream::Event {
            event: "".to_string(),
            data: json_data.to_string(),
            id: "".to_string(),
            retry: None,
        };

        let result = converter.convert_event(event).await;
        assert!(result.is_some());

        if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = result {
            assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        } else {
            panic!("Expected StreamEnd event");
        }
    }
}
