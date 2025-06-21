//! Anthropic Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for Anthropic Claude.

use async_trait::async_trait;
use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::LlmError;
use crate::params::{AnthropicParameterMapper, ParameterMapper};
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::traits::ChatCapability;
use crate::types::*;
use crate::utils::Utf8StreamDecoder;

use super::types::*;
use super::utils::*;

/// Anthropic Chat Capability Implementation
pub struct AnthropicChatCapability {
    pub api_key: String,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
    pub parameter_mapper: AnthropicParameterMapper,
    anthropic_params: AnthropicSpecificParams,
}

impl AnthropicChatCapability {
    /// Create a new Anthropic chat capability instance
    pub const fn new(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
        anthropic_params: AnthropicSpecificParams,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            http_config,
            parameter_mapper: AnthropicParameterMapper,
            anthropic_params,
        }
    }

    /// Build the chat request body
    fn build_chat_request_body(
        &self,
        request: &ChatRequest,
        anthropic_params: Option<&super::types::AnthropicSpecificParams>,
    ) -> Result<serde_json::Value, LlmError> {
        // Map common parameters
        let mut body = self
            .parameter_mapper
            .map_common_params(&request.common_params);

        // Merge provider-specific parameters
        if let Some(ref provider_params) = request.provider_params {
            body = self
                .parameter_mapper
                .merge_provider_params(body, provider_params);
        }

        // Add Anthropic-specific parameters
        if let Some(params) = anthropic_params {
            // Add thinking configuration if present
            if let Some(ref thinking_config) = params.thinking_config {
                body["thinking"] = thinking_config.to_request_params();
            }

            // Add metadata if present
            if let Some(ref metadata) = params.metadata {
                body["metadata"] = metadata.clone();
            }
        }

        // Validate parameters
        self.parameter_mapper.validate_params(&body)?;

        // Convert message format
        let (messages, system) = convert_messages(&request.messages)?;
        body["messages"] = serde_json::to_value(messages)?;

        // If there is a system message, set it separately
        if let Some(system_content) = system {
            body["system"] = serde_json::Value::String(system_content);
        }

        // Add tools if present
        if let Some(ref tools) = request.tools {
            let anthropic_tools = convert_tools_to_anthropic_format(tools)?;
            body["tools"] = serde_json::Value::Array(anthropic_tools);
        }

        // Add streaming if enabled
        if request.stream {
            body["stream"] = serde_json::Value::Bool(true);
        }

        Ok(body)
    }

    /// Parse the Anthropic response
    fn parse_chat_response(
        &self,
        response: AnthropicChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        // Parse content and extract tool calls
        let (content, tool_calls) = parse_response_content_and_tools(&response.content);
        let finish_reason = parse_finish_reason(response.stop_reason.as_deref());
        let usage = create_usage_from_response(response.usage);

        let _metadata = ResponseMetadata {
            id: Some(response.id.clone()),
            model: Some(response.model.clone()),
            created: Some(chrono::Utc::now()), // Anthropic does not provide creation time
            provider: "anthropic".to_string(),
            request_id: None,
        };

        // Extract thinking content if present
        let mut provider_data = HashMap::new();
        if let Some(thinking_content) = extract_thinking_content(&response.content) {
            provider_data.insert(
                "thinking".to_string(),
                serde_json::Value::String(thinking_content),
            );
        }

        // Extract stop sequence if present
        if let Some(stop_seq) = response.stop_sequence {
            provider_data.insert(
                "stop_sequence".to_string(),
                serde_json::Value::String(stop_seq),
            );
        }

        Ok(ChatResponse {
            id: Some(response.id),
            content,
            model: Some(response.model),
            usage,
            finish_reason,
            tool_calls,
            thinking: extract_thinking_content(&response.content),
            metadata: provider_data,
        })
    }
}

#[async_trait]
impl ChatCapability for AnthropicChatCapability {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Create a ChatRequest from messages and tools
        let request = ChatRequest {
            messages,
            tools,
            common_params: CommonParams::default(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: false,
        };

        let headers = build_headers(&self.api_key, &self.http_config.headers)?;
        let body = self.build_chat_request_body(&request, Some(&self.anthropic_params))?;
        let url = format!("{}/v1/messages", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            // Parse Anthropic error response according to official documentation
            // https://docs.anthropic.com/en/api/errors
            if let Ok(error_json) = serde_json::from_str::<serde_json::Value>(&error_text) {
                if let Some(error_obj) = error_json.get("error") {
                    let error_type = error_obj
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("unknown");
                    let error_message = error_obj
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("Unknown error");

                    return Err(map_anthropic_error(
                        status.as_u16(),
                        error_type,
                        error_message,
                        error_json.clone(),
                    ));
                }
            }

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Anthropic API error: {error_text}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let anthropic_response: AnthropicChatResponse = response.json().await?;
        self.parse_chat_response(anthropic_response)
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Create a ChatRequest for streaming
        let request = ChatRequest {
            messages,
            tools,
            common_params: CommonParams::default(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: true,
        };

        let headers = build_headers(&self.api_key, &self.http_config.headers)?;
        let request_body = self.build_chat_request_body(&request, Some(&self.anthropic_params))?;

        let response = self
            .http_client
            .post(&format!("{}/v1/messages", self.base_url))
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
            return Err(LlmError::HttpError(format!(
                "HTTP {}: {}",
                status,
                error_text
            )));
        }

        // Create stream from response with UTF-8 decoder
        let decoder = Arc::new(Mutex::new(Utf8StreamDecoder::new()));
        let decoder_for_flush = decoder.clone();

        let stream = response.bytes_stream();
        let decoded_stream = stream.filter_map(move |chunk_result| {
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
                            if let Some(event) = Self::parse_sse_event(&decoded_chunk) {
                                return Some(event);
                            }
                        }
                        None
                    }
                    Err(e) => Some(Err(LlmError::StreamError(format!("Stream error: {e}")))),
                }
            }
        });

        // Add flush operation
        let flush_stream = futures_util::stream::once(async move {
            let remaining = {
                let mut decoder = decoder_for_flush.lock().unwrap();
                decoder.flush()
            };

            if !remaining.is_empty() {
                Self::parse_sse_event(&remaining)
            } else {
                None
            }
        }).filter_map(|result| async move { result });

        let final_stream = decoded_stream.chain(flush_stream);
        Ok(Box::pin(final_stream))
    }
}

impl AnthropicChatCapability {
    /// Parse SSE event from Anthropic streaming response
    fn parse_sse_event(chunk: &str) -> Option<Result<ChatStreamEvent, LlmError>> {
        for line in chunk.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            // Parse SSE data line
            if let Some(data) = line.strip_prefix("data: ") {
                // Handle end of stream
                if data == "[DONE]" {
                    return None;
                }

                // Parse JSON event
                match serde_json::from_str::<AnthropicStreamEvent>(data) {
                    Ok(event) => {
                        return Self::handle_stream_event(event);
                    }
                    Err(e) => {
                        return Some(Err(LlmError::ParseError(format!(
                            "Failed to parse stream event: {e}"
                        ))));
                    }
                }
            }
        }

        None
    }

    /// Handle different types of Anthropic stream events
    fn handle_stream_event(event: AnthropicStreamEvent) -> Option<Result<ChatStreamEvent, LlmError>> {
        match event.r#type.as_str() {
            "message_start" => {
                // Message started, no content yet
                None
            }
            "content_block_start" => {
                // Content block started, no delta yet
                None
            }
            "content_block_delta" => {
                // Parse content block delta
                match serde_json::from_value::<ContentBlockDeltaEvent>(event.data) {
                    Ok(delta_event) => {
                        match delta_event.delta {
                            AnthropicDelta::TextDelta { text } => {
                                Some(Ok(ChatStreamEvent::ContentDelta {
                                    delta: text,
                                    index: Some(delta_event.index as usize),
                                }))
                            }
                            AnthropicDelta::ThinkingDelta { thinking } => {
                                Some(Ok(ChatStreamEvent::ReasoningDelta {
                                    delta: thinking,
                                }))
                            }
                            AnthropicDelta::InputJsonDelta { partial_json } => {
                                // Handle tool input delta
                                Some(Ok(ChatStreamEvent::ContentDelta {
                                    delta: partial_json,
                                    index: Some(delta_event.index as usize),
                                }))
                            }
                            AnthropicDelta::SignatureDelta { signature } => {
                                // Handle signature delta (for thinking mode)
                                Some(Ok(ChatStreamEvent::ReasoningDelta {
                                    delta: signature,
                                }))
                            }
                        }
                    }
                    Err(e) => Some(Err(LlmError::ParseError(format!(
                        "Failed to parse content block delta: {e}"
                    )))),
                }
            }
            "content_block_stop" => {
                // Content block finished
                None
            }
            "message_delta" => {
                // Message metadata delta (usage, stop reason, etc.)
                None
            }
            "message_stop" => {
                // Message finished
                None
            }
            "ping" => {
                // Heartbeat, ignore
                None
            }
            "error" => {
                // Error event
                let error_msg = event.data.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("Unknown streaming error");
                Some(Err(LlmError::StreamError(error_msg.to_string())))
            }
            _ => {
                // Unknown event type, ignore
                None
            }
        }
    }


}

/// Legacy implementation for backward compatibility
impl AnthropicChatCapability {
    /// Chat with a `ChatRequest` (legacy method)
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools(request.messages, request.tools).await
    }

    /// Chat stream with a `ChatRequest` (legacy method)
    pub async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        ChatCapability::chat_stream(self, request.messages, request.tools).await
    }
}
