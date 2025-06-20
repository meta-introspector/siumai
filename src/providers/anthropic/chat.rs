//! Anthropic Chat Capability Implementation
//!
//! Implements the ChatCapability trait for Anthropic Claude.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::params::{AnthropicParameterMapper, ParameterMapper};
use crate::stream::ChatStream;
use crate::traits::ChatCapability;
use crate::types::*;

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
    pub fn new(
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
                message: format!("Anthropic API error: {}", error_text),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let anthropic_response: AnthropicChatResponse = response.json().await?;
        self.parse_chat_response(anthropic_response)
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Streaming implementation will be added later
        Err(LlmError::UnsupportedOperation(
            "Streaming not yet implemented for Anthropic".to_string(),
        ))
    }
}

/// Legacy implementation for backward compatibility
impl AnthropicChatCapability {
    /// Chat with a ChatRequest (legacy method)
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools(request.messages, request.tools).await
    }

    /// Chat stream with a ChatRequest (legacy method)
    pub async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        ChatCapability::chat_stream(self, request.messages, request.tools).await
    }
}
