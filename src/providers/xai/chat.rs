//! `xAI` Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for `xAI`.

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;

use crate::error::LlmError;
use crate::params::{OpenAiParameterMapper, ParameterMapper};
use crate::stream::ChatStream;
use crate::tracing::ProviderTracer;
use crate::traits::ChatCapability;
use crate::types::*;

use super::types::*;
use super::utils::*;

/// `xAI` Chat Capability Implementation
#[derive(Debug, Clone)]
pub struct XaiChatCapability {
    pub api_key: String,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
    pub parameter_mapper: OpenAiParameterMapper,
    pub common_params: CommonParams,
}

impl XaiChatCapability {
    /// Create a new `xAI` chat capability instance
    pub const fn new(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
        common_params: CommonParams,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            http_config,
            parameter_mapper: OpenAiParameterMapper,
            common_params,
        }
    }

    /// Build the chat request body
    pub fn build_chat_request_body(
        &self,
        request: &ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        // Map common parameters using OpenAI-compatible format
        let mut body = self
            .parameter_mapper
            .map_common_params(&request.common_params);

        // Merge provider-specific parameters
        if let Some(ref provider_params) = request.provider_params {
            body = self
                .parameter_mapper
                .merge_provider_params(body, provider_params);
        }

        // Validate parameters
        self.parameter_mapper.validate_params(&body)?;

        // Convert message format
        let messages = convert_messages(&request.messages)?;
        body["messages"] = serde_json::to_value(messages)?;

        // Add tools if provided
        if let Some(ref tools) = request.tools
            && !tools.is_empty()
        {
            body["tools"] = serde_json::to_value(tools)?;
        }

        // Add stream parameter if needed
        if request.stream {
            body["stream"] = serde_json::Value::Bool(true);
        }

        Ok(body)
    }

    /// Parse the `xAI` response
    fn parse_chat_response(&self, response: XaiChatResponse) -> Result<ChatResponse, LlmError> {
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ApiError {
                code: 500,
                message: "No choices in response".to_string(),
                details: None,
            })?;

        // Extract thinking content and filter it from the main content
        let mut thinking_content: Option<String> = None;

        // Check for reasoning content first (xAI specific)
        if let Some(reasoning) = choice.message.reasoning_content {
            thinking_content = Some(reasoning);
        }

        let content = if let Some(content) = choice.message.content {
            match content {
                serde_json::Value::String(text) => {
                    // Check for <think> tags in the text content if no reasoning_content
                    if thinking_content.is_none() && contains_thinking_tags(&text) {
                        thinking_content = extract_thinking_content(&text);
                        // Filter out thinking tags from the main content
                        let filtered_text = filter_thinking_content(&text);
                        MessageContent::Text(filtered_text)
                    } else {
                        MessageContent::Text(text)
                    }
                }
                serde_json::Value::Array(parts) => {
                    let mut content_parts = Vec::new();
                    for part in parts {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            // Check for thinking tags in each text part
                            if thinking_content.is_none() && contains_thinking_tags(text) {
                                thinking_content = extract_thinking_content(text);
                                // Filter out thinking tags from this part
                                let filtered_text = filter_thinking_content(text);
                                if !filtered_text.is_empty() {
                                    content_parts.push(ContentPart::Text {
                                        text: filtered_text,
                                    });
                                }
                            } else {
                                content_parts.push(ContentPart::Text {
                                    text: text.to_string(),
                                });
                            }
                        }
                    }
                    MessageContent::MultiModal(content_parts)
                }
                _ => MessageContent::Text(String::new()),
            }
        } else {
            MessageContent::Text(String::new())
        };

        let tool_calls = choice.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|call| ToolCall {
                    id: call.id,
                    r#type: call.r#type,
                    function: call.function.map(|f| FunctionCall {
                        name: f.name,
                        arguments: f.arguments,
                    }),
                })
                .collect()
        });

        let finish_reason = Some(parse_finish_reason(choice.finish_reason.as_deref()));

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens.unwrap_or(0),
            completion_tokens: u.completion_tokens.unwrap_or(0),
            total_tokens: u.total_tokens.unwrap_or(0),
            reasoning_tokens: u.reasoning_tokens, // xAI specific
            cached_tokens: None,
        });

        let _metadata = ResponseMetadata {
            id: Some(response.id.clone()),
            model: Some(response.model.clone()),
            created: Some(
                chrono::DateTime::from_timestamp(response.created as i64, 0)
                    .unwrap_or_else(chrono::Utc::now),
            ),
            provider: "xai".to_string(),
            request_id: None, // Needs to be retrieved from the response headers
        };

        Ok(ChatResponse {
            id: Some(response.id),
            content,
            model: Some(response.model),
            usage,
            finish_reason,
            tool_calls,
            thinking: thinking_content, // Now includes extracted thinking content
            metadata: HashMap::new(),
        })
    }
}

#[async_trait]
impl ChatCapability for XaiChatCapability {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let start_time = Instant::now();

        // Create a ChatRequest from messages and tools
        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: false,
        };

        // Extract model name for tracing
        let model = request.common_params.model.clone();
        let tracer = ProviderTracer::new("xai").with_model(model);

        let headers = build_headers(&self.api_key, &self.http_config.headers)?;
        let body = self.build_chat_request_body(&request)?;
        let url = format!("{}/chat/completions", self.base_url);

        tracer.trace_request_start("POST", &url);
        tracer.trace_request_details(&headers, &body);

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

            tracer.trace_request_error(status.as_u16(), &error_text, start_time);

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("xAI API error: {error_text}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        tracer.trace_response_success(response.status().as_u16(), start_time, response.headers());

        // Get response body as text first for logging
        let response_text = response.text().await?;
        tracer.trace_response_body(&response_text);

        let xai_response: XaiChatResponse = serde_json::from_str(&response_text)?;
        let chat_response = self.parse_chat_response(xai_response)?;

        tracer.trace_request_complete(start_time, chat_response.content.all_text().len());

        Ok(chat_response)
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Create a ChatRequest from messages and tools
        let request = ChatRequest {
            messages,
            tools,
            common_params: self.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: true,
        };

        // Create streaming client
        let config = super::config::XaiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            common_params: self.common_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: crate::types::WebSearchConfig::default(),
        };

        let streaming = super::streaming::XaiStreaming::new(config, self.http_client.clone());
        streaming.create_chat_stream(request).await
    }
}

/// Legacy implementation for backward compatibility
impl XaiChatCapability {
    /// Chat with a `ChatRequest` (legacy method)
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let headers = build_headers(&self.api_key, &self.http_config.headers)?;

        let body = self.build_chat_request_body(&request)?;
        let url = format!("{}/chat/completions", self.base_url);

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

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("xAI API error: {error_text}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let xai_response: XaiChatResponse = response.json().await?;
        self.parse_chat_response(xai_response)
    }

    /// Chat stream with a `ChatRequest` (legacy method)
    pub async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        // Create streaming client with the request's configuration
        let config = super::config::XaiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            common_params: request.common_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: crate::types::WebSearchConfig::default(),
        };

        let streaming = super::streaming::XaiStreaming::new(config, self.http_client.clone());
        streaming.create_chat_stream(request).await
    }
}
