//! `OpenAI` Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for `OpenAI`.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::params::{OpenAiParameterMapper, OpenAiParams, ParameterMapper};
use crate::stream::ChatStream;
use crate::traits::ChatCapability;
use crate::types::*;

use super::types::*;
use super::utils::*;

/// `OpenAI` Chat Capability Implementation
pub struct OpenAiChatCapability {
    pub api_key: String,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub organization: Option<String>,
    pub project: Option<String>,
    pub http_config: HttpConfig,
    pub parameter_mapper: OpenAiParameterMapper,
}

impl OpenAiChatCapability {
    /// Create a new `OpenAI` chat capability instance
    pub const fn new(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        organization: Option<String>,
        project: Option<String>,
        http_config: HttpConfig,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            organization,
            project,
            http_config,
            parameter_mapper: OpenAiParameterMapper,
        }
    }

    /// Build the chat request body
    fn build_chat_request_body(
        &self,
        request: &ChatRequest,
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

        // Validate parameters
        self.parameter_mapper.validate_params(&body)?;

        // Convert message format
        let messages = convert_messages(&request.messages)?;
        body["messages"] = serde_json::to_value(messages)?;

        // Add tools if provided
        if let Some(ref tools) = request.tools {
            if !tools.is_empty() {
                body["tools"] = serde_json::to_value(tools)?;
            }
        }

        Ok(body)
    }

    /// Parse the `OpenAI` response
    fn parse_chat_response(&self, response: OpenAiChatResponse) -> Result<ChatResponse, LlmError> {
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

        let content = if let Some(content) = choice.message.content {
            match content {
                serde_json::Value::String(text) => {
                    // Check for <think> tags in the text content
                    if contains_thinking_tags(&text) {
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
                            if contains_thinking_tags(text) {
                                if thinking_content.is_none() {
                                    thinking_content = extract_thinking_content(text);
                                }
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

        let finish_reason = parse_finish_reason(choice.finish_reason.as_deref());

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens.unwrap_or(0),
            completion_tokens: u.completion_tokens.unwrap_or(0),
            total_tokens: u.total_tokens.unwrap_or(0),
            reasoning_tokens: None, // Specific to OpenAI o1, requires special handling
            cached_tokens: None,
        });

        let _metadata = ResponseMetadata {
            id: Some(response.id.clone()),
            model: Some(response.model.clone()),
            created: Some(
                chrono::DateTime::from_timestamp(response.created as i64, 0)
                    .unwrap_or_else(chrono::Utc::now),
            ),
            provider: "openai".to_string(),
            request_id: None, // Needs to be retrieved from the response headers
        };

        Ok(ChatResponse {
            id: Some(response.id),
            content,
            model: Some(response.model),
            usage,
            finish_reason,
            tool_calls,
            thinking: thinking_content, // Now includes extracted <think> content
            metadata: HashMap::new(),
        })
    }
}

#[async_trait]
impl ChatCapability for OpenAiChatCapability {
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

        let headers = build_headers(
            &self.api_key,
            self.organization.as_deref(),
            self.project.as_deref(),
            &self.http_config.headers,
        )?;

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
                message: format!("OpenAI API error: {error_text}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let openai_response: OpenAiChatResponse = response.json().await?;
        self.parse_chat_response(openai_response)
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
            common_params: CommonParams::default(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: true,
        };

        // Create streaming client
        let config = super::config::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: CommonParams::default(),
            openai_params: OpenAiParams::default(),
            http_config: self.http_config.clone(),
            web_search_config: crate::types::WebSearchConfig::default(),
        };

        let streaming = super::streaming::OpenAiStreaming::new(config, self.http_client.clone());
        streaming.create_chat_stream(request).await
    }
}

/// Legacy implementation for backward compatibility
impl OpenAiChatCapability {
    /// Chat with a `ChatRequest` (legacy method)
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools(request.messages, request.tools).await
    }

    /// Chat stream with a `ChatRequest` (legacy method)
    pub async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        ChatCapability::chat_stream(self, request.messages, request.tools).await
    }
}
