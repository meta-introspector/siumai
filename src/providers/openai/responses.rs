//! OpenAI Responses API Implementation
//!
//! This module implements the OpenAI Responses API which combines the simplicity
//! of Chat Completions with the tool-use capabilities of the Assistants API.
//! It supports built-in tools like web search, file search, and computer use.
//!
//! API Reference: https://platform.openai.com/docs/api-reference/responses

use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatResponse, OpenAiBuiltInTool, Tool};
use crate::web_search::{WebSearchCapability, WebSearchProvider};

use super::config::OpenAiConfig;

/// OpenAI Responses API client
#[allow(dead_code)]
pub struct OpenAiResponses {
    /// HTTP client
    http_client: reqwest::Client,
    /// API configuration
    config: OpenAiConfig,
    /// Web search provider
    web_search: Option<WebSearchProvider>,
}

impl OpenAiResponses {
    /// Create a new Responses API client
    pub fn new(http_client: reqwest::Client, config: OpenAiConfig) -> Self {
        let web_search = if config.web_search_config.enabled {
            Some(WebSearchProvider::new(
                "openai".to_string(),
                config.web_search_config.clone(),
            ))
        } else {
            None
        };

        Self {
            http_client,
            config,
            web_search,
        }
    }

    /// Get the responses endpoint
    fn responses_endpoint(&self) -> String {
        format!("{}/responses", self.config.base_url)
    }

    /// Build request body for Responses API
    fn build_request_body(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        built_in_tools: Option<&[OpenAiBuiltInTool]>,
        stream: bool,
        background: bool,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = serde_json::json!({
            "model": self.config.common_params.model,
            "stream": stream,
            "background": background,
        });

        // Convert messages to API format
        let api_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| self.convert_message_to_api_format(msg))
            .collect::<Result<Vec<_>, _>>()?;

        // Handle input format - single message vs array
        if api_messages.len() == 1 {
            body["input"] = api_messages[0].clone();
        } else {
            body["input"] = serde_json::Value::Array(api_messages);
        }

        // Add optional parameters
        if let Some(temp) = self.config.common_params.temperature {
            if let Some(num) = serde_json::Number::from_f64(temp as f64) {
                body["temperature"] = serde_json::Value::Number(num);
            }
        }

        if let Some(max_tokens) = self.config.common_params.max_tokens {
            body["max_tokens"] = serde_json::Value::Number(max_tokens.into());
        }

        if let Some(top_p) = self.config.common_params.top_p {
            if let Some(num) = serde_json::Number::from_f64(top_p as f64) {
                body["top_p"] = serde_json::Value::Number(num);
            }
        }

        // Build tools array
        let mut all_tools = Vec::new();

        // Add function tools
        if let Some(tools) = tools {
            for tool in tools {
                all_tools.push(self.convert_tool_to_responses_format(tool)?);
            }
        }

        // Add built-in tools
        if let Some(built_in_tools) = built_in_tools {
            for tool in built_in_tools {
                all_tools.push(tool.to_json());
            }
        }

        // Add web search tool if enabled
        if self.config.web_search_config.enabled {
            all_tools.push(OpenAiBuiltInTool::WebSearch.to_json());
        }

        if !all_tools.is_empty() {
            body["tools"] = serde_json::Value::Array(all_tools);
        }

        Ok(body)
    }

    /// Convert ChatMessage to API format
    fn convert_message_to_api_format(
        &self,
        message: &ChatMessage,
    ) -> Result<serde_json::Value, LlmError> {
        let mut api_message = serde_json::json!({
            "role": match message.role {
                crate::types::MessageRole::System => "system",
                crate::types::MessageRole::User => "user",
                crate::types::MessageRole::Assistant => "assistant",
                crate::types::MessageRole::Developer => "developer",
                crate::types::MessageRole::Tool => "tool",
            }
        });

        // Handle content
        match &message.content {
            crate::types::MessageContent::Text(text) => {
                api_message["content"] = serde_json::Value::String(text.clone());
            }
            crate::types::MessageContent::MultiModal(parts) => {
                let mut content_parts = Vec::new();
                for part in parts {
                    match part {
                        crate::types::ContentPart::Text { text } => {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": text
                            }));
                        }
                        crate::types::ContentPart::Image { image_url, detail } => {
                            let mut image_part = serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            });
                            if let Some(detail) = detail {
                                image_part["image_url"]["detail"] =
                                    serde_json::Value::String(detail.clone());
                            }
                            content_parts.push(image_part);
                        }
                        crate::types::ContentPart::Audio { audio_url, format } => {
                            content_parts.push(serde_json::json!({
                                "type": "audio",
                                "audio_url": audio_url,
                                "format": format
                            }));
                        }
                    }
                }
                api_message["content"] = serde_json::Value::Array(content_parts);
            }
        }

        // Handle tool calls
        if let Some(tool_calls) = &message.tool_calls {
            let api_tool_calls: Vec<serde_json::Value> = tool_calls
                .iter()
                .map(|call| {
                    serde_json::json!({
                        "id": call.id,
                        "type": call.r#type,
                        "function": call.function.as_ref().map(|f| serde_json::json!({
                            "name": f.name,
                            "arguments": f.arguments
                        }))
                    })
                })
                .collect();
            api_message["tool_calls"] = serde_json::Value::Array(api_tool_calls);
        }

        // Handle tool call ID
        if let Some(tool_call_id) = &message.tool_call_id {
            api_message["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
        }

        Ok(api_message)
    }

    /// Convert Tool to Responses API format
    fn convert_tool_to_responses_format(&self, tool: &Tool) -> Result<serde_json::Value, LlmError> {
        Ok(serde_json::json!({
            "type": tool.r#type,
            "function": {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters
            }
        }))
    }

    /// Parse response from Responses API
    fn parse_response(&self, response_data: serde_json::Value) -> Result<ChatResponse, LlmError> {
        // Extract content from the response
        let content = response_data
            .get("output")
            .and_then(|output| output.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("content"))
            .and_then(|content| content.as_str())
            .unwrap_or("")
            .to_string();

        // Extract usage information
        let usage = response_data
            .get("usage")
            .map(|usage_data| crate::types::Usage {
                prompt_tokens: usage_data
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
                    .unwrap_or(0),
                completion_tokens: usage_data
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
                    .unwrap_or(0),
                total_tokens: usage_data
                    .get("total_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
                    .unwrap_or(0),
                reasoning_tokens: None,
                cached_tokens: None,
            });

        // Extract metadata
        let _metadata = crate::types::ResponseMetadata {
            id: response_data
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            model: Some(self.config.common_params.model.clone()),
            created: None,
            provider: "openai".to_string(),
            request_id: None,
        };

        // Extract provider-specific data
        let mut provider_data = HashMap::new();
        if let Some(reasoning) = response_data.get("reasoning") {
            provider_data.insert("reasoning".to_string(), reasoning.clone());
        }

        Ok(ChatResponse {
            id: response_data
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            content: crate::types::MessageContent::Text(content),
            model: Some(self.config.common_params.model.clone()),
            usage,
            finish_reason: None, // TODO: Extract finish reason
            tool_calls: None, // TODO: Extract tool calls from response
            thinking: provider_data.get("reasoning").and_then(|v| v.as_str()).map(|s| s.to_string()),
            metadata: provider_data,
        })
    }
}

#[async_trait]
impl ChatCapability for OpenAiResponses {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let request_body =
            self.build_request_body(&messages, tools.as_deref(), None, false, false)?;

        let response = self
            .http_client
            .post(&self.responses_endpoint())
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI Responses API error: {}", error_text),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response(response_data)
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let request_body = self.build_request_body(
            &messages,
            tools.as_deref(),
            None,
            true, // stream = true
            false,
        )?;

        // Create SSE stream
        use futures::stream::StreamExt;

        let response = self
            .http_client
            .post(&self.responses_endpoint())
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI Responses API streaming error: {}", error_text),
            ));
        }

        // Convert response to stream
        let model_name = self.config.common_params.model.clone();
        let stream = response
            .bytes_stream()
            .map(move |chunk_result| {
                match chunk_result {
                    Ok(chunk) => {
                        let chunk_str = String::from_utf8_lossy(&chunk);
                        // Parse SSE events and convert to Results
                        Self::parse_sse_chunk_static(&chunk_str, &model_name)
                            .into_iter()
                            .map(Ok)
                            .collect::<Vec<_>>()
                    }
                    Err(e) => vec![Ok(crate::stream::ChatStreamEvent::Error {
                        error: e.to_string(),
                    })],
                }
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(stream))
    }
}

impl OpenAiResponses {
    /// Parse SSE chunk into stream events (static version)
    fn parse_sse_chunk_static(
        chunk: &str,
        _model_name: &str,
    ) -> Vec<crate::stream::ChatStreamEvent> {
        let mut events = Vec::new();

        for line in chunk.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..]; // Remove "data: " prefix

                if data == "[DONE]" {
                    // Stream end event
                    events.push(crate::stream::ChatStreamEvent::Done {
                        finish_reason: Some(crate::types::FinishReason::Stop),
                        usage: None,
                    });
                    continue;
                }

                // Try to parse JSON data
                if let Ok(json_data) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(delta) = json_data.get("delta") {
                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                            events.push(crate::stream::ChatStreamEvent::ContentDelta {
                                delta: content.to_string(),
                                index: None,
                            });
                        }

                        if let Some(tool_calls) =
                            delta.get("tool_calls").and_then(|tc| tc.as_array())
                        {
                            for (index, tool_call) in tool_calls.iter().enumerate() {
                                let id = tool_call
                                    .get("id")
                                    .and_then(|id| id.as_str())
                                    .map(|s| s.to_string())
                                    .unwrap_or_default();

                                let function_name = tool_call
                                    .get("function")
                                    .and_then(|func| func.get("name"))
                                    .and_then(|n| n.as_str())
                                    .map(|s| s.to_string());

                                let arguments_delta = tool_call
                                    .get("function")
                                    .and_then(|func| func.get("arguments"))
                                    .and_then(|a| a.as_str())
                                    .map(|s| s.to_string());

                                events.push(crate::stream::ChatStreamEvent::ToolCallDelta {
                                    id,
                                    function_name,
                                    arguments_delta,
                                    index: Some(index),
                                });
                            }
                        }
                    }

                    // Handle usage updates
                    if let Some(usage) = json_data.get("usage") {
                        let usage_info = crate::types::Usage {
                            prompt_tokens: usage
                                .get("prompt_tokens")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as u32)
                                .unwrap_or(0),
                            completion_tokens: usage
                                .get("completion_tokens")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as u32)
                                .unwrap_or(0),
                            total_tokens: usage
                                .get("total_tokens")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as u32)
                                .unwrap_or(0),
                            reasoning_tokens: usage
                                .get("reasoning_tokens")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as u32),
                            cached_tokens: None,
                        };
                        events.push(crate::stream::ChatStreamEvent::Usage {
                            usage: usage_info,
                        });
                    }
                }
            }
        }

        events
    }

    /// Parse SSE chunk into stream events
    #[allow(dead_code)]
    fn parse_sse_chunk(&self, chunk: &str) -> Vec<crate::stream::ChatStreamEvent> {
        Self::parse_sse_chunk_static(chunk, &self.config.common_params.model)
    }
}

#[async_trait]
impl WebSearchCapability for OpenAiResponses {
    async fn web_search(
        &self,
        query: String,
        _config: Option<crate::types::WebSearchConfig>,
    ) -> Result<Vec<crate::types::WebSearchResult>, LlmError> {
        // Use the built-in web search tool
        let messages = vec![crate::types::ChatMessage {
            role: crate::types::MessageRole::User,
            content: crate::types::MessageContent::Text(query),
            metadata: crate::types::MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }];

        let built_in_tools = vec![OpenAiBuiltInTool::WebSearch];

        let _request_body =
            self.build_request_body(&messages, None, Some(&built_in_tools), false, false)?;

        // TODO: Implement actual web search request and parse results
        // For now, return empty results
        Ok(Vec::new())
    }

    fn supports_web_search(&self) -> bool {
        true
    }

    fn web_search_strategy(&self) -> crate::types::WebSearchStrategy {
        crate::types::WebSearchStrategy::BuiltIn
    }
}
