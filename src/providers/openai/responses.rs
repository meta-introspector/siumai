//! `OpenAI` Responses API Implementation
//!
//! This module implements the `OpenAI` Responses API which combines the simplicity
//! of Chat Completions with the tool-use capabilities of the Assistants API.
//! It supports built-in tools like web search, file search, and computer use.
//!
//! The Responses API provides:
//! - Stateful conversations with automatic context management
//! - Background processing for long-running tasks
//! - Built-in tools (web search, file search, computer use)
//! - Response lifecycle management (create, get, cancel, list)
//!
//! API Reference: <https://platform.openai.com/docs/api-reference/responses>

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatResponse, OpenAiBuiltInTool, Tool};
use crate::web_search::{WebSearchCapability, WebSearchProvider};

use super::config::OpenAiConfig;

/// Response status for background processing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResponseStatus {
    /// Response is being processed
    InProgress,
    /// Response completed successfully
    Completed,
    /// Response failed with an error
    Failed,
    /// Response was cancelled
    Cancelled,
}

/// Response metadata for Responses API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Response ID
    pub id: String,
    /// Response status
    pub status: ResponseStatus,
    /// Creation timestamp
    pub created_at: u64,
    /// Completion timestamp (if completed)
    pub completed_at: Option<u64>,
    /// Model used
    pub model: String,
    /// Whether this was a background request
    pub background: bool,
    /// Previous response ID (if chained)
    pub previous_response_id: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// List responses query parameters
#[derive(Debug, Clone, Default)]
pub struct ListResponsesQuery {
    /// Limit number of results
    pub limit: Option<u32>,
    /// Pagination cursor
    pub after: Option<String>,
    /// Filter by status
    pub status: Option<ResponseStatus>,
    /// Sort order (asc/desc)
    pub order: Option<String>,
}

/// `OpenAI` Responses API client
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

    /// Get a specific response endpoint
    fn response_endpoint(&self, response_id: &str) -> String {
        format!("{}/responses/{}", self.config.base_url, response_id)
    }

    /// Get response cancel endpoint
    fn response_cancel_endpoint(&self, response_id: &str) -> String {
        format!("{}/responses/{}/cancel", self.config.base_url, response_id)
    }

    /// Create a response with background processing
    pub async fn create_response_background(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        built_in_tools: Option<Vec<OpenAiBuiltInTool>>,
        previous_response_id: Option<String>,
    ) -> Result<ResponseMetadata, LlmError> {
        let request_body = self.build_request_body_with_options(
            &messages,
            tools.as_deref(),
            built_in_tools.as_deref(),
            false, // stream = false for background
            true,  // background = true
            previous_response_id,
        )?;

        let response = self
            .http_client
            .post(self.responses_endpoint())
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
                format!("OpenAI Responses API background error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response_metadata(response_data)
    }

    /// Get a response by ID
    pub async fn get_response(&self, response_id: &str) -> Result<ChatResponse, LlmError> {
        let response = self
            .http_client
            .get(self.response_endpoint(response_id))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI get response error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response(response_data)
    }

    /// Cancel a background response
    pub async fn cancel_response(&self, response_id: &str) -> Result<ResponseMetadata, LlmError> {
        let response = self
            .http_client
            .post(self.response_cancel_endpoint(response_id))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI cancel response error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response_metadata(response_data)
    }

    /// List responses with optional filtering
    pub async fn list_responses(
        &self,
        query: Option<ListResponsesQuery>,
    ) -> Result<Vec<ResponseMetadata>, LlmError> {
        let mut url = self.responses_endpoint();

        if let Some(q) = query {
            let mut params = Vec::new();

            if let Some(limit) = q.limit {
                params.push(format!("limit={limit}"));
            }
            if let Some(after) = q.after {
                params.push(format!("after={after}"));
            }
            if let Some(status) = q.status {
                let status_str = match status {
                    ResponseStatus::InProgress => "in_progress",
                    ResponseStatus::Completed => "completed",
                    ResponseStatus::Failed => "failed",
                    ResponseStatus::Cancelled => "cancelled",
                };
                params.push(format!("status={status_str}"));
            }
            if let Some(order) = q.order {
                params.push(format!("order={order}"));
            }

            if !params.is_empty() {
                url.push('?');
                url.push_str(&params.join("&"));
            }
        }

        let response = self
            .http_client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI list responses error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        // Parse the list of responses
        let responses = response_data
            .get("data")
            .and_then(|data| data.as_array())
            .ok_or_else(|| LlmError::ParseError("Invalid response format".to_string()))?;

        let mut result = Vec::new();
        for response_item in responses {
            result.push(self.parse_response_metadata(response_item.clone())?);
        }

        Ok(result)
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
        self.build_request_body_with_options(
            messages,
            tools,
            built_in_tools,
            stream,
            background,
            None,
        )
    }

    /// Build request body for Responses API with additional options
    fn build_request_body_with_options(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        built_in_tools: Option<&[OpenAiBuiltInTool]>,
        stream: bool,
        background: bool,
        previous_response_id: Option<String>,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = serde_json::json!({
            "model": self.config.common_params.model,
            "stream": stream,
            "background": background,
        });

        // Add previous response ID for chaining
        if let Some(prev_id) = previous_response_id {
            body["previous_response_id"] = serde_json::Value::String(prev_id);
        }

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

    /// Parse response metadata from API response
    fn parse_response_metadata(
        &self,
        response_data: serde_json::Value,
    ) -> Result<ResponseMetadata, LlmError> {
        let id = response_data
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::ParseError("Missing response ID".to_string()))?
            .to_string();

        let status_str = response_data
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("in_progress");

        let status = match status_str {
            "in_progress" => ResponseStatus::InProgress,
            "completed" => ResponseStatus::Completed,
            "failed" => ResponseStatus::Failed,
            "cancelled" => ResponseStatus::Cancelled,
            _ => ResponseStatus::InProgress,
        };

        let created_at = response_data
            .get("created_at")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let completed_at = response_data.get("completed_at").and_then(|v| v.as_u64());

        let model = response_data
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.config.common_params.model)
            .to_string();

        let background = response_data
            .get("background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let previous_response_id = response_data
            .get("previous_response_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let error = response_data
            .get("error")
            .and_then(|v| v.get("message"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(ResponseMetadata {
            id,
            status,
            created_at,
            completed_at,
            model,
            background,
            previous_response_id,
            error,
        })
    }

    /// Convert `ChatMessage` to API format
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
                    .and_then(serde_json::Value::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(0),
                completion_tokens: usage_data
                    .get("output_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(0),
                total_tokens: usage_data
                    .get("total_tokens")
                    .and_then(serde_json::Value::as_u64)
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
                .map(std::string::ToString::to_string),
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
                .map(std::string::ToString::to_string),
            content: crate::types::MessageContent::Text(content),
            model: Some(self.config.common_params.model.clone()),
            usage,
            finish_reason: None, // TODO: Extract finish reason
            tool_calls: None,    // TODO: Extract tool calls from response
            thinking: provider_data
                .get("reasoning")
                .and_then(|v| v.as_str())
                .map(std::string::ToString::to_string),
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
            .post(self.responses_endpoint())
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
                format!("OpenAI Responses API error: {error_text}"),
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
            .post(self.responses_endpoint())
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
                format!("OpenAI Responses API streaming error: {error_text}"),
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
            if let Some(data) = line.strip_prefix("data: ") {
                // Remove "data: " prefix

                if data == "[DONE]" {
                    // Stream end event
                    events.push(crate::stream::ChatStreamEvent::StreamEnd {
                        response: crate::types::ChatResponse {
                            id: None,
                            content: crate::types::MessageContent::Text(String::new()),
                            model: None,
                            usage: None,
                            finish_reason: Some(crate::types::FinishReason::Stop),
                            tool_calls: None,
                            thinking: None,
                            metadata: std::collections::HashMap::new(),
                        },
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
                                    .map(std::string::ToString::to_string)
                                    .unwrap_or_default();

                                let function_name = tool_call
                                    .get("function")
                                    .and_then(|func| func.get("name"))
                                    .and_then(|n| n.as_str())
                                    .map(std::string::ToString::to_string);

                                let arguments_delta = tool_call
                                    .get("function")
                                    .and_then(|func| func.get("arguments"))
                                    .and_then(|a| a.as_str())
                                    .map(std::string::ToString::to_string);

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
                                .and_then(serde_json::Value::as_u64)
                                .map(|v| v as u32)
                                .unwrap_or(0),
                            completion_tokens: usage
                                .get("completion_tokens")
                                .and_then(serde_json::Value::as_u64)
                                .map(|v| v as u32)
                                .unwrap_or(0),
                            total_tokens: usage
                                .get("total_tokens")
                                .and_then(serde_json::Value::as_u64)
                                .map(|v| v as u32)
                                .unwrap_or(0),
                            reasoning_tokens: usage
                                .get("reasoning_tokens")
                                .and_then(serde_json::Value::as_u64)
                                .map(|v| v as u32),
                            cached_tokens: None,
                        };
                        events.push(crate::stream::ChatStreamEvent::UsageUpdate {
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

/// Trait for OpenAI Responses API specific functionality
///
/// This trait extends beyond basic chat capabilities to provide
/// stateful conversation management, background processing, and
/// response lifecycle management specific to OpenAI's Responses API.
#[async_trait]
pub trait ResponsesApiCapability {
    /// Create a response with background processing
    async fn create_response_background(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        built_in_tools: Option<Vec<OpenAiBuiltInTool>>,
        previous_response_id: Option<String>,
    ) -> Result<ResponseMetadata, LlmError>;

    /// Get a response by ID
    async fn get_response(&self, response_id: &str) -> Result<ChatResponse, LlmError>;

    /// Cancel a background response
    async fn cancel_response(&self, response_id: &str) -> Result<ResponseMetadata, LlmError>;

    /// List responses with optional filtering
    async fn list_responses(
        &self,
        query: Option<ListResponsesQuery>,
    ) -> Result<Vec<ResponseMetadata>, LlmError>;

    /// Create a response that continues from a previous response
    async fn continue_conversation(
        &self,
        previous_response_id: String,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        background: bool,
    ) -> Result<ChatResponse, LlmError>;

    /// Check if a response is ready (for background responses)
    async fn is_response_ready(&self, response_id: &str) -> Result<bool, LlmError>;
}

#[async_trait]
impl ResponsesApiCapability for OpenAiResponses {
    async fn create_response_background(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        built_in_tools: Option<Vec<OpenAiBuiltInTool>>,
        previous_response_id: Option<String>,
    ) -> Result<ResponseMetadata, LlmError> {
        self.create_response_background(messages, tools, built_in_tools, previous_response_id)
            .await
    }

    async fn get_response(&self, response_id: &str) -> Result<ChatResponse, LlmError> {
        self.get_response(response_id).await
    }

    async fn cancel_response(&self, response_id: &str) -> Result<ResponseMetadata, LlmError> {
        self.cancel_response(response_id).await
    }

    async fn list_responses(
        &self,
        query: Option<ListResponsesQuery>,
    ) -> Result<Vec<ResponseMetadata>, LlmError> {
        self.list_responses(query).await
    }

    async fn continue_conversation(
        &self,
        previous_response_id: String,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        background: bool,
    ) -> Result<ChatResponse, LlmError> {
        let request_body = self.build_request_body_with_options(
            &messages,
            tools.as_deref(),
            None,
            false,
            background,
            Some(previous_response_id),
        )?;

        let response = self
            .http_client
            .post(self.responses_endpoint())
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
                format!("OpenAI continue conversation error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response(response_data)
    }

    async fn is_response_ready(&self, response_id: &str) -> Result<bool, LlmError> {
        let metadata = self.get_response_metadata(response_id).await?;
        Ok(matches!(
            metadata.status,
            ResponseStatus::Completed | ResponseStatus::Failed | ResponseStatus::Cancelled
        ))
    }
}

impl OpenAiResponses {
    /// Get response metadata without full response content
    pub async fn get_response_metadata(
        &self,
        response_id: &str,
    ) -> Result<ResponseMetadata, LlmError> {
        let response = self
            .http_client
            .get(self.response_endpoint(response_id))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("OpenAI get response metadata error: {error_text}"),
            ));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        self.parse_response_metadata(response_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::config::OpenAiConfig;
    use crate::types::{
        ChatMessage, MessageContent, MessageMetadata, MessageRole, OpenAiBuiltInTool,
    };

    fn create_test_config() -> OpenAiConfig {
        OpenAiConfig::new("test-key")
            .with_model("gpt-4o")
            .with_responses_api(true)
            .with_built_in_tool(OpenAiBuiltInTool::WebSearch)
    }

    fn create_test_message() -> ChatMessage {
        ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello, world!".to_string()),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn test_responses_client_creation() {
        let config = create_test_config();
        let client = OpenAiResponses::new(reqwest::Client::new(), config);

        // Test that the client was created successfully
        assert_eq!(client.config.common_params.model, "gpt-4o");
        assert!(client.config.use_responses_api);
        assert_eq!(client.config.built_in_tools.len(), 1);
    }

    #[test]
    fn test_responses_endpoint() {
        let config = create_test_config();
        let client = OpenAiResponses::new(reqwest::Client::new(), config);

        assert_eq!(
            client.responses_endpoint(),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn test_build_request_body_basic() {
        let config = create_test_config();
        let client = OpenAiResponses::new(reqwest::Client::new(), config);
        let messages = vec![create_test_message()];

        let body = client
            .build_request_body(&messages, None, None, false, false)
            .unwrap();

        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], false);
        assert_eq!(body["background"], false);

        // Check input format
        assert!(body["input"].is_object()); // Single message should be an object
    }

    #[test]
    fn test_parse_response_metadata() {
        let config = create_test_config();
        let client = OpenAiResponses::new(reqwest::Client::new(), config);

        let response_data = serde_json::json!({
            "id": "resp_123",
            "status": "completed",
            "created_at": 1234567890,
            "completed_at": 1234567900,
            "model": "gpt-4o",
            "background": true,
            "previous_response_id": "prev_resp_456"
        });

        let metadata = client.parse_response_metadata(response_data).unwrap();

        assert_eq!(metadata.id, "resp_123");
        assert!(matches!(metadata.status, ResponseStatus::Completed));
        assert_eq!(metadata.created_at, 1234567890);
        assert_eq!(metadata.completed_at, Some(1234567900));
        assert_eq!(metadata.model, "gpt-4o");
        assert!(metadata.background);
        assert_eq!(
            metadata.previous_response_id,
            Some("prev_resp_456".to_string())
        );
        assert!(metadata.error.is_none());
    }
}
