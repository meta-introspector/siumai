//! Gemini Chat Capability Implementation
//!
//! This module implements the chat functionality for Google Gemini API.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde_json::json;
use std::time::Instant;

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::tracing::ProviderTracer;
use crate::traits::ChatCapability;
use crate::types::{
    ChatMessage, ChatResponse, FinishReason, MessageContent, ResponseMetadata, Tool, ToolCall,
    Usage,
};

use super::streaming::GeminiStreaming;
use super::types::{
    Content, FunctionCall, FunctionDeclaration, GeminiConfig, GeminiTool, GenerateContentRequest,
    GenerateContentResponse, Part,
};

/// Gemini chat capability implementation
#[derive(Debug, Clone)]
pub struct GeminiChatCapability {
    config: GeminiConfig,
    http_client: HttpClient,
    streaming: GeminiStreaming,
}

impl GeminiChatCapability {
    /// Create a new Gemini chat capability
    pub fn new(config: GeminiConfig, http_client: HttpClient) -> Self {
        let streaming = GeminiStreaming::new(http_client.clone());
        Self {
            config,
            http_client,
            streaming,
        }
    }

    /// Convert `ChatMessage` to Gemini Content
    fn convert_message_to_content(&self, message: &ChatMessage) -> Result<Content, LlmError> {
        let role = match message.role {
            crate::types::MessageRole::User => Some("user".to_string()),
            crate::types::MessageRole::Assistant => Some("model".to_string()),
            crate::types::MessageRole::System => None, // System messages are handled separately
            _ => {
                return Err(LlmError::InvalidInput(format!(
                    "Unsupported role: {:?}",
                    message.role
                )));
            }
        };

        let mut parts = Vec::new();

        // Add text content
        if let crate::types::MessageContent::Text(text) = &message.content {
            if !text.is_empty() {
                parts.push(Part::Text {
                    text: text.clone(),
                    thought: None,
                });
            }
        } else {
            // Handle other content types if needed
        }

        // Add tool calls
        if let Some(tool_calls) = &message.tool_calls {
            for tool_call in tool_calls {
                if let Some(function) = &tool_call.function {
                    let args = serde_json::from_str(&function.arguments).ok();

                    parts.push(Part::FunctionCall {
                        function_call: FunctionCall {
                            name: function.name.clone(),
                            args,
                        },
                    });
                }
            }
        }

        // Add tool results (if tool_call_id is present, this is a tool result message)
        if let Some(tool_call_id) = &message.tool_call_id {
            let response = match &message.content {
                crate::types::MessageContent::Text(text) => json!(text),
                _ => json!({}),
            };

            parts.push(Part::FunctionResponse {
                function_response: super::types::FunctionResponse {
                    name: tool_call_id.clone(),
                    response,
                },
            });
        }

        if parts.is_empty() {
            return Err(LlmError::InvalidInput("Message has no content".to_string()));
        }

        Ok(Content { role, parts })
    }

    /// Convert Tools to Gemini Tools
    fn convert_tools_to_gemini(&self, tools: &[Tool]) -> Result<Vec<GeminiTool>, LlmError> {
        let mut gemini_tools = Vec::new();
        let mut function_declarations = Vec::new();

        for tool in tools {
            if tool.r#type == "function" {
                let parameters = tool.function.parameters.clone();

                function_declarations.push(FunctionDeclaration {
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: Some(parameters),
                    response: None,
                });
            } else {
                return Err(LlmError::UnsupportedOperation(format!(
                    "Tool type {} not supported by Gemini",
                    tool.r#type
                )));
            }
        }

        if !function_declarations.is_empty() {
            gemini_tools.push(GeminiTool::FunctionDeclarations {
                function_declarations,
            });
        }

        Ok(gemini_tools)
    }

    /// Build the request body for Gemini API
    pub fn build_request_body(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<GenerateContentRequest, LlmError> {
        let mut contents = Vec::new();
        let mut system_instruction = None;

        // Process messages
        for message in messages {
            if message.role == crate::types::MessageRole::System {
                // Handle system message as system instruction
                if let crate::types::MessageContent::Text(text) = &message.content {
                    system_instruction = Some(Content::system_text(text.clone()));
                } else {
                    // Handle other content types if needed
                }
            } else {
                contents.push(self.convert_message_to_content(message)?);
            }
        }

        // Convert tools if provided
        let gemini_tools = if let Some(tools) = tools {
            if !tools.is_empty() {
                Some(self.convert_tools_to_gemini(tools)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(GenerateContentRequest {
            model: self.config.model.clone(), // Don't add "models/" prefix here
            contents,
            system_instruction,
            tools: gemini_tools,
            tool_config: None,
            safety_settings: self.config.safety_settings.clone(),
            generation_config: self.config.generation_config.clone(),
            cached_content: None,
        })
    }

    /// Convert Gemini response to `ChatResponse`
    fn convert_response(
        &self,
        response: GenerateContentResponse,
    ) -> Result<ChatResponse, LlmError> {
        if response.candidates.is_empty() {
            return Err(LlmError::api_error(400, "No candidates in response"));
        }

        let candidate = &response.candidates[0];

        let content = candidate
            .content
            .as_ref()
            .ok_or_else(|| LlmError::api_error(400, "No content in candidate"))?;

        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        // Process parts
        let mut thinking_content = String::new();

        for part in &content.parts {
            match part {
                Part::Text { text, thought } => {
                    if thought.unwrap_or(false) {
                        // This is thinking content - collect it separately
                        if !thinking_content.is_empty() {
                            thinking_content.push('\n');
                        }
                        thinking_content.push_str(text);
                    } else {
                        // Regular text content
                        if !text_content.is_empty() {
                            text_content.push('\n');
                        }
                        text_content.push_str(text);
                    }
                }
                Part::FunctionCall { function_call } => {
                    let arguments = if let Some(args) = &function_call.args {
                        serde_json::to_string(args).unwrap_or_default()
                    } else {
                        "{}".to_string()
                    };

                    tool_calls.push(ToolCall {
                        id: format!("call_{}", uuid::Uuid::new_v4()),
                        r#type: "function".to_string(),
                        function: Some(crate::types::FunctionCall {
                            name: function_call.name.clone(),
                            arguments,
                        }),
                    });
                }
                _ => {
                    // Handle other part types if needed
                }
            }
        }

        // Calculate usage
        let usage = response
            .usage_metadata
            .as_ref()
            .map(|usage_metadata| Usage {
                prompt_tokens: usage_metadata.prompt_token_count.unwrap_or(0) as u32,
                completion_tokens: usage_metadata.candidates_token_count.unwrap_or(0) as u32,
                total_tokens: usage_metadata.total_token_count.unwrap_or(0) as u32,
                cached_tokens: None,
                reasoning_tokens: usage_metadata.thoughts_token_count.map(|t| t as u32),
            });

        // Determine finish reason
        let finish_reason = candidate.finish_reason.as_ref().map(|reason| match reason {
            super::types::FinishReason::Stop => FinishReason::Stop,
            super::types::FinishReason::MaxTokens => FinishReason::Length,
            super::types::FinishReason::Safety => FinishReason::ContentFilter,
            _ => FinishReason::Other("unknown".to_string()),
        });

        // Create content
        let content = if text_content.is_empty() {
            MessageContent::Text(String::new())
        } else {
            MessageContent::Text(text_content)
        };

        // Create metadata
        let _metadata = ResponseMetadata {
            id: response.response_id.clone(),
            model: Some(self.config.model.clone()),
            created: Some(chrono::Utc::now()),
            provider: "gemini".to_string(),
            request_id: None,
        };

        // Create metadata (no longer storing thinking content here to avoid duplication)
        let provider_data = std::collections::HashMap::new();

        Ok(ChatResponse {
            id: None, // Gemini doesn't provide response IDs
            content,
            model: None, // Will be set from request context
            usage,
            finish_reason,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            thinking: if thinking_content.is_empty() {
                None
            } else {
                Some(thinking_content)
            },
            metadata: provider_data,
        })
    }

    /// Make a request to the Gemini API
    async fn make_request(
        &self,
        request: GenerateContentRequest,
    ) -> Result<GenerateContentResponse, LlmError> {
        let start_time = Instant::now();

        // Create tracer with model information
        let tracer = ProviderTracer::new("gemini").with_model(&self.config.model);

        let url = crate::utils::url::join_url(
            &self.config.base_url,
            &format!("models/{}:generateContent", self.config.model),
        );

        tracer.trace_request_start("POST", &url);

        // Create headers for tracing
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse().unwrap());
        headers.insert("x-goog-api-key", self.config.api_key.parse().unwrap());

        // Convert request to JSON for tracing
        let request_json = serde_json::to_value(&request)
            .map_err(|e| LlmError::ParseError(format!("Failed to serialize request: {e}")))?;

        tracer.trace_request_details(&headers, &request_json);

        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &self.config.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();

            tracer.trace_request_error(status_code, &error_text, start_time);

            return Err(LlmError::api_error(
                status_code,
                format!("Gemini API error: {status_code} - {error_text}"),
            ));
        }

        tracer.trace_response_success(response.status().as_u16(), start_time, response.headers());

        // Get response body as text first for logging
        let response_text = response
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        tracer.trace_response_body(&response_text);

        let gemini_response: GenerateContentResponse = serde_json::from_str(&response_text)
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {e}")))?;

        // Calculate response length for completion tracing
        let response_length = response_text.len();
        tracer.trace_request_complete(start_time, response_length);

        Ok(gemini_response)
    }
}

#[async_trait]
impl ChatCapability for GeminiChatCapability {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let request = self.build_request_body(&messages, tools.as_deref())?;
        let response = self.make_request(request).await?;
        self.convert_response(response)
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let request = self.build_request_body(&messages, tools.as_deref())?;

        let url = crate::utils::url::join_url(
            &self.config.base_url,
            &format!("models/{}:streamGenerateContent?alt=sse", self.config.model),
        );

        // Use the dedicated streaming capability
        self.streaming
            .clone()
            .create_chat_stream(url, self.config.api_key.clone(), request)
            .await
    }
}
