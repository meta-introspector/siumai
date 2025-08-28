//! `OpenAI` Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for `OpenAI`.

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::time::Instant;

use crate::error::LlmError;
use crate::params::{OpenAiParameterMapper, OpenAiParams, ParameterMapper};
use crate::stream::ChatStream;
use crate::traits::ChatCapability;
use crate::types::*;
use tracing::{debug, error, info};

/// Format JSON for logging based on environment or configuration
fn format_json_for_logging(value: &serde_json::Value) -> String {
    // Check if pretty JSON is requested via environment variable or global config
    let pretty_json = std::env::var("SIUMAI_PRETTY_JSON")
        .unwrap_or_default()
        .to_lowercase()
        == "true"
        || crate::tracing::get_pretty_json();

    if pretty_json {
        serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
    } else {
        serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
    }
}

/// Mask sensitive values in strings for security
fn mask_sensitive_value(value: &str) -> String {
    if !crate::tracing::get_mask_sensitive_values() {
        return value.to_string();
    }

    // Check if this looks like an API key or token
    if let Some(token) = value.strip_prefix("Bearer ") {
        if token.len() > 8 {
            format!("Bearer {}...{}", &token[..4], &token[token.len() - 4..])
        } else {
            "Bearer ***".to_string()
        }
    } else if value.starts_with("sk-") || value.starts_with("xai-") || value.starts_with("gsk_") {
        // OpenAI, xAI, Groq API keys
        if value.len() > 8 {
            format!("{}...{}", &value[..4], &value[value.len() - 4..])
        } else {
            "***".to_string()
        }
    } else if value.len() > 20
        && (value.contains("key") || value.contains("token") || value.contains("secret"))
    {
        // Generic long strings that might be sensitive
        if value.len() > 8 {
            format!("{}...{}", &value[..4], &value[value.len() - 4..])
        } else {
            "***".to_string()
        }
    } else {
        value.to_string()
    }
}

/// Format headers for logging based on pretty JSON configuration
fn format_headers_for_logging(headers: &reqwest::header::HeaderMap) -> String {
    let pretty_json = std::env::var("SIUMAI_PRETTY_JSON")
        .unwrap_or_default()
        .to_lowercase()
        == "true"
        || crate::tracing::get_pretty_json();

    let header_map: std::collections::HashMap<&str, String> = headers
        .iter()
        .map(|(k, v)| {
            let value = v.to_str().unwrap_or("<invalid>");
            let masked_value = if k.as_str().to_lowercase().contains("authorization")
                || k.as_str().to_lowercase().contains("key")
                || k.as_str().to_lowercase().contains("token")
            {
                mask_sensitive_value(value)
            } else {
                value.to_string()
            };
            (k.as_str(), masked_value)
        })
        .collect();

    if pretty_json {
        serde_json::to_string_pretty(&header_map).unwrap_or_else(|_| format!("{header_map:?}"))
    } else {
        serde_json::to_string(&header_map).unwrap_or_else(|_| format!("{header_map:?}"))
    }
}

use super::request::OpenAiRequestBuilder;
use super::types::*;
use super::utils::*;
use crate::request_factory::RequestBuilder;

/// `OpenAI` Chat Capability Implementation
#[derive(Clone)]
pub struct OpenAiChatCapability {
    pub api_key: SecretString,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub organization: Option<String>,
    pub project: Option<String>,
    pub http_config: HttpConfig,
    pub parameter_mapper: OpenAiParameterMapper,
    pub common_params: CommonParams,
    pub request_builder: OpenAiRequestBuilder,
}

impl OpenAiChatCapability {
    /// Create a new `OpenAI` chat capability instance
    pub fn new(
        api_key: SecretString,
        base_url: String,
        http_client: reqwest::Client,
        organization: Option<String>,
        project: Option<String>,
        http_config: HttpConfig,
        common_params: CommonParams,
    ) -> Self {
        let request_builder = OpenAiRequestBuilder::new(
            common_params.clone(),
            crate::params::openai::OpenAiParams::default(),
        );

        Self {
            api_key,
            base_url,
            http_client,
            organization,
            project,
            http_config,
            parameter_mapper: OpenAiParameterMapper,
            common_params,
            request_builder,
        }
    }

    /// Build the chat request body
    pub fn build_chat_request_body(
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
        if let Some(ref tools) = request.tools
            && !tools.is_empty()
        {
            body["tools"] = serde_json::to_value(tools)?;
        }

        // Clean up null values that might cause API errors
        self.clean_null_values(&mut body);

        Ok(body)
    }

    /// Clean null values from request body to prevent API errors
    fn clean_null_values(&self, body: &mut serde_json::Value) {
        if let serde_json::Value::Object(obj) = body {
            // Remove null values that can cause OpenAI API errors
            let keys_to_remove: Vec<String> = obj
                .iter()
                .filter_map(|(key, value)| {
                    if value.is_null() {
                        Some(key.clone())
                    } else {
                        None
                    }
                })
                .collect();

            for key in keys_to_remove {
                obj.remove(&key);
            }
        }
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
        let start_time = Instant::now();

        info!("Starting OpenAI chat request");

        // Use the request builder to create a properly configured ChatRequest
        let request =
            self.request_builder
                .build_chat_request(messages.clone(), tools.clone(), false)?;

        let headers = build_headers(
            self.api_key.expose_secret(),
            self.organization.as_deref(),
            self.project.as_deref(),
            &self.http_config.headers,
        )?;

        let body = self.build_chat_request_body(&request)?;
        let url = crate::utils::url::join_url(&self.base_url, "chat/completions");

        debug!(
            url = %url,
            request_body = %format_json_for_logging(&body),
            request_headers = %format_headers_for_logging(&headers),
            "Sending OpenAI API request"
        );

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let duration = start_time.elapsed();

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            error!(
                status_code = status.as_u16(),
                error_text = %error_text,
                duration_ms = duration.as_millis(),
                "OpenAI API request failed"
            );

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("OpenAI API error: {error_text}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        debug!(
            status_code = response.status().as_u16(),
            duration_ms = duration.as_millis(),
            response_headers = %format_headers_for_logging(response.headers()),
            "OpenAI API request successful"
        );

        // Get response body as text first for logging
        let response_text = response.text().await?;

        debug!(
            response_body = %response_text,
            "OpenAI API response body"
        );

        let openai_response: OpenAiChatResponse = serde_json::from_str(&response_text)?;
        let chat_response = self.parse_chat_response(openai_response)?;

        info!(
            duration_ms = duration.as_millis(),
            response_length = chat_response.content.all_text().len(),
            "OpenAI chat request completed"
        );

        Ok(chat_response)
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Use the request builder to create a properly configured ChatRequest
        let request = self
            .request_builder
            .build_chat_request(messages, tools, true)?;

        // Create streaming client
        let config = super::config::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: OpenAiParams::default(),
            http_config: self.http_config.clone(),
            web_search_config: crate::types::WebSearchConfig::default(),
            use_responses_api: false,
            previous_response_id: None,
            built_in_tools: Vec::new(),
        };

        let streaming = super::streaming::OpenAiStreaming::new(config, self.http_client.clone());
        streaming.create_chat_stream(request).await
    }
}

/// Legacy implementation for backward compatibility
impl OpenAiChatCapability {
    /// Chat with a `ChatRequest` (legacy method)
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let start_time = Instant::now();

        info!("Starting OpenAI chat request");

        let headers = build_headers(
            self.api_key.expose_secret(),
            self.organization.as_deref(),
            self.project.as_deref(),
            &self.http_config.headers,
        )?;

        let body = self.build_chat_request_body(&request)?;
        let url = crate::utils::url::join_url(&self.base_url, "chat/completions");

        debug!(
            url = %url,
            request_body = %format_json_for_logging(&body),
            request_headers = %format_headers_for_logging(&headers),
            "Sending OpenAI API request"
        );

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let duration = start_time.elapsed();

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            error!(
                status_code = status.as_u16(),
                error_text = %error_text,
                duration_ms = duration.as_millis(),
                "OpenAI API request failed"
            );

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("OpenAI API error: {error_text}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        debug!(
            status_code = response.status().as_u16(),
            duration_ms = duration.as_millis(),
            response_headers = %format_headers_for_logging(response.headers()),
            "OpenAI API request successful"
        );

        // Get response body as text first for logging
        let response_text = response.text().await?;

        debug!(
            response_body = %response_text,
            "OpenAI API response body"
        );

        let openai_response: OpenAiChatResponse = serde_json::from_str(&response_text)?;
        let chat_response = self.parse_chat_response(openai_response)?;

        info!(
            duration_ms = duration.as_millis(),
            response_length = chat_response.content.all_text().len(),
            "OpenAI chat request completed"
        );

        Ok(chat_response)
    }

    /// Chat stream with a `ChatRequest` (legacy method)
    pub async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        // Create streaming client with the request's configuration
        let config = super::config::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: request.common_params.clone(),
            openai_params: OpenAiParams::default(),
            http_config: self.http_config.clone(),
            web_search_config: crate::types::WebSearchConfig::default(),
            use_responses_api: false,
            previous_response_id: None,
            built_in_tools: Vec::new(),
        };

        let streaming = super::streaming::OpenAiStreaming::new(config, self.http_client.clone());
        streaming.create_chat_stream(request).await
    }
}
