//! Ollama Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for Ollama using the /api/chat endpoint.

use async_trait::async_trait;
use futures_util::StreamExt;

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::traits::ChatCapability;
use crate::types::*;

use super::config::OllamaParams;
use super::types::*;
use super::utils::*;

/// Ollama Chat Capability Implementation
pub struct OllamaChatCapability {
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
    pub ollama_params: OllamaParams,
}

impl OllamaChatCapability {
    /// Creates a new Ollama chat capability
    pub const fn new(
        base_url: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
        ollama_params: OllamaParams,
    ) -> Self {
        Self {
            base_url,
            http_client,
            http_config,
            ollama_params,
        }
    }

    /// Build chat request body
    pub fn build_chat_request_body(&self, request: &ChatRequest) -> Result<OllamaChatRequest, LlmError> {
        // Get model from request
        let model = request.common_params.model.clone();
        if model.is_empty() {
            return Err(LlmError::ConfigurationError("Model is required".to_string()));
        }

        validate_model_name(&model)?;

        // Convert messages
        let messages: Vec<OllamaChatMessage> = request
            .messages
            .iter()
            .map(convert_chat_message)
            .collect();

        // Convert tools if present
        let tools = request.tools.as_ref().map(|tools| {
            tools.iter().map(convert_tool).collect()
        });

        // Build model options
        let options = build_model_options(
            request.common_params.temperature,
            request.common_params.max_tokens,
            request.common_params.top_p,
            None, // frequency_penalty not in CommonParams
            None, // presence_penalty not in CommonParams
            self.ollama_params.options.as_ref(),
        );

        // Build format if specified
        let format = if let Some(format_str) = &self.ollama_params.format {
            if format_str == "json" {
                Some(serde_json::Value::String("json".to_string()))
            } else {
                // Try to parse as JSON schema
                match serde_json::from_str(format_str) {
                    Ok(schema) => Some(schema),
                    Err(_) => Some(serde_json::Value::String(format_str.clone())),
                }
            }
        } else {
            None
        };

        Ok(OllamaChatRequest {
            model,
            messages,
            tools,
            stream: Some(request.stream),
            format,
            options: if options.is_empty() { None } else { Some(options) },
            keep_alive: self.ollama_params.keep_alive.clone(),
        })
    }

    /// Parse chat response
    fn parse_chat_response(&self, response: OllamaChatResponse) -> ChatResponse {
        let message = convert_from_ollama_message(&response.message);

        // Calculate usage if metrics are available
        let usage = if response.prompt_eval_count.is_some() || response.eval_count.is_some() {
            Some(Usage {
                prompt_tokens: response.prompt_eval_count.unwrap_or(0),
                completion_tokens: response.eval_count.unwrap_or(0),
                total_tokens: response.prompt_eval_count.unwrap_or(0) + response.eval_count.unwrap_or(0),
                cached_tokens: None,
                reasoning_tokens: None,
            })
        } else {
            None
        };

        // Parse finish reason
        let finish_reason = response.done_reason.as_deref().map(|reason| {
            match reason {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                _ => FinishReason::Other(reason.to_string()),
            }
        }).or({
            if response.done { Some(FinishReason::Stop) } else { None }
        });

        // Create metadata with performance metrics
        let mut metadata = std::collections::HashMap::new();
        if let Some(tokens_per_second) = calculate_tokens_per_second(response.eval_count, response.eval_duration) {
            metadata.insert("tokens_per_second".to_string(), serde_json::Value::Number(
                serde_json::Number::from_f64(tokens_per_second).unwrap_or_else(|| serde_json::Number::from(0))
            ));
        }
        if let Some(total_duration) = response.total_duration {
            metadata.insert("total_duration_ms".to_string(), serde_json::Value::Number(
                serde_json::Number::from(total_duration / 1_000_000)
            ));
        }

        ChatResponse {
            id: Some(format!("ollama-{}", chrono::Utc::now().timestamp_millis())),
            content: message.content,
            model: Some(response.model),
            usage,
            finish_reason,
            tool_calls: message.tool_calls,
            thinking: None,
            metadata,
        }
    }
}

#[async_trait]
impl ChatCapability for OllamaChatCapability {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let request = ChatRequest {
            messages,
            tools,
            common_params: Default::default(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: false,
        };
        self.chat(request).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Note: This method should not be called directly.
        // Use OllamaClient::chat_stream instead which provides proper common_params.
        let mut request = ChatRequest {
            messages,
            tools,
            common_params: Default::default(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: true,
        };
        request.stream = true;

        let headers = build_headers(&self.http_config.headers)?;
        let body = self.build_chat_request_body(&request)?;
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpError(format!(
                "Chat request failed: {status} - {error_text}"
            )));
        }

        // Create stream from response
        let stream = response.bytes_stream();
        let mapped_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    for line in chunk_str.lines() {
                        if let Ok(Some(json_value)) = parse_streaming_line(line) {
                            if let Ok(ollama_response) = serde_json::from_value::<OllamaChatResponse>(json_value) {
                                let content_delta = ollama_response.message.content.clone();
                                return Ok(ChatStreamEvent::ContentDelta {
                                    delta: content_delta,
                                    index: Some(0),
                                });
                            }
                        }
                    }
                    Ok(ChatStreamEvent::ContentDelta {
                        delta: String::new(),
                        index: Some(0),
                    })
                }
                Err(e) => Err(LlmError::StreamError(format!("Stream error: {e}"))),
            }
        });

        Ok(Box::pin(mapped_stream))
    }
}

impl OllamaChatCapability {
    /// Chat implementation (internal)
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let headers = build_headers(&self.http_config.headers)?;
        let body = self.build_chat_request_body(&request)?;
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpError(format!(
                "Chat request failed: {status} - {error_text}"
            )));
        }

        let ollama_response: OllamaChatResponse = response.json().await?;
        Ok(self.parse_chat_response(ollama_response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CommonParams;

    #[test]
    fn test_build_chat_request_body() {
        let capability = OllamaChatCapability::new(
            "http://localhost:11434".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
            OllamaParams::default(),
        );

        let common_params = CommonParams {
            model: "llama3.2".to_string(),
            temperature: Some(0.7),
            ..Default::default()
        };

        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: crate::types::MessageRole::User,
                content: crate::types::MessageContent::Text("Hello".to_string()),
                metadata: crate::types::MessageMetadata::default(),
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            common_params,
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: false,
        };

        let body = capability.build_chat_request_body(&request).unwrap();
        assert_eq!(body.model, "llama3.2");
        assert_eq!(body.messages.len(), 1);
        assert_eq!(body.messages[0].content, "Hello");
        assert_eq!(body.stream, Some(false));
    }

    #[test]
    fn test_parse_chat_response() {
        let capability = OllamaChatCapability::new(
            "http://localhost:11434".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
            OllamaParams::default(),
        );

        let ollama_response = OllamaChatResponse {
            model: "llama3.2".to_string(),
            created_at: "2023-01-01T00:00:00Z".to_string(),
            message: OllamaChatMessage {
                role: "assistant".to_string(),
                content: "Hello there!".to_string(),
                images: None,
                tool_calls: None,
            },
            done: true,
            done_reason: Some("stop".to_string()),
            total_duration: Some(1_000_000_000),
            load_duration: Some(100_000_000),
            prompt_eval_count: Some(10),
            prompt_eval_duration: Some(200_000_000),
            eval_count: Some(20),
            eval_duration: Some(700_000_000),
        };

        let response = capability.parse_chat_response(ollama_response);
        assert_eq!(response.model, Some("llama3.2".to_string()));
        assert_eq!(response.content, crate::types::MessageContent::Text("Hello there!".to_string()));
        assert_eq!(response.finish_reason, Some(crate::types::FinishReason::Stop));
        assert!(response.usage.is_some());
        assert!(response.metadata.contains_key("total_duration_ms"));
    }
}
