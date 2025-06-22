//! `xAI` Streaming Implementation
//!
//! Implements streaming chat completions for the `xAI` provider.

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::types::*;

use super::config::XaiConfig;
use super::utils::*;

/// `xAI` Streaming Client
pub struct XaiStreaming {
    config: XaiConfig,
    http_client: reqwest::Client,
}

impl XaiStreaming {
    /// Create a new `xAI` streaming client
    pub const fn new(config: XaiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Create a chat stream
    pub async fn create_chat_stream(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let headers = build_headers(&self.config.api_key, &self.config.http_config.headers)?;

        // Build request body with streaming enabled
        let mut body = serde_json::json!({
            "model": request.common_params.model,
            "messages": convert_messages(&request.messages)?,
            "stream": true
        });

        // Add common parameters
        if let Some(temp) = request.common_params.temperature {
            body["temperature"] =
                serde_json::Value::Number(serde_json::Number::from_f64(temp as f64).unwrap());
        }
        if let Some(max_tokens) = request.common_params.max_tokens {
            body["max_tokens"] = serde_json::Value::Number(serde_json::Number::from(max_tokens));
        }
        if let Some(top_p) = request.common_params.top_p {
            body["top_p"] =
                serde_json::Value::Number(serde_json::Number::from_f64(top_p as f64).unwrap());
        }

        // Add tools if provided
        if let Some(ref tools) = request.tools {
            if !tools.is_empty() {
                body["tools"] = serde_json::to_value(tools)?;
            }
        }

        let url = format!("{}/chat/completions", self.config.base_url);

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

            return Err(LlmError::HttpError(format!(
                "xAI API error ({}): {error_text}",
                status.as_u16()
            )));
        }

        // For now, return an error indicating streaming is not fully implemented
        Err(LlmError::UnsupportedOperation(
            "xAI streaming not fully implemented yet".to_string(),
        ))
    }
}
