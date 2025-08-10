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

        // Use the same request building logic as non-streaming
        let chat_capability = super::chat::XaiChatCapability::new(
            self.config.api_key.clone(),
            self.config.base_url.clone(),
            self.http_client.clone(),
            self.config.http_config.clone(),
        );

        let mut body = chat_capability.build_chat_request_body(&request)?;

        // Override with streaming-specific settings
        body["stream"] = serde_json::Value::Bool(true);

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
