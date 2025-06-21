//! Ollama Streaming Implementation
//!
//! This module provides Ollama-specific streaming functionality for chat and completion.

use futures::{Stream, StreamExt};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::Mutex;

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::utils::Utf8StreamDecoder;

use super::types::*;
use super::utils::parse_streaming_line;

/// Ollama streaming client for handling JSON Lines format
#[derive(Clone)]
pub struct OllamaStreaming {
    /// HTTP client
    http_client: reqwest::Client,
    /// JSON line buffer for handling incomplete lines
    json_buffer: Arc<Mutex<String>>,
}

impl OllamaStreaming {
    /// Create a new Ollama streaming client
    pub fn new(http_client: reqwest::Client) -> Self {
        Self {
            http_client,
            json_buffer: Arc::new(Mutex::new(String::new())),
        }
    }

    /// Create a streaming chat completion request
    pub async fn create_chat_stream(
        &self,
        url: String,
        headers: reqwest::header::HeaderMap,
        body: OllamaChatRequest,
    ) -> Result<ChatStream, LlmError> {
        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::HttpError(format!(
                "Ollama API error {status}: {error_text}"
            )));
        }

        // Create the stream
        let stream = self.clone().create_event_stream(response).await?;
        Ok(Box::pin(stream))
    }

    /// Create a streaming completion request
    pub async fn create_completion_stream(
        &self,
        url: String,
        headers: reqwest::header::HeaderMap,
        body: OllamaGenerateRequest,
    ) -> Result<ChatStream, LlmError> {
        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::HttpError(format!(
                "Ollama API error {status}: {error_text}"
            )));
        }

        // Create the stream
        let stream = self
            .clone()
            .create_completion_event_stream(response)
            .await?;
        Ok(Box::pin(stream))
    }

    /// Create an event stream from the HTTP response for chat
    async fn create_event_stream(
        self,
        response: reqwest::Response,
    ) -> Result<impl Stream<Item = Result<ChatStreamEvent, LlmError>>, LlmError> {
        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));

        // Create a UTF-8 decoder for this stream
        let decoder = Arc::new(StdMutex::new(Utf8StreamDecoder::new()));
        let decoder_for_flush = decoder.clone();
        let streaming_for_flush = self.clone();

        // Create a stream that handles UTF-8 decoding
        let decoded_stream = stream.filter_map(move |chunk_result| {
            let streaming = self.clone();
            let decoder = decoder.clone();
            async move {
                match chunk_result {
                    Ok(chunk) => {
                        // Use UTF-8 decoder to handle incomplete sequences
                        let decoded_chunk = {
                            let mut decoder = decoder.lock().unwrap();
                            decoder.decode(&chunk)
                        };

                        if !decoded_chunk.is_empty() {
                            streaming.parse_json_lines_buffered(&decoded_chunk).await
                        } else {
                            None
                        }
                    }
                    Err(e) => Some(Err(e)),
                }
            }
        });

        // Add a final flush operation
        let flush_stream = futures::stream::once(async move {
            let remaining = {
                let mut decoder = decoder_for_flush.lock().unwrap();
                decoder.flush()
            };

            if !remaining.is_empty() {
                streaming_for_flush
                    .parse_json_lines_buffered(&remaining)
                    .await
            } else {
                // Also flush any remaining JSON buffer content
                streaming_for_flush.flush_json_buffer().await
            }
        })
        .filter_map(|result| async move { result });

        Ok(decoded_stream.chain(flush_stream))
    }

    /// Create an event stream from the HTTP response for completion
    async fn create_completion_event_stream(
        self,
        response: reqwest::Response,
    ) -> Result<impl Stream<Item = Result<ChatStreamEvent, LlmError>>, LlmError> {
        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));

        // Create a UTF-8 decoder for this stream
        let decoder = Arc::new(StdMutex::new(Utf8StreamDecoder::new()));
        let decoder_for_flush = decoder.clone();
        let streaming_for_flush = self.clone();

        // Create a stream that handles UTF-8 decoding
        let decoded_stream = stream.filter_map(move |chunk_result| {
            let streaming = self.clone();
            let decoder = decoder.clone();
            async move {
                match chunk_result {
                    Ok(chunk) => {
                        // Use UTF-8 decoder to handle incomplete sequences
                        let decoded_chunk = {
                            let mut decoder = decoder.lock().unwrap();
                            decoder.decode(&chunk)
                        };

                        if !decoded_chunk.is_empty() {
                            streaming
                                .parse_completion_lines_buffered(&decoded_chunk)
                                .await
                        } else {
                            None
                        }
                    }
                    Err(e) => Some(Err(e)),
                }
            }
        });

        // Add a final flush operation
        let flush_stream = futures::stream::once(async move {
            let remaining = {
                let mut decoder = decoder_for_flush.lock().unwrap();
                decoder.flush()
            };

            if !remaining.is_empty() {
                streaming_for_flush
                    .parse_completion_lines_buffered(&remaining)
                    .await
            } else {
                // Also flush any remaining JSON buffer content
                streaming_for_flush.flush_json_buffer().await
            }
        })
        .filter_map(|result| async move { result });

        Ok(decoded_stream.chain(flush_stream))
    }

    /// Parse JSON Lines with buffering for incomplete lines (chat)
    async fn parse_json_lines_buffered(
        &self,
        chunk: &str,
    ) -> Option<Result<ChatStreamEvent, LlmError>> {
        // Add chunk to buffer
        {
            let mut buffer = self.json_buffer.lock().await;
            buffer.push_str(chunk);
        }

        // Process complete lines from buffer
        self.process_buffered_chat_lines().await
    }

    /// Parse JSON Lines with buffering for incomplete lines (completion)
    async fn parse_completion_lines_buffered(
        &self,
        chunk: &str,
    ) -> Option<Result<ChatStreamEvent, LlmError>> {
        // Add chunk to buffer
        {
            let mut buffer = self.json_buffer.lock().await;
            buffer.push_str(chunk);
        }

        // Process complete lines from buffer
        self.process_buffered_completion_lines().await
    }

    /// Process complete lines from the JSON buffer (chat)
    async fn process_buffered_chat_lines(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        let mut buffer = self.json_buffer.lock().await;

        // Find the last complete line (ending with \n)
        if let Some(last_newline_pos) = buffer.rfind('\n') {
            // Extract complete lines
            let complete_lines = buffer[..=last_newline_pos].to_string();
            // Keep incomplete line in buffer
            let remaining = buffer[last_newline_pos + 1..].to_string();
            *buffer = remaining;

            // Release the lock before processing
            drop(buffer);

            // Process the complete lines
            return Self::parse_chat_json_lines(&complete_lines);
        }

        // No complete lines yet
        None
    }

    /// Process complete lines from the JSON buffer (completion)
    async fn process_buffered_completion_lines(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        let mut buffer = self.json_buffer.lock().await;

        // Find the last complete line (ending with \n)
        if let Some(last_newline_pos) = buffer.rfind('\n') {
            // Extract complete lines
            let complete_lines = buffer[..=last_newline_pos].to_string();
            // Keep incomplete line in buffer
            let remaining = buffer[last_newline_pos + 1..].to_string();
            *buffer = remaining;

            // Release the lock before processing
            drop(buffer);

            // Process the complete lines
            return Self::parse_completion_json_lines(&complete_lines);
        }

        // No complete lines yet
        None
    }

    /// Flush any remaining content in the JSON buffer
    async fn flush_json_buffer(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        let remaining = {
            let mut buffer = self.json_buffer.lock().await;
            let content = buffer.clone();
            buffer.clear();
            content
        };

        if !remaining.is_empty() {
            // Try to parse remaining content as if it were complete
            Self::parse_chat_json_lines(&remaining)
        } else {
            None
        }
    }

    /// Parse JSON Lines for chat (original method for complete lines)
    fn parse_chat_json_lines(chunk: &str) -> Option<Result<ChatStreamEvent, LlmError>> {
        for line in chunk.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            match parse_streaming_line(line) {
                Ok(Some(json_value)) => {
                    match serde_json::from_value::<OllamaChatResponse>(json_value) {
                        Ok(ollama_response) => {
                            let content_delta = ollama_response.message.content;
                            return Some(Ok(ChatStreamEvent::ContentDelta {
                                delta: content_delta,
                                index: Some(0),
                            }));
                        }
                        Err(e) => {
                            return Some(Err(LlmError::ParseError(format!(
                                "Failed to parse chat response: {e}"
                            ))));
                        }
                    }
                }
                Ok(None) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
        None
    }

    /// Parse JSON Lines for completion (original method for complete lines)
    fn parse_completion_json_lines(chunk: &str) -> Option<Result<ChatStreamEvent, LlmError>> {
        for line in chunk.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            match parse_streaming_line(line) {
                Ok(Some(json_value)) => {
                    match serde_json::from_value::<OllamaGenerateResponse>(json_value) {
                        Ok(ollama_response) => {
                            let content = ollama_response.response;
                            return Some(Ok(ChatStreamEvent::ContentDelta {
                                delta: content,
                                index: Some(0),
                            }));
                        }
                        Err(e) => {
                            return Some(Err(LlmError::ParseError(format!(
                                "Failed to parse completion response: {e}"
                            ))));
                        }
                    }
                }
                Ok(None) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
        None
    }
}
