//! Gemini Streaming Implementation
//!
//! This module provides Gemini-specific streaming functionality for chat.

use futures::{Stream, StreamExt};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::Mutex;

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::utils::Utf8StreamDecoder;

use super::types::{GenerateContentRequest, GenerateContentResponse, Part};

/// Gemini streaming client for handling JSON format
#[derive(Debug, Clone)]
pub struct GeminiStreaming {
    /// HTTP client
    http_client: reqwest::Client,
    /// JSON buffer for handling incomplete JSON objects
    json_buffer: Arc<Mutex<String>>,
}

impl GeminiStreaming {
    /// Create a new Gemini streaming client
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
        api_key: String,
        body: GenerateContentRequest,
    ) -> Result<ChatStream, LlmError> {
        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("Gemini streaming API error: {status_code} - {error_text}"),
            ));
        }

        // Create the stream
        let stream = self.clone().create_event_stream(response).await?;
        Ok(Box::pin(stream))
    }

    /// Create an event stream from the HTTP response
    async fn create_event_stream(
        self,
        response: reqwest::Response,
    ) -> Result<impl Stream<Item = Result<ChatStreamEvent, LlmError>>, LlmError> {
        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(e.to_string())));

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
                            streaming.parse_json_buffered(&decoded_chunk).await
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
                streaming_for_flush.parse_json_buffered(&remaining).await
            } else {
                // Also flush any remaining JSON buffer content
                streaming_for_flush.flush_json_buffer().await
            }
        })
        .filter_map(|result| async move { result });

        Ok(decoded_stream.chain(flush_stream))
    }

    /// Parse JSON with buffering for incomplete JSON objects
    async fn parse_json_buffered(&self, chunk: &str) -> Option<Result<ChatStreamEvent, LlmError>> {
        // Add chunk to buffer
        {
            let mut buffer = self.json_buffer.lock().await;
            buffer.push_str(chunk);
        }

        // Try to extract complete JSON objects from buffer
        self.process_buffered_json().await
    }

    /// Process complete JSON objects from the buffer
    async fn process_buffered_json(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        let mut buffer = self.json_buffer.lock().await;

        // Gemini sends JSON objects separated by newlines, but they can be multi-line
        // We need to find complete JSON objects by counting braces
        let mut brace_count = 0;
        let mut start_pos = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, ch) in buffer.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match ch {
                '"' if !escape_next => in_string = !in_string,
                '\\' if in_string => escape_next = true,
                '{' if !in_string => {
                    if brace_count == 0 {
                        start_pos = i;
                    }
                    brace_count += 1;
                }
                '}' if !in_string => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        // Found a complete JSON object
                        let json_str = buffer[start_pos..=i].to_string();
                        let remaining = buffer[i + 1..].to_string();
                        *buffer = remaining;

                        // Release the lock before processing
                        drop(buffer);

                        // Process the complete JSON object
                        return Self::parse_json_object(&json_str);
                    }
                }
                _ => {}
            }
        }

        // No complete JSON objects yet
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

        if !remaining.trim().is_empty() {
            // Try to parse remaining content as if it were complete
            Self::parse_json_object(&remaining)
        } else {
            None
        }
    }

    /// Parse a complete JSON object
    fn parse_json_object(json_str: &str) -> Option<Result<ChatStreamEvent, LlmError>> {
        let json_str = json_str.trim();
        if json_str.is_empty() {
            return None;
        }

        match serde_json::from_str::<GenerateContentResponse>(json_str) {
            Ok(response) => {
                if let Some(candidate) = response.candidates.first() {
                    if let Some(content) = &candidate.content {
                        for part in &content.parts {
                            if let Part::Text { text, thought } = part {
                                // Handle both regular text and thought summaries
                                let event = if thought.unwrap_or(false) {
                                    ChatStreamEvent::ThinkingDelta {
                                        delta: text.clone(),
                                    }
                                } else {
                                    ChatStreamEvent::ContentDelta {
                                        delta: text.clone(),
                                        index: None,
                                    }
                                };
                                return Some(Ok(event));
                            }
                        }
                    }
                }
                // Return empty delta if no content found
                Some(Ok(ChatStreamEvent::ContentDelta {
                    delta: String::new(),
                    index: None,
                }))
            }
            Err(e) => {
                // Return parse error
                Some(Err(LlmError::ParseError(format!(
                    "Failed to parse Gemini response: {e}"
                ))))
            }
        }
    }
}
