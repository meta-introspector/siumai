//! Common Streaming Utilities
//!
//! This module provides common utilities for handling streaming responses
//! across different providers, including line buffering, UTF-8 handling,
//! and unified SSE processing using eventsource-stream.

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::utils::SseStreamExt;
use eventsource_stream::Event;
use futures_util::StreamExt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Type alias for SSE event conversion future
type SseEventFuture<'a> =
    Pin<Box<dyn Future<Output = Option<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>;

/// Type alias for JSON event conversion future
type JsonEventFuture<'a> =
    Pin<Box<dyn Future<Output = Option<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>;

/// Generic line buffer for handling incomplete lines in streaming responses
#[derive(Debug, Clone)]
pub struct LineBuffer {
    /// Buffer for incomplete lines
    buffer: Arc<Mutex<String>>,
}

impl LineBuffer {
    /// Create a new line buffer
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(String::new())),
        }
    }

    /// Add chunk to buffer and extract complete lines
    pub async fn add_chunk(&self, chunk: &str) -> Vec<String> {
        let mut buffer = self.buffer.lock().await;
        buffer.push_str(chunk);

        // Find complete lines (ending with \n)
        let mut lines = Vec::new();
        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].to_string();
            *buffer = buffer[newline_pos + 1..].to_string();
            if !line.trim().is_empty() {
                lines.push(line);
            }
        }

        lines
    }

    /// Flush any remaining content in the buffer
    pub async fn flush(&self) -> Option<String> {
        let mut buffer = self.buffer.lock().await;
        if buffer.trim().is_empty() {
            None
        } else {
            let content = buffer.clone();
            buffer.clear();
            Some(content)
        }
    }
}

impl Default for LineBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON object buffer for handling incomplete JSON objects in streaming responses
#[derive(Debug, Clone)]
pub struct JsonBuffer {
    /// Buffer for incomplete JSON objects
    buffer: Arc<Mutex<String>>,
}

impl JsonBuffer {
    /// Create a new JSON buffer
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(String::new())),
        }
    }

    /// Add chunk to buffer and extract complete JSON objects
    pub async fn add_chunk(&self, chunk: &str) -> Vec<String> {
        let mut buffer = self.buffer.lock().await;
        buffer.push_str(chunk);

        let mut objects = Vec::new();
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
                        objects.push(json_str);

                        // Update buffer to remove processed object
                        let remaining = buffer[i + 1..].to_string();
                        *buffer = remaining;

                        // Reset for next object search
                        return objects; // Return early to avoid index issues
                    }
                }
                _ => {}
            }
        }

        objects
    }

    /// Flush any remaining content in the buffer
    pub async fn flush(&self) -> Option<String> {
        let mut buffer = self.buffer.lock().await;
        if buffer.trim().is_empty() {
            None
        } else {
            let content = buffer.clone();
            buffer.clear();
            Some(content)
        }
    }
}

impl Default for JsonBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for converting provider-specific SSE events to ChatStreamEvent
pub trait SseEventConverter: Send + Sync {
    /// Convert an SSE event to a ChatStreamEvent
    fn convert_event(&self, event: Event) -> SseEventFuture<'_>;

    /// Handle the end of stream
    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        None
    }
}

/// Trait for converting JSON data to ChatStreamEvent (for providers like Gemini)
pub trait JsonEventConverter: Send + Sync {
    /// Convert JSON data to a ChatStreamEvent
    fn convert_json<'a>(&'a self, json_data: &'a str) -> JsonEventFuture<'a>;
}

/// Stream processor for all providers using eventsource-stream
pub struct StreamProcessor;

impl StreamProcessor {
    /// Create a chat stream from an HTTP response using eventsource-stream
    ///
    /// This is the unified method that all providers should use for SSE streaming.
    /// It handles:
    /// - UTF-8 decoding across chunk boundaries
    /// - SSE parsing and validation
    /// - Event conversion using provider-specific converters
    /// - Error handling and recovery
    pub async fn create_sse_stream<C>(
        response: reqwest::Response,
        converter: C,
    ) -> Result<ChatStream, LlmError>
    where
        C: SseEventConverter + Clone + 'static,
    {
        let byte_stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));

        // Use eventsource-stream to handle SSE parsing and UTF-8 decoding
        let sse_stream = byte_stream.into_sse_stream();

        // Convert SSE events to ChatStreamEvent using provider-specific converter
        let chat_stream = sse_stream.filter_map(move |event_result| {
            let converter = converter.clone();
            async move {
                match event_result {
                    Ok(event) => {
                        // Handle [DONE] events
                        if event.data == "[DONE]" {
                            return converter.handle_stream_end();
                        }

                        // Skip empty events
                        if event.data.trim().is_empty() {
                            return None;
                        }

                        // Convert using provider-specific logic
                        converter.convert_event(event).await
                    }
                    Err(e) => Some(Err(LlmError::ParseError(format!("SSE parsing error: {e}")))),
                }
            }
        });

        // Explicitly type the boxed stream to help the compiler
        let boxed_stream: ChatStream = Box::pin(chat_stream);
        Ok(boxed_stream)
    }

    /// Create a chat stream for JSON-based streaming (like Gemini)
    ///
    /// Some providers use JSON streaming instead of SSE. This method handles
    /// JSON object parsing across chunk boundaries using UTF-8 safe processing.
    pub async fn create_json_stream<C>(
        response: reqwest::Response,
        converter: C,
    ) -> Result<ChatStream, LlmError>
    where
        C: JsonEventConverter + Clone + 'static,
    {
        let byte_stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));

        // Use eventsource-stream for UTF-8 handling, then parse as JSON
        let sse_stream = byte_stream.into_sse_stream();

        let chat_stream = sse_stream.filter_map(move |event_result| {
            let converter = converter.clone();
            async move {
                match event_result {
                    Ok(event) => {
                        // For JSON streaming, we treat the data as raw JSON
                        if event.data.trim().is_empty() {
                            return None;
                        }

                        converter.convert_json(&event.data).await
                    }
                    Err(e) => Some(Err(LlmError::ParseError(format!(
                        "JSON parsing error: {e}"
                    )))),
                }
            }
        });

        // Explicitly type the boxed stream to help the compiler
        let boxed_stream: ChatStream = Box::pin(chat_stream);
        Ok(boxed_stream)
    }
}

/// Helper macro to create SSE event converters
#[macro_export]
macro_rules! impl_sse_converter {
    ($converter_type:ty, $event_type:ty, $convert_fn:ident) => {
        impl $crate::utils::streaming::SseEventConverter for $converter_type {
            fn convert_event(
                &self,
                event: eventsource_stream::Event,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Option<
                                Result<$crate::stream::ChatStreamEvent, $crate::error::LlmError>,
                            >,
                        > + Send
                        + Sync
                        + '_,
                >,
            > {
                Box::pin(async move {
                    match serde_json::from_str::<$event_type>(&event.data) {
                        Ok(parsed_event) => Some(Ok(self.$convert_fn(parsed_event))),
                        Err(e) => Some(Err($crate::error::LlmError::ParseError(format!(
                            "Failed to parse event: {e}"
                        )))),
                    }
                })
            }
        }
    };
}

/// Helper macro to create JSON event converters
#[macro_export]
macro_rules! impl_json_converter {
    ($converter_type:ty, $event_type:ty, $convert_fn:ident) => {
        impl $crate::utils::streaming::JsonEventConverter for $converter_type {
            fn convert_json<'a>(
                &'a self,
                json_data: &'a str,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Option<
                                Result<$crate::stream::ChatStreamEvent, $crate::error::LlmError>,
                            >,
                        > + Send
                        + Sync
                        + 'a,
                >,
            > {
                Box::pin(async move {
                    match serde_json::from_str::<$event_type>(json_data) {
                        Ok(parsed_event) => Some(Ok(self.$convert_fn(parsed_event))),
                        Err(e) => Some(Err($crate::error::LlmError::ParseError(format!(
                            "Failed to parse JSON: {e}"
                        )))),
                    }
                })
            }
        }
    };
}
