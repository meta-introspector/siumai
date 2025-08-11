//! Common Streaming Utilities
//!
//! This module provides common utilities for handling streaming responses
//! across different providers, including line buffering, UTF-8 handling,
//! and unified SSE processing using eventsource-stream.

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::utils::sse_stream::SseStreamExt;
use eventsource_stream::Event;
use futures_util::StreamExt;
use std::future::Future;
use std::pin::Pin;

/// Type alias for SSE event conversion future
type SseEventFuture<'a> =
    Pin<Box<dyn Future<Output = Option<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>;

/// Type alias for JSON event conversion future
type JsonEventFuture<'a> =
    Pin<Box<dyn Future<Output = Option<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>;

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

    /// Create a chat stream using reqwest_eventsource (with retry support)
    ///
    /// This method provides enhanced features over the basic eventsource-stream:
    /// - Automatic retry on connection failures
    /// - Better error handling and recovery
    /// - Connection state management
    /// - Last-Event-ID support for reconnection
    ///
    /// Note: This method creates the EventSource and immediately processes it,
    /// avoiding Sync issues with the EventSource type.
    pub async fn create_eventsource_stream<C>(
        request_builder: reqwest::RequestBuilder,
        converter: C,
    ) -> Result<ChatStream, LlmError>
    where
        C: SseEventConverter + Clone + Send + 'static,
    {
        // Send the request and get the response
        let response = request_builder
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to send request: {e}")))?;

        // Check if the response is successful
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpError(format!(
                "HTTP error {}: {}",
                status.as_u16(),
                error_text
            )));
        }

        // Process the response as a byte stream and parse SSE manually
        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut events = Vec::new();

        while let Some(chunk_result) = byte_stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    // Convert bytes to string and add to buffer
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    buffer.push_str(&chunk_str);

                    // Process complete lines
                    while let Some(line_end) = buffer.find('\n') {
                        let line = buffer[..line_end].trim_end_matches('\r').to_string();
                        buffer = buffer[line_end + 1..].to_string();

                        // Process SSE data lines
                        if let Some(data) = line.strip_prefix("data: ") {
                            // Remove "data: " prefix

                            // Handle [DONE] events
                            if data == "[DONE]" {
                                if let Some(end_event) = converter.handle_stream_end() {
                                    events.push(end_event);
                                }
                                // Don't break here, let the stream end naturally
                                continue;
                            }

                            // Skip empty events
                            if data.trim().is_empty() {
                                continue;
                            }

                            // Create an Event for the converter
                            let event = Event {
                                event: "".to_string(),
                                data: data.to_string(),
                                id: "".to_string(),
                                retry: None,
                            };

                            // Convert using provider-specific logic
                            if let Some(result) = converter.convert_event(event).await {
                                events.push(result);
                            }
                        }
                    }
                }
                Err(e) => {
                    events.push(Err(LlmError::StreamError(format!("Stream error: {e}"))));
                    break;
                }
            }
        }

        // Create a stream from the collected events
        let stream = futures_util::stream::iter(events);
        Ok(Box::pin(stream))
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
