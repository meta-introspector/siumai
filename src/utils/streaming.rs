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

/// Stream factory for creating provider-specific streams
///
/// This factory provides utilities for creating SSE and JSON streams
/// using eventsource-stream for proper UTF-8 handling.
pub struct StreamFactory;

impl StreamFactory {
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

    /// Create a chat stream using eventsource-stream
    ///
    /// This method creates an SSE stream using the eventsource-stream crate,
    /// which handles UTF-8 boundaries, line buffering, and SSE parsing automatically.
    pub async fn create_eventsource_stream<C>(
        request_builder: reqwest::RequestBuilder,
        converter: C,
    ) -> Result<ChatStream, LlmError>
    where
        C: SseEventConverter + Clone + Send + 'static,
    {
        use crate::utils::sse_stream::SseStreamExt;

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

        // Convert response to byte stream
        let byte_stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));

        // Use eventsource-stream to parse SSE
        let sse_stream = byte_stream.into_sse_stream();

        // Convert SSE events to ChatStreamEvents
        let chat_stream = sse_stream.filter_map(move |event_result| {
            let converter = converter.clone();
            async move {
                match event_result {
                    Ok(event) => {
                        // Handle special [DONE] event
                        if event.data.trim() == "[DONE]" {
                            return converter.handle_stream_end();
                        }

                        // Skip empty events
                        if event.data.trim().is_empty() {
                            return None;
                        }

                        // Convert using provider-specific logic
                        converter.convert_event(event).await
                    }
                    Err(e) => Some(Err(LlmError::StreamError(format!(
                        "SSE parsing error: {e}"
                    )))),
                }
            }
        });

        Ok(Box::pin(chat_stream))
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
