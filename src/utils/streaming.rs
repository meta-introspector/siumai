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
use reqwest_eventsource::{Event as ReqwestEvent, RequestBuilderExt};
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
        // Create the EventSource
        let mut event_source = request_builder
            .eventsource()
            .map_err(|e| LlmError::HttpError(format!("Failed to create EventSource: {e}")))?;

        // Collect all events into a vector first, then create a stream from it
        let mut events = Vec::new();

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(ReqwestEvent::Open) => {
                    // Connection established, continue
                    tracing::debug!("EventSource connection opened");
                    continue;
                }
                Ok(ReqwestEvent::Message(message)) => {
                    // Handle [DONE] events
                    if message.data == "[DONE]" {
                        if let Some(end_event) = converter.handle_stream_end() {
                            events.push(end_event);
                        }
                        break;
                    }

                    // Skip empty events
                    if message.data.trim().is_empty() {
                        continue;
                    }

                    // The message is already an eventsource_stream::Event
                    let sse_event = message;

                    // Convert using provider-specific logic
                    if let Some(result) = converter.convert_event(sse_event).await {
                        events.push(result);
                    }
                }
                Err(err) => {
                    events.push(Err(LlmError::StreamError(format!(
                        "EventSource error: {err}"
                    ))));
                    break; // Stop on error
                }
            }
        }

        // Create a stream from the collected events
        let chat_stream = futures_util::stream::iter(events);
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
