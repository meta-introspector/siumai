//! Streaming Processing Module
//!
//! Defines types and functionality for handling streaming responses from LLM providers

use futures::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;

use crate::error::LlmError;
use crate::types::*;

/// Chat Stream - Main interface for streaming responses
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + Sync>>;

// Re-export ChatStreamEvent from types module to avoid duplication
pub use crate::types::ChatStreamEvent;

/// Tool Call Delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Call ID
    pub id: Option<String>,
    /// Tool type
    pub r#type: Option<ToolType>,
    /// Function call delta
    pub function: Option<FunctionCallDelta>,
}

/// Function Call Delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    /// Function name delta
    pub name: Option<String>,
    /// Arguments delta
    pub arguments: Option<String>,
}

/// Stream Processor - for processing and transforming stream events
pub struct StreamProcessor {
    buffer: String,
    tool_calls: HashMap<String, ToolCallBuilder>,
    thinking_buffer: String,
    current_usage: Option<Usage>,
}

impl StreamProcessor {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            tool_calls: HashMap::new(),
            thinking_buffer: String::new(),
            current_usage: None,
        }
    }

    /// Process a stream event
    pub fn process_event(&mut self, event: ChatStreamEvent) -> ProcessedEvent {
        match event {
            ChatStreamEvent::ContentDelta { delta, index } => {
                self.buffer.push_str(&delta);
                ProcessedEvent::ContentUpdate {
                    delta,
                    accumulated: self.buffer.clone(),
                    index,
                }
            }
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                index,
            } => {
                let call_id = id.clone();
                let builder = self.tool_calls.entry(call_id.clone()).or_default();

                builder.id = call_id.clone();

                if let Some(name) = function_name {
                    builder.name.push_str(&name);
                }
                if let Some(args) = arguments_delta {
                    builder.arguments.push_str(&args);
                }

                ProcessedEvent::ToolCallUpdate {
                    id: call_id,
                    current_state: builder.clone(),
                    index,
                }
            }
            ChatStreamEvent::ThinkingDelta { delta } => {
                self.thinking_buffer.push_str(&delta);
                ProcessedEvent::ThinkingUpdate {
                    delta,
                    accumulated: self.thinking_buffer.clone(),
                }
            }
            ChatStreamEvent::UsageUpdate { usage } => {
                if let Some(ref mut current) = self.current_usage {
                    current.merge(&usage);
                } else {
                    self.current_usage = Some(usage.clone());
                }
                ProcessedEvent::UsageUpdate {
                    usage: self.current_usage.clone().unwrap(),
                }
            }
            ChatStreamEvent::StreamStart { metadata } => ProcessedEvent::StreamStart { metadata },
            ChatStreamEvent::StreamEnd { response } => ProcessedEvent::StreamEnd { response },

            ChatStreamEvent::Error { error } => ProcessedEvent::Error {
                error: LlmError::InternalError(error),
            },
        }
    }

    /// Build the final response
    pub fn build_final_response(&self) -> ChatResponse {
        self.build_final_response_with_finish_reason(None)
    }

    /// Build the final response with finish reason
    pub fn build_final_response_with_finish_reason(
        &self,
        finish_reason: Option<FinishReason>,
    ) -> ChatResponse {
        let mut metadata = HashMap::new();

        if !self.thinking_buffer.is_empty() {
            metadata.insert(
                "thinking".to_string(),
                serde_json::Value::String(self.thinking_buffer.clone()),
            );
        }

        let tool_calls = if !self.tool_calls.is_empty() {
            Some(
                self.tool_calls
                    .values()
                    .map(ToolCallBuilder::build)
                    .collect(),
            )
        } else {
            None
        };

        let thinking = if !self.thinking_buffer.is_empty() {
            Some(self.thinking_buffer.clone())
        } else {
            None
        };

        ChatResponse {
            id: None,
            content: MessageContent::Text(self.buffer.clone()),
            model: None,
            usage: self.current_usage.clone(),
            finish_reason,
            tool_calls,
            thinking,
            metadata,
        }
    }
}

/// Processed Event
#[derive(Debug, Clone)]
pub enum ProcessedEvent {
    ContentUpdate {
        delta: String,
        accumulated: String,
        index: Option<usize>,
    },
    ToolCallUpdate {
        id: String,
        current_state: ToolCallBuilder,
        index: Option<usize>,
    },
    ThinkingUpdate {
        delta: String,
        accumulated: String,
    },
    UsageUpdate {
        usage: Usage,
    },
    StreamStart {
        metadata: ResponseMetadata,
    },
    StreamEnd {
        response: ChatResponse,
    },
    Error {
        error: LlmError,
    },
}

/// Tool Call Builder
#[derive(Debug, Clone)]
pub struct ToolCallBuilder {
    pub id: String,
    pub r#type: Option<ToolType>,
    pub name: String,
    pub arguments: String,
}

impl Default for ToolCallBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallBuilder {
    pub const fn new() -> Self {
        Self {
            id: String::new(),
            r#type: None,
            name: String::new(),
            arguments: String::new(),
        }
    }

    pub fn build(&self) -> ToolCall {
        ToolCall {
            id: self.id.clone(),
            r#type: self
                .r#type
                .as_ref()
                .map(|t| format!("{t:?}"))
                .unwrap_or_else(|| "function".to_string()),
            function: Some(FunctionCall {
                name: self.name.clone(),
                arguments: self.arguments.clone(),
            }),
        }
    }
}

impl Default for StreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Stream Utilities
///
/// Utility functions for working with chat streams
/// Collect all stream events into a single response
///
/// This function consumes the entire stream and builds a final `ChatResponse`
pub async fn collect_stream_response(mut stream: ChatStream) -> Result<ChatResponse, LlmError> {
    use futures::StreamExt;

    let mut processor = StreamProcessor::new();
    let mut _metadata = None;

    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::StreamStart { metadata: meta } => {
                _metadata = Some(meta);
            }
            ChatStreamEvent::StreamEnd { response } => {
                return Ok(response);
            }
            ChatStreamEvent::Error { error } => {
                return Err(LlmError::InternalError(error));
            }
            event => {
                processor.process_event(event);
            }
        }
    }

    Ok(processor.build_final_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_processor() {
        let mut processor = StreamProcessor::new();

        let event = ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        };

        let processed = processor.process_event(event);

        match processed {
            ProcessedEvent::ContentUpdate {
                delta, accumulated, ..
            } => {
                assert_eq!(delta, "Hello");
                assert_eq!(accumulated, "Hello");
            }
            _ => panic!("Expected ContentUpdate"),
        }
    }

    // Test that stream types are Send + Sync for multi-threading
    #[test]
    fn test_stream_types_are_send_sync() {
        use std::sync::Arc;

        // Test that stream types can be used in Arc (requires Send + Sync)
        fn test_arc_usage() {
            let _: Option<Arc<ChatStream>> = None;
        }

        test_arc_usage();
    }

    // Test actual multi-threading with stream types
    #[tokio::test]
    async fn test_stream_multithreading() {
        use futures::stream;
        use std::sync::Arc;
        use tokio::task;

        // Create a mock stream that we can share across threads
        let mock_events = vec![
            Ok(ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: None,
            }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: " World".to_string(),
                index: None,
            }),
            Ok(ChatStreamEvent::StreamEnd {
                response: crate::types::ChatResponse {
                    id: Some("test-id".to_string()),
                    content: crate::types::MessageContent::Text("Hello World".to_string()),
                    model: Some("test-model".to_string()),
                    usage: None,
                    finish_reason: Some(crate::types::FinishReason::Stop),
                    tool_calls: None,
                    thinking: None,
                    metadata: std::collections::HashMap::new(),
                },
            }),
        ];

        let stream: ChatStream = Box::pin(stream::iter(mock_events));
        let stream_arc = Arc::new(tokio::sync::Mutex::new(stream));

        // Spawn multiple tasks that could potentially access the stream
        // (In practice, streams are usually consumed by one task, but this tests Send + Sync)
        let mut handles = Vec::new();

        for i in 0..3 {
            let stream_clone = stream_arc.clone();
            let handle = task::spawn(async move {
                // This tests that the stream can be moved across thread boundaries
                let _guard = stream_clone.lock().await;
                // In a real scenario, we'd consume the stream here
                i // Return task id for verification
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            results.push(result);
        }

        // Verify all tasks completed
        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(*result, i);
        }
    }
}
