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
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send>>;

/// Chat Stream Events
///
/// Represents different types of events that can occur during streaming
#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    /// Content delta event - incremental text content
    ContentDelta { delta: String, index: Option<usize> },
    /// Tool call delta event - incremental tool call information
    ToolCallDelta {
        tool_call: ToolCallDelta,
        index: Option<usize>,
    },
    /// Thinking process delta (DeepSeek/Anthropic specific)
    ThinkingDelta { delta: String },
    /// Reasoning process delta (OpenAI o1 specific)
    ReasoningDelta { delta: String },
    /// Usage statistics update
    UsageUpdate { usage: Usage },
    /// Stream start event with metadata
    StreamStart { metadata: ResponseMetadata },
    /// Stream completion event with final response
    StreamEnd { response: ChatResponse },
    /// Error event
    Error { error: LlmError },
}

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
    reasoning_buffer: String,
    current_usage: Option<Usage>,
}

impl StreamProcessor {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            tool_calls: HashMap::new(),
            thinking_buffer: String::new(),
            reasoning_buffer: String::new(),
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
            ChatStreamEvent::ToolCallDelta { tool_call, index } => {
                let id = tool_call.id.clone().unwrap_or_default();
                let builder = self
                    .tool_calls
                    .entry(id.clone())
                    .or_insert_with(ToolCallBuilder::new);

                if let Some(function) = tool_call.function {
                    if let Some(name) = function.name {
                        builder.name.push_str(&name);
                    }
                    if let Some(args) = function.arguments {
                        builder.arguments.push_str(&args);
                    }
                }

                ProcessedEvent::ToolCallUpdate {
                    id,
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
            ChatStreamEvent::ReasoningDelta { delta } => {
                self.reasoning_buffer.push_str(&delta);
                ProcessedEvent::ReasoningUpdate {
                    delta,
                    accumulated: self.reasoning_buffer.clone(),
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
            ChatStreamEvent::Error { error } => ProcessedEvent::Error { error },
        }
    }

    /// Build the final response
    pub fn build_final_response(&self, metadata: ResponseMetadata) -> ChatResponse {
        let mut provider_data = HashMap::new();

        if !self.thinking_buffer.is_empty() {
            provider_data.insert(
                "thinking".to_string(),
                serde_json::Value::String(self.thinking_buffer.clone()),
            );
        }

        if !self.reasoning_buffer.is_empty() {
            provider_data.insert(
                "reasoning".to_string(),
                serde_json::Value::String(self.reasoning_buffer.clone()),
            );
        }

        let tool_calls = if !self.tool_calls.is_empty() {
            Some(
                self.tool_calls
                    .values()
                    .map(|builder| builder.build())
                    .collect(),
            )
        } else {
            None
        };

        ChatResponse {
            content: MessageContent::Text(self.buffer.clone()),
            tool_calls,
            usage: self.current_usage.clone(),
            finish_reason: None, // Needs to be obtained from the final event
            metadata,
            provider_data,
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
    ReasoningUpdate {
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

impl ToolCallBuilder {
    pub fn new() -> Self {
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
                .map(|t| format!("{:?}", t))
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
/// This function consumes the entire stream and builds a final ChatResponse
pub async fn collect_stream_response(mut stream: ChatStream) -> Result<ChatResponse, LlmError> {
    use futures::StreamExt;

    let mut processor = StreamProcessor::new();
    let mut metadata = None;

    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::StreamStart { metadata: meta } => {
                metadata = Some(meta);
            }
            ChatStreamEvent::StreamEnd { response } => {
                return Ok(response);
            }
            ChatStreamEvent::Error { error } => {
                return Err(error);
            }
            event => {
                processor.process_event(event);
            }
        }
    }

    let metadata = metadata.unwrap_or_else(|| ResponseMetadata {
        id: None,
        model: None,
        created: Some(chrono::Utc::now()),
        provider: "unknown".to_string(),
        request_id: None,
    });

    Ok(processor.build_final_response(metadata))
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
}
