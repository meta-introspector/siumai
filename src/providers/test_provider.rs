//! Test Provider for UTF-8 and Thinking Content Testing
//!
//! This module provides a test provider that simulates real-world scenarios
//! including UTF-8 byte truncation and thinking content processing.

use async_trait::async_trait;
use futures::{StreamExt, stream};
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatResponse, MessageContent, Tool, Usage};
use crate::utils::Utf8StreamDecoder;

/// Content part types for parsing mixed content
#[derive(Debug, Clone)]
enum ContentPart {
    /// Thinking content (inside <think> tags)
    Thinking(String),
    /// Regular content (outside thinking tags)
    Regular(String),
}

/// Test provider configuration
#[derive(Debug, Clone)]
pub struct TestProviderConfig {
    /// Whether to simulate UTF-8 truncation
    pub simulate_utf8_truncation: bool,
    /// Whether to include thinking content
    pub include_thinking: bool,
    /// Chunk size for simulating network packets
    pub chunk_size: usize,
}

impl Default for TestProviderConfig {
    fn default() -> Self {
        Self {
            simulate_utf8_truncation: true,
            include_thinking: true,
            chunk_size: 3, // Small chunks to force UTF-8 truncation
        }
    }
}

/// Test provider for UTF-8 and thinking content testing
#[derive(Debug, Clone)]
pub struct TestProvider {
    config: TestProviderConfig,
}

impl TestProvider {
    /// Create a new test provider
    pub fn new(config: TestProviderConfig) -> Self {
        Self { config }
    }

    /// Create a test response with mixed content
    fn create_test_response(&self) -> String {
        let mut response = String::new();

        if self.config.include_thinking {
            response.push_str("<think>\n这是一个复杂的问题，需要仔细思考。让我分析一下：\n1. 用户询问了关于UTF-8编码的问题\n2. 我需要提供准确的技术信息\n3. 同时要考虑中文字符的处理\n🤔 这涉及到字节边界的问题...\n</think>\n\n");
        }

        response.push_str("你好！关于UTF-8编码的问题，我来详细解释一下：\n\n");
        response.push_str("UTF-8是一种可变长度的字符编码，中文字符通常占用3个字节。");
        response.push_str("例如：'中'字的UTF-8编码是 0xE4 0xB8 0xAD。\n\n");
        response.push_str("在网络传输中，如果数据包在字符边界被截断，就可能出现乱码。");
        response.push_str("这就是为什么需要UTF-8流式解码器的原因。🌍✨\n\n");

        if self.config.include_thinking {
            response.push_str(
                "<think>\n用户应该明白了基本概念，我再补充一些实际应用的例子。\n</think>\n\n",
            );
        }

        response.push_str("实际应用中，我们需要缓冲不完整的字节序列，直到收到完整的字符。");
        response.push_str("这样就能确保正确解码多字节字符了！🚀");

        response
    }

    /// Simulate SSE stream with potential UTF-8 truncation
    fn create_sse_chunks(&self, content: &str) -> Vec<Vec<u8>> {
        // Parse content to separate thinking and regular content
        let content_parts = self.parse_content_parts(content);
        let mut all_chunks = Vec::new();

        for part in content_parts {
            let sse_chunk = match part {
                ContentPart::Thinking(thinking_content) => {
                    // Create reasoning delta event
                    format!(
                        "data: {}\n\n",
                        json!({
                            "id": "test-123",
                            "object": "chat.completion.chunk",
                            "created": 1677652288,
                            "model": "test-model",
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "reasoning": thinking_content
                                },
                                "finish_reason": null
                            }]
                        })
                    )
                }
                ContentPart::Regular(regular_content) => {
                    // Create content delta event
                    format!(
                        "data: {}\n\n",
                        json!({
                            "id": "test-123",
                            "object": "chat.completion.chunk",
                            "created": 1677652288,
                            "model": "test-model",
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": regular_content
                                },
                                "finish_reason": null
                            }]
                        })
                    )
                }
            };

            if self.config.simulate_utf8_truncation {
                // Split the SSE chunk bytes at arbitrary boundaries to simulate network truncation
                let sse_bytes = sse_chunk.as_bytes();
                let mut i = 0;

                while i < sse_bytes.len() {
                    let end = std::cmp::min(i + self.config.chunk_size, sse_bytes.len());
                    all_chunks.push(sse_bytes[i..end].to_vec());
                    i = end;
                }
            } else {
                all_chunks.push(sse_chunk.into_bytes());
            }
        }

        // Add final chunk
        all_chunks.push(b"data: [DONE]\n\n".to_vec());
        all_chunks
    }

    /// Parse content into thinking and regular parts
    fn parse_content_parts(&self, content: &str) -> Vec<ContentPart> {
        let mut parts = Vec::new();
        let mut remaining = content;

        while !remaining.is_empty() {
            if let Some(think_start) = remaining.find("<think>") {
                // Add any content before the thinking tag
                if think_start > 0 {
                    let before_think = &remaining[..think_start];
                    if !before_think.trim().is_empty() {
                        parts.push(ContentPart::Regular(before_think.to_string()));
                    }
                }

                // Find the end of thinking tag
                if let Some(think_end) = remaining.find("</think>") {
                    let thinking_content = &remaining[think_start + 7..think_end];
                    if !thinking_content.trim().is_empty() {
                        parts.push(ContentPart::Thinking(thinking_content.to_string()));
                    }
                    remaining = &remaining[think_end + 8..];
                } else {
                    // Incomplete thinking tag, treat as regular content
                    parts.push(ContentPart::Regular(remaining.to_string()));
                    break;
                }
            } else {
                // No more thinking tags, add remaining as regular content
                if !remaining.trim().is_empty() {
                    parts.push(ContentPart::Regular(remaining.to_string()));
                }
                break;
            }
        }

        parts
    }

    /// Parse SSE chunk and extract content or reasoning
    /// This method needs to handle partial SSE data that may be split across chunks
    fn parse_sse_chunk(&self, chunk: &str) -> Option<ChatStreamEvent> {
        // Look for complete SSE data lines
        for line in chunk.lines() {
            let line = line.trim();

            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    return None;
                }

                if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(choices) = json_value["choices"].as_array() {
                        if let Some(choice) = choices.first() {
                            if let Some(delta) = choice["delta"].as_object() {
                                // Check for reasoning content first
                                if let Some(reasoning) =
                                    delta.get("reasoning").and_then(|v| v.as_str())
                                {
                                    return Some(ChatStreamEvent::ReasoningDelta {
                                        delta: reasoning.to_string(),
                                    });
                                }
                                // Then check for regular content
                                if let Some(content) = delta.get("content").and_then(|v| v.as_str())
                                {
                                    return Some(ChatStreamEvent::ContentDelta {
                                        delta: content.to_string(),
                                        index: Some(0),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }
}

#[async_trait]
impl ChatCapability for TestProvider {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let content = self.create_test_response();

        Ok(ChatResponse {
            id: Some("test-123".to_string()),
            content: MessageContent::Text(content),
            model: Some("test-model".to_string()),
            usage: Some(Usage {
                prompt_tokens: 50,
                completion_tokens: 100,
                total_tokens: 150,
                cached_tokens: None,
                reasoning_tokens: Some(25),
            }),
            finish_reason: Some(crate::types::FinishReason::Stop),
            tool_calls: None,
            thinking: if self.config.include_thinking {
                Some("这是模拟的思考过程，包含中文字符和emoji 🤔".to_string())
            } else {
                None
            },
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let content = self.create_test_response();
        let chunks = self.create_sse_chunks(&content);

        // Create a UTF-8 decoder and SSE buffer for this stream
        let decoder = Arc::new(Mutex::new(Utf8StreamDecoder::new()));
        let sse_buffer = Arc::new(Mutex::new(String::new()));
        let decoder_for_flush = decoder.clone();
        let sse_buffer_for_flush = sse_buffer.clone();

        // Create stream from chunks
        let chunk_stream = stream::iter(chunks).then(|chunk| async move {
            // Simulate network delay
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok::<Vec<u8>, LlmError>(chunk)
        });

        // Clone provider for use in async closures
        let provider_clone = self.clone();

        // Process chunks with UTF-8 decoder and SSE buffer
        let decoded_stream = chunk_stream.filter_map(move |chunk_result| {
            let decoder = decoder.clone();
            let sse_buffer = sse_buffer.clone();
            let provider = provider_clone.clone();
            async move {
                match chunk_result {
                    Ok(chunk) => {
                        // Use UTF-8 decoder to handle incomplete sequences
                        let decoded_chunk = {
                            let mut decoder = decoder.lock().unwrap();
                            decoder.decode(&chunk)
                        };

                        if !decoded_chunk.is_empty() {
                            // Add to SSE buffer and try to parse complete lines
                            let mut buffer = sse_buffer.lock().unwrap();
                            buffer.push_str(&decoded_chunk);

                            // Look for complete SSE events (ending with \n\n)
                            while let Some(double_newline_pos) = buffer.find("\n\n") {
                                let complete_part = buffer[..double_newline_pos + 2].to_string();
                                *buffer = buffer[double_newline_pos + 2..].to_string();

                                if let Some(event) = provider.parse_sse_chunk(&complete_part) {
                                    return Some(Ok(event));
                                }
                            }
                        }
                        None
                    }
                    Err(e) => Some(Err(e)),
                }
            }
        });

        // Add flush operation to handle any remaining data
        let flush_stream = stream::once(async move {
            let remaining_utf8 = {
                let mut decoder = decoder_for_flush.lock().unwrap();
                decoder.flush()
            };

            let remaining_sse = {
                let mut buffer = sse_buffer_for_flush.lock().unwrap();
                let remaining = buffer.clone();
                buffer.clear();
                remaining
            };

            // Try to parse any remaining SSE data
            if !remaining_sse.is_empty() {
                // For the test provider, we can try to parse incomplete SSE data
                // In a real implementation, this would be more sophisticated
                if remaining_sse.contains("data: [DONE]") {
                    return None; // End of stream
                }
            }

            if !remaining_utf8.is_empty() {
                Some(Ok(ChatStreamEvent::ContentDelta {
                    delta: remaining_utf8,
                    index: Some(0),
                }))
            } else {
                None
            }
        })
        .filter_map(|result| async move { result });

        let final_stream = decoded_stream.chain(flush_stream);
        Ok(Box::pin(final_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_utf8_truncation_handling() {
        // Test with small chunks that will cause UTF-8 truncation but still allow SSE parsing
        let config = TestProviderConfig {
            simulate_utf8_truncation: true,
            include_thinking: true,
            chunk_size: 100, // Large enough for SSE parsing but small enough to cause UTF-8 truncation
        };

        let provider = TestProvider::new(config);
        let messages = vec![ChatMessage::user("测试UTF-8截断处理").build()];

        let stream = provider.chat_stream(messages, None).await.unwrap();
        let events: Vec<_> = stream.collect().await;

        // Should have successfully processed all events without corruption
        assert!(!events.is_empty());

        // Check that we got some content (either reasoning or regular content)
        let has_content = events.iter().any(|event| {
            matches!(
                event,
                Ok(ChatStreamEvent::ContentDelta { .. })
                    | Ok(ChatStreamEvent::ReasoningDelta { .. })
            )
        });
        assert!(has_content);

        // Verify that UTF-8 characters are properly decoded (no replacement characters)
        for event in &events {
            if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event {
                assert!(
                    !delta.contains('�'),
                    "Found replacement character in content: {}",
                    delta
                );
            }
            if let Ok(ChatStreamEvent::ReasoningDelta { delta }) = event {
                assert!(
                    !delta.contains('�'),
                    "Found replacement character in reasoning: {}",
                    delta
                );
            }
        }
    }

    #[tokio::test]
    async fn test_thinking_content_extraction() {
        let config = TestProviderConfig {
            simulate_utf8_truncation: false,
            include_thinking: true,
            chunk_size: 100,
        };

        let provider = TestProvider::new(config);
        let messages = vec![ChatMessage::user("测试思考内容提取").build()];

        let stream = provider.chat_stream(messages, None).await.unwrap();
        let events: Vec<_> = stream.collect().await;

        // Should have reasoning deltas for thinking content
        let has_reasoning = events
            .iter()
            .any(|event| matches!(event, Ok(ChatStreamEvent::ReasoningDelta { .. })));
        assert!(has_reasoning);
    }
}
