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
        if self.config.simulate_utf8_truncation {
            // Create a single SSE chunk with the content, then split the raw bytes
            let sse_chunk = format!(
                "data: {}\n\n",
                json!({
                    "id": "test-123",
                    "object": "chat.completion.chunk",
                    "created": 1677652288,
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": content
                        },
                        "finish_reason": null
                    }]
                })
            );

            // Split the SSE chunk bytes at arbitrary boundaries to simulate network truncation
            let sse_bytes = sse_chunk.as_bytes();
            let mut chunks = Vec::new();
            let mut i = 0;

            while i < sse_bytes.len() {
                let end = std::cmp::min(i + self.config.chunk_size, sse_bytes.len());
                chunks.push(sse_bytes[i..end].to_vec());
                i = end;
            }

            // Add final chunk
            chunks.push(b"data: [DONE]\n\n".to_vec());
            chunks
        } else {
            // Send complete content in one chunk
            let sse_chunk = format!(
                "data: {}\n\n",
                json!({
                    "id": "test-123",
                    "object": "chat.completion.chunk",
                    "created": 1677652288,
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": content
                        },
                        "finish_reason": null
                    }]
                })
            );
            vec![sse_chunk.into_bytes(), b"data: [DONE]\n\n".to_vec()]
        }
    }

    /// Parse SSE chunk and extract content
    /// This method needs to handle partial SSE data that may be split across chunks
    fn parse_sse_chunk(&self, chunk: &str) -> Option<String> {
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
                            if let Some(content) = choice["delta"]["content"].as_str() {
                                return Some(content.to_string());
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
        let _sse_buffer = Arc::new(Mutex::new(String::new()));
        let decoder_for_flush = decoder.clone();

        // Create stream from chunks
        let chunk_stream = stream::iter(chunks).then(|chunk| async move {
            // Simulate network delay
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok::<Vec<u8>, LlmError>(chunk)
        });

        // Clone provider for use in async closures
        let provider_clone = self.clone();

        // Process chunks with UTF-8 decoder
        let decoded_stream = chunk_stream.filter_map(move |chunk_result| {
            let decoder = decoder.clone();
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
                            if let Some(content) = provider.parse_sse_chunk(&decoded_chunk) {
                                // Check for thinking tags
                                if content.contains("<think>") || content.contains("</think>") {
                                    // Extract thinking content
                                    if let Some(start) = content.find("<think>") {
                                        if let Some(end) = content.find("</think>") {
                                            let thinking = &content[start + 7..end];
                                            return Some(Ok(ChatStreamEvent::ReasoningDelta {
                                                delta: thinking.to_string(),
                                            }));
                                        }
                                    }
                                    // Filter out thinking tags for regular content
                                    let filtered =
                                        content.replace("<think>", "").replace("</think>", "");
                                    if !filtered.trim().is_empty() {
                                        return Some(Ok(ChatStreamEvent::ContentDelta {
                                            delta: filtered,
                                            index: Some(0),
                                        }));
                                    }
                                } else {
                                    return Some(Ok(ChatStreamEvent::ContentDelta {
                                        delta: content,
                                        index: Some(0),
                                    }));
                                }
                            }
                        }
                        None
                    }
                    Err(e) => Some(Err(e)),
                }
            }
        });

        // Add flush operation
        let flush_stream = stream::once(async move {
            let remaining = {
                let mut decoder = decoder_for_flush.lock().unwrap();
                decoder.flush()
            };

            if !remaining.is_empty() {
                Some(Ok(ChatStreamEvent::ContentDelta {
                    delta: remaining,
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
        let config = TestProviderConfig {
            simulate_utf8_truncation: true,
            include_thinking: true,
            chunk_size: 2, // Very small chunks to force truncation
        };

        let provider = TestProvider::new(config);
        let messages = vec![ChatMessage::user("测试UTF-8截断处理").build()];

        let stream = provider.chat_stream(messages, None).await.unwrap();
        let events: Vec<_> = stream.collect().await;

        // Should have successfully processed all events without corruption
        assert!(!events.is_empty());

        // Check that we got some content
        let has_content = events
            .iter()
            .any(|event| matches!(event, Ok(ChatStreamEvent::ContentDelta { .. })));
        assert!(has_content);
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
