//! UTF-8 Streaming Integration Tests
//!
//! These tests verify that the UTF-8 stream decoder correctly handles
//! multi-byte characters in streaming responses and thinking content.

use futures::StreamExt;
use siumai::providers::test_provider::{TestProvider, TestProviderConfig};
use siumai::stream::ChatStreamEvent;
use siumai::traits::ChatCapability;
use siumai::types::ChatMessage;

#[tokio::test]
async fn test_utf8_truncation_with_chinese_characters() {
    let config = TestProviderConfig {
        simulate_utf8_truncation: true,
        include_thinking: false,
        chunk_size: 50, // Small enough to cause UTF-8 truncation but large enough for SSE parsing
    };

    let provider = TestProvider::new(config);
    let messages = vec![ChatMessage::user("测试中文字符的UTF-8截断处理").build()];

    let stream = provider.chat_stream(messages, None).await.unwrap();
    let events: Vec<Result<ChatStreamEvent, _>> = stream.collect().await;

    // Debug output
    println!("Number of events: {}", events.len());
    for (i, event) in events.iter().enumerate() {
        println!("Event {}: {:?}", i, event);
    }

    // Verify we got events without errors
    assert!(!events.is_empty(), "Should have received some events");

    // Check that all events are successful
    for event in &events {
        assert!(event.is_ok(), "Event should be successful: {:?}", event);
    }

    // Collect all content deltas
    let mut content = String::new();
    for event in events {
        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event {
            content.push_str(&delta);
        }
    }

    // Verify that Chinese characters are correctly decoded
    assert!(content.contains("你好"), "Should contain Chinese greeting");
    assert!(content.contains("UTF-8"), "Should contain UTF-8 reference");
    assert!(
        content.contains("中文字符"),
        "Should contain Chinese characters"
    );

    // Verify no replacement characters (�) indicating corruption
    assert!(
        !content.contains('�'),
        "Should not contain replacement characters"
    );
}

#[tokio::test]
async fn test_emoji_handling_in_truncated_stream() {
    let config = TestProviderConfig {
        simulate_utf8_truncation: true,
        include_thinking: false,
        chunk_size: 60, // Small enough to split emoji but large enough for SSE parsing
    };

    let provider = TestProvider::new(config);
    let messages = vec![ChatMessage::user("测试emoji处理 🌍🚀✨").build()];

    let stream = provider.chat_stream(messages, None).await.unwrap();
    let events: Vec<Result<ChatStreamEvent, _>> = stream.collect().await;

    // Collect all content
    let mut content = String::new();
    for event in events {
        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event {
            content.push_str(&delta);
        }
    }

    // Verify emoji are correctly handled
    assert!(content.contains("🌍"), "Should contain earth emoji");
    assert!(content.contains("🚀"), "Should contain rocket emoji");
    assert!(content.contains("✨"), "Should contain sparkles emoji");

    // Verify no corruption
    assert!(
        !content.contains('�'),
        "Should not contain replacement characters"
    );
}

#[tokio::test]
async fn test_thinking_content_with_utf8_truncation() {
    let config = TestProviderConfig {
        simulate_utf8_truncation: true,
        include_thinking: true,
        chunk_size: 80, // Small enough to test UTF-8 truncation but large enough for SSE parsing
    };

    let provider = TestProvider::new(config);
    let messages = vec![ChatMessage::user("测试思考内容的UTF-8处理").build()];

    let stream = provider.chat_stream(messages, None).await.unwrap();
    let events: Vec<Result<ChatStreamEvent, _>> = stream.collect().await;

    // Separate reasoning and content events
    let mut reasoning_content = String::new();
    let mut regular_content = String::new();

    for event in events {
        match event {
            Ok(ChatStreamEvent::ThinkingDelta { delta }) => {
                reasoning_content.push_str(&delta);
            }
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                regular_content.push_str(&delta);
            }
            Ok(_) => {} // Other event types
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    // Verify thinking content was extracted
    assert!(
        !reasoning_content.is_empty(),
        "Should have reasoning content"
    );
    assert!(
        reasoning_content.contains("思考"),
        "Reasoning should contain thinking"
    );

    // Verify regular content doesn't contain thinking tags
    assert!(
        !regular_content.contains("<think>"),
        "Regular content should not contain thinking tags"
    );
    assert!(
        !regular_content.contains("</think>"),
        "Regular content should not contain thinking tags"
    );

    // Verify Chinese characters in both contents
    assert!(
        !reasoning_content.contains('�'),
        "Reasoning should not have corruption"
    );
    assert!(
        !regular_content.contains('�'),
        "Regular content should not have corruption"
    );
}

#[tokio::test]
async fn test_mixed_content_with_extreme_fragmentation() {
    let config = TestProviderConfig {
        simulate_utf8_truncation: true,
        include_thinking: true,
        chunk_size: 30, // Small chunks but still parseable for SSE
    };

    let provider = TestProvider::new(config);
    let messages = vec![ChatMessage::user("极端分片测试：中文🤔emoji混合内容").build()];

    let stream = provider.chat_stream(messages, None).await.unwrap();
    let events: Vec<Result<ChatStreamEvent, _>> = stream.collect().await;

    // Should handle extreme fragmentation without errors
    assert!(!events.is_empty());

    // Verify all events are successful
    for event in &events {
        assert!(
            event.is_ok(),
            "All events should be successful with extreme fragmentation"
        );
    }

    // Collect all content
    let mut all_content = String::new();
    for event in events {
        match event {
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                all_content.push_str(&delta);
            }
            Ok(ChatStreamEvent::ThinkingDelta { delta }) => {
                all_content.push_str(&delta);
            }
            _ => {}
        }
    }

    // Verify content integrity despite extreme fragmentation
    assert!(
        !all_content.contains('�'),
        "Should handle extreme fragmentation without corruption"
    );
    assert!(all_content.len() > 0, "Should have some content");
}

#[tokio::test]
async fn test_thinking_tag_boundary_splitting() {
    let config = TestProviderConfig {
        simulate_utf8_truncation: true,
        include_thinking: true,
        chunk_size: 70, // Small enough to test boundaries but large enough for SSE parsing
    };

    let provider = TestProvider::new(config);
    let messages = vec![ChatMessage::user("测试思考标签边界分割").build()];

    let stream = provider.chat_stream(messages, None).await.unwrap();
    let events: Vec<Result<ChatStreamEvent, _>> = stream.collect().await;

    // Should handle tag boundary splitting correctly
    let mut has_reasoning = false;
    let mut has_content = false;

    for event in events {
        match event {
            Ok(ChatStreamEvent::ThinkingDelta { .. }) => {
                has_reasoning = true;
            }
            Ok(ChatStreamEvent::ContentDelta { .. }) => {
                has_content = true;
            }
            Ok(_) => {}
            Err(e) => panic!(
                "Should not have errors with tag boundary splitting: {:?}",
                e
            ),
        }
    }

    assert!(
        has_reasoning,
        "Should extract reasoning content even with tag splitting"
    );
    assert!(
        has_content,
        "Should extract regular content even with tag splitting"
    );
}

#[tokio::test]
async fn test_comparison_with_and_without_truncation() {
    // Test with truncation
    let config_with_truncation = TestProviderConfig {
        simulate_utf8_truncation: true,
        include_thinking: true,
        chunk_size: 90, // Small enough to test truncation but large enough for SSE parsing
    };

    // Test without truncation
    let config_without_truncation = TestProviderConfig {
        simulate_utf8_truncation: false,
        include_thinking: true,
        chunk_size: 1000,
    };

    let provider_with = TestProvider::new(config_with_truncation);
    let provider_without = TestProvider::new(config_without_truncation);

    let messages = vec![ChatMessage::user("对比测试：UTF-8截断vs完整传输").build()];

    // Get results from both providers
    let stream_with = provider_with
        .chat_stream(messages.clone(), None)
        .await
        .unwrap();
    let events_with: Vec<Result<ChatStreamEvent, _>> = stream_with.collect().await;

    let stream_without = provider_without.chat_stream(messages, None).await.unwrap();
    let events_without: Vec<Result<ChatStreamEvent, _>> = stream_without.collect().await;

    // Both should succeed
    assert!(!events_with.is_empty());
    assert!(!events_without.is_empty());

    // Extract content from both
    let extract_content = |events: Vec<Result<ChatStreamEvent, siumai::error::LlmError>>| {
        let mut content = String::new();
        for event in events {
            if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event {
                content.push_str(&delta);
            }
        }
        content
    };

    let content_with = extract_content(events_with);
    let content_without = extract_content(events_without);

    // Both should produce valid content without corruption
    assert!(
        !content_with.contains('�'),
        "Truncated version should not have corruption"
    );
    assert!(
        !content_without.contains('�'),
        "Non-truncated version should not have corruption"
    );

    // Both should contain expected Chinese content
    assert!(
        content_with.contains("UTF-8"),
        "Truncated version should contain UTF-8"
    );
    assert!(
        content_without.contains("UTF-8"),
        "Non-truncated version should contain UTF-8"
    );
}
