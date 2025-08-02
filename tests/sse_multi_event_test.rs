//! SSE Multi-Event Parsing Tests
//!
//! These tests verify that our SSE parsing logic correctly handles
//! HTTP chunks containing multiple SSE events, which was the root cause
//! of the tool call streaming issue.

use siumai::providers::openai::OpenAiConfig;
use siumai::providers::openai::streaming::OpenAiStreaming;
use siumai::stream::ChatStreamEvent;

fn create_test_config() -> OpenAiConfig {
    OpenAiConfig {
        api_key: secrecy::SecretString::from("test-key"),
        base_url: "https://api.openai.com/v1".to_string(),
        ..Default::default()
    }
}

#[tokio::test]
async fn test_single_chunk_multiple_sse_events() {
    let config = create_test_config();
    let streaming = OpenAiStreaming::new(config, reqwest::Client::new());

    // Simulate a chunk with multiple SSE events (like what we saw in debugging)
    let multi_event_chunk = r#"data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_test","type":"function","function":{"name":"query_search","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"count\":"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"50,\"query\":\""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"rust programming\"}"}}]},"finish_reason":null}]}

"#;

    let events = streaming.parse_sse_chunk_all_events(multi_event_chunk);

    // Should parse 4 events from this chunk
    assert_eq!(events.len(), 4, "Should parse 4 SSE events from the chunk");

    // All events should be successful
    for (i, event) in events.iter().enumerate() {
        assert!(event.is_ok(), "Event {i} should be successful");
    }

    // Verify the events are tool call deltas
    let tool_call_events = events
        .into_iter()
        .flatten()
        .filter(|event| matches!(event, ChatStreamEvent::ToolCallDelta { .. }))
        .count();

    assert_eq!(tool_call_events, 4, "Should have 4 tool call delta events");
}

#[tokio::test]
async fn test_multiple_events_parsing() {
    let config = create_test_config();
    let streaming = OpenAiStreaming::new(config, reqwest::Client::new());

    // Test that we can parse multiple events from a single chunk
    let multi_event_chunk = r#"data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_test","type":"function","function":{"name":"test_function","arguments":"part1"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"part2"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"part3"}}]},"finish_reason":null}]}

"#;

    let events = streaming.parse_sse_chunk_all_events(multi_event_chunk);

    // Should parse 3 events from this chunk
    assert_eq!(events.len(), 3, "Should parse 3 SSE events from the chunk");

    // All events should be successful
    for (i, event) in events.iter().enumerate() {
        assert!(event.is_ok(), "Event {i} should be successful");
    }

    // Verify the events are tool call deltas
    let tool_call_events = events
        .into_iter()
        .flatten()
        .filter(|event| matches!(event, ChatStreamEvent::ToolCallDelta { .. }))
        .count();

    assert_eq!(tool_call_events, 3, "Should have 3 tool call delta events");
}

#[tokio::test]
async fn test_empty_and_malformed_sse_chunks() {
    let config = create_test_config();

    let streaming = OpenAiStreaming::new(config, reqwest::Client::new());

    // Test empty chunk
    let events = streaming.parse_sse_chunk_all_events("");
    assert_eq!(events.len(), 0, "Empty chunk should produce no events");

    // Test chunk with only comments and empty lines
    let comment_chunk = r#"
: this is a comment
: another comment

"#;
    let events = streaming.parse_sse_chunk_all_events(comment_chunk);
    assert_eq!(
        events.len(),
        0,
        "Comment-only chunk should produce no events"
    );

    // Test chunk with malformed JSON
    let malformed_chunk = r#"data: {"invalid": json}

"#;
    let events = streaming.parse_sse_chunk_all_events(malformed_chunk);
    assert_eq!(
        events.len(),
        1,
        "Should produce one event for malformed JSON"
    );
    assert!(
        events[0].is_err(),
        "Malformed JSON should produce error event"
    );

    // Test chunk with [DONE] marker
    let done_chunk = r#"data: [DONE]

"#;
    let events = streaming.parse_sse_chunk_all_events(done_chunk);
    assert_eq!(
        events.len(),
        1,
        "Should produce stream end event for [DONE]"
    );
    if let Ok(ChatStreamEvent::StreamEnd { .. }) = &events[0] {
        // Expected
    } else {
        panic!("Should produce StreamEnd event for [DONE] marker");
    }
}

#[tokio::test]
async fn test_mixed_event_types_in_single_chunk() {
    let config = create_test_config();

    let streaming = OpenAiStreaming::new(config, reqwest::Client::new());

    // Chunk with content delta and tool call delta
    let mixed_chunk = r#"data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_test","type":"function","function":{"name":"test_function","arguments":"test"}}]},"finish_reason":null}]}

"#;

    let events = streaming.parse_sse_chunk_all_events(mixed_chunk);

    assert_eq!(events.len(), 2, "Should parse 2 events from mixed chunk");

    // Verify event types
    let mut content_events = 0;
    let mut tool_call_events = 0;

    for event in events {
        match event {
            Ok(ChatStreamEvent::ContentDelta { .. }) => content_events += 1,
            Ok(ChatStreamEvent::ToolCallDelta { .. }) => tool_call_events += 1,
            _ => {}
        }
    }

    assert_eq!(content_events, 1, "Should have 1 content delta event");
    assert_eq!(tool_call_events, 1, "Should have 1 tool call delta event");
}
