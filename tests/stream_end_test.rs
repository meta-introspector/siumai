//! Stream End Event Tests
//!
//! These tests verify that StreamEnd events are properly sent
//! in various streaming scenarios.

use siumai::FinishReason;
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
async fn test_stream_end_with_stop_reason() {
    let config = create_test_config();
    let streaming = OpenAiStreaming::new(config, reqwest::Client::new());

    // Simulate a chunk with finish_reason: "stop"
    let stop_chunk = r#"data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

"#;

    let events = streaming.parse_sse_chunk_all_events(stop_chunk);

    assert_eq!(events.len(), 1, "Should parse 1 event");

    match &events[0] {
        Ok(ChatStreamEvent::StreamEnd { response }) => {
            assert_eq!(response.finish_reason, Some(FinishReason::Stop));
            assert_eq!(response.id, Some("chatcmpl-test".to_string()));
            assert_eq!(response.model, Some("gpt-3.5-turbo".to_string()));
        }
        _ => panic!("Expected StreamEnd event with Stop finish reason"),
    }
}

#[tokio::test]
async fn test_stream_end_with_tool_calls_reason() {
    let config = create_test_config();
    let streaming = OpenAiStreaming::new(config, reqwest::Client::new());

    // Simulate a chunk with finish_reason: "tool_calls"
    let tool_calls_chunk = r#"data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

"#;

    let events = streaming.parse_sse_chunk_all_events(tool_calls_chunk);

    assert_eq!(events.len(), 1, "Should parse 1 event");

    match &events[0] {
        Ok(ChatStreamEvent::StreamEnd { response }) => {
            assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
            assert_eq!(response.id, Some("chatcmpl-test".to_string()));
        }
        _ => panic!("Expected StreamEnd event with ToolCalls finish reason"),
    }
}

#[tokio::test]
async fn test_stream_end_with_length_reason() {
    let config = create_test_config();
    let streaming = OpenAiStreaming::new(config, reqwest::Client::new());

    // Simulate a chunk with finish_reason: "length"
    let length_chunk = r#"data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}

"#;

    let events = streaming.parse_sse_chunk_all_events(length_chunk);

    assert_eq!(events.len(), 1, "Should parse 1 event");

    match &events[0] {
        Ok(ChatStreamEvent::StreamEnd { response }) => {
            assert_eq!(response.finish_reason, Some(FinishReason::Length));
        }
        _ => panic!("Expected StreamEnd event with Length finish reason"),
    }
}

#[tokio::test]
async fn test_done_marker_creates_stream_end() {
    let config = create_test_config();
    let streaming = OpenAiStreaming::new(config, reqwest::Client::new());

    // Simulate [DONE] marker
    let done_chunk = r#"data: [DONE]

"#;

    let events = streaming.parse_sse_chunk_all_events(done_chunk);

    assert_eq!(events.len(), 1, "Should parse 1 event");

    match &events[0] {
        Ok(ChatStreamEvent::StreamEnd { response }) => {
            assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        }
        _ => panic!("Expected StreamEnd event for [DONE] marker"),
    }
}

#[tokio::test]
async fn test_usage_update_separate_from_stream_end() {
    let config = create_test_config();
    let streaming = OpenAiStreaming::new(config, reqwest::Client::new());

    // Simulate a chunk with usage information (no choices)
    let usage_chunk = r#"data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

"#;

    let events = streaming.parse_sse_chunk_all_events(usage_chunk);

    assert_eq!(events.len(), 1, "Should parse 1 event");

    match &events[0] {
        Ok(ChatStreamEvent::UsageUpdate { usage }) => {
            assert_eq!(usage.prompt_tokens, 10);
            assert_eq!(usage.completion_tokens, 5);
            assert_eq!(usage.total_tokens, 15);
        }
        _ => panic!("Expected UsageUpdate event"),
    }
}

#[tokio::test]
async fn test_multiple_events_with_stream_end() {
    let config = create_test_config();
    let streaming = OpenAiStreaming::new(config, reqwest::Client::new());

    // Simulate multiple events including tool calls and stream end
    let multi_event_chunk = r#"data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"test"}}]}}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

"#;

    let events = streaming.parse_sse_chunk_all_events(multi_event_chunk);

    assert_eq!(events.len(), 3, "Should parse 3 events");

    // First event should be tool call delta
    match &events[0] {
        Ok(ChatStreamEvent::ToolCallDelta {
            arguments_delta, ..
        }) => {
            assert_eq!(arguments_delta, &Some("test".to_string()));
        }
        _ => panic!("Expected ToolCallDelta event"),
    }

    // Second event should be stream end
    match &events[1] {
        Ok(ChatStreamEvent::StreamEnd { response }) => {
            assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        }
        _ => panic!("Expected StreamEnd event"),
    }

    // Third event should be usage update
    match &events[2] {
        Ok(ChatStreamEvent::UsageUpdate { usage }) => {
            assert_eq!(usage.total_tokens, 15);
        }
        _ => panic!("Expected UsageUpdate event"),
    }
}
