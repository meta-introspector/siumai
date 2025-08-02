//! Integration tests for the new streaming infrastructure
//!
//! These tests verify that all providers can use the new eventsource-stream
//! based streaming infrastructure correctly.

use eventsource_stream::Event;
use siumai::providers::anthropic::streaming::AnthropicEventConverter;
use siumai::providers::gemini::streaming::GeminiEventConverter;
use siumai::providers::ollama::streaming::OllamaEventConverter;
use siumai::providers::openai::streaming::OpenAiEventConverter;
use siumai::stream::ChatStreamEvent;
use siumai::utils::streaming::{JsonEventConverter, SseEventConverter};

#[tokio::test]
async fn test_openai_event_conversion() {
    let config = siumai::providers::openai::config::OpenAiConfig::default();
    let converter = OpenAiEventConverter::new(config);

    // Test content delta
    let event = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"delta":{"content":"Hello"}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(result.is_some());

    if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = result {
        assert_eq!(delta, "Hello");
    } else {
        panic!("Expected ContentDelta event");
    }
}

#[tokio::test]
async fn test_anthropic_event_conversion() {
    let config = siumai::params::AnthropicParams::default();
    let converter = AnthropicEventConverter::new(config);

    // Test content delta
    let event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(result.is_some());

    if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = result {
        assert_eq!(delta, "Hello");
    } else {
        panic!("Expected ContentDelta event");
    }
}

#[tokio::test]
async fn test_gemini_json_conversion() {
    let config = siumai::providers::gemini::types::GeminiConfig::default();
    let converter = GeminiEventConverter::new(config);

    // Test content delta
    let json_data = r#"{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}"#;

    let result = converter.convert_json(json_data).await;
    assert!(result.is_some());

    if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = result {
        assert_eq!(delta, "Hello");
    } else {
        panic!("Expected ContentDelta event");
    }
}

#[tokio::test]
async fn test_ollama_json_conversion() {
    let converter = OllamaEventConverter::new();

    // Test content delta
    let json_data =
        r#"{"model":"llama2","message":{"role":"assistant","content":"Hello"},"done":false}"#;

    let result = converter.convert_json(json_data).await;
    assert!(result.is_some());

    if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = result {
        assert_eq!(delta, "Hello");
    } else {
        panic!("Expected ContentDelta event");
    }
}

#[tokio::test]
async fn test_openai_thinking_conversion() {
    let config = siumai::providers::openai::config::OpenAiConfig::default();
    let converter = OpenAiEventConverter::new(config);

    // Test thinking delta
    let event = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"delta":{"thinking":"Let me think..."}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(result.is_some());

    if let Some(Ok(ChatStreamEvent::ThinkingDelta { delta })) = result {
        assert_eq!(delta, "Let me think...");
    } else {
        panic!("Expected ThinkingDelta event");
    }
}

#[tokio::test]
async fn test_openai_usage_conversion() {
    let config = siumai::providers::openai::config::OpenAiConfig::default();
    let converter = OpenAiEventConverter::new(config);

    // Test usage update
    let event = Event {
        event: "".to_string(),
        data: r#"{"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(result.is_some());

    if let Some(Ok(ChatStreamEvent::UsageUpdate { usage })) = result {
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    } else {
        panic!("Expected UsageUpdate event");
    }
}

#[tokio::test]
async fn test_openai_finish_reason_conversion() {
    let config = siumai::providers::openai::config::OpenAiConfig::default();
    let converter = OpenAiEventConverter::new(config);

    // Test finish reason
    let event = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"finish_reason":"stop"}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(result.is_some());

    if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = result {
        assert_eq!(
            response.finish_reason,
            Some(siumai::types::FinishReason::Stop)
        );
    } else {
        panic!("Expected StreamEnd event");
    }
}

#[tokio::test]
async fn test_ollama_stream_end() {
    let converter = OllamaEventConverter::new();

    // Test stream end with usage
    let json_data = r#"{"model":"llama2","done":true,"prompt_eval_count":10,"eval_count":20}"#;

    let result = converter.convert_json(json_data).await;
    assert!(result.is_some());

    if let Some(Ok(ChatStreamEvent::UsageUpdate { usage })) = result {
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    } else {
        panic!("Expected UsageUpdate event");
    }
}

#[tokio::test]
async fn test_gemini_finish_reason() {
    let config = siumai::providers::gemini::types::GeminiConfig::default();
    let converter = GeminiEventConverter::new(config);

    // Test finish reason
    let json_data = r#"{"candidates":[{"finishReason":"STOP"}]}"#;

    let result = converter.convert_json(json_data).await;
    assert!(result.is_some());

    if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = result {
        assert_eq!(
            response.finish_reason,
            Some(siumai::types::FinishReason::Stop)
        );
    } else {
        panic!("Expected StreamEnd event");
    }
}

#[tokio::test]
async fn test_anthropic_stream_end() {
    let config = siumai::params::AnthropicParams::default();
    let converter = AnthropicEventConverter::new(config);

    // Test stream end
    let event = Event {
        event: "".to_string(),
        data: r#"{"type":"message_stop"}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(result.is_some());

    if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = result {
        assert_eq!(
            response.finish_reason,
            Some(siumai::types::FinishReason::Stop)
        );
    } else {
        panic!("Expected StreamEnd event");
    }
}

#[tokio::test]
async fn test_error_handling() {
    let config = siumai::providers::openai::config::OpenAiConfig::default();
    let converter = OpenAiEventConverter::new(config);

    // Test invalid JSON
    let event = Event {
        event: "".to_string(),
        data: "invalid json".to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(result.is_some());

    if let Some(Err(_)) = result {
        // Expected error
    } else {
        panic!("Expected error for invalid JSON");
    }
}
