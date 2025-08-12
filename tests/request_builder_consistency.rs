//! Request Builder Consistency Tests
//!
//! Tests to verify that all providers use consistent ChatRequest construction
//! patterns and that the new RequestBuilder trait works correctly.

use siumai::params::{AnthropicParams, OpenAiParams};
use siumai::providers::anthropic::request::AnthropicRequestBuilder;
use siumai::providers::openai::request::OpenAiRequestBuilder;
use siumai::request_factory::{RequestBuilder, RequestBuilderConfig, StandardRequestBuilder};
use siumai::types::{ChatMessage, CommonParams, MessageContent, MessageRole, Tool, ToolFunction};

/// Test that OpenAI RequestBuilder creates consistent requests
#[test]
fn test_openai_request_builder_consistency() {
    let common_params = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        ..Default::default()
    };

    let openai_params = OpenAiParams::default();
    let builder = OpenAiRequestBuilder::new(common_params.clone(), openai_params);

    let messages = vec![ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Hello, world!".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    // Test non-streaming request
    let request = builder
        .build_chat_request(messages.clone(), None, false)
        .expect("Should build non-streaming request");

    assert_eq!(request.common_params.model, "gpt-4");
    assert!(!request.stream);
    assert!(request.provider_params.is_some());
    assert_eq!(request.messages.len(), 1);

    // Test streaming request
    let streaming_request = builder
        .build_chat_request(messages, None, true)
        .expect("Should build streaming request");

    assert!(streaming_request.stream);
    assert!(streaming_request.provider_params.is_some());
}

/// Test that Anthropic RequestBuilder creates consistent requests
#[test]
fn test_anthropic_request_builder_consistency() {
    let common_params = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        ..Default::default()
    };

    let anthropic_params = AnthropicParams::default();
    let builder = AnthropicRequestBuilder::new(common_params.clone(), anthropic_params);

    let messages = vec![ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Hello, Claude!".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    // Test non-streaming request
    let request = builder
        .build_chat_request(messages.clone(), None, false)
        .expect("Should build non-streaming request");

    assert_eq!(request.common_params.model, "claude-3-5-sonnet-20241022");
    assert!(!request.stream);
    assert!(request.provider_params.is_some());
    assert_eq!(request.messages.len(), 1);

    // Test streaming request
    let streaming_request = builder
        .build_chat_request(messages, None, true)
        .expect("Should build streaming request");

    assert!(streaming_request.stream);
    assert!(streaming_request.provider_params.is_some());
}

/// Test that all RequestBuilders handle tools consistently
#[test]
fn test_request_builders_with_tools() {
    let tools = vec![Tool {
        r#type: "function".to_string(),
        function: ToolFunction {
            name: "get_weather".to_string(),
            description: "Get current weather".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }),
        },
    }];

    let messages = vec![ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("What's the weather?".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    // Test OpenAI with tools
    let openai_common = CommonParams {
        model: "gpt-4".to_string(),
        ..Default::default()
    };
    let openai_builder = OpenAiRequestBuilder::new(openai_common, OpenAiParams::default());
    let openai_request = openai_builder
        .build_chat_request(messages.clone(), Some(tools.clone()), false)
        .expect("OpenAI should handle tools");

    assert!(openai_request.tools.is_some());
    assert_eq!(openai_request.tools.as_ref().unwrap().len(), 1);

    // Test Anthropic with tools
    let anthropic_common = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        ..Default::default()
    };
    let anthropic_builder =
        AnthropicRequestBuilder::new(anthropic_common, AnthropicParams::default());
    let anthropic_request = anthropic_builder
        .build_chat_request(messages, Some(tools), false)
        .expect("Anthropic should handle tools");

    assert!(anthropic_request.tools.is_some());
    assert_eq!(anthropic_request.tools.as_ref().unwrap().len(), 1);
}

/// Test validation consistency across providers
#[test]
fn test_validation_consistency() {
    let messages = vec![ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Hello".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    // Test empty model validation
    let empty_model_params = CommonParams {
        model: "".to_string(),
        ..Default::default()
    };

    let config = RequestBuilderConfig {
        strict_validation: true,
        provider_validation: true,
    };

    let openai_builder =
        OpenAiRequestBuilder::new(empty_model_params.clone(), OpenAiParams::default());
    let openai_result =
        openai_builder.build_chat_request_with_config(messages.clone(), None, false, &config);
    assert!(openai_result.is_err());

    let anthropic_builder =
        AnthropicRequestBuilder::new(empty_model_params, AnthropicParams::default());
    let anthropic_result =
        anthropic_builder.build_chat_request_with_config(messages.clone(), None, false, &config);
    assert!(anthropic_result.is_err());

    // Test empty messages validation
    let valid_params = CommonParams {
        model: "test-model".to_string(),
        ..Default::default()
    };

    let standard_builder = StandardRequestBuilder::new(valid_params, None);
    let empty_messages_result = standard_builder.build_chat_request(vec![], None, false);
    assert!(empty_messages_result.is_err());
}

/// Test that provider-specific parameters are always included
#[test]
fn test_provider_params_inclusion() {
    let common_params = CommonParams {
        model: "test-model".to_string(),
        ..Default::default()
    };

    let messages = vec![ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Test".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    // OpenAI should always include provider params
    let openai_builder = OpenAiRequestBuilder::new(common_params.clone(), OpenAiParams::default());
    let openai_request = openai_builder
        .build_chat_request(messages.clone(), None, false)
        .expect("Should build OpenAI request");
    assert!(openai_request.provider_params.is_some());

    // Anthropic should always include provider params
    let anthropic_common = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        ..Default::default()
    };
    let anthropic_builder =
        AnthropicRequestBuilder::new(anthropic_common, AnthropicParams::default());
    let anthropic_request = anthropic_builder
        .build_chat_request(messages, None, false)
        .expect("Should build Anthropic request");
    assert!(anthropic_request.provider_params.is_some());
}

/// Test that streaming flag is properly set
#[test]
fn test_streaming_flag_consistency() {
    let common_params = CommonParams {
        model: "gpt-4".to_string(),
        ..Default::default()
    };

    let messages = vec![ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Test streaming".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    let builder = OpenAiRequestBuilder::new(common_params, OpenAiParams::default());

    // Test non-streaming
    let non_streaming = builder
        .build_chat_request(messages.clone(), None, false)
        .expect("Should build non-streaming request");
    assert!(!non_streaming.stream);

    // Test streaming
    let streaming = builder
        .build_chat_request(messages, None, true)
        .expect("Should build streaming request");
    assert!(streaming.stream);
}
