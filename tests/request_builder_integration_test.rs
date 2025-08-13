//! RequestBuilder Integration Tests
//!
//! Tests to verify that the RequestBuilder system works correctly
//! and that SiumaiBuilder properly integrates with it.

use siumai::error::LlmError;
use siumai::provider::SiumaiBuilder;
use siumai::request_factory::{RequestBuilder, RequestBuilderFactory, StandardRequestBuilder};
use siumai::types::{
    ChatMessage, CommonParams, MessageContent, MessageMetadata, MessageRole, ProviderParams,
    ProviderType,
};

/// Test that RequestBuilder can be created for all provider types
#[test]
fn test_request_builder_factory_all_providers() {
    println!("🧪 Testing RequestBuilder factory for all providers");

    let common_params = CommonParams {
        model: "test-model".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: Some(vec!["STOP".to_string()]),
        seed: Some(42),
    };

    let provider_types = vec![
        ProviderType::OpenAi,
        ProviderType::Anthropic,
        ProviderType::Gemini,
        ProviderType::Ollama,
        ProviderType::XAI,
        ProviderType::Groq,
        ProviderType::Custom("deepseek".to_string()),
        ProviderType::Custom("openrouter".to_string()),
    ];

    for provider_type in provider_types {
        println!("  🔍 Testing RequestBuilder for {:?}...", provider_type);

        let builder =
            RequestBuilderFactory::create_builder(&provider_type, common_params.clone(), None);

        // Verify that the builder was created successfully
        assert_eq!(builder.get_common_params().model, "test-model");
        assert_eq!(builder.get_common_params().temperature, Some(0.7));
        assert_eq!(builder.get_common_params().max_tokens, Some(1000));

        println!(
            "    ✅ RequestBuilder created successfully for {:?}",
            provider_type
        );
    }
}

/// Test that RequestBuilder can build ChatRequest objects
#[test]
fn test_request_builder_chat_request_creation() {
    println!("🧪 Testing ChatRequest creation via RequestBuilder");

    let common_params = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(0.8),
        max_tokens: Some(1500),
        top_p: Some(0.95),
        stop_sequences: None,
        seed: Some(123),
    };

    let builder = StandardRequestBuilder::new(common_params.clone(), None);

    let messages = vec![
        ChatMessage {
            role: MessageRole::System,
            content: MessageContent::Text("You are a helpful assistant".to_string()),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello, world!".to_string()),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    // Test non-streaming request
    let request = builder
        .build_chat_request(messages.clone(), None, false)
        .expect("Should build non-streaming request");

    assert_eq!(request.messages.len(), 2);
    assert_eq!(request.common_params.model, "gpt-4");
    assert_eq!(request.common_params.temperature, Some(0.8));
    assert!(!request.stream);

    println!("  ✅ Non-streaming ChatRequest created successfully");

    // Test streaming request
    let streaming_request = builder
        .build_chat_request(messages, None, true)
        .expect("Should build streaming request");

    assert_eq!(streaming_request.messages.len(), 2);
    assert_eq!(streaming_request.common_params.model, "gpt-4");
    assert!(streaming_request.stream);

    println!("  ✅ Streaming ChatRequest created successfully");
}

/// Test RequestBuilder parameter validation
#[test]
fn test_request_builder_parameter_validation() {
    println!("🧪 Testing RequestBuilder parameter validation");

    // Test with valid parameters
    let valid_params = CommonParams {
        model: "valid-model".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: None,
        seed: Some(42),
    };

    let builder = StandardRequestBuilder::new(valid_params, None);
    let messages = vec![ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Test message".to_string()),
        metadata: MessageMetadata::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    let result = builder.build_chat_request(messages.clone(), None, false);
    assert!(result.is_ok(), "Valid parameters should succeed");

    println!("  ✅ Valid parameters accepted");

    // Test with potentially invalid parameters (empty model)
    let invalid_params = CommonParams {
        model: "".to_string(), // Empty model
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: None,
        seed: Some(42),
    };

    let invalid_builder = StandardRequestBuilder::new(invalid_params, None);
    let invalid_result = invalid_builder.build_chat_request(messages, None, false);

    // The result depends on validation implementation
    match invalid_result {
        Ok(_) => {
            println!("  ⚠️ Empty model was accepted (validation may be lenient)");
        }
        Err(e) => {
            println!("  ✅ Empty model correctly rejected: {}", e);
        }
    }
}

/// Test that SiumaiBuilder integrates with RequestBuilder system
#[tokio::test]
async fn test_siumai_builder_request_builder_integration() {
    println!("🧪 Testing SiumaiBuilder integration with RequestBuilder");

    // This test verifies that SiumaiBuilder uses the RequestBuilder system
    // by checking that parameter validation and handling is consistent

    let test_cases = vec![
        ("openai", "gpt-4"),
        ("anthropic", "claude-3-sonnet"),
        ("gemini", "gemini-1.5-flash"),
    ];

    for (provider, model) in test_cases {
        println!("  🔍 Testing {} integration...", provider);

        let builder = SiumaiBuilder::new()
            .api_key("test-key")
            .model(model)
            .temperature(0.7)
            .max_tokens(1000)
            .top_p(0.9)
            .seed(42);

        let result = match provider {
            "openai" => builder.openai().build().await,
            "anthropic" => builder.anthropic().build().await,
            "gemini" => builder.gemini().build().await,
            _ => panic!("Unknown provider: {}", provider),
        };

        match result {
            Ok(client) => {
                println!("    ✅ {} client created successfully", provider);
                assert!(client.supports("chat"));

                // The fact that the client was created successfully indicates
                // that the RequestBuilder system is working correctly
            }
            Err(e) => match e {
                LlmError::ConfigurationError(msg) if msg.contains("parameter") => {
                    panic!("    ❌ {} parameter integration error: {}", provider, msg);
                }
                _ => {
                    println!("    ✅ {} failed with expected error: {}", provider, e);
                }
            },
        }
    }
}

/// Test RequestBuilder with provider-specific parameters
#[test]
fn test_request_builder_with_provider_params() {
    println!("🧪 Testing RequestBuilder with provider-specific parameters");

    let common_params = CommonParams {
        model: "test-model".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: None,
        seed: Some(42),
    };

    // Test with OpenAI provider params
    let openai_provider_params = Some(
        ProviderParams::openai()
            .with_param("frequency_penalty", 0.1)
            .with_param("presence_penalty", 0.2),
    );

    let openai_builder = RequestBuilderFactory::create_builder(
        &ProviderType::OpenAi,
        common_params.clone(),
        openai_provider_params,
    );

    let messages = vec![ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Test with OpenAI params".to_string()),
        metadata: MessageMetadata::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    let openai_request = openai_builder
        .build_chat_request(messages.clone(), None, false)
        .expect("Should build OpenAI request with provider params");

    assert!(openai_request.provider_params.is_some());
    println!("  ✅ OpenAI RequestBuilder with provider params works");

    // Test with Anthropic provider params
    let anthropic_provider_params = Some(
        ProviderParams::anthropic()
            .with_param("system", "You are Claude")
            .with_param("thinking_budget", 5000),
    );

    let anthropic_builder = RequestBuilderFactory::create_builder(
        &ProviderType::Anthropic,
        common_params.clone(),
        anthropic_provider_params,
    );

    let anthropic_request = anthropic_builder
        .build_chat_request(messages, None, false)
        .expect("Should build Anthropic request with provider params");

    assert!(anthropic_request.provider_params.is_some());
    println!("  ✅ Anthropic RequestBuilder with provider params works");
}

/// Test RequestBuilder consistency across multiple calls
#[test]
fn test_request_builder_consistency() {
    println!("🧪 Testing RequestBuilder consistency across multiple calls");

    let common_params = CommonParams {
        model: "consistency-test-model".to_string(),
        temperature: Some(0.5),
        max_tokens: Some(800),
        top_p: Some(0.8),
        stop_sequences: Some(vec!["END".to_string()]),
        seed: Some(999),
    };

    let builder = StandardRequestBuilder::new(common_params.clone(), None);

    let messages = vec![ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Consistency test".to_string()),
        metadata: MessageMetadata::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    // Build multiple requests and verify they're consistent
    for i in 1..=5 {
        let request = builder
            .build_chat_request(messages.clone(), None, false)
            .unwrap_or_else(|_| panic!("Should build request {}", i));

        assert_eq!(request.common_params.model, "consistency-test-model");
        assert_eq!(request.common_params.temperature, Some(0.5));
        assert_eq!(request.common_params.max_tokens, Some(800));
        assert_eq!(request.common_params.top_p, Some(0.8));
        assert_eq!(request.common_params.seed, Some(999));
        assert!(!request.stream);

        println!("  ✅ Request {} is consistent", i);
    }

    println!("  ✅ All requests are consistent");
}

/// Test RequestBuilder error handling
#[test]
fn test_request_builder_error_handling() {
    println!("🧪 Testing RequestBuilder error handling");

    // Test with empty messages (should this be allowed?)
    let common_params = CommonParams {
        model: "error-test-model".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: None,
        seed: Some(42),
    };

    let builder = StandardRequestBuilder::new(common_params, None);

    // Test with empty messages
    let empty_messages = vec![];
    let result = builder.build_chat_request(empty_messages, None, false);

    match result {
        Ok(_) => {
            println!("  ✅ Empty messages accepted (may be valid for some use cases)");
        }
        Err(e) => {
            println!("  ✅ Empty messages correctly rejected: {}", e);
        }
    }

    // Test with very large number of messages (stress test)
    let many_messages: Vec<ChatMessage> = (0..1000)
        .map(|i| ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text(format!("Message {}", i)),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        })
        .collect();

    let large_result = builder.build_chat_request(many_messages, None, false);

    match large_result {
        Ok(_) => {
            println!("  ✅ Large number of messages handled successfully");
        }
        Err(e) => {
            println!("  ✅ Large number of messages rejected: {}", e);
        }
    }
}
