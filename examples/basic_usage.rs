//! Basic Usage Examples for Siumai LLM Library
//!
//! This example demonstrates the core functionality of the unified LLM interface

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("🚀 Siumai LLM Library - Basic Usage Examples");
    println!("===========================================\n");

    // Example 1: Basic Message Creation
    basic_message_creation();

    // Example 2: Multimodal Messages
    multimodal_message_example();

    // Example 3: Chat Request Building
    chat_request_building();

    // Example 4: Provider Configuration
    provider_configuration_example();

    // Example 5: Error Handling
    error_handling_example();

    // Example 6: Parameter Mapping
    parameter_mapping_example();

    // Example 7: Stream Processing (conceptual)
    stream_processing_example();

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Example 1: Basic Message Creation
///
/// Demonstrates how to create different types of messages using convenience macros
fn basic_message_creation() {
    println!("📝 Example 1: Basic Message Creation");
    println!("-----------------------------------");

    // Create messages using convenience macros - now returns ChatMessage directly
    let system_msg = system!("You are a helpful AI assistant specialized in Rust programming.");
    let user_msg = user!("How do I create a vector in Rust?");
    let assistant_msg = assistant!("You can create a vector using Vec::new() or the vec! macro.");

    println!("System message: {:?}", system_msg.role);
    println!("User message: {:?}", user_msg.role);
    println!("Assistant message: {:?}", assistant_msg.role);

    // Extract text content
    if let MessageContent::Text(text) = &user_msg.content {
        println!("User question: {}", text);
    }

    println!();
}

/// Example 2: Multimodal Messages
///
/// Shows how to create messages with multiple content types (text + images)
fn multimodal_message_example() {
    println!("🖼️  Example 2: Multimodal Messages");
    println!("----------------------------------");

    // Create a multimodal message with text and image
    // For complex messages with additional parameters, use the builder pattern
    let multimodal_msg = ChatMessage::user("What do you see in this image?")
        .with_image(
            "https://example.com/rust-logo.png".to_string(),
            Some("high".to_string()),
        )
        .build();

    match &multimodal_msg.content {
        MessageContent::MultiModal(parts) => {
            println!("Multimodal message with {} parts:", parts.len());
            for (i, part) in parts.iter().enumerate() {
                match part {
                    ContentPart::Text { text } => println!("  Part {}: Text - {}", i + 1, text),
                    ContentPart::Image { image_url, detail } => {
                        println!(
                            "  Part {}: Image - {} (detail: {:?})",
                            i + 1,
                            image_url,
                            detail
                        );
                    }
                    ContentPart::Audio { audio_url, format } => {
                        println!(
                            "  Part {}: Audio - {} (format: {})",
                            i + 1,
                            audio_url,
                            format
                        );
                    }
                }
            }
        }
        MessageContent::Text(text) => println!("Text message: {}", text),
    }

    println!();
}

/// Example 3: Chat Request Building
///
/// Demonstrates building complete chat requests with parameters
fn chat_request_building() {
    println!("🔧 Example 3: Chat Request Building");
    println!("-----------------------------------");

    // Build a conversation - simple messages use macros directly
    let messages = vec![
        system!("You are a helpful assistant."),
        user!("Explain quantum computing in simple terms."),
    ];

    // Create common parameters
    let common_params = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: Some(vec!["END".to_string()]),
        seed: Some(42),
    };

    // Create provider-specific parameters
    let provider_params = ProviderParams::new()
        .with_param("frequency_penalty", 0.1)
        .with_param("presence_penalty", 0.1);

    // Build the complete request
    let request = ChatRequest::builder()
        .messages(messages)
        .common_params(common_params)
        .provider_params(provider_params)
        .build();

    println!(
        "Chat request created with {} messages",
        request.messages.len()
    );
    println!("Model: {}", request.common_params.model);
    println!("Temperature: {:?}", request.common_params.temperature);
    println!("Max tokens: {:?}", request.common_params.max_tokens);

    println!();
}

/// Example 4: Provider Configuration
///
/// Shows how to work with different LLM providers and their capabilities
fn provider_configuration_example() {
    println!("🏢 Example 4: Provider Configuration");
    println!("------------------------------------");

    // Get information about supported providers
    let providers = siumai::providers::get_supported_providers();
    println!("Supported providers: {}", providers.len());

    for provider in &providers {
        println!("  📋 {}: {}", provider.name, provider.description);
        println!("     Default URL: {}", provider.default_base_url);
        println!(
            "     Capabilities: Chat={}, Streaming={}, Tools={}, Vision={}",
            provider.capabilities.chat,
            provider.capabilities.streaming,
            provider.capabilities.tools,
            provider.capabilities.vision
        );
        println!("     Models: {:?}", provider.supported_models);
        println!();
    }

    // Check model support
    let is_gpt4_supported = siumai::providers::is_model_supported(&ProviderType::OpenAi, "gpt-4");
    let is_claude_supported = siumai::providers::is_model_supported(
        &ProviderType::Anthropic,
        "claude-3-5-sonnet-20241022",
    );

    println!("GPT-4 supported by OpenAI: {}", is_gpt4_supported);
    println!(
        "Claude 3.5 Sonnet supported by Anthropic: {}",
        is_claude_supported
    );

    println!();
}

/// Example 5: Error Handling
///
/// Demonstrates different types of errors and their properties
fn error_handling_example() {
    println!("⚠️  Example 5: Error Handling");
    println!("-----------------------------");

    // Create different types of errors
    let errors = vec![
        LlmError::api_error(404, "Model not found"),
        LlmError::api_error(429, "Rate limit exceeded"),
        LlmError::api_error(500, "Internal server error"),
        LlmError::AuthenticationError("Invalid API key".to_string()),
        LlmError::RateLimitError("Too many requests".to_string()),
        LlmError::TimeoutError("Request timed out".to_string()),
        LlmError::ModelNotSupported("unsupported-model".to_string()),
    ];

    for error in errors {
        println!("Error: {}", error);
        println!("  Status code: {:?}", error.status_code());
        println!("  Is retryable: {}", error.is_retryable());
        println!("  Is auth error: {}", error.is_auth_error());
        println!("  Is rate limit: {}", error.is_rate_limit_error());
        println!();
    }
}

/// Example 6: Parameter Mapping
///
/// Shows how parameters are mapped between common format and provider-specific format
fn parameter_mapping_example() {
    println!("🔄 Example 6: Parameter Mapping");
    println!("-------------------------------");

    use siumai::params::{AnthropicParameterMapper, OpenAiParameterMapper, ParameterMapper};

    // Common parameters
    let common_params = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(0.8),
        max_tokens: Some(2000),
        top_p: Some(0.95),
        stop_sequences: Some(vec!["STOP".to_string(), "END".to_string()]),
        seed: Some(123),
    };

    // Map to OpenAI format
    let openai_mapper = OpenAiParameterMapper;
    let openai_params = openai_mapper.map_common_params(&common_params);
    println!(
        "OpenAI format: {}",
        serde_json::to_string_pretty(&openai_params).unwrap()
    );

    // Map to Anthropic format
    let anthropic_mapper = AnthropicParameterMapper;
    let anthropic_params = anthropic_mapper.map_common_params(&common_params);
    println!(
        "Anthropic format: {}",
        serde_json::to_string_pretty(&anthropic_params).unwrap()
    );

    println!();
}

/// Example 7: Stream Processing (Conceptual)
///
/// Demonstrates how stream processing would work (without actual streaming)
fn stream_processing_example() {
    println!("🌊 Example 7: Stream Processing (Conceptual)");
    println!("--------------------------------------------");

    // Create a stream processor
    let mut processor = StreamProcessor::new();

    // Simulate processing stream events
    let events = vec![
        ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("stream_001".to_string()),
                model: Some("gpt-4".to_string()),
                created: Some(chrono::Utc::now()),
                provider: "openai".to_string(),
                request_id: Some("req_123".to_string()),
            },
        },
        ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: Some(0),
        },
        ChatStreamEvent::ContentDelta {
            delta: " there!".to_string(),
            index: Some(0),
        },
        ChatStreamEvent::UsageUpdate {
            usage: Usage {
                prompt_tokens: Some(10),
                completion_tokens: Some(2),
                total_tokens: Some(12),
                reasoning_tokens: None,
                cache_hit_tokens: None,
                cache_creation_tokens: None,
            },
        },
    ];

    println!("Processing {} stream events:", events.len());

    for (i, event) in events.into_iter().enumerate() {
        let processed = processor.process_event(event);
        match processed {
            ProcessedEvent::ContentUpdate {
                delta, accumulated, ..
            } => {
                println!(
                    "  Event {}: Content delta '{}' -> accumulated '{}'",
                    i + 1,
                    delta,
                    accumulated
                );
            }
            ProcessedEvent::UsageUpdate { usage } => {
                println!(
                    "  Event {}: Usage update - {} total tokens",
                    i + 1,
                    usage.total_tokens.unwrap_or(0)
                );
            }
            _ => {
                println!("  Event {}: Other event type", i + 1);
            }
        }
    }

    println!();
}
