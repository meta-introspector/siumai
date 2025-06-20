//! OpenAI Enhanced Features Example
//!
//! This example demonstrates the newly implemented OpenAI API features:
//! - Developer role messages
//! - New parameters: modalities, reasoning_effort, max_completion_tokens, etc.
//! - Enhanced parameter validation

use siumai::{
    params::{OpenAiParamsBuilder, ReasoningEffort},
    providers::openai::{OpenAiClient, OpenAiConfig},
    types::{ChatMessage, CommonParams, ProviderParams},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the OpenAI client
    let config = OpenAiConfig::new("your-api-key-here");
    let http_client = reqwest::Client::new();
    let client = OpenAiClient::new(config, http_client);

    println!("🚀 OpenAI Enhanced Features Demo");
    println!("=================================\n");

    // Example 1: Using the new Developer role
    developer_role_example(&client).await?;

    // Example 2: Using new parameters
    enhanced_parameters_example(&client).await?;

    // Example 3: Reasoning effort for o1 models
    reasoning_effort_example(&client).await?;

    // Example 4: Modalities parameter
    modalities_example(&client).await?;

    Ok(())
}

/// Example 1: Demonstrate the new Developer role
async fn developer_role_example(_client: &OpenAiClient) -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Example 1: Developer Role Messages");
    println!("-------------------------------------");

    let messages = vec![
        ChatMessage::system("You are a helpful AI assistant.").build(),
        ChatMessage::developer("Always respond in a structured format with clear sections.")
            .build(),
        ChatMessage::user("Explain what machine learning is.").build(),
    ];

    // Note: This is a demonstration - actual API call would require valid credentials
    println!("Messages created with Developer role:");
    for (i, msg) in messages.iter().enumerate() {
        println!("  {}. Role: {:?}", i + 1, msg.role);
        if let Some(text) = msg.content_text() {
            println!(
                "     Content: {}",
                text.chars().take(50).collect::<String>()
            );
        }
    }
    println!();

    Ok(())
}

/// Example 2: Demonstrate enhanced parameters
async fn enhanced_parameters_example(
    _client: &OpenAiClient,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚙️  Example 2: Enhanced Parameters");
    println!("----------------------------------");

    // Create enhanced OpenAI parameters
    let openai_params = OpenAiParamsBuilder::new()
        .frequency_penalty(0.5)
        .presence_penalty(-0.2)
        .max_completion_tokens(1000)
        .service_tier("default".to_string())
        .user("user-123".to_string())
        .build();

    // Create logit bias for specific tokens
    let mut logit_bias = HashMap::new();
    logit_bias.insert("50256".to_string(), -100.0); // Suppress end-of-text token

    let enhanced_params = OpenAiParamsBuilder::new().logit_bias(logit_bias).build();

    println!("Enhanced parameters configured:");
    println!(
        "  - Frequency penalty: {:?}",
        openai_params.frequency_penalty
    );
    println!("  - Presence penalty: {:?}", openai_params.presence_penalty);
    println!(
        "  - Max completion tokens: {:?}",
        openai_params.max_completion_tokens
    );
    println!("  - Service tier: {:?}", openai_params.service_tier);
    println!("  - User ID: {:?}", openai_params.user);
    println!(
        "  - Logit bias configured: {}",
        enhanced_params.logit_bias.is_some()
    );
    println!();

    Ok(())
}

/// Example 3: Demonstrate reasoning effort for o1 models
async fn reasoning_effort_example(
    _client: &OpenAiClient,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Example 3: Reasoning Effort (o1 Models)");
    println!("-------------------------------------------");

    // Configure parameters for reasoning models
    let reasoning_params = OpenAiParamsBuilder::new()
        .reasoning_effort(ReasoningEffort::High)
        .max_completion_tokens(2000)
        .build();

    println!("Reasoning model parameters:");
    println!(
        "  - Reasoning effort: {:?}",
        reasoning_params.reasoning_effort
    );
    println!(
        "  - Max completion tokens: {:?}",
        reasoning_params.max_completion_tokens
    );
    println!("  - Note: Temperature and top_p are not supported for reasoning models");
    println!();

    Ok(())
}

/// Example 4: Demonstrate modalities parameter
async fn modalities_example(_client: &OpenAiClient) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎭 Example 4: Response Modalities");
    println!("---------------------------------");

    // Configure modalities for multimodal responses
    let multimodal_params = OpenAiParamsBuilder::new()
        .modalities(vec!["text".to_string(), "audio".to_string()])
        .build();

    println!("Multimodal parameters:");
    println!(
        "  - Supported modalities: {:?}",
        multimodal_params.modalities
    );
    println!("  - This enables both text and audio responses");
    println!();

    Ok(())
}

/// Example 5: Parameter validation demonstration
#[allow(dead_code)]
fn parameter_validation_example() {
    println!("✅ Example 5: Parameter Validation");
    println!("----------------------------------");

    // These would cause validation errors:

    // Invalid frequency penalty (outside -2.0 to 2.0 range)
    // let invalid_params = OpenAiParamsBuilder::new()
    //     .frequency_penalty(3.0)  // This would fail validation
    //     .build();

    // Invalid service tier
    // let invalid_tier = OpenAiParamsBuilder::new()
    //     .service_tier("premium".to_string())  // Only "auto" and "default" are supported
    //     .build();

    // Invalid modality
    // let invalid_modality = OpenAiParamsBuilder::new()
    //     .modalities(vec!["video".to_string()])  // Only "text" and "audio" are supported
    //     .build();

    println!("Parameter validation ensures:");
    println!("  - Frequency/presence penalties are in [-2.0, 2.0] range");
    println!("  - Service tier is 'auto' or 'default'");
    println!("  - Modalities are 'text' or 'audio'");
    println!("  - Top logprobs is between 0 and 20");
    println!();
}

/// Example 6: Complete chat request with all new features
#[allow(dead_code)]
async fn complete_example(_client: &OpenAiClient) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Example 6: Complete Enhanced Request");
    println!("--------------------------------------");

    // Create messages with developer role
    let _messages = vec![
        ChatMessage::system("You are an expert software engineer.").build(),
        ChatMessage::developer(
            "Always provide code examples and explain your reasoning step by step.",
        )
        .build(),
        ChatMessage::user("How do I implement a binary search algorithm in Rust?").build(),
    ];

    // Create comprehensive parameters
    let openai_params = OpenAiParamsBuilder::new()
        .frequency_penalty(0.1)
        .presence_penalty(0.1)
        .max_completion_tokens(1500)
        .service_tier("default".to_string())
        .user("developer-123".to_string())
        .modalities(vec!["text".to_string()])
        .build();

    // Convert to provider params
    let _provider_params = ProviderParams::from_openai(openai_params);

    // Create common parameters
    let _common_params = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(0.7),
        max_tokens: None, // Using max_completion_tokens instead
        top_p: Some(0.9),
        stop_sequences: None,
        seed: Some(42),
    };

    println!("Complete request configured with:");
    println!("  - Developer role message included");
    println!("  - Enhanced parameters set");
    println!("  - Validation will be performed automatically");
    println!();

    // Note: Actual API call would be:
    // let response = client.chat_with_params(messages, None, common_params, Some(provider_params)).await?;

    Ok(())
}
