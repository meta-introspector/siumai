//! 💬 `OpenAI` Basic Chat - Essential `OpenAI` functionality
//!
//! This example demonstrates core `OpenAI` chat features:
//! - Model selection and configuration
//! - Parameter tuning for different use cases
//! - Token usage monitoring and optimization
//! - Response format options
//! - Cost-effective usage patterns
//!
//! Before running, set your API key:
//! ```bash
//! export OPENAI_API_KEY="your-openai-api-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example basic_chat
//! ```

use siumai::models;
use siumai::params::ResponseFormat;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💬 OpenAI Basic Chat - Essential OpenAI functionality\n");

    // Check for API key
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            println!("❌ OPENAI_API_KEY environment variable not set or empty");
            println!("💡 Set it with: export OPENAI_API_KEY=\"your-api-key\"");
            return Ok(());
        }
    };

    // Demonstrate different aspects of OpenAI chat
    demonstrate_model_selection(&api_key).await;
    demonstrate_parameter_tuning(&api_key).await;
    demonstrate_token_optimization(&api_key).await;
    demonstrate_response_formats(&api_key).await;
    demonstrate_conversation_patterns(&api_key).await;

    println!("\n✅ OpenAI basic chat examples completed!");
    Ok(())
}

/// Demonstrate different `OpenAI` models and their characteristics
async fn demonstrate_model_selection(api_key: &str) {
    println!("🤖 Model Selection:\n");

    let models = vec![
        (models::openai::GPT_4O_MINI, "Balanced cost and performance"),
        (models::openai::GPT_4O, "Best overall performance"),
        (models::openai::GPT_3_5_TURBO, "Fast and economical"),
    ];

    let test_prompt = "Explain quantum computing in exactly 50 words.";

    for (model, description) in models {
        println!("   Testing {model}: {description}");

        match LlmBuilder::new()
            .openai()
            .api_key(api_key)
            .model(model)
            .temperature(0.7)
            .max_tokens(100)
            .build()
            .await
        {
            Ok(client) => {
                let messages = vec![user!(test_prompt)];

                let start_time = std::time::Instant::now();
                match client.chat(messages).await {
                    Ok(response) => {
                        let duration = start_time.elapsed();

                        if let Some(text) = response.content_text() {
                            println!("      Response: {text}");
                        }

                        if let Some(usage) = &response.usage {
                            println!(
                                "      Tokens: {} total ({} prompt + {} completion)",
                                usage.total_tokens, usage.prompt_tokens, usage.completion_tokens
                            );

                            // Estimate cost (approximate rates)
                            let cost = match model {
                                m if m == models::openai::GPT_4O_MINI => {
                                    usage.total_tokens as f64 * 0.00015 / 1000.0
                                }
                                m if m == models::openai::GPT_4O => {
                                    usage.total_tokens as f64 * 0.005 / 1000.0
                                }
                                m if m == models::openai::GPT_3_5_TURBO => {
                                    usage.total_tokens as f64 * 0.0015 / 1000.0
                                }
                                _ => 0.0,
                            };
                            println!("      Estimated cost: ${cost:.6}");
                        }

                        println!("      Response time: {}ms", duration.as_millis());
                        println!("      ✅ Success");
                    }
                    Err(e) => {
                        println!("      ❌ Failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("      ❌ Client creation failed: {e}");
            }
        }
        println!();
    }
}

/// Demonstrate parameter tuning for different use cases
async fn demonstrate_parameter_tuning(api_key: &str) {
    println!("🎛️ Parameter Tuning:\n");

    let scenarios = vec![
        (
            "Creative Writing",
            0.9,
            "Write a creative short story opening about a time traveler.",
        ),
        (
            "Technical Analysis",
            0.1,
            "Explain the technical differences between REST and GraphQL APIs.",
        ),
        (
            "Balanced Response",
            0.5,
            "What are the pros and cons of remote work?",
        ),
    ];

    for (scenario, temperature, prompt) in scenarios {
        println!("   Scenario: {scenario} (temperature: {temperature})");

        match LlmBuilder::new()
            .openai()
            .api_key(api_key)
            .model(models::openai::GPT_4O_MINI)
            .temperature(temperature)
            .max_tokens(150)
            .build()
            .await
        {
            Ok(client) => {
                let messages = vec![user!(prompt)];

                match client.chat(messages).await {
                    Ok(response) => {
                        if let Some(text) = response.content_text() {
                            let preview = if text.len() > 200 {
                                format!("{}...", &text[..200])
                            } else {
                                text.to_string()
                            };
                            println!("      Response: {preview}");
                        }
                        println!("      ✅ Success");
                    }
                    Err(e) => {
                        println!("      ❌ Failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("      ❌ Client creation failed: {e}");
            }
        }
        println!();
    }
}

/// Demonstrate token usage optimization
async fn demonstrate_token_optimization(api_key: &str) {
    println!("🔧 Token Optimization:\n");

    let optimization_strategies = vec![
        (
            "Unlimited tokens",
            None,
            "Write a comprehensive guide about machine learning.",
        ),
        (
            "Limited tokens",
            Some(50),
            "Write a comprehensive guide about machine learning.",
        ),
        (
            "Concise prompt",
            Some(100),
            "Explain machine learning briefly.",
        ),
    ];

    for (strategy, max_tokens, prompt) in optimization_strategies {
        println!("   Strategy: {strategy}");

        let mut builder = LlmBuilder::new()
            .openai()
            .api_key(api_key)
            .model(models::openai::GPT_4O_MINI)
            .temperature(0.7);

        if let Some(tokens) = max_tokens {
            builder = builder.max_tokens(tokens);
        }

        match builder.build().await {
            Ok(client) => {
                let messages = vec![user!(prompt)];

                match client.chat(messages).await {
                    Ok(response) => {
                        if let Some(text) = response.content_text() {
                            println!("      Response length: {} characters", text.len());
                            let preview = if text.len() > 150 {
                                format!("{}...", &text[..150])
                            } else {
                                text.to_string()
                            };
                            println!("      Preview: {preview}");
                        }

                        if let Some(usage) = &response.usage {
                            println!("      Token usage: {} total", usage.total_tokens);
                            println!(
                                "      Efficiency: {:.2} chars/token",
                                response.content_text().map_or(0.0, |t| t.len() as f64)
                                    / usage.total_tokens as f64
                            );
                        }

                        println!("      ✅ Success");
                    }
                    Err(e) => {
                        println!("      ❌ Failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("      ❌ Client creation failed: {e}");
            }
        }
        println!();
    }
}

/// Demonstrate response format options
async fn demonstrate_response_formats(api_key: &str) {
    println!("📋 Response Formats:\n");

    // Standard text response
    println!("   Format: Standard Text");
    match create_openai_client(api_key, models::openai::GPT_4O_MINI).await {
        Ok(client) => {
            let messages = vec![user!(
                "What are the three primary colors? Answer in a simple list."
            )];

            match client.chat(messages).await {
                Ok(response) => {
                    if let Some(text) = response.content_text() {
                        println!("      Response: {text}");
                    }
                    println!("      ✅ Standard format success");
                }
                Err(e) => {
                    println!("      ❌ Failed: {e}");
                }
            }
        }
        Err(e) => {
            println!("      ❌ Client creation failed: {e}");
        }
    }

    // JSON format (if supported)
    println!("\n   Format: JSON (structured)");
    match LlmBuilder::new()
        .openai()
        .api_key(api_key)
        .model(models::openai::GPT_4O_MINI)
        .response_format(ResponseFormat::JsonObject)
        .build()
        .await
    {
        Ok(client) => {
            let messages = vec![
                system!(
                    "Respond in valid JSON format with 'colors' array containing the three primary colors."
                ),
                user!("What are the three primary colors?"),
            ];

            match client.chat(messages).await {
                Ok(response) => {
                    if let Some(text) = response.content_text() {
                        println!("      JSON Response: {text}");
                    }
                    println!("      ✅ JSON format success");
                }
                Err(e) => {
                    println!("      ❌ Failed: {e}");
                }
            }
        }
        Err(e) => {
            println!("      ❌ JSON client creation failed: {e}");
        }
    }

    println!();
}

/// Demonstrate conversation management patterns
async fn demonstrate_conversation_patterns(api_key: &str) {
    println!("💬 Conversation Patterns:\n");

    match create_openai_client(api_key, models::openai::GPT_4O_MINI).await {
        Ok(client) => {
            // Build a conversation step by step
            let mut conversation = vec![system!(
                "You are a helpful math tutor. Keep responses concise and encouraging."
            )];

            // First exchange
            conversation.push(user!("I need help with algebra. What are variables?"));

            match client.chat(conversation.clone()).await {
                Ok(response) => {
                    if let Some(text) = response.content_text() {
                        conversation.push(assistant!(text));
                        println!("   User: I need help with algebra. What are variables?");
                        println!("   Assistant: {text}");
                    }
                }
                Err(e) => {
                    println!("   ❌ First exchange failed: {e}");
                    return;
                }
            }

            // Second exchange - AI remembers context
            conversation.push(user!("Can you give me a simple example with variables?"));

            match client.chat(conversation.clone()).await {
                Ok(response) => {
                    if let Some(text) = response.content_text() {
                        conversation.push(assistant!(text));
                        println!("   User: Can you give me a simple example with variables?");
                        println!("   Assistant: {text}");
                    }
                }
                Err(e) => {
                    println!("   ❌ Second exchange failed: {e}");
                    return;
                }
            }

            // Third exchange - test memory
            conversation.push(user!("What was my original question?"));

            match client.chat(conversation).await {
                Ok(response) => {
                    if let Some(text) = response.content_text() {
                        println!("   User: What was my original question?");
                        println!("   Assistant: {text}");
                    }
                    println!("   ✅ Conversation memory maintained successfully");
                }
                Err(e) => {
                    println!("   ❌ Third exchange failed: {e}");
                }
            }
        }
        Err(e) => {
            println!("   ❌ Client creation failed: {e}");
        }
    }

    println!();
}

/// Helper function to create `OpenAI` client
async fn create_openai_client(api_key: &str, model: &str) -> Result<impl ChatCapability, LlmError> {
    LlmBuilder::new()
        .openai()
        .api_key(api_key)
        .model(model)
        .temperature(0.7)
        .build()
        .await
}

/*
🎯 Key OpenAI Concepts:

Model Selection:
- gpt-4o-mini: Best balance of cost and performance
- gpt-4o: Highest quality, more expensive
- gpt-3.5-turbo: Fast and economical for simple tasks

Parameter Tuning:
- Temperature: 0.0-1.0 (deterministic to creative)
- Max tokens: Limit response length and cost
- Top-p: Alternative to temperature for nucleus sampling

Cost Optimization:
- Choose appropriate model for task complexity
- Limit max_tokens for cost control
- Monitor token usage patterns
- Use shorter prompts when possible

Response Formats:
- Standard text: Default format
- JSON mode: Structured data output
- Function calling: Tool integration

Best Practices:
1. Start with gpt-4o-mini for development
2. Use environment variables for API keys
3. Implement proper error handling
4. Monitor token usage and costs
5. Use conversation history appropriately
6. Set reasonable timeouts

Production Considerations:
- Rate limiting and backoff strategies
- Cost monitoring and budgets
- Error logging and alerting
- Performance metrics tracking
- Security and API key management

Next Steps:
- enhanced_features.rs: Advanced OpenAI capabilities
- vision_processing.rs: GPT-4 Vision features
- audio_processing.rs: Whisper and TTS integration
*/
