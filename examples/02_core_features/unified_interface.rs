//! 🔄 Unified Interface - Provider-agnostic AI interactions
//!
//! This example demonstrates how to use the same code with different AI providers:
//! - Provider abstraction and switching
//! - Capability detection and graceful degradation
//! - Fallback strategies for reliability
//! - Dynamic provider selection
//!
//! Before running, set your API keys:
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export ANTHROPIC_API_KEY="your-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example unified_interface
//! ```

use siumai::prelude::*;
use siumai::traits::ChatCapability;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Unified Interface - Provider-agnostic AI interactions\n");

    // Demonstrate different aspects of unified interface
    demonstrate_provider_abstraction().await;
    demonstrate_dynamic_provider_selection().await;
    demonstrate_fallback_strategies().await;
    demonstrate_capability_detection().await;
    demonstrate_provider_independent_functions().await;

    println!("\n✅ Unified interface examples completed!");
    Ok(())
}

/// Demonstrate basic provider abstraction
async fn demonstrate_provider_abstraction() {
    println!("🎭 Provider Abstraction:\n");

    // Create different providers
    let providers = create_available_providers().await;

    if providers.is_empty() {
        println!("   ⚠️  No providers available. Set API keys or start Ollama.");
        return;
    }

    let test_message = "Hello! Please introduce yourself in one sentence.";

    for (name, client) in providers {
        println!("   Testing with {name}:");

        let messages = vec![user!(test_message)];
        match client.chat(messages).await {
            Ok(response) => {
                if let Some(text) = response.content_text() {
                    println!("      Response: {text}");
                    println!("      ✅ Success");
                }
            }
            Err(e) => {
                println!("      ❌ Failed: {e}");
            }
        }
        println!();
    }
}

/// Demonstrate dynamic provider selection based on task
async fn demonstrate_dynamic_provider_selection() {
    println!("🎯 Dynamic Provider Selection:\n");

    let tasks = vec![
        ("simple_question", "What is 2+2?"),
        ("creative_writing", "Write a haiku about programming"),
        ("analysis", "Analyze the pros and cons of remote work"),
    ];

    for (task_type, prompt) in tasks {
        println!("   Task: {task_type} - \"{prompt}\"");

        match select_best_provider_for_task(task_type).await {
            Ok((provider_name, client)) => {
                println!("      Selected provider: {provider_name}");

                let messages = vec![user!(prompt)];
                match client.chat(messages).await {
                    Ok(response) => {
                        if let Some(text) = response.content_text() {
                            let preview = if text.len() > 100 {
                                format!("{}...", &text[..100])
                            } else {
                                text.to_string()
                            };
                            println!("      Response: {preview}");
                            println!("      ✅ Success");
                        }
                    }
                    Err(e) => {
                        println!("      ❌ Failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("      ❌ No suitable provider: {e}");
            }
        }
        println!();
    }
}

/// Demonstrate fallback strategies
async fn demonstrate_fallback_strategies() {
    println!("🛡️  Fallback Strategies:\n");

    let message = "Explain machine learning in simple terms";

    match chat_with_fallback(message).await {
        Ok((provider_name, response)) => {
            println!("   Successfully used provider: {provider_name}");
            if let Some(text) = response.content_text() {
                let preview = if text.len() > 150 {
                    format!("{}...", &text[..150])
                } else {
                    text.to_string()
                };
                println!("   Response: {preview}");
            }
            println!("   ✅ Fallback strategy successful");
        }
        Err(e) => {
            println!("   ❌ All providers failed: {e}");
        }
    }
    println!();
}

/// Demonstrate capability detection
async fn demonstrate_capability_detection() {
    println!("🔍 Capability Detection:\n");

    let providers = create_available_providers().await;

    for (name, client) in providers {
        println!("   Provider: {name}");

        // Test basic chat capability
        println!("      Chat: ✅ (all providers support this)");

        // Test streaming capability (simplified check)
        let supports_streaming = test_streaming_capability(client.as_ref()).await;
        println!(
            "      Streaming: {}",
            if supports_streaming { "✅" } else { "❌" }
        );

        // Note: In a real implementation, you would check actual capabilities
        println!("      Vision: 🔍 (would need capability detection)");
        println!("      Tools: 🔍 (would need capability detection)");

        println!();
    }
}

/// Demonstrate provider-independent functions
async fn demonstrate_provider_independent_functions() {
    println!("🔧 Provider-Independent Functions:\n");

    let providers = create_available_providers().await;

    if let Some((name, client)) = providers.into_iter().next() {
        println!("   Using provider: {name}");

        // Use the same function with any provider
        match ask_question(client.as_ref(), "What is the meaning of life?").await {
            Ok(answer) => {
                println!("   Question: What is the meaning of life?");
                println!("   Answer: {answer}");
                println!("   ✅ Provider-independent function successful");
            }
            Err(e) => {
                println!("   ❌ Function failed: {e}");
            }
        }
    } else {
        println!("   ⚠️  No providers available for testing");
    }
}

/// Create all available providers
async fn create_available_providers() -> Vec<(String, Box<dyn ChatCapability + Send + Sync>)> {
    let mut providers = Vec::new();

    // Try OpenAI
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY")
        && let Ok(client) = LlmBuilder::new()
            .openai()
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .build()
            .await
    {
        providers.push((
            "OpenAI".to_string(),
            Box::new(client) as Box<dyn ChatCapability + Send + Sync>,
        ));
    }

    // Try Anthropic
    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY")
        && let Ok(client) = LlmBuilder::new()
            .anthropic()
            .api_key(&api_key)
            .model("claude-3-5-haiku-20241022")
            .build()
            .await
    {
        providers.push((
            "Anthropic".to_string(),
            Box::new(client) as Box<dyn ChatCapability + Send + Sync>,
        ));
    }

    // Try Ollama
    if let Ok(client) = LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .build()
        .await
    {
        // Test if Ollama is actually available
        let test_messages = vec![user!("Hi")];
        if client.chat(test_messages).await.is_ok() {
            providers.push((
                "Ollama".to_string(),
                Box::new(client) as Box<dyn ChatCapability + Send + Sync>,
            ));
        }
    }

    providers
}

/// Select the best provider for a specific task
async fn select_best_provider_for_task(
    task_type: &str,
) -> Result<(String, Box<dyn ChatCapability + Send + Sync>), LlmError> {
    let mut providers = create_available_providers().await;

    if providers.is_empty() {
        return Err(LlmError::InternalError(
            "No providers available".to_string(),
        ));
    }

    // Simple task-based selection logic
    match task_type {
        "simple_question" => {
            // Prefer faster providers for simple questions
            for i in 0..providers.len() {
                let (name, _) = &providers[i];
                if name == "Ollama" || name == "OpenAI" {
                    let (name, client) = providers.remove(i);
                    return Ok((name, client));
                }
            }
            // If no preferred provider found, use first available
            if let Some((name, client)) = providers.pop() {
                return Ok((name, client));
            }
        }
        "creative_writing" => {
            // Prefer providers known for creativity
            for i in 0..providers.len() {
                let (name, _) = &providers[i];
                if name == "Anthropic" || name == "OpenAI" {
                    let (name, client) = providers.remove(i);
                    return Ok((name, client));
                }
            }
            if let Some((name, client)) = providers.pop() {
                return Ok((name, client));
            }
        }
        "analysis" => {
            // Prefer providers good at analysis
            for i in 0..providers.len() {
                let (name, _) = &providers[i];
                if name == "Anthropic" {
                    let (name, client) = providers.remove(i);
                    return Ok((name, client));
                }
            }
            if let Some((name, client)) = providers.pop() {
                return Ok((name, client));
            }
        }
        _ => {}
    }

    // Fallback to first available provider
    providers
        .into_iter()
        .next()
        .ok_or_else(|| LlmError::InternalError("No suitable provider found".to_string()))
}

/// Chat with fallback strategy
async fn chat_with_fallback(message: &str) -> Result<(String, ChatResponse), LlmError> {
    let providers = create_available_providers().await;

    for (name, client) in providers {
        let messages = vec![user!(message)];
        if let Ok(response) = client.chat(messages).await {
            return Ok((name, response));
        } else {
            // Continue to next provider
        }
    }

    Err(LlmError::InternalError("All providers failed".to_string()))
}

/// Test if a provider supports streaming (simplified)
async fn test_streaming_capability(_client: &dyn ChatCapability) -> bool {
    // In a real implementation, you would check the provider's capabilities
    // For now, we'll assume all providers support streaming
    true
}

/// Provider-independent function that works with any client
async fn ask_question(client: &dyn ChatCapability, question: &str) -> Result<String, LlmError> {
    let messages = vec![user!(question)];
    let response = client.chat(messages).await?;
    Ok(response.content_text().unwrap_or_default().to_string())
}

/*
🎯 Key Unified Interface Concepts:

Provider Abstraction:
- Use trait objects to abstract over different providers
- Same interface works with OpenAI, Anthropic, Ollama, etc.
- Dynamic provider creation and selection

Capability Detection:
- Check what features each provider supports
- Graceful degradation when features aren't available
- Runtime feature discovery

Fallback Strategies:
- Try multiple providers in order of preference
- Handle provider failures gracefully
- Ensure application reliability

Best Practices:
1. Always have fallback providers configured
2. Choose providers based on task requirements
3. Handle provider-specific errors appropriately
4. Monitor provider performance and costs
5. Use capability detection for optional features

Benefits:
- Vendor independence and flexibility
- Improved reliability through redundancy
- Cost optimization through provider selection
- Easy testing with local providers

Next Steps:
- error_handling.rs: Robust error management
- capability_detection.rs: Advanced feature detection
- ../04_providers/: Provider-specific optimizations
*/
