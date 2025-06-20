//! Siumai Interface Example
//!
//! This example demonstrates the main siumai interface for unified LLM provider access.
//! The siumai interface provides a clean, type-safe way to work with different LLM providers
//! while maintaining the flexibility to switch between them dynamically.

use futures_util::stream::StreamExt;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Siumai Interface Demo");
    println!("========================\n");

    // Create a siumai provider from an OpenAI client
    let openai_client = llm()
        .openai()
        .api_key("your-openai-key")
        .model("gpt-4")
        .temperature(0.7)
        .build()
        .await?;

    let siumai = Siumai::new(Box::new(openai_client));

    // Basic usage
    basic_chat_example(&siumai).await?;

    // Capability detection
    capability_detection_example(&siumai).await?;

    // Streaming example
    streaming_example(&siumai).await?;

    // Provider comparison
    provider_comparison_example().await?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Basic chat example using the siumai interface
async fn basic_chat_example(siumai: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Basic Chat Example");
    println!("---------------------");

    let messages = vec![
        system!("You are a helpful assistant."),
        user!("What is the capital of France?"),
    ];

    let response = siumai.chat(messages).await?;
    println!("🤖 Response: {}", response.text().unwrap_or("No response"));
    println!();

    Ok(())
}

/// Demonstrate capability detection
async fn capability_detection_example(siumai: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Capability Detection Example");
    println!("--------------------------------");

    println!("Provider: {}", siumai.provider_name());
    println!("Provider Type: {:?}", siumai.metadata().provider_type);

    let caps = siumai.capabilities();
    println!("\nCapabilities:");
    println!("  ✅ Chat: {}", caps.chat);
    println!("  🎵 Audio: {}", caps.audio);
    println!("  👁️  Vision: {}", caps.vision);
    println!("  🛠️  Tools: {}", caps.tools);
    println!("  📊 Embedding: {}", caps.embedding);
    println!("  🌊 Streaming: {}", caps.streaming);

    // Dynamic capability checking
    if siumai.supports("audio") {
        println!("\n🎵 This provider supports audio processing!");
    } else {
        println!("\n❌ This provider does not support audio processing.");
    }

    if siumai.supports("vision") {
        println!("👁️  This provider supports vision capabilities!");
    } else {
        println!("❌ This provider does not support vision capabilities.");
    }

    println!();
    Ok(())
}

/// Demonstrate streaming capabilities
async fn streaming_example(siumai: &Siumai) -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Streaming Example");
    println!("--------------------");

    if !siumai.supports("streaming") {
        println!("❌ This provider does not support streaming.");
        return Ok(());
    }

    let messages = vec![user!("Count from 1 to 5, one number per line.")];
    let mut stream = siumai.chat_stream(messages, None).await?;

    print!("🤖 Streaming response: ");
    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::ContentDelta { delta, .. } => {
                print!("{}", delta);
            }
            ChatStreamEvent::StreamEnd { .. } => {
                println!("\n✅ Stream completed");
                break;
            }
            _ => {}
        }
    }

    println!();
    Ok(())
}

/// Compare multiple providers using the siumai interface
async fn provider_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚖️  Provider Comparison Example");
    println!("-------------------------------");

    let providers = create_test_providers().await?;

    let test_message = "What is artificial intelligence?";
    println!("Test question: {}\n", test_message);

    for (i, provider) in providers.iter().enumerate() {
        println!("Provider {}: {}", i + 1, provider.provider_name());

        let messages = vec![user!(test_message)];
        match provider.chat(messages).await {
            Ok(response) => {
                let text = response.text().unwrap_or("No response");
                println!("Response: {}", truncate_text(text, 100));
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
        println!();
    }

    Ok(())
}

/// Create multiple providers for testing
async fn create_test_providers() -> Result<Vec<Siumai>, Box<dyn std::error::Error>> {
    let mut providers = Vec::new();

    // OpenAI provider
    if let Ok(client) = llm()
        .openai()
        .api_key("your-openai-key")
        .model("gpt-4")
        .build()
        .await
    {
        providers.push(Siumai::new(Box::new(client)));
    }

    // Anthropic provider (when available)
    // if let Ok(client) = llm()
    //     .anthropic()
    //     .api_key("your-anthropic-key")
    //     .model("claude-3-sonnet")
    //     .build()
    //     .await
    // {
    //     providers.push(Siumai::new(Box::new(client)));
    // }

    // Add more providers as they become available
    // providers.push(create_gemini_provider().await?);
    // providers.push(create_xai_provider().await?);

    Ok(providers)
}

/// Utility function to truncate text for display
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
}

/// Example of task-based provider selection
#[allow(dead_code)]
async fn task_based_selection_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Task-Based Provider Selection");
    println!("--------------------------------");

    // Select provider based on task requirements
    let image_provider = select_provider_for_task("image_generation").await?;
    println!("For image generation: {}", image_provider.provider_name());

    let reasoning_provider = select_provider_for_task("reasoning").await?;
    println!(
        "For reasoning tasks: {}",
        reasoning_provider.provider_name()
    );

    let fast_provider = select_provider_for_task("fast_inference").await?;
    println!("For fast inference: {}", fast_provider.provider_name());

    Ok(())
}

/// Select the best provider for a specific task
async fn select_provider_for_task(task: &str) -> Result<Siumai, Box<dyn std::error::Error>> {
    match task {
        "image_generation" => {
            // Prefer OpenAI for image generation (DALL-E)
            let client = llm()
                .openai()
                .api_key("your-openai-key")
                .model("dall-e-3")
                .build()
                .await?;
            Ok(Siumai::new(Box::new(client)))
        }
        "reasoning" => {
            // Prefer Anthropic for complex reasoning
            let client = llm()
                .anthropic()
                .api_key("your-anthropic-key")
                .model("claude-3-opus")
                .build()
                .await?;
            Ok(Siumai::new(Box::new(client)))
        }
        "fast_inference" => {
            // Use a faster model for quick responses
            let client = llm()
                .openai()
                .api_key("your-openai-key")
                .model("gpt-3.5-turbo")
                .build()
                .await?;
            Ok(Siumai::new(Box::new(client)))
        }
        _ => {
            // Default to GPT-4
            let client = llm()
                .openai()
                .api_key("your-openai-key")
                .model("gpt-4")
                .build()
                .await?;
            Ok(Siumai::new(Box::new(client)))
        }
    }
}

/// Example of provider fallback strategy
#[allow(dead_code)]
async fn fallback_strategy_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Provider Fallback Strategy");
    println!("-----------------------------");

    let message = "Explain quantum computing in simple terms.";

    match chat_with_fallback(message).await {
        Ok(response) => {
            println!("✅ Success: {}", truncate_text(&response, 100));
        }
        Err(e) => {
            println!("❌ All providers failed: {}", e);
        }
    }

    Ok(())
}

/// Chat with automatic fallback between providers
async fn chat_with_fallback(message: &str) -> Result<String, Box<dyn std::error::Error>> {
    let provider_configs = vec![
        ("openai", "gpt-4"),
        ("anthropic", "claude-3-sonnet"),
        ("openai", "gpt-3.5-turbo"), // Fallback to cheaper model
    ];

    for (provider_name, model) in provider_configs {
        println!("Trying {} with model {}...", provider_name, model);

        match provider_name {
            "openai" => {
                if let Ok(client) = llm()
                    .openai()
                    .api_key("your-openai-key")
                    .model(model)
                    .build()
                    .await
                {
                    let siumai = Siumai::new(Box::new(client));
                    if let Ok(response) = siumai.chat(vec![user!(message)]).await {
                        return Ok(response.text().unwrap_or("").to_string());
                    }
                }
            }
            "anthropic" => {
                if let Ok(client) = llm()
                    .anthropic()
                    .api_key("your-anthropic-key")
                    .model(model)
                    .build()
                    .await
                {
                    let siumai = Siumai::new(Box::new(client));
                    if let Ok(response) = siumai.chat(vec![user!(message)]).await {
                        return Ok(response.text().unwrap_or("").to_string());
                    }
                }
            }
            _ => continue,
        }
    }

    Err("All providers failed".into())
}
