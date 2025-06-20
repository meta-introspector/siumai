//! Thinking Feature Demo
//!
//! This example demonstrates the corrected thinking feature implementation
//! according to Anthropic's official documentation.

use siumai::prelude::*;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("🧠 Anthropic Thinking Feature Demo\n");

    if let Ok(anthropic_key) = env::var("ANTHROPIC_API_KEY") {
        println!("📝 Demo 1: Basic Thinking with Unified Interface");
        
        // Test unified interface with thinking
        let provider = Siumai::builder()
            .anthropic()
            .api_key(anthropic_key.clone())
            .model("claude-3-5-sonnet-20241022")
            .temperature(0.7)
            .build()
            .await?;

        // Test if provider supports thinking
        println!("   Provider supports thinking: {}", provider.supports("thinking"));

        // Create a thinking-enabled client using the existing builder pattern
        let thinking_client = Provider::anthropic()
            .api_key(anthropic_key.clone())
            .model("claude-3-5-sonnet-20241022")
            .with_thinking_enabled() // Enable thinking with default budget
            .build()
            .await?;

        println!("   Thinking client created successfully");

        // Test with a complex reasoning task
        let messages = vec![
            system!("You are a helpful assistant that shows your reasoning process."),
            user!("Solve this step by step: If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is the average speed for the entire journey?"),
        ];

        println!("   Sending request with thinking enabled...");
        let response = thinking_client.chat(messages).await?;

        if let Some(text) = response.content_text() {
            println!("   Response: {}\n", text);
        }

        // Check if thinking content was captured
        if let Some(thinking_value) = response.metadata.get("thinking") {
            if let Some(thinking_text) = thinking_value.as_str() {
                println!("🤔 Claude's Thinking Process:");
                println!("   {}\n", thinking_text);
            }
        } else {
            println!("   Note: No thinking content captured in metadata\n");
        }

        println!("📝 Demo 2: Thinking with Custom Budget");
        
        // Test with custom thinking budget
        let custom_thinking_client = Provider::anthropic()
            .api_key(anthropic_key.clone())
            .model("claude-3-5-sonnet-20241022")
            .with_thinking_mode(Some(5000)) // Custom budget of 5000 tokens
            .build()
            .await?;

        let complex_messages = vec![
            system!("You are a logic puzzle expert. Show your detailed reasoning."),
            user!("Three friends - Alice, Bob, and Charlie - each have a different pet (cat, dog, fish) and live in different colored houses (red, blue, green). Given these clues, determine who has which pet and lives in which house:\n1. Alice doesn't live in the red house\n2. The person with the cat lives in the blue house\n3. Bob doesn't have the fish\n4. Charlie lives in the green house\n5. The person in the red house has the dog"),
        ];

        println!("   Sending complex logic puzzle...");
        let complex_response = custom_thinking_client.chat(complex_messages).await?;

        if let Some(text) = complex_response.content_text() {
            println!("   Response: {}\n", text);
        }

        // Check thinking content
        if let Some(thinking_value) = complex_response.metadata.get("thinking") {
            if let Some(thinking_text) = thinking_value.as_str() {
                println!("🤔 Claude's Detailed Reasoning:");
                println!("   {}\n", thinking_text);
            }
        }

        println!("📝 Demo 3: Thinking Configuration Validation");
        
        // Test thinking configuration validation
        println!("   Testing thinking configuration...");
        
        // This should work (budget >= 1024)
        let valid_client = Provider::anthropic()
            .api_key(anthropic_key.clone())
            .model("claude-3-5-sonnet-20241022")
            .with_thinking_mode(Some(2048))
            .build()
            .await;

        match valid_client {
            Ok(_) => println!("   ✅ Valid thinking budget (2048) accepted"),
            Err(e) => println!("   ❌ Unexpected error with valid budget: {}", e),
        }

        // Test with minimum budget
        let min_client = Provider::anthropic()
            .api_key(anthropic_key)
            .model("claude-3-5-sonnet-20241022")
            .with_thinking_mode(Some(1024)) // Minimum budget
            .build()
            .await;

        match min_client {
            Ok(_) => println!("   ✅ Minimum thinking budget (1024) accepted"),
            Err(e) => println!("   ❌ Unexpected error with minimum budget: {}", e),
        }

    } else {
        println!("❌ ANTHROPIC_API_KEY environment variable not set");
        println!("   Please set your Anthropic API key to test thinking functionality");
    }

    println!("\n✅ Thinking Feature Demo completed!");
    println!("\n💡 Key Features Demonstrated:");
    println!("   • Unified interface with thinking support");
    println!("   • Default thinking budget (10k tokens)");
    println!("   • Custom thinking budget configuration");
    println!("   • Thinking content extraction from metadata");
    println!("   • Configuration validation");
    println!("   • Complex reasoning tasks");

    Ok(())
}
