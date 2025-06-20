//! Simplified Gemini Thinking Example
//!
//! This example demonstrates the simplified thinking API that lets the
//! Google API handle model-specific limitations and validation.

use siumai::llm;
use siumai::stream::ChatStreamEvent;
use siumai::types::ChatMessage;
use siumai::traits::ChatCapability;
use std::env;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("GEMINI_API_KEY")
        .expect("Please set GEMINI_API_KEY environment variable");

    println!("🧠 Simplified Gemini Thinking API Examples");
    println!("{}", "=".repeat(60));

    // Example 1: Different thinking configurations
    thinking_configurations_examples(&api_key).await?;

    // Example 2: Error handling (let API handle model limitations)
    api_error_handling_examples(&api_key).await?;

    // Example 3: Dynamic thinking vs fixed budget
    thinking_strategies_examples(&api_key).await?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Demonstrate different thinking configurations
async fn thinking_configurations_examples(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📋 Example 1: Different Thinking Configurations");
    println!("{}", "-".repeat(50));

    // Try to disable thinking
    println!("\n🔹 Attempting to disable thinking:");
    let client = llm()
        .gemini()
        .api_key(api_key)
        .model("gemini-2.5-flash")
        .disable_thinking() // Let API decide if this is supported
        .build()
        .await?;

    let simple_query = vec![
        ChatMessage::user("What is the capital of France?".to_string()).build()
    ];

    let response = client.chat(simple_query).await?;
    if let Some(text) = response.content_text() {
        println!("Response: {}", text);
    }

    // Use dynamic thinking
    println!("\n🔹 Using dynamic thinking:");
    let client = llm()
        .gemini()
        .api_key(api_key)
        .model("gemini-2.5-pro")
        .thinking() // Dynamic thinking
        .thought_summaries(true)
        .build()
        .await?;

    let complex_query = vec![
        ChatMessage::user("Explain the mathematical proof of why the square root of 2 is irrational.".to_string()).build()
    ];

    let response = client.chat(complex_query).await?;
    if let Some(text) = response.content_text() {
        println!("Response: {}", text);
    }

    Ok(())
}

/// Demonstrate API error handling (let Google API handle model limitations)
async fn api_error_handling_examples(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n❌ Example 2: API Error Handling");
    println!("{}", "-".repeat(50));

    // Try a configuration that might not be supported
    println!("\n🔹 Trying potentially unsupported configuration:");
    let client = llm()
        .gemini()
        .api_key(api_key)
        .model("gemini-2.5-pro")
        .thinking_budget(0) // Might not be supported
        .build()
        .await?; // Build succeeds, validation happens at API level

    let messages = vec![
        ChatMessage::user("Test message".to_string()).build()
    ];

    // The error (if any) will come from the Google API
    match client.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.content_text() {
                println!("✅ Configuration was accepted: {}", text);
            }
        }
        Err(e) => {
            println!("ℹ️  API returned error (expected for unsupported config): {}", e);
            println!("   This is how you handle model-specific limitations.");
        }
    }

    // Use a safe configuration
    println!("\n🔹 Using safe dynamic thinking configuration:");
    let safe_client = llm()
        .gemini()
        .api_key(api_key)
        .model("gemini-2.5-pro")
        .thinking() // Dynamic thinking is usually safe
        .build()
        .await?;

    let response = safe_client.chat(
        vec![ChatMessage::user("What is 2+2?".to_string()).build()]
    ).await?;

    if let Some(text) = response.content_text() {
        println!("✅ Safe configuration worked: {}", text);
    }

    Ok(())
}

/// Demonstrate different thinking strategies
async fn thinking_strategies_examples(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Example 3: Different Thinking Strategies");
    println!("{}", "-".repeat(50));

    let problem = "Calculate the compound interest on $1000 invested at 5% annual rate for 10 years, compounded quarterly.";

    // Strategy 1: Dynamic thinking (model decides)
    println!("\n🔹 Strategy 1: Dynamic Thinking (Model Decides)");
    let dynamic_client = llm()
        .gemini()
        .api_key(api_key)
        .model("gemini-2.5-flash")
        .thinking() // Dynamic thinking
        .build()
        .await?;

    let response = dynamic_client.chat(
        vec![ChatMessage::user(problem.to_string()).build()]
    ).await?;

    if let Some(text) = response.content_text() {
        println!("Dynamic thinking response: {}", text);
    }

    // Strategy 2: Fixed budget thinking
    println!("\n🔹 Strategy 2: Fixed Budget Thinking (1024 tokens)");
    let fixed_client = llm()
        .gemini()
        .api_key(api_key)
        .model("gemini-2.5-flash")
        .thinking_budget(1024)
        .thought_summaries(true)
        .build()
        .await?;

    let response = fixed_client.chat(
        vec![ChatMessage::user(problem.to_string()).build()]
    ).await?;

    if let Some(text) = response.content_text() {
        println!("Fixed budget response: {}", text);
    }

    // Strategy 3: No thinking (for comparison)
    println!("\n🔹 Strategy 3: No Thinking (Flash only)");
    let no_thinking_client = llm()
        .gemini()
        .api_key(api_key)
        .model("gemini-2.5-flash")
        .thinking_budget(0) // Disable thinking
        .build()
        .await?;

    let response = no_thinking_client.chat(
        vec![ChatMessage::user(problem.to_string()).build()]
    ).await?;

    if let Some(text) = response.content_text() {
        println!("No thinking response: {}", text);
    }

    Ok(())
}
