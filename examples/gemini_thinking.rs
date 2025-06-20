//! Gemini Thinking Example
//!
//! This example demonstrates how to handle Google Gemini's thinking output
//! with the Siumai LLM library. Thinking is automatically included when the
//! model produces it - no special configuration needed.
//!
//! The example shows the improved API that provides clean, type-safe access
//! to thinking content without redundant metadata lookups.

use siumai::providers::gemini::GeminiBuilder;
use siumai::stream::ChatStreamEvent;
use siumai::types::ChatMessage;
use siumai::traits::ChatCapability;
use std::env;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("GEMINI_API_KEY")
        .expect("GEMINI_API_KEY environment variable must be set");

    println!("🧠 Gemini Thinking Output Demo\n");

    // Example 1: Basic chat with thinking detection
    basic_thinking_example(&api_key).await?;

    // Example 2: Streaming with thinking content
    streaming_thinking_example(&api_key).await?;

    // Example 3: Complex reasoning task
    complex_reasoning_example(&api_key).await?;

    Ok(())
}

/// Example 1: Basic chat with thinking detection
async fn basic_thinking_example(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Example 1: Basic Chat with Thinking Detection");
    println!("{}", "=".repeat(50));

    let client = GeminiBuilder::new()
        .api_key(api_key.to_string())
        .model("gemini-2.5-pro".to_string())
        .build()?;

    let messages = vec![
        ChatMessage::user("What is the sum of the first 50 prime numbers? Show your reasoning.".to_string()).build()
    ];

    let response = client.chat_with_tools(messages, None).await?;

    println!("🤖 Response: {}", response.content.text().unwrap_or("No content"));

    // Check if thinking content was included using the new convenience methods
    if response.has_thinking() {
        println!("\n🧠 Thinking process detected:");
        println!("{}", "-".repeat(40));
        println!("{}", response.get_thinking().unwrap());
        println!("{}", "-".repeat(40));
    }

    if let Some(usage) = &response.usage {
        println!("\n📊 Usage:");
        println!("  - Prompt tokens: {}", usage.prompt_tokens);
        println!("  - Completion tokens: {}", usage.completion_tokens);
        if let Some(reasoning_tokens) = usage.reasoning_tokens {
            println!("  - Thinking tokens: {}", reasoning_tokens);
        }
        println!("  - Total tokens: {}", usage.total_tokens);
    }

    println!("\n");
    Ok(())
}

/// Example 2: Streaming with thinking content
async fn streaming_thinking_example(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Example 2: Streaming with Thinking Content");
    println!("{}", "=".repeat(50));

    let client = GeminiBuilder::new()
        .api_key(api_key.to_string())
        .model("gemini-2.5-pro".to_string())
        .build()?;

    let messages = vec![
        ChatMessage::user("Calculate the compound interest on $1000 invested at 5% annual rate for 10 years, compounded quarterly. Show your step-by-step calculation.".to_string()).build()
    ];

    println!("🤖 Streaming response:\n");

    let mut stream = client.chat_stream(messages, None).await?;
    let mut thinking_content = String::new();
    let mut response_content = String::new();

    use futures::StreamExt;
    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::ThinkingDelta { delta } => {
                thinking_content.push_str(&delta);
                // Print thinking in gray color
                print!("\x1B[90m{}\x1B[0m", delta);
                io::stdout().flush()?;
            }
            ChatStreamEvent::ContentDelta { delta, .. } => {
                if !thinking_content.is_empty() && response_content.is_empty() {
                    println!("\n\n🎯 Final Answer:");
                    println!("{}", "-".repeat(40));
                }
                response_content.push_str(&delta);
                print!("{}", delta);
                io::stdout().flush()?;
            }
            ChatStreamEvent::StreamEnd { response } => {
                println!("\n{}", "-".repeat(40));
                println!("✅ Streaming completed!");

                if let Some(usage) = &response.usage {
                    if let Some(reasoning_tokens) = usage.reasoning_tokens {
                        println!("🧠 Thinking tokens: {}", reasoning_tokens);
                    }
                    println!("📊 Total tokens: {}", usage.total_tokens);
                }
                break;
            }
            _ => {} // Handle other events
        }
    }

    println!("\n");
    Ok(())
}

/// Example 3: Complex reasoning task
async fn complex_reasoning_example(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧩 Example 3: Complex Reasoning Task");
    println!("{}", "=".repeat(50));

    let client = GeminiBuilder::new()
        .api_key(api_key.to_string())
        .model("gemini-2.5-pro".to_string())
        .build()?;

    let messages = vec![
        ChatMessage::user(r#"
You are given a 3x3 grid where each cell can be either empty (0) or filled (1).
The grid starts as:
```
0 1 0
1 0 1
0 1 0
```

Rules:
1. In each turn, you can flip any cell (0→1 or 1→0)
2. When you flip a cell, all adjacent cells (up, down, left, right, not diagonal) also flip
3. Goal: Make all cells 0

Find the minimum number of moves and which cells to flip.
"#.to_string()).build()
    ];

    let response = client.chat_with_tools(messages, None).await?;

    println!("🤖 Solution: {}", response.content.text().unwrap_or("No content"));

    // Check for thinking content using convenience methods
    if response.has_thinking() {
        println!("\n🧠 Model's thinking process:");
        println!("{}", "-".repeat(40));
        println!("{}", response.get_thinking().unwrap());
        println!("{}", "-".repeat(40));
    }

    if let Some(usage) = &response.usage {
        println!("\n📊 Token Usage:");
        if let Some(reasoning_tokens) = usage.reasoning_tokens {
            println!("  - Thinking: {} tokens", reasoning_tokens);
        }
        println!("  - Response: {} tokens", usage.completion_tokens);
        println!("  - Total: {} tokens", usage.total_tokens);
    }

    println!("\n");
    Ok(())
}
