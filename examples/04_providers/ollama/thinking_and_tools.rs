//! Ollama Thinking and Tools Example
//!
//! This example demonstrates the new unified architecture for Ollama,
//! including thinking functionality and streaming tool calls.

use futures::StreamExt;
use siumai::prelude::*;
use siumai::stream::ChatStreamEvent;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦙 Ollama Thinking and Tools Example");
    println!("=====================================\n");

    // Test 1: Thinking functionality
    println!("📋 Test 1: Thinking Functionality");
    test_thinking_functionality().await?;

    println!("\n{}\n", "=".repeat(50));

    // Test 2: Streaming with tools
    println!("📋 Test 2: Streaming Tool Calls");
    test_streaming_tools().await?;

    println!("\n{}\n", "=".repeat(50));

    // Test 3: Independent ChatCapability usage
    println!("📋 Test 3: Independent ChatCapability");
    test_independent_capability().await?;

    Ok(())
}

/// Test thinking functionality
async fn test_thinking_functionality() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating Ollama client with thinking enabled...");

    let client = LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model("deepseek-r1:8b") // Use a thinking model
        .reasoning(true) // Enable reasoning
        .temperature(0.7)
        .build()
        .await?;

    let messages = vec![user!(
        "Solve this step by step: What is 15% of 240? Show your reasoning."
    )];

    println!("   User: Solve this step by step: What is 15% of 240?");
    println!("   🧠 DeepSeek-R1 (with thinking): ");

    match client.chat_stream(messages, None).await {
        Ok(mut stream) => {
            let mut thinking_content = String::new();
            let mut response_content = String::new();
            let mut in_thinking = false;

            while let Some(event) = stream.next().await {
                match event {
                    Ok(ChatStreamEvent::ThinkingDelta { delta }) => {
                        if !in_thinking {
                            println!("\n   🧠 Thinking:");
                            in_thinking = true;
                        }
                        thinking_content.push_str(&delta);
                        print!("{}", delta);
                        io::stdout().flush().unwrap();
                    }
                    Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                        if in_thinking {
                            println!("\n\n   💬 Response:");
                            in_thinking = false;
                        }
                        response_content.push_str(&delta);
                        print!("{}", delta);
                        io::stdout().flush().unwrap();
                    }
                    Ok(ChatStreamEvent::StreamEnd { .. }) => {
                        println!("\n   ✅ Thinking completed");
                        break;
                    }
                    Err(e) => {
                        println!("\n   ❌ Stream error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }

            if !thinking_content.is_empty() {
                println!(
                    "   📊 Thinking content length: {} characters",
                    thinking_content.len()
                );
            }
            if !response_content.is_empty() {
                println!(
                    "   📊 Response content length: {} characters",
                    response_content.len()
                );
            }
        }
        Err(e) => {
            println!("   ❌ Thinking test failed: {}", e);
        }
    }

    Ok(())
}

/// Test streaming tool calls
async fn test_streaming_tools() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating Ollama client with tool support...");

    let client = LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model("qwen3") // Use a tool-capable model
        .temperature(0.7)
        .build()
        .await?;

    // Define a simple math tool
    let math_tool = Tool {
        r#type: "function".to_string(),
        function: ToolFunction {
            name: "calculate".to_string(),
            description: "Perform basic mathematical calculations".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }),
        },
    };

    let messages = vec![user!(
        "What is 25 * 4 + 10? Use the calculate tool to solve this."
    )];

    println!("   User: What is 25 * 4 + 10? Use the calculate tool.");
    println!("   🔧 Qwen3 (with tools): ");

    match client.chat_stream(messages, Some(vec![math_tool])).await {
        Ok(mut stream) => {
            let mut tool_calls = Vec::new();

            while let Some(event) = stream.next().await {
                match event {
                    Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                        print!("{}", delta);
                        io::stdout().flush().unwrap();
                    }
                    Ok(ChatStreamEvent::ToolCallDelta {
                        id,
                        function_name,
                        arguments_delta,
                        ..
                    }) => {
                        if let Some(name) = function_name {
                            println!("\n   🔧 Tool call: {}", name);
                            tool_calls.push((id.clone(), name, String::new()));
                        }
                        if let Some(args) = arguments_delta {
                            if let Some((_, _, accumulated_args)) = tool_calls.last_mut() {
                                accumulated_args.push_str(&args);
                            }
                            print!("   📝 Args: {}", args);
                            io::stdout().flush().unwrap();
                        }
                    }
                    Ok(ChatStreamEvent::StreamEnd { .. }) => {
                        println!("\n   ✅ Tool call streaming completed");
                        break;
                    }
                    Err(e) => {
                        println!("\n   ❌ Stream error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }

            if !tool_calls.is_empty() {
                println!("   📊 Tool calls received: {}", tool_calls.len());
                for (id, name, args) in tool_calls {
                    println!("     - {} ({}): {}", name, id, args);
                }
            }
        }
        Err(e) => {
            println!("   ❌ Tool streaming test failed: {}", e);
        }
    }

    Ok(())
}

/// Test independent ChatCapability usage
async fn test_independent_capability() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing independent ChatCapability usage...");

    // This tests the fix for the architecture inconsistency
    // Previously, OllamaChatCapability would return an error when used directly
    use siumai::providers::ollama::chat::OllamaChatCapability;
    use siumai::providers::ollama::config::OllamaParams;
    use siumai::types::HttpConfig;

    let capability = OllamaChatCapability::new(
        "http://localhost:11434".to_string(),
        reqwest::Client::new(),
        HttpConfig::default(),
        OllamaParams::default(),
    );

    let messages = vec![user!(
        "Hello! This is a test of independent capability usage."
    )];

    println!("   User: Hello! This is a test of independent capability usage.");
    println!("   🦙 Direct capability: ");

    match capability.chat_with_tools(messages, None).await {
        Ok(response) => {
            println!("{}", response.content_text().unwrap_or_default());
            println!("   ✅ Independent capability test passed");
        }
        Err(e) => {
            println!("   ❌ Independent capability test failed: {}", e);
        }
    }

    Ok(())
}
