//! 🚀 Basic Usage Examples for Siumai LLM Library
//!
//! This example demonstrates the core functionality of the unified LLM interface.
//! Perfect for getting started with Siumai and understanding the basic concepts.
//!
//! Before running, set your API key:
//! ```bash
//! export OPENAI_API_KEY="your-openai-key"
//! export GROQ_API_KEY="your-groq-key"
//! ```
//!
//! Usage:
//! ```bash
//! cargo run --example basic_usage
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Siumai LLM Library - Basic Usage Examples");
    println!("===========================================\n");

    // Get API key
    let api_key = std::env::var("GROQ_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .unwrap_or_else(|_| {
            println!("⚠️  No API key found, using demo key");
            "demo-key".to_string()
        });

    // Example 1: Simple Chat
    println!("💬 Example 1: Simple Chat");
    simple_chat_example(&api_key).await?;
    println!();

    // Example 2: Message Types
    println!("📝 Example 2: Different Message Types");
    message_types_example();
    println!();

    // Example 3: Provider Configuration
    println!("⚙️ Example 3: Provider Configuration");
    provider_configuration_example(&api_key).await?;
    println!();

    // Example 4: Parameter Customization
    println!("🔧 Example 4: Parameter Customization");
    parameter_customization_example(&api_key).await?;
    println!();

    // Example 5: Error Handling
    println!("⚠️ Example 5: Error Handling");
    error_handling_example(&api_key).await?;
    println!();

    println!("✅ All basic usage examples completed successfully!");
    Ok(())
}

/// Example 1: Simple Chat
/// Demonstrates the most basic way to chat with an AI
async fn simple_chat_example(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating a simple AI client...");
    
    // Create AI client with minimal configuration
    let ai = Siumai::builder()
        .openai()  // Using OpenAI as it's available in Provider
        .api_key(api_key)
        .model("gpt-4o-mini")
        .build()
        .await?;

    // Simple question
    let messages = vec![
        ChatMessage::user("What is the capital of France?").build()
    ];

    println!("   Asking: What is the capital of France?");
    
    match ai.chat(messages).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("   🤖 AI: {}", text.trim());
            }
            
            // Show usage information if available
            if let Some(usage) = &response.usage {
                println!("   📊 Tokens used: {}", usage.total_tokens);
            }
        }
        Err(e) => println!("   ❌ Error: {e}"),
    }

    Ok(())
}

/// Example 2: Different Message Types
/// Shows how to create different types of messages
fn message_types_example() {
    println!("   Creating different message types...");

    // System message - sets the AI's behavior
    let system_msg = ChatMessage::system(
        "You are a helpful assistant that explains things simply and clearly."
    ).build();
    println!("   📋 System: {:?}", system_msg.role);

    // User message - your input
    let user_msg = ChatMessage::user("Explain quantum computing in simple terms").build();
    println!("   👤 User: {:?}", user_msg.role);

    // Assistant message - AI's previous response (for conversation context)
    let assistant_msg = ChatMessage::assistant(
        "Quantum computing uses quantum mechanics to process information..."
    ).build();
    println!("   🤖 Assistant: {:?}", assistant_msg.role);

    // Show how to build a conversation
    let conversation = vec![system_msg, user_msg, assistant_msg];
    println!("   💬 Conversation has {} messages", conversation.len());
}

/// Example 3: Provider Configuration
/// Demonstrates how to configure different providers
async fn provider_configuration_example(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing different provider configurations...");

    // Configuration 1: OpenAI (fast inference)
    println!("   🚀 OpenAI Configuration:");
    let openai_ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .temperature(0.3)
        .max_tokens(100)
        .build()
        .await?;

    let test_message = vec![ChatMessage::user("Say hello in a creative way").build()];

    match openai_ai.chat(test_message).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("     Response: {}", text.trim());
            }
        }
        Err(e) => println!("     ❌ Error: {e}"),
    }

    // Configuration 2: Anthropic (if key available)
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        println!("   🧠 Anthropic Configuration:");
        let anthropic_ai = Siumai::builder()
            .anthropic()
            .api_key(&std::env::var("ANTHROPIC_API_KEY")?)
            .model("claude-3-5-sonnet-20241022")
            .temperature(0.3)
            .max_tokens(100)
            .build()
            .await?;

        let test_message = vec![ChatMessage::user("Say hello in a creative way").build()];
        match anthropic_ai.chat(test_message).await {
            Ok(response) => {
                if let Some(text) = response.text() {
                    println!("     Response: {}", text.trim());
                }
            }
            Err(e) => println!("     ❌ Error: {e}"),
        }
    } else {
        println!("   🧠 Anthropic: Skipped (no API key)");
    }

    Ok(())
}

/// Example 4: Parameter Customization
/// Shows how to customize AI behavior with parameters
async fn parameter_customization_example(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing different parameter configurations...");

    let prompt = "Write a short poem about programming";

    // Configuration 1: Creative (high temperature)
    println!("   🎨 Creative Configuration (temperature: 0.9):");
    let creative_ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .temperature(0.9)
        .max_tokens(150)
        .build()
        .await?;

    match creative_ai.chat(vec![ChatMessage::user(prompt).build()]).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("     {}", text.trim());
            }
        }
        Err(e) => println!("     ❌ Error: {e}"),
    }

    println!();

    // Configuration 2: Focused (low temperature)
    println!("   🎯 Focused Configuration (temperature: 0.1):");
    let focused_ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .temperature(0.1)
        .max_tokens(150)
        .build()
        .await?;

    match focused_ai.chat(vec![ChatMessage::user(prompt).build()]).await {
        Ok(response) => {
            if let Some(text) = response.text() {
                println!("     {}", text.trim());
            }
        }
        Err(e) => println!("     ❌ Error: {e}"),
    }

    Ok(())
}

/// Example 5: Error Handling
/// Demonstrates proper error handling patterns
async fn error_handling_example(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing error handling scenarios...");

    // Test 1: Invalid model
    println!("   🔍 Testing invalid model:");
    let result = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("invalid-model-name")
        .build()
        .await;

    match result {
        Ok(_) => println!("     ✅ Model accepted (might fail at runtime)"),
        Err(e) => println!("     ❌ Expected error: {e}"),
    }

    // Test 2: Empty message handling
    println!("   📝 Testing empty message handling:");
    let ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .build()
        .await?;

    let empty_messages: Vec<ChatMessage> = vec![];

    match ai.chat(empty_messages).await {
        Ok(response) => {
            println!("     ✅ Handled empty messages: {}",
                    response.text().unwrap_or("No response".to_string()));
        }
        Err(e) => println!("     ❌ Expected error with empty messages: {e}"),
    }

    // Test 3: Timeout handling (simulated)
    println!("   ⏱️ Error handling best practices:");
    println!("     • Always check for API key availability");
    println!("     • Handle network timeouts gracefully");
    println!("     • Implement retry logic for transient errors");
    println!("     • Validate input before sending requests");
    println!("     • Log errors for debugging");

    Ok(())
}

/// 🎯 Key Basic Usage Concepts Summary:
///
/// Core Components:
/// - `SiumaiBuilder`: Creates AI clients with configuration
/// - `ChatMessage`: Represents messages in conversations
/// - Provider: Specifies which AI service to use
/// - Parameters: Control AI behavior (temperature, tokens, etc.)
///
/// Message Types:
/// - System: Sets AI behavior and context
/// - User: Your input/questions
/// - Assistant: AI responses (for conversation history)
///
/// Essential Parameters:
/// - model: Which AI model to use
/// - temperature: Creativity level (0.0-1.0)
/// - `max_tokens`: Maximum response length
/// - `api_key`: Authentication for the service
///
/// Error Handling:
/// - Always use Result types
/// - Check for API key availability
/// - Handle network and service errors
/// - Validate inputs before requests
///
/// Best Practices:
/// - Start with simple configurations
/// - Test with different providers
/// - Experiment with parameters
/// - Implement proper error handling
/// - Monitor token usage
///
/// Next Steps:
/// - Explore streaming responses
/// - Try different providers
/// - Learn about advanced features
/// - Build real applications
const fn _documentation() {}
