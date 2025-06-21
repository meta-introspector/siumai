//! 💬 Chat Basics - Foundation of AI Interactions
//!
//! This example demonstrates the fundamental concepts of chat-based AI:
//! - Creating and managing conversations
//! - Different message types and their purposes
//! - Handling responses and metadata
//! - Managing conversation context and history
//!
//! Before running, set your API key:
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export ANTHROPIC_API_KEY="your-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example chat_basics
//! ```

use siumai::prelude::*;
use siumai::traits::ChatCapability;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💬 Chat Basics - Foundation of AI Interactions\n");

    // Demonstrate different aspects of chat
    demonstrate_basic_chat().await;
    demonstrate_message_types().await;
    demonstrate_conversation_history().await;
    demonstrate_response_metadata().await;
    demonstrate_context_management().await;

    println!("\n✅ Chat basics completed!");
    Ok(())
}

/// Demonstrate basic chat functionality
async fn demonstrate_basic_chat() {
    println!("🔤 Basic Chat:\n");

    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        match LlmBuilder::new()
            .openai()
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .build()
            .await
        {
            Ok(client) => {
                let messages = vec![user!("What is the capital of Japan?")];
                match client.chat(messages).await {
                    Ok(response) => {
                        println!("   User: What is the capital of Japan?");
                        if let Some(text) = response.content_text() {
                            println!("   AI: {text}");
                        }
                        println!("   ✅ Basic chat successful\n");
                    }
                    Err(e) => {
                        println!("   ❌ Basic chat failed: {e}\n");
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Client creation failed: {e}\n");
            }
        }
    } else {
        println!("   ⚠️  OPENAI_API_KEY not set, skipping basic chat example\n");
    }
}

/// Demonstrate different message types
async fn demonstrate_message_types() {
    println!("📝 Message Types:\n");

    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        match LlmBuilder::new()
            .openai()
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .build()
            .await
        {
            Ok(client) => {
                // Different message types in conversation
                let messages = vec![
                    // System message - sets AI behavior
                    system!("You are a helpful math tutor. Explain concepts clearly and encourage learning."),

                    // User message - user input
                    user!("I'm struggling with algebra. Can you help me understand variables?"),

                    // Assistant message - previous AI response (for context)
                    assistant!("Of course! Variables in algebra are like containers that hold unknown values. Think of them as boxes with labels like \"x\" or \"y\" that can contain different numbers."),

                    // Follow-up user message
                    user!("Can you give me a simple example?"),
                ];

                match client.chat(messages).await {
                    Ok(response) => {
                        println!("   System: Math tutor personality set");
                        println!("   User: Asking about algebra variables");
                        println!("   Assistant: Previous explanation about variables");
                        println!("   User: Requesting an example");
                        if let Some(text) = response.content_text() {
                            println!("   AI: {text}");
                        }
                        println!("   ✅ Message types demonstration successful\n");
                    }
                    Err(e) => {
                        println!("   ❌ Message types failed: {e}\n");
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Client creation failed: {e}\n");
            }
        }
    } else {
        println!("   ⚠️  OPENAI_API_KEY not set, skipping message types example\n");
    }
}

/// Demonstrate conversation history management
async fn demonstrate_conversation_history() {
    println!("📚 Conversation History:\n");
    println!("   ⚠️  Conversation history example requires implementation\n");
}

/// Demonstrate response metadata and usage statistics
async fn demonstrate_response_metadata() {
    println!("📊 Response Metadata:\n");
    println!("   ⚠️  Metadata example requires implementation\n");
}

/// Demonstrate context management strategies
async fn demonstrate_context_management() {
    println!("🧠 Context Management:\n");
    println!("   ⚠️  Context management example requires implementation\n");
}

/*
🎯 Key Chat Concepts Summary:

Message Types:
- System: Sets AI behavior and personality
- User: Human input and questions
- Assistant: AI responses (for conversation history)

Best Practices:
1. Use system messages to define AI behavior
2. Maintain conversation history for context
3. Monitor token usage and costs
4. Handle errors gracefully
5. Manage context window size appropriately

Response Data:
- content_text(): The AI's response content
- usage: Token consumption statistics
- metadata: Response metadata

Next Steps:
- streaming_chat.rs: Real-time response streaming
- error_handling.rs: Production-ready error management
- ../04_providers/: Provider-specific features
*/
