//! Quick Start - Basic Siumai usage
//!
//! This example demonstrates the simplest way to get started with Siumai.
//! Set environment variables before running:
//! 
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export ANTHROPIC_API_KEY="your-key"
//! export GROQ_API_KEY="your-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example quick_start
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Siumai Quick Start\n");

    // Try different providers
    quick_start_with_openai().await;
    quick_start_with_anthropic().await;
    quick_start_with_ollama().await;

    println!("\n✅ Quick start completed!");
    Ok(())
}

/// Method 1: `OpenAI` (most popular)
async fn quick_start_with_openai() {
    println!("Method 1: OpenAI");

    match std::env::var("OPENAI_API_KEY") {
        Ok(api_key) if !api_key.is_empty() => {
            match LlmBuilder::new()
                .openai()
                .api_key(&api_key)
                .model("gpt-4o-mini")
                .temperature(0.7)
                .build()
                .await
            {
                Ok(client) => {
                    let messages = vec![
                        user!("Hello! Please introduce yourself in one sentence.")
                    ];

                    match client.chat(messages).await {
                        Ok(response) => {
                            if let Some(text) = response.content_text() {
                                println!("   AI Reply: {text}");
                                println!("   ✅ Success\n");
                            }
                        }
                        Err(e) => {
                            println!("   ❌ Chat failed: {e}");
                        }
                    }
                }
                Err(e) => {
                    println!("   ❌ Client creation failed: {e}");
                }
            }
        }
        _ => {
            println!("   ⚠️  OPENAI_API_KEY not set, skipping OpenAI example\n");
        }
    }
}

/// Method 2: Anthropic (Claude)
async fn quick_start_with_anthropic() {
    println!("Method 2: Anthropic (Claude)");

    match std::env::var("ANTHROPIC_API_KEY") {
        Ok(api_key) if !api_key.is_empty() => {
            match LlmBuilder::new()
                .anthropic()
                .api_key(&api_key)
                .model("claude-3-5-haiku-20241022")
                .temperature(0.7)
                .build()
                .await
            {
                Ok(client) => {
                    let messages = vec![
                        user!("What is the capital of France? Answer in one sentence.")
                    ];

                    match client.chat(messages).await {
                        Ok(response) => {
                            if let Some(text) = response.content_text() {
                                println!("   AI Reply: {text}");
                                println!("   ✅ Success\n");
                            }
                        }
                        Err(e) => {
                            println!("   ❌ Chat failed: {e}");
                        }
                    }
                }
                Err(e) => {
                    println!("   ❌ Client creation failed: {e}");
                }
            }
        }
        _ => {
            println!("   ⚠️  ANTHROPIC_API_KEY not set, skipping Anthropic example\n");
        }
    }
}

/// Method 3: Ollama (local)
async fn quick_start_with_ollama() {
    println!("Method 3: Ollama (local)");

    match LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.7)
        .build()
        .await
    {
        Ok(client) => {
            let messages = vec![
                user!("Hello! Introduce yourself in one sentence.")
            ];

            match client.chat(messages).await {
                Ok(response) => {
                    if let Some(text) = response.content_text() {
                        println!("   AI Reply: {text}");
                        println!("   ✅ Success\n");
                    }
                }
                Err(e) => {
                    println!("   ❌ Chat failed: {e}");
                    println!("   💡 Ensure Ollama is running: ollama serve");
                    println!("   💡 Install model: ollama pull llama3.2\n");
                }
            }
        }
        Err(e) => {
            println!("   ❌ Client creation failed: {e}");
            println!("   💡 Ensure Ollama is running: ollama serve\n");
        }
    }
}

/*
🎯 Key Points:

Provider creation:
- LlmBuilder::new().openai() / .anthropic() / .ollama()
- Configure with .api_key(), .model(), .temperature()
- Build with .build().await

Configuration:
- api_key: Your API key (from environment variables)
- model: The AI model to use
- temperature: Creativity level (0.0 = deterministic, 1.0 = creative)

Messages:
- user!("message") - User input
- system!("message") - System instructions
- assistant!("message") - AI responses (for conversation history)

Response:
- response.content_text() - Get the AI's text response
- response.usage - Token usage information
- response.metadata - Response metadata

Next Steps:
- basic_usage.rs: Learn about message types and conversation management
- provider_comparison.rs: Compare different AI providers
- ../02_core_features/chat_basics.rs: Deep dive into chat functionality
*/
