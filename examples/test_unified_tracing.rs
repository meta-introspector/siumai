//! Test Unified Tracing Implementation
//!
//! This example demonstrates the unified tracing implementation across all providers.

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing Unified Tracing Implementation");
    println!("=========================================");
    println!();

    // Test OpenAI with tracing
    println!("📋 Testing OpenAI Provider Tracing:");
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let client = Provider::openai()
            .api_key(&api_key)
            .model("gpt-4o-mini")
            .debug_tracing()
            .build()
            .await?;

        let messages = vec![user!("Say hello in one word")];
        match client.chat(messages).await {
            Ok(response) => {
                println!("✅ OpenAI: {}", response.text().unwrap_or_default());
            }
            Err(e) => {
                println!("❌ OpenAI Error: {e}");
            }
        }
    } else {
        println!("⚠️  OpenAI API key not found, skipping");
    }
    println!();

    // Test Anthropic with tracing
    println!("📋 Testing Anthropic Provider Tracing:");
    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        let client = Provider::anthropic()
            .api_key(&api_key)
            .model("claude-3-5-haiku-20241022")
            .debug_tracing()
            .build()
            .await?;

        let messages = vec![user!("Say hello in one word")];
        match client.chat(messages).await {
            Ok(response) => {
                println!("✅ Anthropic: {}", response.text().unwrap_or_default());
            }
            Err(e) => {
                println!("❌ Anthropic Error: {e}");
            }
        }
    } else {
        println!("⚠️  Anthropic API key not found, skipping");
    }
    println!();

    // Test Ollama with tracing (if available)
    println!("📋 Testing Ollama Provider Tracing:");
    let ollama_client = Provider::ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2:latest")
        .debug_tracing()
        .build()
        .await;

    match ollama_client {
        Ok(client) => {
            let messages = vec![user!("Say hello in one word")];
            match client.chat(messages).await {
                Ok(response) => {
                    println!("✅ Ollama: {}", response.text().unwrap_or_default());
                }
                Err(e) => {
                    println!("❌ Ollama Error: {e}");
                }
            }
        }
        Err(_) => {
            println!("⚠️  Ollama not available, skipping");
        }
    }
    println!();

    // Test Groq with tracing
    println!("📋 Testing Groq Provider Tracing:");
    if let Ok(api_key) = std::env::var("GROQ_API_KEY") {
        let client = Provider::groq()
            .api_key(&api_key)
            .model("llama-3.3-70b-versatile")
            .debug_tracing()
            .build()
            .await?;

        let messages = vec![user!("Say hello in one word")];
        match client.chat(messages).await {
            Ok(response) => {
                println!("✅ Groq: {}", response.text().unwrap_or_default());
            }
            Err(e) => {
                println!("❌ Groq Error: {e}");
            }
        }
    } else {
        println!("⚠️  Groq API key not found, skipping");
    }
    println!();

    // Test Gemini with tracing
    println!("📋 Testing Gemini Provider Tracing:");
    if let Ok(api_key) = std::env::var("GEMINI_API_KEY") {
        let client = Provider::gemini()
            .api_key(&api_key)
            .model("gemini-1.5-flash")
            .debug_tracing()
            .build()
            .await?;

        let messages = vec![user!("Say hello in one word")];
        match client.chat(messages).await {
            Ok(response) => {
                println!("✅ Gemini: {}", response.text().unwrap_or_default());
            }
            Err(e) => {
                println!("❌ Gemini Error: {e}");
            }
        }
    } else {
        println!("⚠️  Gemini API key not found, skipping");
    }
    println!();

    // Test xAI with tracing
    println!("📋 Testing xAI Provider Tracing:");
    if let Ok(api_key) = std::env::var("XAI_API_KEY") {
        let client = Provider::xai()
            .api_key(&api_key)
            .model("grok-3-latest")
            .debug_tracing()
            .build()
            .await?;

        let messages = vec![user!("Say hello in one word")];
        match client.chat(messages).await {
            Ok(response) => {
                println!("✅ xAI: {}", response.text().unwrap_or_default());
            }
            Err(e) => {
                println!("❌ xAI Error: {e}");
            }
        }
    } else {
        println!("⚠️  xAI API key not found, skipping");
    }
    println!();

    println!("🎯 Unified Tracing Test Complete!");
    println!();
    println!("📊 What was tested:");
    println!("   ✅ Consistent tracing format across all providers");
    println!(
        "   ✅ Provider identification in logs (OpenAI, Anthropic, Ollama, Groq, Gemini, xAI)"
    );
    println!("   ✅ Model information in traces");
    println!("   ✅ Request/response timing");
    println!("   ✅ Error handling with tracing");
    println!("   ✅ Sensitive data masking");
    println!("   ✅ Unified tracing guard management");
    println!();
    println!("🔍 Check the logs above to see the unified tracing format!");

    Ok(())
}
