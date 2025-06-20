//! Unified Interface Demo
//!
//! This example demonstrates the new unified interface for siumai,
//! similar to llm_dart's ai() function. It shows how to use different
//! providers through a single, consistent API.

use siumai::prelude::*;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("🚀 Siumai Unified Interface Demo\n");

    // Demo 1: OpenAI using provider name (dynamic dispatch)
    if let Ok(openai_key) = env::var("OPENAI_API_KEY") {
        println!("📝 Demo 1: OpenAI using provider name");
        
        let provider = Siumai::builder()
            .provider_name("openai")
            .api_key(openai_key)
            .model("gpt-4o-mini")
            .temperature(0.7)
            .max_tokens(100)
            .build()
            .await?;

        let response = provider.ask("What is the capital of France?".to_string()).await?;
        println!("   Response: {}\n", response);
    }

    // Demo 2: OpenAI using convenience method
    if let Ok(openai_key) = env::var("OPENAI_API_KEY") {
        println!("📝 Demo 2: OpenAI using convenience method");
        
        let provider = Siumai::builder()
            .openai()
            .api_key(openai_key)
            .model("gpt-4o-mini")
            .temperature(0.7)
            .max_tokens(100)
            .build()
            .await?;

        let response = provider.ask("What is 2 + 2?".to_string()).await?;
        println!("   Response: {}\n", response);
    }

    // Demo 3: Anthropic using convenience method
    if let Ok(anthropic_key) = env::var("ANTHROPIC_API_KEY") {
        println!("📝 Demo 3: Anthropic using convenience method");
        
        let provider = Siumai::builder()
            .anthropic()
            .api_key(anthropic_key)
            .model("claude-3-5-sonnet-20241022")
            .temperature(0.7)
            .max_tokens(100)
            .build()
            .await?;

        let response = provider.ask("Explain quantum computing in one sentence.".to_string()).await?;
        println!("   Response: {}\n", response);
    }

    // Demo 4: DeepSeek using convenience method
    if let Ok(deepseek_key) = env::var("DEEPSEEK_API_KEY") {
        println!("📝 Demo 4: DeepSeek using convenience method");
        
        let provider = Siumai::builder()
            .deepseek()
            .api_key(deepseek_key)
            .model("deepseek-chat")
            .temperature(0.1)
            .max_tokens(100)
            .build()
            .await?;

        let response = provider.ask("What is machine learning?".to_string()).await?;
        println!("   Response: {}\n", response);
    }

    // Demo 5: OpenRouter using convenience method
    if let Ok(openrouter_key) = env::var("OPENROUTER_API_KEY") {
        println!("📝 Demo 5: OpenRouter using convenience method");
        
        let provider = Siumai::builder()
            .openrouter()
            .api_key(openrouter_key)
            .model("openai/gpt-3.5-turbo")
            .temperature(0.7)
            .max_tokens(100)
            .build()
            .await?;

        let response = provider.ask("What is the meaning of life?".to_string()).await?;
        println!("   Response: {}\n", response);
    }

    // Demo 6: Groq using convenience method
    if let Ok(groq_key) = env::var("GROQ_API_KEY") {
        println!("📝 Demo 6: Groq using convenience method");
        
        let provider = Siumai::builder()
            .groq()
            .api_key(groq_key)
            .model("llama-3.3-70b-versatile")
            .temperature(0.7)
            .max_tokens(100)
            .build()
            .await?;

        let response = provider.ask("What is artificial intelligence?".to_string()).await?;
        println!("   Response: {}\n", response);
    }

    // Demo 7: Provider capabilities check
    if let Ok(openai_key) = env::var("OPENAI_API_KEY") {
        println!("📝 Demo 7: Provider capabilities check");
        
        let provider = Siumai::builder()
            .openai()
            .api_key(openai_key)
            .model("gpt-4o-mini")
            .build()
            .await?;

        println!("   Provider: {}", provider.provider_name());
        println!("   Supports chat: {}", provider.supports("chat"));
        println!("   Supports audio: {}", provider.supports("audio"));
        println!("   Supports vision: {}", provider.supports("vision"));
        println!("   Supports embedding: {}", provider.supports("embedding"));
        println!("   Supports streaming: {}", provider.supports("streaming"));
        println!("   Supported models: {:?}", provider.supported_models().iter().take(3).collect::<Vec<_>>());
    }

    // Demo 8: Chat with tools (if supported)
    if let Ok(openai_key) = env::var("OPENAI_API_KEY") {
        println!("\n📝 Demo 8: Chat with tools");
        
        let provider = Siumai::builder()
            .openai()
            .api_key(openai_key)
            .model("gpt-4o-mini")
            .build()
            .await?;

        let messages = vec![
            system!("You are a helpful assistant."),
            user!("Hello! How are you today?"),
        ];

        let response = provider.chat(messages).await?;
        if let Some(text) = response.content_text() {
            println!("   Response: {}\n", text);
        }
    }

    println!("✅ Unified Interface Demo completed!");
    println!("\n💡 Key Features Demonstrated:");
    println!("   • Dynamic provider dispatch using provider_name()");
    println!("   • Convenience methods for specific providers");
    println!("   • Consistent API across all providers");
    println!("   • Capability checking");
    println!("   • Parameter configuration");
    println!("   • Chat functionality");

    Ok(())
}
