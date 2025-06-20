//! OpenAI-Compatible Providers with Model Constants Example
//!
//! This example demonstrates how to use model constants instead of convenience
//! builder methods for OpenAI-compatible providers. This approach keeps the
//! API surface area manageable and maintainable.

use siumai::{types::ChatMessage, Provider};
use siumai::providers::openai_compatible::providers::{deepseek, openrouter, recommendations};
use siumai::traits::ChatCapability;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("🤖 OpenAI-Compatible Providers with Model Constants Example\n");

    // Example 1: DeepSeek with model constants
    println!("1. DeepSeek Examples:");
    
    // Using specific model constants
    let deepseek_chat = Provider::deepseek()
        .api_key(std::env::var("DEEPSEEK_API_KEY")?)
        .model(deepseek::CHAT)  // Using constant instead of convenience method
        .temperature(0.7)
        .build()
        .await?;

    let deepseek_coder = Provider::deepseek()
        .api_key(std::env::var("DEEPSEEK_API_KEY")?)
        .model(deepseek::CODER)  // Using constant for coding model
        .temperature(0.1)
        .build()
        .await?;

    let deepseek_reasoner = Provider::deepseek()
        .api_key(std::env::var("DEEPSEEK_API_KEY")?)
        .model(deepseek::REASONER)  // Using constant for reasoning model
        .temperature(0.3)
        .build()
        .await?;

    println!("   ✓ Chat model: {}", deepseek::CHAT);
    println!("   ✓ Coder model: {}", deepseek::CODER);
    println!("   ✓ Reasoner model: {}", deepseek::REASONER);

    // Example 2: OpenRouter with model constants
    println!("\n2. OpenRouter Examples:");
    
    let openrouter_gpt4 = Provider::openrouter()
        .api_key(std::env::var("OPENROUTER_API_KEY")?)
        .model(openrouter::openai::GPT_4O)  // Using OpenAI model through OpenRouter
        .site_url("https://example.com")?
        .app_name("Model Constants Example")?
        .build()
        .await?;

    let openrouter_claude = Provider::openrouter()
        .api_key(std::env::var("OPENROUTER_API_KEY")?)
        .model(openrouter::anthropic::CLAUDE_3_5_SONNET)  // Using Anthropic model
        .build()
        .await?;

    let openrouter_gemini = Provider::openrouter()
        .api_key(std::env::var("OPENROUTER_API_KEY")?)
        .model(openrouter::google::GEMINI_1_5_PRO)  // Using Google model
        .build()
        .await?;

    println!("   ✓ GPT-4o: {}", openrouter::openai::GPT_4O);
    println!("   ✓ Claude 3.5 Sonnet: {}", openrouter::anthropic::CLAUDE_3_5_SONNET);
    println!("   ✓ Gemini 1.5 Pro: {}", openrouter::google::GEMINI_1_5_PRO);

    // Example 3: Using recommendation helpers
    println!("\n3. Recommendation Helpers:");
    
    let chat_client = Provider::openrouter()
        .api_key(std::env::var("OPENROUTER_API_KEY")?)
        .model(recommendations::for_chat())  // Gets recommended chat model
        .build()
        .await?;

    let coding_client = Provider::deepseek()
        .api_key(std::env::var("DEEPSEEK_API_KEY")?)
        .model(recommendations::for_coding())  // Gets recommended coding model
        .build()
        .await?;

    let reasoning_client = Provider::deepseek()
        .api_key(std::env::var("DEEPSEEK_API_KEY")?)
        .model(recommendations::for_reasoning())  // Gets recommended reasoning model
        .build()
        .await?;

    println!("   ✓ Recommended for chat: {}", recommendations::for_chat());
    println!("   ✓ Recommended for coding: {}", recommendations::for_coding());
    println!("   ✓ Recommended for reasoning: {}", recommendations::for_reasoning());
    println!("   ✓ Recommended for fast response: {}", recommendations::for_fast_response());
    println!("   ✓ Recommended for cost-effective: {}", recommendations::for_cost_effective());
    println!("   ✓ Recommended for vision: {}", recommendations::for_vision());

    // Example 4: Demonstrating the benefits
    println!("\n4. Benefits of Model Constants:");
    println!("   ✓ Type safety: Constants are checked at compile time");
    println!("   ✓ Maintainability: Easy to update model names in one place");
    println!("   ✓ Discoverability: IDE autocomplete shows available models");
    println!("   ✓ Consistency: Same model names across different providers");
    println!("   ✓ No API bloat: No need for .use_gpt4(), .use_coder() methods");

    // Example 5: All available models
    println!("\n5. Available Model Collections:");
    println!("   DeepSeek models: {:?}", deepseek::ALL);
    println!("   Popular OpenRouter models:");
    println!("     - {}", openrouter::popular::GPT_4);
    println!("     - {}", openrouter::popular::GPT_4O);
    println!("     - {}", openrouter::popular::CLAUDE_3_5_SONNET);
    println!("     - {}", openrouter::popular::GEMINI_PRO);

    // Example 6: Actual chat usage
    if std::env::var("DEEPSEEK_API_KEY").is_ok() {
        println!("\n6. Testing with actual chat:");
        
        let messages = vec![
            ChatMessage::user("What's the difference between using model constants vs convenience methods?").build()
        ];

        let response = deepseek_chat.chat(messages).await?;
        if let Some(text) = response.content_text() {
            println!("   DeepSeek response: {}", text);
        }
    }

    println!("\n✅ Model constants example completed!");
    println!("\n💡 Key takeaway: Use model constants like `deepseek::CODER` instead of");
    println!("   convenience methods like `.use_coder()` to keep the API clean and maintainable.");

    Ok(())
}

/// Helper function to demonstrate model selection logic
fn select_model_for_task(task: &str) -> &'static str {
    match task {
        "coding" | "programming" | "debug" => deepseek::CODER,
        "reasoning" | "analysis" | "thinking" => deepseek::REASONER,
        "chat" | "conversation" => recommendations::for_chat(),
        "vision" | "image" => recommendations::for_vision(),
        "fast" | "quick" => recommendations::for_fast_response(),
        "cheap" | "cost-effective" => recommendations::for_cost_effective(),
        _ => deepseek::CHAT, // Default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_selection() {
        assert_eq!(select_model_for_task("coding"), deepseek::CODER);
        assert_eq!(select_model_for_task("reasoning"), deepseek::REASONER);
        assert_eq!(select_model_for_task("unknown"), deepseek::CHAT);
    }

    #[test]
    fn test_model_constants() {
        // Ensure constants are not empty
        assert!(!deepseek::CHAT.is_empty());
        assert!(!deepseek::CODER.is_empty());
        assert!(!deepseek::REASONER.is_empty());
        
        // Ensure OpenRouter models have correct format
        assert!(openrouter::openai::GPT_4.contains('/'));
        assert!(openrouter::anthropic::CLAUDE_3_5_SONNET.contains('/'));
        assert!(openrouter::google::GEMINI_PRO.contains('/'));
    }
}
