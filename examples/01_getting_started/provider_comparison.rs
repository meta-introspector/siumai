//! 🔍 Provider Comparison - Understanding Different AI Providers
//!
//! This example demonstrates the differences between AI providers:
//! - Performance characteristics
//! - Model capabilities
//! - Cost considerations
//! - Use case recommendations
//!
//! Before running, set your API keys:
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export ANTHROPIC_API_KEY="your-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example provider_comparison
//! ```

use siumai::models;
use siumai::prelude::*;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Provider Comparison - Understanding Different AI Providers\n");

    // Test the same prompt with different providers
    let test_prompt = "Explain the concept of recursion in programming with a simple example.";

    compare_providers(test_prompt).await;
    demonstrate_provider_strengths().await;
    show_cost_considerations().await;
    provide_recommendations().await;

    println!("\n✅ Provider comparison completed!");
    Ok(())
}

/// Compare the same prompt across different providers
async fn compare_providers(prompt: &str) {
    println!("⚖️  Provider Performance Comparison:\n");
    println!("   Test prompt: \"{prompt}\"\n");

    // Test OpenAI
    test_provider_performance("OpenAI", || async {
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            let client = LlmBuilder::new()
                .openai()
                .api_key(&api_key)
                .model(models::openai::GPT_4O_MINI)
                .temperature(0.7)
                .build()
                .await?;

            let messages = vec![user!(prompt)];
            let response = client.chat(messages).await?;
            Ok(response)
        } else {
            Err(LlmError::AuthenticationError("No API key".to_string()))
        }
    })
    .await;

    // Test Anthropic
    test_provider_performance("Anthropic", || async {
        if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
            let client = LlmBuilder::new()
                .anthropic()
                .api_key(&api_key)
                .model(models::anthropic::CLAUDE_HAIKU_3_5)
                .temperature(0.7)
                .build()
                .await?;

            let messages = vec![user!(prompt)];
            let response = client.chat(messages).await?;
            Ok(response)
        } else {
            Err(LlmError::AuthenticationError("No API key".to_string()))
        }
    })
    .await;

    // Test Ollama (local)
    test_provider_performance("Ollama", || async {
        let client = LlmBuilder::new()
            .ollama()
            .base_url("http://localhost:11434")
            .model("llama3.2")
            .temperature(0.7)
            .build()
            .await?;

        let messages = vec![user!(prompt)];
        let response = client.chat(messages).await?;
        Ok(response)
    })
    .await;
}

/// Test a single provider's performance
async fn test_provider_performance<F, Fut>(provider_name: &str, test_fn: F)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<ChatResponse, LlmError>>,
{
    println!("   🧪 Testing {provider_name}:");

    let start_time = Instant::now();

    match test_fn().await {
        Ok(response) => {
            let duration = start_time.elapsed();

            if let Some(text) = response.content_text() {
                println!("      ✅ Success");
                println!("      ⏱️  Response time: {}ms", duration.as_millis());
                println!("      📝 Response length: {} characters", text.len());

                if let Some(usage) = &response.usage {
                    println!(
                        "      🔢 Tokens: {} total ({} prompt + {} completion)",
                        usage.total_tokens, usage.prompt_tokens, usage.completion_tokens
                    );
                }

                // Show first 100 characters of response
                let preview = if text.len() > 100 {
                    format!("{}...", &text[..100])
                } else {
                    text.to_string()
                };
                println!("      💬 Preview: {preview}");
            }
        }
        Err(e) => {
            println!("      ❌ Failed: {e}");
            match provider_name {
                "OpenAI" => println!("      💡 Set OPENAI_API_KEY environment variable"),
                "Anthropic" => println!("      💡 Set ANTHROPIC_API_KEY environment variable"),
                "Ollama" => println!("      💡 Ensure Ollama is running: ollama serve"),
                _ => {}
            }
        }
    }

    println!();
}

/// Demonstrate each provider's unique strengths
async fn demonstrate_provider_strengths() {
    println!("💪 Provider Strengths:\n");

    println!("   🤖 OpenAI:");
    println!("      • Most popular and well-documented");
    println!("      • Excellent general-purpose performance");
    println!("      • Strong multimodal capabilities (vision, audio)");
    println!("      • Large ecosystem and community");
    println!("      • Best for: General applications, prototyping");

    println!("\n   🧠 Anthropic (Claude):");
    println!("      • Excellent reasoning and analysis");
    println!("      • Strong safety and alignment focus");
    println!("      • Great for complex, nuanced tasks");
    println!("      • Transparent thinking process");
    println!("      • Best for: Research, analysis, complex reasoning");

    println!("\n   🏠 Ollama (Local):");
    println!("      • Complete privacy and data control");
    println!("      • No API costs after setup");
    println!("      • Works offline");
    println!("      • Customizable and fine-tunable");
    println!("      • Best for: Privacy-sensitive applications, development");

    println!("\n   ⚡ Groq (if available):");
    println!("      • Extremely fast inference");
    println!("      • Cost-effective for high-volume usage");
    println!("      • Great for real-time applications");
    println!("      • Best for: High-throughput, latency-sensitive apps");
}

/// Show cost considerations for different providers
async fn show_cost_considerations() {
    println!("\n💰 Cost Considerations:\n");

    println!("   📊 Approximate Pricing (per 1M tokens):");
    println!("      • OpenAI GPT-4o-mini: ~$0.15 input, ~$0.60 output");
    println!("      • Anthropic Claude Haiku: ~$0.25 input, ~$1.25 output");
    println!("      • Ollama: Free after hardware investment");
    println!("      • Groq: ~$0.05 input, ~$0.08 output (very fast)");

    println!("\n   💡 Cost Optimization Tips:");
    println!("      • Use smaller models for simple tasks");
    println!("      • Implement caching for repeated queries");
    println!("      • Monitor token usage in production");
    println!("      • Consider local models for development");
    println!("      • Use streaming to provide better UX while processing");
}

/// Provide recommendations for different use cases
async fn provide_recommendations() {
    println!("\n🎯 Use Case Recommendations:\n");

    println!("   🚀 Getting Started / Prototyping:");
    println!("      → OpenAI GPT-4o-mini");
    println!("      • Easy to use, well-documented");
    println!("      • Good balance of cost and performance");

    println!("\n   🏢 Production Applications:");
    println!("      → Multiple providers with fallback");
    println!("      • Primary: Based on your specific needs");
    println!("      • Fallback: Different provider for reliability");

    println!("\n   🔒 Privacy-Sensitive Applications:");
    println!("      → Ollama (local deployment)");
    println!("      • Complete data control");
    println!("      • No external API calls");

    println!("\n   📊 High-Volume / Real-Time:");
    println!("      → Groq or OpenAI with caching");
    println!("      • Fast response times");
    println!("      • Cost-effective at scale");

    println!("\n   🧪 Research / Complex Analysis:");
    println!("      → Anthropic Claude");
    println!("      • Superior reasoning capabilities");
    println!("      • Transparent thinking process");

    println!("\n   💻 Development / Testing:");
    println!("      → Ollama for development, cloud for production");
    println!("      • Free local testing");
    println!("      • Easy transition to production");
}

/// 🎯 Key Comparison Points:
///
/// Performance Factors:
/// - Response time and latency
/// - Quality and accuracy
/// - Token efficiency
/// - Reliability and uptime
///
/// Cost Factors:
/// - Per-token pricing
/// - Volume discounts
/// - Hidden costs (rate limits, etc.)
/// - Total cost of ownership
///
/// Feature Differences:
/// - Model capabilities
/// - Multimodal support
/// - Context window size
/// - Special features (thinking, tools, etc.)
///
/// Selection Criteria:
/// 1. Define your primary use case
/// 2. Consider performance requirements
/// 3. Evaluate cost constraints
/// 4. Test with your specific data
/// 5. Plan for scaling and reliability
///
/// Next Steps:
/// - `basic_usage.rs`: Learn core functionality
/// - ../`02_core_features/`: Explore advanced features
/// - ../`04_providers/`: Provider-specific capabilities
const fn _documentation() {}
