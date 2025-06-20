//! New API Design Demo
//!
//! This example demonstrates the new API design with Provider and Siumai structures.
//! It shows the difference between provider-specific clients and unified interface.

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 New Siumai API Design Demo");
    println!("==============================\n");

    // Demo 1: Provider-specific clients
    provider_specific_demo().await?;

    // Demo 2: Unified interface
    unified_interface_demo().await?;

    // Demo 3: Comparison
    comparison_demo().await?;

    println!("\n✅ All demos completed successfully!");
    Ok(())
}

/// Demo 1: Provider-specific clients
/// Use this when you need access to provider-specific features
async fn provider_specific_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Provider-Specific Clients Demo");
    println!("----------------------------------");

    // OpenAI client with provider-specific features
    let openai_client = Provider::openai()
        .api_key("your-openai-key")
        .model("gpt-4")
        .temperature(0.7)
        .build()
        .await?;

    println!("✅ Created OpenAI client");
    println!("   - Can access OpenAI-specific features");
    println!("   - Direct access to all OpenAI capabilities");

    // Anthropic client with provider-specific features
    let anthropic_client = Provider::anthropic()
        .api_key("your-anthropic-key")
        .model("claude-3-sonnet-20240229")
        .temperature(0.7)
        .build()
        .await?;

    println!("✅ Created Anthropic client");
    println!("   - Can access Anthropic-specific features");
    println!("   - Direct access to all Anthropic capabilities");

    // Example usage (commented out to avoid API calls)
    // let response = openai_client.chat(vec![user!("Hello from OpenAI!")]).await?;
    // println!("🤖 OpenAI: {}", response.text().unwrap_or_default());

    println!();
    Ok(())
}

/// Demo 2: Unified interface
/// Use this when you want provider-agnostic code
async fn unified_interface_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Unified Interface Demo");
    println!("--------------------------");

    // Create a unified client backed by OpenAI
    let openai_unified = Siumai::builder()
        .openai()
        .api_key("your-openai-key")
        .model("gpt-4")
        .temperature(0.7)
        .build()
        .await?;

    println!("✅ Created unified client (OpenAI backend)");
    println!("   - Uses standard Siumai interface");
    println!("   - Easy to switch providers");

    // Create a unified client backed by Anthropic
    let anthropic_unified = Siumai::builder()
        .anthropic()
        .api_key("your-anthropic-key")
        .model("claude-3-sonnet-20240229")
        .temperature(0.7)
        .build()
        .await?;

    println!("✅ Created unified client (Anthropic backend)");
    println!("   - Same interface as OpenAI client");
    println!("   - Code remains identical");

    // Example usage (commented out to avoid API calls)
    // let response = openai_unified.chat(vec![user!("Hello from unified interface!")]).await?;
    // println!("🤖 Unified: {}", response.text().unwrap_or_default());

    println!();
    Ok(())
}

/// Demo 3: Comparison between approaches
async fn comparison_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚖️  Comparison Demo");
    println!("-------------------");

    println!("📋 When to use Provider::openai():");
    println!("   ✅ Need OpenAI-specific features (assistants, structured output, etc.)");
    println!("   ✅ Want direct access to all provider capabilities");
    println!("   ✅ Building provider-specific integrations");
    println!("   ✅ Performance-critical applications");

    println!("\n📋 When to use Siumai::builder():");
    println!("   ✅ Want provider-agnostic code");
    println!("   ✅ Need to switch providers easily");
    println!("   ✅ Building multi-provider applications");
    println!("   ✅ Prototyping and experimentation");

    println!("\n🔄 Migration path:");
    println!("   1. Start with Siumai::builder() for flexibility");
    println!("   2. Switch to Provider::* when you need specific features");
    println!("   3. Mix both approaches in the same application");

    println!("\n💡 Best practices:");
    println!("   - Use Siumai::builder() for business logic");
    println!("   - Use Provider::* for provider integrations");
    println!("   - Keep provider-specific code isolated");
    println!("   - Use traits for abstraction");

    println!();
    Ok(())
}

/// Example of how to abstract over both approaches
trait ChatProvider {
    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse, Box<dyn std::error::Error>>;
}

// You can implement this trait for both approaches
// impl ChatProvider for OpenAiClient { ... }
// impl ChatProvider for Siumai { ... }

/// Example of provider-agnostic function
async fn _chat_with_any_provider<P: ChatProvider>(
    provider: &P,
    message: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let messages = vec![user!(message)];
    let response = provider.chat(messages).await?;
    Ok(response.text().unwrap_or_default())
}
