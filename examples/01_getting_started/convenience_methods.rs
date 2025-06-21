//! 🛠️ Convenience Methods Examples
//!
//! This example demonstrates simplified APIs and convenience methods
//! that make the library easier to use for common tasks.
//!
//! Before running, set your API key:
//! ```bash
//! export OPENAI_API_KEY="your-openai-key"
//! export GROQ_API_KEY="your-groq-key"
//! ```
//!
//! Usage:
//! ```bash
//! cargo run --example convenience_methods
//! ```

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛠️ Convenience Methods Examples");
    println!("===============================\n");

    // Get API key
    let api_key = std::env::var("GROQ_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .unwrap_or_else(|_| {
            println!("⚠️  No API key found, using demo key");
            "demo-key".to_string()
        });

    // Example 1: Quick Client Creation
    println!("🚀 Example 1: Quick Client Creation");
    quick_client_creation(&api_key).await?;
    println!();

    // Example 2: Common Task Helpers
    println!("🎯 Example 2: Common Task Helpers");
    common_task_helpers(&api_key).await?;
    println!();

    // Example 3: Preset Configurations
    println!("⚙️ Example 3: Preset Configurations");
    preset_configurations(&api_key).await?;
    println!();

    // Example 4: Message Creation Shortcuts
    println!("💬 Example 4: Message Creation Shortcuts");
    message_creation_shortcuts();
    println!();

    // Example 5: Conversation Helpers
    println!("🗣️ Example 5: Conversation Helpers");
    conversation_helpers(&api_key).await?;

    println!("\n✅ All convenience method examples completed!");
    Ok(())
}

/// Example 1: Quick client creation with minimal configuration
async fn quick_client_creation(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating clients with minimal configuration...");

    // Quick OpenAI client (fastest setup)
    let openai_client = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .build()
        .await?;
    
    println!("   ✅ Created OpenAI client with default settings");

    // Test the quick client
    let response = openai_client.chat(vec![
        ChatMessage::user("What is 2+2?").build()
    ]).await?;
    
    if let Some(text) = response.text() {
        println!("   🤖 Quick response: {}", text.trim());
    }

    // Quick Anthropic client (if available)
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        let anthropic_client = Siumai::builder()
            .anthropic()
            .api_key(&std::env::var("ANTHROPIC_API_KEY")?)
            .model("claude-3-5-sonnet-20241022")
            .build()
            .await?;

        println!("   ✅ Created Anthropic client with default settings");
    } else {
        println!("   ⏭️ Skipped Anthropic (no API key)");
    }

    Ok(())
}

/// Example 2: Helper functions for common tasks
async fn common_task_helpers(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating common task helpers...");

    let ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .temperature(0.3)
        .max_tokens(200)
        .build()
        .await?;

    // Helper 1: Simple question
    println!("   📝 Simple question helper:");
    let response = simple_ask(&ai, "What is the capital of Japan?").await?;
    println!("      Answer: {}", response);

    // Helper 2: Translation
    println!("   🌍 Translation helper:");
    let response = translate(&ai, "Hello, how are you?", "Spanish").await?;
    println!("      Translation: {}", response);

    // Helper 3: Explanation
    println!("   💡 Explanation helper:");
    let response = explain(&ai, "blockchain", "a beginner").await?;
    println!("      Explanation: {}", &response[..response.len().min(100)]);

    // Helper 4: Creative generation
    println!("   🎨 Creative generation helper:");
    let response = generate_creative(&ai, "haiku", "about programming").await?;
    println!("      Creative content: {}", response);

    Ok(())
}

/// Example 3: Preset configurations for different use cases
async fn preset_configurations(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing preset configurations...");

    // Fast configuration for interactive apps
    println!("   ⚡ Fast configuration (low latency):");
    let fast_ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .temperature(0.3)
        .max_tokens(100)
        .build()
        .await?;

    let response = fast_ai.chat(vec![
        ChatMessage::user("Say hello briefly").build()
    ]).await?;
    
    if let Some(text) = response.text() {
        println!("      Response: {}", text.trim());
    }

    // Balanced configuration for general use
    println!("   ⚖️ Balanced configuration:");
    let balanced_ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-4o")
        .temperature(0.5)
        .max_tokens(300)
        .build()
        .await?;

    println!("      ✅ Created balanced client for general use");

    // Creative configuration for content generation
    println!("   🎨 Creative configuration:");
    let creative_ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .temperature(0.8)
        .max_tokens(500)
        .build()
        .await?;

    println!("      ✅ Created creative client for content generation");

    Ok(())
}

/// Example 4: Message creation shortcuts
fn message_creation_shortcuts() {
    println!("   Demonstrating message creation shortcuts...");

    // Basic message types
    let system_msg = ChatMessage::system("You are a helpful assistant").build();
    let user_msg = ChatMessage::user("Hello!").build();
    let assistant_msg = ChatMessage::assistant("Hi there! How can I help you?").build();

    println!("   ✅ Created system message: {:?}", system_msg.role);
    println!("   ✅ Created user message: {:?}", user_msg.role);
    println!("   ✅ Created assistant message: {:?}", assistant_msg.role);

    // Conversation builder
    let conversation = vec![
        ChatMessage::system("You are a helpful coding assistant").build(),
        ChatMessage::user("How do I create a vector in Rust?").build(),
        ChatMessage::assistant("You can create a vector using Vec::new() or the vec! macro.").build(),
        ChatMessage::user("Can you show me an example?").build(),
    ];

    println!("   💬 Built conversation with {} messages", conversation.len());

    // Message with context
    let contextual_message = ChatMessage::user(
        "Based on our previous discussion about vectors, \
        how do I add elements to a vector?"
    ).build();

    println!("   🔗 Created contextual message");
}

/// Example 5: Conversation helpers
async fn conversation_helpers(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating conversation helpers...");

    let ai = Siumai::builder()
        .openai()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .temperature(0.3)
        .max_tokens(200)
        .build()
        .await?;

    // Start a conversation
    let mut conversation = vec![
        ChatMessage::system("You are a helpful programming tutor").build(),
    ];

    // Helper function to continue conversation
    println!("   🗣️ Starting conversation:");
    let response = continue_conversation(&ai, &mut conversation, "What is Rust?").await?;
    println!("      AI: {}", &response[..response.len().min(80)]);

    // Continue the conversation
    let response = continue_conversation(&ai, &mut conversation, "What makes it special?").await?;
    println!("      AI: {}", &response[..response.len().min(80)]);

    println!("   📊 Conversation now has {} messages", conversation.len());

    Ok(())
}

// Helper functions

/// Simple ask helper
async fn simple_ask<T: ChatCapability + Sync>(ai: &T, question: &str) -> Result<String, Box<dyn std::error::Error>> {
    let response = ai.chat(vec![ChatMessage::user(question).build()]).await?;
    Ok(response.text().unwrap_or_default())
}

/// Translation helper
async fn translate<T: ChatCapability + Sync>(ai: &T, text: &str, target_language: &str) -> Result<String, Box<dyn std::error::Error>> {
    let prompt = format!("Translate this text to {}: {}", target_language, text);
    let response = ai.chat(vec![ChatMessage::user(&prompt).build()]).await?;
    Ok(response.text().unwrap_or_default())
}

/// Explanation helper
async fn explain<T: ChatCapability + Sync>(ai: &T, topic: &str, audience: &str) -> Result<String, Box<dyn std::error::Error>> {
    let prompt = format!("Explain {} to {}", topic, audience);
    let response = ai.chat(vec![ChatMessage::user(&prompt).build()]).await?;
    Ok(response.text().unwrap_or_default())
}

/// Creative generation helper
async fn generate_creative<T: ChatCapability + Sync>(ai: &T, content_type: &str, topic: &str) -> Result<String, Box<dyn std::error::Error>> {
    let prompt = format!("Write a {} about {}", content_type, topic);
    let response = ai.chat(vec![
        ChatMessage::system("You are a creative writer").build(),
        ChatMessage::user(&prompt).build()
    ]).await?;
    Ok(response.text().unwrap_or_default())
}

/// Continue conversation helper
async fn continue_conversation<T: ChatCapability + Sync>(
    ai: &T,
    conversation: &mut Vec<ChatMessage>,
    user_input: &str
) -> Result<String, Box<dyn std::error::Error>> {
    // Add user message
    conversation.push(ChatMessage::user(user_input).build());

    // Get AI response
    let response = ai.chat(conversation.clone()).await?;
    let ai_response = response.text().unwrap_or_default();

    // Add AI response to conversation
    conversation.push(ChatMessage::assistant(&ai_response).build());

    Ok(ai_response)
}

/// 🎯 Key Convenience Features Summary:
///
/// Quick Setup:
/// - Minimal configuration builders
/// - Environment variable integration
/// - Sensible defaults for common use cases
/// - Provider-specific optimizations
///
/// Common Task Helpers:
/// - Simple question answering
/// - Translation assistance
/// - Topic explanation
/// - Creative content generation
/// - Conversation management
///
/// Preset Configurations:
/// - Fast: Optimized for speed and interactivity
/// - Balanced: Good quality and reasonable speed
/// - Creative: High temperature for varied outputs
/// - Production: Reliable settings for deployment
///
/// Message Shortcuts:
/// - Simple message creation
/// - Conversation building
/// - Context management
/// - Role-based messaging
///
/// Conversation Helpers:
/// - Automatic conversation tracking
/// - Context preservation
/// - Multi-turn dialogue support
/// - State management
///
/// Best Practices:
/// - Use appropriate presets for your use case
/// - Leverage helper functions for common tasks
/// - Build conversations incrementally
/// - Handle errors gracefully
/// - Monitor token usage
///
/// Next Steps:
/// - Explore streaming responses
/// - Try advanced configurations
/// - Build custom helpers
/// - Integrate with applications
fn _documentation() {}
