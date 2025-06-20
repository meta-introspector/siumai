//! Convenience Methods Examples
//!
//! This example demonstrates the simplified APIs and convenience methods
//! that make the library easier to use for common tasks.

use siumai::{quick_openai, quick_anthropic, user, system, assistant, user_with_image, LlmError, LlmBuilder};
use siumai::traits::ChatCapability;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Convenience Methods Examples ===\n");

    // Example 1: Quick client creation
    example_quick_clients().await?;

    // Example 2: Simple convenience methods
    example_convenience_methods().await?;

    // Example 3: Preset configurations
    example_preset_configurations().await?;

    // Example 4: Message macros
    example_message_macros().await?;

    Ok(())
}

/// Example 1: Quick client creation with minimal configuration
async fn example_quick_clients() -> Result<(), LlmError> {
    println!("1. Quick Client Creation");
    println!("========================");

    // Quick OpenAI client (uses OPENAI_API_KEY env var)
    if std::env::var("OPENAI_API_KEY").is_ok() {
        let client = quick_openai().await?;
        println!("✓ Created OpenAI client with default settings");

        // Quick response
        let response = client.ask("What is 2+2?".to_string()).await?;
        println!("Quick ask: {}", response);
    }

    // Quick Anthropic client (uses ANTHROPIC_API_KEY env var)
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        let client = quick_anthropic().await?;
        println!("✓ Created Anthropic client with default settings");
    }

    println!();
    Ok(())
}

/// Example 2: Convenience methods for common tasks
async fn example_convenience_methods() -> Result<(), LlmError> {
    println!("2. Convenience Methods");
    println!("======================");

    if std::env::var("OPENAI_API_KEY").is_ok() {
        let client = quick_openai().await?;

        // Simple ask
        let response = client.ask("Explain quantum computing in one sentence".to_string()).await?;
        println!("Simple ask: {}", response);

        // Ask with system prompt
        let response = client.ask_with_system(
            "You are a helpful math tutor".to_string(),
            "Explain the Pythagorean theorem".to_string()
        ).await?;
        println!("With system prompt: {}", response);

        // Translation
        let response = client.translate("Hello, how are you?".to_string(), "Spanish".to_string()).await?;
        println!("Translation: {}", response);

        // Explanation
        let response = client.explain("blockchain".to_string(), Some("a 10-year-old".to_string())).await?;
        println!("Explanation: {}", response);

        // Creative generation
        let response = client.generate("haiku".to_string(), "about programming".to_string()).await?;
        println!("Creative content: {}", response);

        // Conversation continuation
        let mut conversation = vec![
            system!("You are a helpful assistant"),
        ];

        let (response, updated_conversation) = client
            .continue_conversation(conversation, "Hello!".to_string())
            .await?;
        println!("Conversation response: {}", response);
        println!("Conversation has {} messages", updated_conversation.len());
    }

    println!();
    Ok(())
}

/// Example 3: Preset configurations for different use cases
async fn example_preset_configurations() -> Result<(), LlmError> {
    println!("3. Preset Configurations");
    println!("========================");

    if std::env::var("OPENAI_API_KEY").is_ok() {
        // Fast configuration for interactive apps
        let client = LlmBuilder::fast()
            .openai()
            .model("gpt-4o-mini")
            .build()
            .await?;
        println!("✓ Created fast client for interactive use");

        // Long-running configuration for batch processing
        let client = LlmBuilder::long_running()
            .openai()
            .model("gpt-4")
            .build()
            .await?;
        println!("✓ Created long-running client for batch processing");

        // Production defaults
        let client = LlmBuilder::with_defaults()
            .openai()
            .model("gpt-4")
            .build()
            .await?;
        println!("✓ Created client with production defaults");
    }

    println!();
    Ok(())
}

/// Example 4: Message creation macros
async fn example_message_macros() -> Result<(), LlmError> {
    println!("4. Message Creation Macros");
    println!("==========================");

    // Create messages using macros
    let messages = vec![
        system!("You are a helpful assistant"),
        user!("Hello!"),
        assistant!("Hi there! How can I help you?"),
        user!("What's the weather like?"),
    ];

    println!("Created {} messages using macros", messages.len());

    // Message with image (if supported)
    let user_with_image = user_with_image!(
        "Describe this image",
        "https://example.com/image.jpg"
    );
    println!("✓ Created user message with image");

    // Message with image and detail
    let user_with_detailed_image = user_with_image!(
        "Analyze this image in detail",
        "https://example.com/image.jpg",
        detail: "high"
    );
    println!("✓ Created user message with detailed image");

    if std::env::var("OPENAI_API_KEY").is_ok() {
        let client = quick_openai().await?;
        
        // Use the messages in a conversation
        let response = client.chat(messages).await?;
        if let Some(text) = response.content_text() {
            println!("Response: {}", text);
        }
    }

    println!();
    Ok(())
}
