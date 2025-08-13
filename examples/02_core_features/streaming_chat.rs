//! 🌊 Streaming Chat - Real-time Response Streaming
//!
//! This example demonstrates how to handle real-time streaming responses:
//! - Processing stream events as they arrive
//! - Building responsive user interfaces
//! - Handling different event types
//! - Error recovery and stream management
//!
//! Before running, set your API key:
//! ```bash
//! export OPENAI_API_KEY="your-key"
//! export ANTHROPIC_API_KEY="your-key"
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example streaming_chat
//! ```

use futures_util::StreamExt;
use siumai::models;
use siumai::prelude::*;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Streaming Chat - Real-time Response Streaming\n");

    // Get API key and create provider
    let provider: Box<dyn ChatCapability> = if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        Box::new(
            LlmBuilder::new()
                .openai()
                .api_key(&api_key)
                .model(models::openai::GPT_4O_MINI)
                .temperature(0.7)
                .max_tokens(500)
                .build()
                .await?,
        )
    } else if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        Box::new(
            LlmBuilder::new()
                .anthropic()
                .api_key(&api_key)
                .model(models::anthropic::CLAUDE_HAIKU_3_5)
                .temperature(0.7)
                .max_tokens(500)
                .build()
                .await?,
        )
    } else {
        println!("⚠️  No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY");
        return Ok(());
    };

    // Demonstrate different streaming scenarios
    demonstrate_basic_streaming(provider.as_ref()).await;
    demonstrate_stream_event_types(provider.as_ref()).await;
    demonstrate_stream_error_handling(provider.as_ref()).await;
    demonstrate_stream_performance(provider.as_ref()).await;

    println!("\n✅ Streaming chat completed!");
    Ok(())
}

/// Demonstrate basic streaming functionality
async fn demonstrate_basic_streaming(provider: &dyn ChatCapability) {
    println!("⚡ Basic Streaming:\n");

    match async {
        let messages = vec![
            user!("Please write a detailed explanation about the Rust programming language, including its features, advantages, and use cases. Write about 300-500 words and take your time to explain thoroughly.")
        ];

        println!("   User: Write a detailed explanation about Rust programming...");
        println!("   AI: ");

        // Stream the response
        let mut stream = provider.chat_stream(messages, None).await?;
        while let Some(event) = stream.next().await {
            match event? {
                ChatStreamEvent::ContentDelta { delta, .. } => {
                    // Print each text chunk as it arrives
                    print!("{delta}");
                    io::stdout().flush().unwrap();
                }
                ChatStreamEvent::StreamEnd { .. } => {
                    // Stream completed
                    println!("\n   ✅ Basic streaming successful\n");
                    break;
                }
                _ => {
                    // Handle other event types
                }
            }
        }
        Ok::<_, LlmError>(())
    }.await {
        Ok(()) => {}
        Err(e) => {
            println!("\n   ❌ Basic streaming failed: {e}\n");
        }
    }
}

/// Demonstrate different stream event types
async fn demonstrate_stream_event_types(provider: &dyn ChatCapability) {
    println!("📡 Stream Event Types:\n");

    match async {
        let messages = vec![user!(
            "Write a short poem about programming and explain the metaphors you used."
        )];

        println!("   User: Write a short poem about programming...");
        println!("   Processing events:\n");

        let mut text_chunks = 0;
        let mut total_text = String::new();
        let mut stream = provider.chat_stream(messages, None).await?;

        while let Some(event) = stream.next().await {
            match event? {
                ChatStreamEvent::ContentDelta { delta, index } => {
                    text_chunks += 1;
                    total_text.push_str(&delta);
                    println!("   📝 Text chunk {text_chunks} (index: {index:?}): \"{delta}\"");
                }
                ChatStreamEvent::StreamEnd { response } => {
                    println!("\n   🏁 Completion event received");
                    if let Some(reason) = response.finish_reason {
                        println!("   🏁 Finish reason: {reason:?}");
                    }
                    if let Some(usage) = response.usage {
                        println!("   📊 Usage: {} total tokens", usage.total_tokens);
                    }
                    break;
                }
                ChatStreamEvent::UsageUpdate { usage } => {
                    println!("   📈 Usage update: {} tokens", usage.total_tokens);
                }
                _ => {
                    // Handle other event types
                }
            }
        }

        println!("\n   📈 Stream Statistics:");
        println!("      • Total text chunks: {text_chunks}");
        println!("      • Final text length: {} characters", total_text.len());
        println!("   ✅ Event types demonstration successful\n");

        Ok::<_, LlmError>(())
    }
    .await
    {
        Ok(()) => {}
        Err(e) => {
            println!("   ❌ Event types demonstration failed: {e}\n");
        }
    }
}

/// Demonstrate stream error handling
async fn demonstrate_stream_error_handling(_provider: &dyn ChatCapability) {
    println!("🛡️  Stream Error Handling:\n");

    match async {
        // Create a provider with invalid settings to trigger errors
        let invalid_provider = LlmBuilder::new()
            .openai()
            .api_key("invalid-key") // Invalid API key
            .model(models::openai::GPT_4O_MINI)
            .build()
            .await?;

        let messages = vec![user!("This should fail due to invalid API key.")];

        println!("   Testing error handling with invalid API key...");

        let mut stream = invalid_provider.chat_stream(messages, None).await?;

        while let Some(event) = stream.next().await {
            match event {
                Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                    println!("   📝 Unexpected text: {delta}");
                }
                Ok(ChatStreamEvent::StreamEnd { .. }) => {
                    println!("   ❌ Unexpected completion");
                    break;
                }
                Err(e) => {
                    println!("   ✅ Caught error in stream: {e}");
                    break;
                }
                _ => {}
            }
        }

        Ok::<_, LlmError>(())
    }
    .await
    {
        Ok(()) => {}
        Err(e) => {
            println!("   ✅ Caught exception: {e}");
        }
    }

    println!("\n   💡 Error Handling Best Practices:");
    println!("      • Always wrap stream processing in try-catch");
    println!("      • Handle errors within the stream loop");
    println!("      • Implement retry logic for transient errors");
    println!("      • Provide user feedback for stream interruptions");
    println!("   ✅ Error handling demonstration completed\n");
}

/// Demonstrate stream performance characteristics
async fn demonstrate_stream_performance(provider: &dyn ChatCapability) {
    println!("🚀 Stream Performance:\n");

    match async {
        let messages = vec![
            user!("Write a detailed explanation of machine learning in about 200 words, covering key concepts and applications.")
        ];

        println!("   User: Write a detailed explanation of machine learning...");
        println!("   Measuring performance...\n");

        let start_time = std::time::Instant::now();
        let mut first_chunk_time = None;
        let mut chunk_count = 0;
        let mut total_chars = 0;
        let mut chunk_times = Vec::new();

        let mut stream = provider.chat_stream(messages, None).await?;

        while let Some(event) = stream.next().await {
            match event? {
                ChatStreamEvent::ContentDelta { delta, .. } => {
                    chunk_count += 1;
                    total_chars += delta.len();

                    let elapsed = start_time.elapsed();
                    if first_chunk_time.is_none() {
                        first_chunk_time = Some(elapsed);
                        println!("   ⚡ First chunk received: {}ms", elapsed.as_millis());
                    }

                    chunk_times.push(elapsed);
                }
                ChatStreamEvent::StreamEnd { .. } => {
                    break;
                }
                _ => {}
            }
        }

        let total_time = start_time.elapsed();
        let first_chunk_ms = first_chunk_time.unwrap_or_default().as_millis();
        let avg_chunk_interval = if chunk_times.len() > 1 {
            (total_time.as_millis() - first_chunk_ms) / (chunk_times.len() - 1) as u128
        } else {
            0
        };

        println!("\n   📊 Performance Metrics:");
        println!("      • Time to first chunk: {first_chunk_ms}ms");
        println!("      • Total response time: {}ms", total_time.as_millis());
        println!("      • Total chunks: {chunk_count}");
        println!("      • Total characters: {total_chars}");
        println!("      • Average chunk interval: {avg_chunk_interval}ms");
        if total_time.as_millis() > 0 {
            println!("      • Characters per second: {:.1}", 
                (total_chars as f64 * 1000.0) / total_time.as_millis() as f64);
        }

        println!("\n   💡 Performance Benefits:");
        println!("      • Reduced perceived latency (first chunk arrives quickly)");
        println!("      • Better user experience (progressive content display)");
        println!("      • Ability to process content as it arrives");
        println!("      • Early error detection and handling");

        println!("   ✅ Performance demonstration completed\n");
        Ok::<_, LlmError>(())
    }.await {
        Ok(()) => {}
        Err(e) => {
            println!("   ❌ Performance demonstration failed: {e}\n");
        }
    }
}

/// 🎯 Key Streaming Concepts Summary:
///
/// Stream Events:
/// - `ContentDelta`: Incremental text content as it's generated
/// - StreamEnd: Stream completion with final response metadata
/// - `UsageUpdate`: Token usage information during streaming
///
/// Benefits:
/// - Reduced perceived latency
/// - Real-time user feedback
/// - Progressive content display
/// - Better error handling
///
/// Best Practices:
/// 1. Handle all event types appropriately
/// 2. Implement proper error handling within the stream
/// 3. Measure and optimize performance
/// 4. Provide user feedback during streaming
/// 5. Consider buffering for UI updates
///
/// Next Steps:
/// - `error_handling.rs`: Production-ready error management
/// - ../`03_advanced_features/`: Advanced streaming patterns
/// - ../`04_providers/`: Provider-specific streaming features
const fn _documentation() {}
