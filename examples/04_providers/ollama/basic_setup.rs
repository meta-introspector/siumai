//! 🦙 Ollama Basic Setup - Local AI Models
//!
//! This example demonstrates how to use Ollama for local AI inference:
//! - Setting up Ollama connection
//! - Basic chat functionality
//! - Streaming responses
//! - Model management
//! - Local deployment benefits
//!
//! Prerequisites:
//! - Ollama must be installed and running on your system
//! - At least one model must be available (e.g., llama3.2)
//!
//! Setup:
//! ```bash
//! # Install Ollama (visit https://ollama.ai)
//! # Start Ollama service
//! ollama serve
//!
//! # Pull a model
//! ollama pull llama3.2
//! ollama pull llama3.2:1b  # Smaller model for testing
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example basic_setup
//! ```

use futures_util::StreamExt;
use siumai::prelude::*;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦙 Ollama Basic Setup - Local AI Models");
    println!("=======================================\n");

    // Test Ollama connection and functionality
    test_ollama_connection().await;
    demonstrate_basic_chat().await;
    demonstrate_streaming_chat().await;
    demonstrate_model_management().await;
    show_ollama_benefits().await;

    println!("\n✅ Ollama examples completed!");
    Ok(())
}

/// Test basic Ollama connection
async fn test_ollama_connection() {
    println!("🔌 Testing Ollama Connection:\n");

    match LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.7)
        .build()
        .await
    {
        Ok(client) => {
            println!("   ✅ Ollama client created successfully");

            // Test with a simple request
            let messages = vec![user!(
                "Hello! Please respond with just 'Hi there!' to test the connection."
            )];

            match client.chat(messages).await {
                Ok(response) => {
                    if let Some(text) = response.content_text() {
                        println!("   ✅ Connection test successful");
                        println!("   🤖 Response: {text}");
                    }
                }
                Err(e) => {
                    println!("   ❌ Connection test failed: {e}");
                    print_ollama_troubleshooting();
                }
            }
        }
        Err(e) => {
            println!("   ❌ Failed to create Ollama client: {e}");
            print_ollama_troubleshooting();
        }
    }

    println!();
}

/// Demonstrate basic chat functionality
async fn demonstrate_basic_chat() {
    println!("💬 Basic Chat Functionality:\n");

    match create_ollama_client().await {
        Ok(client) => {
            let messages = vec![user!("Explain what Ollama is in 2-3 sentences.")];

            match client.chat(messages).await {
                Ok(response) => {
                    println!("   User: Explain what Ollama is in 2-3 sentences.");
                    if let Some(text) = response.content_text() {
                        println!("   🦙 Ollama: {text}");
                    }

                    if let Some(usage) = response.usage {
                        println!("   📊 Usage: {} tokens total", usage.total_tokens);
                    }

                    println!("   ✅ Basic chat successful");
                }
                Err(e) => {
                    println!("   ❌ Chat failed: {e}");
                }
            }
        }
        Err(e) => {
            println!("   ❌ Failed to create client: {e}");
        }
    }

    println!();
}

/// Demonstrate streaming chat
async fn demonstrate_streaming_chat() {
    println!("🌊 Streaming Chat:\n");

    match create_ollama_client().await {
        Ok(client) => {
            let messages = vec![user!(
                "Write a short story about a robot learning to paint. Make it about 150 words."
            )];

            println!("   User: Write a short story about a robot learning to paint...");
            println!("   🦙 Ollama (streaming): ");

            match client.chat_stream(messages, None).await {
                Ok(mut stream) => {
                    while let Some(event) = stream.next().await {
                        match event {
                            Ok(ChatStreamEvent::ContentDelta { delta, .. }) => {
                                print!("{delta}");
                                io::stdout().flush().unwrap();
                            }
                            Ok(ChatStreamEvent::Done { .. }) => {
                                println!("\n   ✅ Streaming completed");
                                break;
                            }
                            Err(e) => {
                                println!("\n   ❌ Stream error: {e}");
                                break;
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    println!("   ❌ Streaming failed: {e}");
                }
            }
        }
        Err(e) => {
            println!("   ❌ Failed to create client: {e}");
        }
    }

    println!();
}

/// Demonstrate model management concepts
async fn demonstrate_model_management() {
    println!("🔧 Model Management:\n");

    // Show how to work with different models
    let models_to_try = vec![
        ("llama3.2", "General purpose, good balance"),
        ("llama3.2:1b", "Smaller, faster, less capable"),
        ("codellama", "Specialized for code generation"),
        ("mistral", "Alternative general purpose model"),
    ];

    println!("   📋 Available Model Options:");
    for (model, description) in &models_to_try {
        println!("      • {model}: {description}");
    }

    println!("\n   🧪 Testing Model Availability:");

    // Test the primary model
    match test_model("llama3.2").await {
        Ok(()) => println!("      ✅ llama3.2 is available and working"),
        Err(e) => {
            println!("      ❌ llama3.2 failed: {e}");
            println!("      💡 Try: ollama pull llama3.2");
        }
    }

    // Test smaller model
    if let Ok(()) = test_model("llama3.2:1b").await {
        println!("      ✅ llama3.2:1b is available (faster option)")
    } else {
        println!("      ⚠️  llama3.2:1b not available");
        println!("      💡 Try: ollama pull llama3.2:1b");
    }

    println!("\n   💡 Model Selection Tips:");
    println!("      • Start with llama3.2 for general use");
    println!("      • Use smaller models (1b, 3b) for faster responses");
    println!("      • Use specialized models (codellama) for specific tasks");
    println!("      • Check available models: ollama list");
}

/// Test a specific model
async fn test_model(model_name: &str) -> Result<(), LlmError> {
    let client = LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model(model_name)
        .temperature(0.7)
        .build()
        .await?;

    let messages = vec![user!("Hi")];
    client.chat(messages).await?;
    Ok(())
}

/// Show Ollama benefits and use cases
async fn show_ollama_benefits() {
    println!("🌟 Ollama Benefits:\n");

    println!("   🔒 Privacy & Security:");
    println!("      • All data stays on your machine");
    println!("      • No external API calls");
    println!("      • Perfect for sensitive data");

    println!("\n   💰 Cost Efficiency:");
    println!("      • No per-token charges");
    println!("      • Only hardware costs");
    println!("      • Unlimited usage after setup");

    println!("\n   🌐 Offline Capability:");
    println!("      • Works without internet");
    println!("      • Reliable for air-gapped environments");
    println!("      • No dependency on external services");

    println!("\n   🛠️  Development Benefits:");
    println!("      • Fast iteration during development");
    println!("      • No API key management");
    println!("      • Consistent environment");

    println!("\n   ⚡ Performance:");
    println!("      • Low latency (local processing)");
    println!("      • Scalable with hardware");
    println!("      • Customizable model parameters");

    println!("\n   🎯 Best Use Cases:");
    println!("      • Development and testing");
    println!("      • Privacy-sensitive applications");
    println!("      • High-volume processing");
    println!("      • Offline or air-gapped environments");
    println!("      • Learning and experimentation");
}

/// Helper function to create Ollama client
async fn create_ollama_client() -> Result<siumai::providers::ollama::OllamaClient, LlmError> {
    LlmBuilder::new()
        .ollama()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .temperature(0.7)
        .max_tokens(1000)
        .build()
        .await
}

/// Print troubleshooting information
fn print_ollama_troubleshooting() {
    println!("\n   🔧 Troubleshooting:");
    println!("      1. Install Ollama: https://ollama.ai");
    println!("      2. Start Ollama: ollama serve");
    println!("      3. Pull a model: ollama pull llama3.2");
    println!("      4. Check status: ollama list");
    println!("      5. Verify URL: http://localhost:11434");
}

/// 🎯 Key Ollama Concepts:
///
/// Setup Requirements:
/// - Ollama installed and running
/// - At least one model pulled
/// - Sufficient system resources
///
/// Configuration:
/// - `base_url`: Ollama server URL (default: localhost:11434)
/// - model: Model name (e.g., llama3.2, codellama)
/// - Standard parameters: temperature, `max_tokens`, etc.
///
/// Model Management:
/// - ollama pull <model>: Download models
/// - ollama list: Show available models
/// - ollama rm <model>: Remove models
/// - ollama ps: Show running models
///
/// Best Practices:
/// 1. Start with smaller models for testing
/// 2. Monitor system resources
/// 3. Use appropriate models for tasks
/// 4. Keep models updated
/// 5. Plan storage requirements
///
/// Next Steps:
/// - `advanced_features.rs`: Advanced Ollama configurations
/// - ../../`02_core_features/`: Core functionality with any provider
/// - ../../`05_use_cases/`: Real-world applications
const fn _documentation() {}
